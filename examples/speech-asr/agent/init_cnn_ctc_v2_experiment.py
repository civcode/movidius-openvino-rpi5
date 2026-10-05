#!/usr/bin/env python3
"""Create the reviewed Phase 11 generation-1 cnn_ctc_v2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.orchestration import V2_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v2" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"
DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "ami-smoke-v1" / "manifest.jsonl"
)
DEFAULT_PARENT = "exp-f915ec624a63caf6"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def write_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    args = parser.parse_args()

    try:
        if not args.manifest.is_file():
            raise ValueError(f"manifest missing: {args.manifest}")
        parent_path = (
            ROOT / "work" / "speech-asr" / "experiments" / args.parent
        )
        if not parent_path.is_dir():
            raise ValueError(
                f"Phase 11 parent experiment is missing locally: {parent_path}"
            )

        package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
        manifest_sha = sha256_path(args.manifest)
        head = git_head()

        seed = int(package["training"]["seed"]) if args.seed is None else args.seed
        epochs = int(package["training"]["epochs"]) if args.epochs is None else args.epochs
        batch_size = (
            int(package["training"]["batch_size"])
            if args.batch_size is None
            else args.batch_size
        )
        learning_rate = (
            float(package["training"]["learning_rate"])
            if args.learning_rate is None
            else args.learning_rate
        )
        if seed < 0 or epochs < 0 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("invalid training override")

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "Phase 11 generation 1: residual temporal cnn_ctc_v2",
            "hypothesis": (
                "Moving encoder depth to the 4x-downsampled frame rate and "
                "adding large-kernel residual temporal context will improve "
                "cnn_ctc_v1 learnability without sacrificing MA2450 realtime "
                "execution."
            ),
            "rationale": (
                "FastConformer/Zipformer motivate low-rate encoder compute; "
                "QuartzNet/Citrinet motivate residual temporal CTC. The design "
                "uses ordinary convolution instead of grouped/depthwise paths "
                "to respect the pinned OpenVINO 2020.3/MYRIAD envelope."
            ),
            "changes": [
                "replace three-layer encoder with two stride-2 stems plus five residual temporal blocks",
                "increase encoder width to 96 channels after temporal reduction",
                "use large kernels 11,19,27,35,43 for full-window receptive field",
                "add BatchNorm and training-only dropout while retaining ReLU and fixed shapes"
            ],
            "notes": (
                "Generation 1 changes encoder architecture only; frontend, "
                "vocabulary, CTC decoder, benchmark and one-epoch training "
                "policy remain controlled against the accepted v1 lineage."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v2",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V2_ARCHITECTURE),
            "export": {
                "format": "onnx",
                "onnx_opset": 11,
                "fixed_shapes": True,
            },
        }
        manifest_ref = {
            "id": "ami-smoke-v1",
            "path": args.manifest.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": manifest_sha,
        }
        train_config = {
            "schema": "speech-asr/train-config",
            "version": 1,
            "seed": seed,
            "device": args.device,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_samples": args.max_samples,
            "optimizer": {
                "kind": "adam",
                "learning_rate": learning_rate,
            },
            "training_manifest": dict(manifest_ref),
            "validation_manifest": dict(manifest_ref),
        }
        acceptance = {
            "schema": "speech-asr/acceptance-policy",
            "version": 1,
            "required_gates": [
                "training",
                "onnx_export",
                "openvino_conversion",
                "myriad_execution",
                "accuracy_evaluation",
            ],
            "thresholds": {
                "max_wer": 1.0,
                "max_cer": 0.913043,
                "max_realtime_factor": 0.05,
                "max_latency_p95_ms": 100.0,
                "min_frame_argmax_agreement": 1.0,
            },
            "retry_policy": {
                "max_attempts": 3,
                "retryable_failure_classes": [
                    "transport_preflight",
                    "worker_busy",
                    "hardware_transient",
                ],
            },
        }

        draft = (
            ROOT
            / "work"
            / "speech-asr"
            / "experiment-drafts"
            / f"cnn_ctc_v2-{head[:8]}-{manifest_sha[:8]}"
        )
        paths = {
            "proposal": draft / "proposal.json",
            "model": draft / "model-spec.json",
            "train": draft / "train-config.json",
            "acceptance": draft / "acceptance.json",
        }
        write_json(paths["proposal"], proposal)
        write_json(paths["model"], experiment_model)
        write_json(paths["train"], train_config)
        write_json(paths["acceptance"], acceptance)

        command = [
            str(PYTHON),
            str(MANAGER),
            "init",
            "--proposal",
            str(paths["proposal"]),
            "--model-spec",
            str(paths["model"]),
            "--train-config",
            str(paths["train"]),
            "--acceptance",
            str(paths["acceptance"]),
            "--benchmark-id",
            "ami-smoke-v1",
            "--manifest",
            str(args.manifest),
            "--repo-commit",
            head,
            "--parent",
            args.parent,
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout.strip())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(proc.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
