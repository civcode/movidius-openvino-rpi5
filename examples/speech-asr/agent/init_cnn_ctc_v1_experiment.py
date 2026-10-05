#!/usr/bin/env python3
"""Create the reviewed baseline cnn_ctc_v1 request used to accept Phase 10."""

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
MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v1" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"
DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "ami-smoke-v1" / "manifest.jsonl"
)


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
    parser.add_argument("--parent")
    args = parser.parse_args()

    try:
        if not args.manifest.is_file():
            raise ValueError(f"manifest missing: {args.manifest}")
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
            "title": "Phase 10 cnn_ctc_v1 orchestration acceptance",
            "hypothesis": (
                "The frozen cnn_ctc_v1 lifecycle can be executed from oberon "
                "through the dedicated edge MYRIAD worker without manual handoff."
            ),
            "rationale": (
                "Phase 8 qualified the model lifecycle and Phase 9 froze the "
                "experiment/deployment contracts; this run qualifies orchestration."
            ),
            "changes": [
                "execute the already-qualified cnn_ctc_v1 through Phase 10 orchestration"
            ],
            "notes": "Bootstrap/acceptance experiment; no architecture mutation.",
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v1",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": {"kind": "cnn_ctc_v1"},
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
                "max_wer": None,
                "max_cer": None,
                "max_realtime_factor": 1.0,
                "max_latency_p95_ms": None,
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
            / f"cnn_ctc_v1-{head[:8]}-{manifest_sha[:8]}"
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
        ]
        if args.parent:
            command.extend(["--parent", args.parent])
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
