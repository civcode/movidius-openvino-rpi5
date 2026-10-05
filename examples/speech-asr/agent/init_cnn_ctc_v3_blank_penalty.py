#!/usr/bin/env python3
"""Initialize the cnn_ctc_v3 blank-logit CTC objective diagnostic."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import canonical_sha256  # noqa: E402
from speech_asr.orchestration import V3_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

QUALITY_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v1"
QUALIFICATION = QUALITY_DIR / "qualification.json"
TRAIN_MANIFEST = QUALITY_DIR / "train.manifest.jsonl"
VALIDATION_MANIFEST = QUALITY_DIR / "validation.manifest.jsonl"

SOURCE_MANIFEST_SHA256 = "9f4444c6c0e54cf25d92a69723727bb46a73632e34ec33402c71beca1b64cd3e"
TRAIN_MANIFEST_SHA256 = "89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a"
VALIDATION_MANIFEST_SHA256 = "07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923"
MODEL_SPEC_SHA256 = "beb28c4666687d634aa4a954b0c5cdabd82c373b7d028d6ffa96c9f56aaf275f"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"


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


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_qualification() -> dict:
    for path in (QUALIFICATION, TRAIN_MANIFEST, VALIDATION_MANIFEST):
        if not path.is_file():
            raise ValueError(f"model-quality input missing: {path}")

    if sha256_path(TRAIN_MANIFEST) != TRAIN_MANIFEST_SHA256:
        raise ValueError("model-quality training manifest hash differs from reviewed baseline")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("model-quality validation manifest hash differs from reviewed baseline")

    package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
    vocab_path = MODEL_SPEC.parent / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if canonical_sha256(package) != MODEL_SPEC_SHA256:
        raise ValueError("checked-in cnn_ctc_v3 model spec hash differs from reviewed baseline")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("checked-in cnn_ctc_v3 vocabulary hash differs from reviewed baseline")

    value = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if value.get("status") != "structurally_valid":
        raise ValueError("model-quality qualification is not structurally_valid")
    if value.get("source", {}).get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("model-quality source manifest hash differs from reviewed baseline")
    if value.get("model", {}).get("spec_sha256") != MODEL_SPEC_SHA256:
        raise ValueError("model-quality model spec hash differs from reviewed baseline")
    if value.get("model", {}).get("vocab_sha256") != VOCAB_SHA256:
        raise ValueError("model-quality vocabulary hash differs from reviewed baseline")
    if value.get("partition") != {
        "record_overlap": 0,
        "speaker_overlap": 0,
        "train_speakers": ["A", "B", "C"],
        "validation_speakers": ["D"],
    }:
        raise ValueError("model-quality speaker partition differs from reviewed baseline")
    if value.get("train", {}).get("manifest_sha256") != TRAIN_MANIFEST_SHA256:
        raise ValueError("qualification training hash differs from reviewed baseline")
    if value.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("qualification validation hash differs from reviewed baseline")
    if value.get("train", {}).get("stats", {}).get("records") != 125:
        raise ValueError("reviewed model-quality training record count must remain 125")
    if value.get("validation", {}).get("stats", {}).get("records") != 95:
        raise ValueError("reviewed model-quality validation record count must remain 95")
    return value


def main() -> int:
    try:
        qualification = validate_qualification()
        parent_path = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
        if not parent_path.is_dir():
            raise ValueError(
                f"accepted CER-selection parent is missing: {parent_path}"
            )

        package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
        head = git_head()

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v3 blank-logit CTC objective diagnostic",
            "hypothesis": (
                "Holding the accepted cnn_ctc_v3 graph, logmel-v1 frontend, data, "
                "optimizer schedule and validation-CER checkpoint selection fixed "
                "while subtracting a mild constant from the CTC blank logit during "
                "training will reduce blank-dominated alignments and improve held-out "
                "character accuracy and emission."
            ),
            "rationale": (
                "The accepted CER-selected v3 model remains strongly under-emitting "
                "with 61.33 percent blank argmax frames and only 47.98 percent as many "
                "emitted non-space characters as the reference. SpecAugment increased "
                "blank collapse, and valid-frame CMVN also regressed CER and empty "
                "hypotheses. The next isolated test therefore changes the CTC training "
                "objective rather than the encoder, frontend, decoder or hardware graph."
            ),
            "changes": [
                "branch from accepted exp-3c7727ca3f37ba2c",
                "reuse exact reviewed model-quality-v1 train/validation manifests",
                "keep cnn_ctc_v3 graph and logmel-v1 frontend unchanged",
                "keep 32 epochs, Adam/cosine schedule and CER checkpoint selection",
                "apply blank-logit-penalty-v1 with penalty 0.25 during training CTC loss only",
                "leave validation CTC loss, greedy decoding, export and inference logits unmodified",
                "use no SpecAugment",
                "retain fixed-shape OpenVINO/MYRIAD compatibility and latency gates",
            ],
            "notes": (
                "A 0.25 logit penalty is intentionally conservative. This diagnostic "
                "tests training alignment pressure only; it is not an inference-time "
                "blank bias and does not change the deployed MA2450 graph."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v3",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V3_ARCHITECTURE),
            "export": {
                "format": "onnx",
                "onnx_opset": 11,
                "fixed_shapes": True,
            },
        }
        train_ref = {
            "id": "ami-model-quality-v1-train",
            "path": repo_relative(TRAIN_MANIFEST),
            "sha256": TRAIN_MANIFEST_SHA256,
        }
        validation_ref = {
            "id": "ami-model-quality-v1-validation",
            "path": repo_relative(VALIDATION_MANIFEST),
            "sha256": VALIDATION_MANIFEST_SHA256,
        }
        train_config = {
            "schema": "speech-asr/train-config",
            "version": 1,
            "seed": int(package["training"]["seed"]),
            "device": "cuda",
            "epochs": int(package["training"]["epochs"]),
            "batch_size": int(package["training"]["batch_size"]),
            "max_samples": None,
            "checkpoint_selection": "validation_cer",
            "ctc_objective": {
                "kind": "blank-logit-penalty-v1",
                "blank_logit_penalty": 0.25,
            },
            "optimizer": {
                "kind": "adam",
                "learning_rate": float(package["training"]["learning_rate"]),
            },
            "training_manifest": train_ref,
            "validation_manifest": validation_ref,
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
                "max_realtime_factor": None,
                "max_latency_p95_ms": 25.0,
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
            / f"cnn_ctc_v3-blank-penalty-{head[:8]}-{VALIDATION_MANIFEST_SHA256[:8]}"
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
            "ami-model-quality-v1-validation",
            "--manifest",
            str(VALIDATION_MANIFEST),
            "--repo-commit",
            head,
            "--parent",
            DEFAULT_PARENT,
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

    output = json.loads(proc.stdout)
    output["qualification"] = {
        "train_records": qualification["train"]["stats"]["records"],
        "validation_records": qualification["validation"]["stats"]["records"],
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
