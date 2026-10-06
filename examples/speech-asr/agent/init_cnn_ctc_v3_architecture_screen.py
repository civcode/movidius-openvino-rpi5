#!/usr/bin/env python3
"""Initialize the cnn_ctc_v3 architecture-screen-v1 control."""

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
SCREEN_TOOL = SPEECH_ROOT / "tools" / "prepare_architecture_screen_v1.py"

SCREEN_DIR = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1"
)
SCREEN_MANIFEST = SCREEN_DIR / "train.manifest.jsonl"
SCREEN_PROVENANCE = SCREEN_DIR / "screen.json"
VALIDATION_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"
)

MODEL_SPEC_SHA256 = "beb28c4666687d634aa4a954b0c5cdabd82c373b7d028d6ffa96c9f56aaf275f"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
FULL_TRAIN_MANIFEST_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
DEFAULT_PARENT = "exp-63fdb8d218673527"
SCREEN_ID = "ami-model-quality-v4-architecture-screen-v1"
SCREEN_EPOCHS = 12
EXPECTED_MEETINGS = 48


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


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_parent() -> None:
    parent = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
    attempt = parent / "attempts" / "attempt-0001"
    required = (
        parent / "request" / "model-spec.json",
        attempt / "attempt.json",
        attempt / "results" / "acceptance-evaluation.json",
        attempt / "results" / "result.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "model-quality-v4 v3 parent evidence missing: " + ", ".join(missing)
        )
    model = json.loads(required[0].read_text(encoding="utf-8"))
    attempt_doc = json.loads(required[1].read_text(encoding="utf-8"))
    acceptance = json.loads(required[2].read_text(encoding="utf-8"))
    result = json.loads(required[3].read_text(encoding="utf-8"))
    if model.get("model_id") != "cnn_ctc_v3":
        raise ValueError("architecture-screen parent is not cnn_ctc_v3")
    if attempt_doc.get("outcome") != "completed":
        raise ValueError("architecture-screen parent did not complete")
    if acceptance.get("status") != "accepted":
        raise ValueError("architecture-screen parent acceptance did not pass")
    if result.get("metrics", {}).get("cer") != 0.7588550365720209:
        raise ValueError("architecture-screen parent CER changed")


def validate_screen() -> dict:
    proc = subprocess.run(
        [str(PYTHON), str(SCREEN_TOOL), "--verify-only"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(proc.stdout.strip())
    screen = json.loads(SCREEN_PROVENANCE.read_text(encoding="utf-8"))
    if screen.get("id") != SCREEN_ID or screen.get("status") != "valid":
        raise ValueError("architecture-screen boundary identity changed")
    if screen.get("source", {}).get("train_manifest_sha256") != FULL_TRAIN_MANIFEST_SHA256:
        raise ValueError("architecture-screen source training manifest changed")
    if screen.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen validation manifest changed")
    if screen.get("selection", {}).get("rule_id") != "sha256-id-first-u64-be-mod4-eq0-v1":
        raise ValueError("architecture-screen selection rule changed")
    if screen.get("budget", {}).get("epochs") != SCREEN_EPOCHS:
        raise ValueError("architecture-screen epoch budget changed")
    train_stats = screen.get("train", {}).get("stats", {})
    if len(train_stats.get("meetings", [])) != EXPECTED_MEETINGS:
        raise ValueError("architecture-screen no longer represents all train meetings")
    if not (0 < int(train_stats.get("records", 0)) < 15738):
        raise ValueError("architecture-screen training population is invalid")
    if sha256_path(SCREEN_MANIFEST) != screen["train"]["manifest_sha256"]:
        raise ValueError("architecture-screen train manifest hash mismatch")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen validation manifest hash mismatch")
    return screen


def validate_model() -> dict:
    package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
    vocab = json.loads((MODEL_SPEC.parent / "vocab.json").read_text(encoding="utf-8"))
    if canonical_sha256(package) != MODEL_SPEC_SHA256:
        raise ValueError("cnn_ctc_v3 model spec changed")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v3 vocabulary changed")
    return package


def main() -> int:
    try:
        validate_parent()
        screen = validate_screen()
        package = validate_model()
        head = git_head()
        train_sha = sha256_path(SCREEN_MANIFEST)

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v3 architecture-screen-v1 control",
            "hypothesis": (
                "A deterministic 25-percent training proxy with full ES2011 validation "
                "can provide a faster architecture-ranking reference while preserving "
                "meeting diversity and the model-quality-v4 selection boundary."
            ),
            "rationale": (
                "Full-budget v3/v7 runs take about one hour, while current model error "
                "is still large. Architecture screening therefore prioritizes large, "
                "repeatable directional effects before promotion to larger budgets."
            ),
            "changes": [
                f"branch from full-budget v3 baseline {DEFAULT_PARENT}",
                "keep cnn_ctc_v3 graph, frontend, objective and decoder unchanged",
                "derive training examples by frozen sha256(sample_id) modulo-4 rule",
                "retain examples from all 48 training meetings",
                "use 12 epochs instead of 32",
                "use the full frozen 1273-record ES2011 validation manifest",
                "select checkpoints by validation CER",
                "use no augmentation, blank penalty or auxiliary CTC objective",
                "keep sealed held-out data forbidden for architecture ranking",
            ],
            "notes": (
                "This establishes the proxy reference only. Candidates must be compared "
                "against this control under the exact same screen manifest and budget. "
                "Promising candidates require larger-budget confirmation."
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
        train_config = {
            "schema": "speech-asr/train-config",
            "version": 1,
            "seed": int(package["training"]["seed"]),
            "device": "cuda",
            "epochs": SCREEN_EPOCHS,
            "batch_size": 1,
            "max_samples": None,
            "checkpoint_selection": "validation_cer",
            "optimizer": {
                "kind": "adam",
                "learning_rate": float(package["training"]["learning_rate"]),
            },
            "training_manifest": {
                "id": f"{SCREEN_ID}-train",
                "path": repo_relative(SCREEN_MANIFEST),
                "sha256": train_sha,
            },
            "validation_manifest": {
                "id": "ami-model-quality-v4-es2011-validation",
                "path": repo_relative(VALIDATION_MANIFEST),
                "sha256": VALIDATION_MANIFEST_SHA256,
            },
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
            / f"cnn_ctc_v3-architecture-screen-v1-{head[:8]}-{train_sha[:8]}"
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
            "ami-model-quality-v4-architecture-screen-v1-es2011-validation",
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
    output["screen"] = {
        "id": SCREEN_ID,
        "train_records": screen["train"]["stats"]["records"],
        "train_audio_seconds": screen["train"]["stats"]["audio_seconds"],
        "train_manifest_sha256": screen["train"]["manifest_sha256"],
        "validation_records": screen["validation"]["stats"]["records"],
        "validation_audio_seconds": screen["validation"]["stats"]["audio_seconds"],
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "epochs": SCREEN_EPOCHS,
        "selection_rule": screen["selection"]["rule_id"],
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
