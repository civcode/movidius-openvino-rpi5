#!/usr/bin/env python3
"""Initialize the unchanged cnn_ctc_v3 baseline on model-quality-v4."""

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

QUALITY_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4"
QUALIFICATION = QUALITY_DIR / "qualification.json"
TRAIN_MANIFEST = QUALITY_DIR / "train.manifest.jsonl"
VALIDATION_MANIFEST = QUALITY_DIR / "validation.manifest.jsonl"
POLICY = (
    SPEECH_ROOT / "datasets" / "ami" / "splits" / "model-quality-v4-edinburgh-v1.json"
)
LOCK_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-source-lock"
SOURCE_LOCK = LOCK_DIR / "source-lock.json"

MODEL_SPEC_SHA256 = "beb28c4666687d634aa4a954b0c5cdabd82c373b7d028d6ffa96c9f56aaf275f"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
DEFAULT_PARENT = "exp-87538823d2bf1562"
BOUNDARY_ID = "ami-model-quality-v4-edinburgh-full-corpus-asr-v1"


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
    for path in (
        QUALIFICATION,
        TRAIN_MANIFEST,
        VALIDATION_MANIFEST,
        POLICY,
        SOURCE_LOCK,
        MODEL_SPEC,
    ):
        if not path.is_file():
            raise ValueError(f"model-quality-v4 input missing: {path}")

    package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
    vocab_path = MODEL_SPEC.parent / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if canonical_sha256(package) != MODEL_SPEC_SHA256:
        raise ValueError("cnn_ctc_v3 model spec differs from frozen baseline")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v3 vocabulary differs from frozen baseline")

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if qualification.get("schema") != "speech-asr/model-quality-manifests":
        raise ValueError("model-quality-v4 qualification schema mismatch")
    if (
        qualification.get("version") != 2
        or qualification.get("status") != "structurally_valid"
    ):
        raise ValueError("model-quality-v4 qualification is not structurally_valid")

    boundary = qualification.get("boundary", {})
    if boundary.get("id") != BOUNDARY_ID:
        raise ValueError("model-quality-v4 boundary id mismatch")
    if boundary.get("official_partition") != "Full-corpus-ASR":
        raise ValueError("model-quality-v4 official partition mismatch")
    if boundary.get("policy_sha256") != canonical_sha256(policy):
        raise ValueError("model-quality-v4 policy hash mismatch")
    if boundary.get("source_lock_sha256") != sha256_path(SOURCE_LOCK):
        raise ValueError("model-quality-v4 source lock hash mismatch")
    if boundary.get("excluded_prior_selection_groups") != ["ES2002"]:
        raise ValueError("model-quality-v4 prior-selection exclusion changed")
    if boundary.get("sealed_test_groups") != [
        "EN2002",
        "ES2004",
        "IS1009",
        "TS3003",
    ]:
        raise ValueError("model-quality-v4 sealed-test exclusions changed")
    if boundary.get("training_allowed_on_train") is not True:
        raise ValueError("model-quality-v4 training role mismatch")
    if boundary.get("training_allowed_on_validation") is not False:
        raise ValueError("model-quality-v4 validation must forbid training")
    if boundary.get("checkpoint_selection_allowed_on_validation") is not True:
        raise ValueError("model-quality-v4 validation must allow checkpoint selection")
    if boundary.get("heldout_metrics_allowed_for_model_selection") is not False:
        raise ValueError("sealed held-out metrics must remain forbidden for selection")

    if qualification.get("model", {}).get("spec_sha256") != MODEL_SPEC_SHA256:
        raise ValueError("model-quality-v4 model spec hash mismatch")
    if qualification.get("model", {}).get("vocab_sha256") != VOCAB_SHA256:
        raise ValueError("model-quality-v4 vocabulary hash mismatch")
    if qualification.get("partition") != {
        "record_overlap": 0,
        "meeting_overlap": 0,
    }:
        raise ValueError("model-quality-v4 train/validation partition is not isolated")
    if qualification.get("train", {}).get("manifest_sha256") != sha256_path(
        TRAIN_MANIFEST
    ):
        raise ValueError("model-quality-v4 training manifest hash mismatch")
    if qualification.get("validation", {}).get("manifest_sha256") != sha256_path(
        VALIDATION_MANIFEST
    ):
        raise ValueError("model-quality-v4 validation manifest hash mismatch")

    train_stats = qualification.get("train", {}).get("stats", {})
    validation_stats = qualification.get("validation", {}).get("stats", {})
    if train_stats.get("records", 0) <= 4429:
        raise ValueError("model-quality-v4 training population is not larger than v3")
    if train_stats.get("audio_seconds", 0) <= 7150.12:
        raise ValueError("model-quality-v4 training duration is not larger than v3")
    if set(validation_stats.get("meetings", [])) != {
        "ES2011a",
        "ES2011b",
        "ES2011c",
        "ES2011d",
    }:
        raise ValueError("model-quality-v4 validation meetings must remain ES2011a-d")
    return qualification


def main() -> int:
    try:
        qualification = validate_qualification()
        parent_path = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
        if not parent_path.is_dir():
            raise ValueError(
                f"frozen scaled-data parent is missing: {parent_path}"
            )

        package = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
        head = git_head()
        train_sha = sha256_path(TRAIN_MANIFEST)
        validation_sha = sha256_path(VALIDATION_MANIFEST)
        train_stats = qualification["train"]["stats"]
        validation_stats = qualification["validation"]["stats"]
        boundary = qualification["boundary"]

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v3 model-quality-v4 fresh-development baseline",
            "hypothesis": (
                "Expanding the unchanged v3 training population across the remaining "
                "Edinburgh Full-corpus-ASR training groups and selecting checkpoints "
                "on fresh ES2011 development meetings will provide a materially more "
                "reliable model-development baseline."
            ),
            "rationale": (
                "The prior controlled data expansions improved CER and emission while "
                "the MA2450 runtime retained large latency margin. ES2002 has already "
                "participated in repeated model/data selection, so it is retired. "
                "This experiment changes the development data boundary only."
            ),
            "changes": [
                f"branch from frozen scaled-data reference {DEFAULT_PARENT}",
                (
                    "expand eligible training records from 4429 to "
                    f"{train_stats['records']}"
                ),
                (
                    "expand eligible training audio from 7150.12 to "
                    f"{train_stats['audio_seconds']} seconds"
                ),
                (
                    "train on Full-corpus-ASR Edinburgh SA groups "
                    "ES2003, ES2005-ES2010 and ES2012-ES2016"
                ),
                "retire all ES2002 meetings from model/checkpoint selection",
                "use official Full-corpus-ASR SB meeting family ES2011a-d for validation",
                "keep sealed EN2002/ES2004/IS1009/TS3003 held-out meetings excluded",
                "keep cnn_ctc_v3 graph, logmel-v1 and vocabulary unchanged",
                "keep 32 epochs, Adam/cosine and validation-CER checkpoint selection",
                "use standard CTC with no augmentation or blank bias",
                "retain fixed-shape OpenVINO/MYRIAD compatibility gates",
            ],
            "notes": (
                "This is a new development baseline, not a held-out rerun. "
                f"data_policy_sha256={boundary['policy_sha256']} "
                f"source_lock_sha256={boundary['source_lock_sha256']}. "
                "Sealed held-out metrics are forbidden for model selection."
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
            "id": "ami-model-quality-v4-train",
            "path": repo_relative(TRAIN_MANIFEST),
            "sha256": train_sha,
        }
        validation_ref = {
            "id": "ami-model-quality-v4-es2011-validation",
            "path": repo_relative(VALIDATION_MANIFEST),
            "sha256": validation_sha,
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
            / f"cnn_ctc_v3-model-quality-v4-{head[:8]}-{train_sha[:8]}"
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
            "ami-model-quality-v4-es2011-validation",
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
        "boundary_id": BOUNDARY_ID,
        "train_records": train_stats["records"],
        "train_audio_seconds": train_stats["audio_seconds"],
        "validation_records": validation_stats["records"],
        "validation_audio_seconds": validation_stats["audio_seconds"],
        "train_manifest_sha256": train_sha,
        "validation_manifest_sha256": validation_sha,
        "policy_sha256": boundary["policy_sha256"],
        "source_lock_sha256": boundary["source_lock_sha256"],
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
