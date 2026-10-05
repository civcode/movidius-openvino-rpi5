#!/usr/bin/env python3
"""Initialize cnn_ctc_v3 with expanded meeting-disjoint AMI training data."""

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

QUALITY_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v2"
QUALIFICATION = QUALITY_DIR / "qualification.json"
TRAIN_MANIFEST = QUALITY_DIR / "train.manifest.jsonl"
VALIDATION_MANIFEST = (
    ROOT
    / "work"
    / "speech-asr"
    / "ami"
    / "model-quality-v1"
    / "validation.manifest.jsonl"
)
TRAIN_SPLIT_SPEC = (
    SPEECH_ROOT / "datasets" / "ami" / "splits" / "train-es2005-v1.json"
)

TRAIN_SOURCE_MANIFEST_SHA256 = "4e4ba1b8df2e3041b9dbbf654b761777abd256c4a272f8103a24575e49114c6b"
TRAIN_MANIFEST_SHA256 = "abb4ed8fcf054497e1d81b63acfc8aa42ad0a49e542184581299be8496fefbb6"
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
    for path in (
        QUALIFICATION,
        TRAIN_MANIFEST,
        VALIDATION_MANIFEST,
        TRAIN_SPLIT_SPEC,
    ):
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

    split = json.loads(TRAIN_SPLIT_SPEC.read_text(encoding="utf-8"))
    if split.get("expected") != {
        "records": 1865,
        "manifest_sha256": TRAIN_SOURCE_MANIFEST_SHA256,
        "logical_tree_sha256": "74ee9c46a7b5c68a5313b0c7c7f8e7f5262209cd991a05ebfbb60efce58269bd",
    }:
        raise ValueError("expanded AMI split identity differs from reviewed data")

    value = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if value.get("version") != 2 or value.get("status") != "structurally_valid":
        raise ValueError("model-quality-v2 qualification is not structurally_valid")
    if (
        value.get("sources", {}).get("train", {}).get("manifest_sha256")
        != TRAIN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("expanded training source differs from reviewed identity")
    if (
        value.get("sources", {}).get("validation", {}).get("manifest_sha256")
        != VALIDATION_MANIFEST_SHA256
    ):
        raise ValueError("validation source differs from accepted v1 identity")
    if value.get("model", {}).get("spec_sha256") != MODEL_SPEC_SHA256:
        raise ValueError("model-quality model spec hash differs from reviewed baseline")
    if value.get("model", {}).get("vocab_sha256") != VOCAB_SHA256:
        raise ValueError("model-quality vocabulary hash differs from reviewed baseline")
    if value.get("partition") != {
        "record_overlap": 0,
        "meeting_overlap": 0,
    }:
        raise ValueError("model-quality-v2 meeting partition differs from review")
    if value.get("train", {}).get("manifest_sha256") != TRAIN_MANIFEST_SHA256:
        raise ValueError("qualification training hash differs from reviewed baseline")
    if value.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("qualification validation hash differs from reviewed baseline")
    if value.get("train", {}).get("stats", {}).get("records") != 1487:
        raise ValueError("expanded training record count must remain 1487")
    if value.get("train", {}).get("stats", {}).get("audio_seconds") != 2393.989:
        raise ValueError("expanded training duration differs from review")
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
            "title": "cnn_ctc_v3 expanded meeting-disjoint AMI training",
            "hypothesis": (
                "Increasing training from 220 to 2394 seconds and separating training "
                "meetings from the frozen ES2002a validation meeting will materially "
                "improve v3 CER and emission without changing model semantics."
            ),
            "rationale": (
                "Blank penalty, frontend correction, augmentation and two InterCTC "
                "encoder designs all failed on the 125-utterance training boundary. "
                "The accepted v3 graph is restored while training expands to 1487 "
                "eligible utterances from official AMI scenario session ES2005."
            ),
            "changes": [
                "branch from accepted exp-3c7727ca3f37ba2c",
                "expand training from 125 to 1487 eligible utterances",
                "expand training audio from 219.998 to 2393.989 seconds",
                "train on ES2005a,b,c,d with zero validation meeting overlap",
                "retain the exact ES2002a speaker-D validation manifest",
                "keep cnn_ctc_v3 graph, logmel-v1 and vocabulary unchanged",
                "keep 32 epochs, Adam/cosine and validation-CER selection",
                "use standard CTC with no augmentation or blank bias",
                "retain fixed-shape OpenVINO/MYRIAD compatibility gates",
            ],
            "notes": (
                "This changes only the reviewed training population. The validation "
                "manifest and hardware benchmark are byte-identical to the v3 reference."
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
            "id": "ami-model-quality-v2-es2005-train",
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
            / f"cnn_ctc_v3-expanded-data-{head[:8]}-{TRAIN_MANIFEST_SHA256[:8]}"
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
