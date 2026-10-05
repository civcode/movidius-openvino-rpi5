#!/usr/bin/env python3
"""Initialize cnn_ctc_v4 valid-frame CMVN diagnostic."""

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

from speech_asr.cnn_ctc import canonical_sha256, load_spec, load_vocab  # noqa: E402
from speech_asr.orchestration import V4_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v4" / "model_spec.json"
V3_MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

QUALITY_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v1"
QUALIFICATION = QUALITY_DIR / "qualification.json"
TRAIN_MANIFEST = QUALITY_DIR / "train.manifest.jsonl"
VALIDATION_MANIFEST = QUALITY_DIR / "validation.manifest.jsonl"

SOURCE_MANIFEST_SHA256 = "9f4444c6c0e54cf25d92a69723727bb46a73632e34ec33402c71beca1b64cd3e"
TRAIN_MANIFEST_SHA256 = "89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a"
VALIDATION_MANIFEST_SHA256 = "07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923"
QUALIFIED_V3_SPEC_SHA256 = "beb28c4666687d634aa4a954b0c5cdabd82c373b7d028d6ffa96c9f56aaf275f"
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


def validate_quality_inputs() -> dict:
    for path in (QUALIFICATION, TRAIN_MANIFEST, VALIDATION_MANIFEST):
        if not path.is_file():
            raise ValueError(f"model-quality input missing: {path}")
    if sha256_path(TRAIN_MANIFEST) != TRAIN_MANIFEST_SHA256:
        raise ValueError("training manifest hash differs from reviewed identity")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest hash differs from reviewed identity")

    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if qualification.get("status") != "structurally_valid":
        raise ValueError("model-quality qualification is not structurally_valid")
    if qualification.get("source", {}).get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("qualification source hash differs from reviewed identity")
    if qualification.get("model", {}).get("spec_sha256") != QUALIFIED_V3_SPEC_SHA256:
        raise ValueError("qualification no longer matches reviewed v3 eligibility spec")
    if qualification.get("model", {}).get("vocab_sha256") != VOCAB_SHA256:
        raise ValueError("qualification vocabulary hash differs from reviewed identity")
    if qualification.get("partition") != {
        "record_overlap": 0,
        "speaker_overlap": 0,
        "train_speakers": ["A", "B", "C"],
        "validation_speakers": ["D"],
    }:
        raise ValueError("model-quality speaker partition differs from reviewed identity")
    if qualification.get("train", {}).get("manifest_sha256") != TRAIN_MANIFEST_SHA256:
        raise ValueError("qualification training hash differs from reviewed identity")
    if qualification.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("qualification validation hash differs from reviewed identity")

    v3 = load_spec(V3_MODEL_SPEC)
    v4 = load_spec(MODEL_SPEC)
    v4_vocab = load_vocab(MODEL_SPEC.parent / "vocab.json")
    if canonical_sha256(v4_vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v4 vocabulary differs from reviewed identity")

    # Eligibility depends on fixed audio/feature geometry, output-time geometry,
    # and vocabulary. Those contracts must remain identical to the qualified
    # v3 manifests; only frontend normalization semantics may change.
    frontend_keys = (
        "sample_rate_hz",
        "window_samples",
        "hop_samples",
        "fft_size",
        "mel_bins",
        "fixed_frames",
        "fixed_audio_samples",
    )
    for key in frontend_keys:
        if v4["frontend"][key] != v3["frontend"][key]:
            raise ValueError(f"cnn_ctc_v4 frontend geometry changed at {key}")
    for key in ("input_contract", "output_contract", "network"):
        if v4[key] != v3[key]:
            raise ValueError(f"cnn_ctc_v4 {key} must match frozen v3 graph")
    if v4["frontend"]["kind"] != "logmel-v2":
        raise ValueError("cnn_ctc_v4 frontend kind must be logmel-v2")
    if (
        v4["frontend"]["normalization"]
        != "per_mel_bin_mean_valid_zero_pad"
    ):
        raise ValueError("cnn_ctc_v4 valid-frame normalization policy changed")
    return qualification


def main() -> int:
    try:
        qualification = validate_quality_inputs()
        parent_path = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
        if not parent_path.is_dir():
            raise ValueError(
                f"accepted CER-selection parent is missing: {parent_path}"
            )

        package = load_spec(MODEL_SPEC)
        head = git_head()

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v4 valid-frame CMVN correction",
            "hypothesis": (
                "Keeping the accepted cnn_ctc_v3 residual temporal graph, data, "
                "optimizer schedule and CER-aligned checkpoint selection fixed while "
                "excluding fixed-shape padding from per-mel-bin mean normalization "
                "will improve held-out emission and character accuracy."
            ),
            "rationale": (
                "logmel-v1 normalizes after padding every utterance to 512 feature "
                "frames, so synthetic padding contributes to CMVN. On the held-out "
                "speaker set, average clips are much shorter than the fixed window. "
                "SpecAugment increased blank collapse, so the next change corrects "
                "the host frontend rather than adding more regularization."
            ),
            "changes": [
                "branch from accepted exp-3c7727ca3f37ba2c rather than negative SpecAugment sibling",
                "keep the v3 residual temporal Conv1D/ReLU graph unchanged",
                "replace logmel-v1 with logmel-v2 valid-frame CMVN",
                "compute per-mel-bin mean from real feature frames only",
                "set padded normalized feature frames exactly to zero",
                "reuse exact reviewed model-quality-v1 train/validation manifests",
                "retain 32 epochs and validation-CER checkpoint selection",
                "retain fixed-shape OpenVINO/MYRIAD latency and compatibility gates",
            ],
            "notes": (
                "This is a frontend-semantic generation, not a larger hardware graph. "
                "The validation set is still same-meeting speaker holdout and is not "
                "an unbiased final generalization benchmark."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v4",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v2"},
            "architecture": dict(V4_ARCHITECTURE),
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
            / f"cnn_ctc_v4-valid-cmvn-{head[:8]}-{VALIDATION_MANIFEST_SHA256[:8]}"
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
        "frontend": "logmel-v2",
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
