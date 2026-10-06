#!/usr/bin/env python3
"""Initialize the model-quality-v4 cnn_ctc_v7 112-channel capacity ablation."""

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

from speech_asr.cnn_ctc import (  # noqa: E402
    canonical_sha256,
    load_spec,
    load_vocab,
    model_resource_estimate,
)
from speech_asr.orchestration import V7_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v7" / "model_spec.json"
V3_MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

QUALITY_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4"
QUALIFICATION = QUALITY_DIR / "qualification.json"
TRAIN_MANIFEST = QUALITY_DIR / "train.manifest.jsonl"
VALIDATION_MANIFEST = QUALITY_DIR / "validation.manifest.jsonl"

MODEL_SPEC_SHA256 = "a4e53c89044dad473fb321e2187f5de598f8623b66de42d29e2c2c775c82ebc8"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
TRAIN_MANIFEST_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
TRAIN_RECORDS = 15738
TRAIN_AUDIO_SECONDS = 25846.327
VALIDATION_RECORDS = 1273
VALIDATION_AUDIO_SECONDS = 2279.385
DEFAULT_PARENT = "exp-63fdb8d218673527"
PARENT_ATTEMPT = "attempt-0001"
PARENT_CER = 0.7588550365720209
PARENT_WER = 1.0427553444180522
PARENT_P95_MS = 16.7812226
EXPECTED_PARAMETERS = 1818183
EXPECTED_MACS = 235177984
EXPECTED_RECEPTIVE_FIELD = 533


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def validate_parent() -> dict:
    parent = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
    result_path = parent / "attempts" / PARENT_ATTEMPT / "results" / "result.json"
    if not result_path.is_file():
        raise ValueError(f"reviewed model-quality-v4 parent result is missing: {result_path}")
    result = load_json(result_path)
    if result.get("outcome") != "completed" or result.get("acceptance") != "accepted":
        raise ValueError("model-quality-v4 parent is not a completed accepted execution")
    if result.get("model") != "cnn_ctc_v3":
        raise ValueError("model-quality-v4 parent model is not cnn_ctc_v3")
    benchmark = result.get("benchmark", {})
    if benchmark.get("id") != "ami-model-quality-v4-es2011-validation":
        raise ValueError("model-quality-v4 parent benchmark id changed")
    if benchmark.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("model-quality-v4 parent validation manifest changed")
    metrics = result.get("metrics", {})
    if metrics.get("cer") != PARENT_CER:
        raise ValueError("model-quality-v4 parent CER changed")
    if metrics.get("wer") != PARENT_WER:
        raise ValueError("model-quality-v4 parent WER changed")
    if metrics.get("inference_latency_p95_ms") != PARENT_P95_MS:
        raise ValueError("model-quality-v4 parent p95 latency changed")
    return result


def validate_inputs() -> dict:
    for path in (MODEL_SPEC, V3_MODEL_SPEC, QUALIFICATION, TRAIN_MANIFEST, VALIDATION_MANIFEST):
        if not path.is_file():
            raise ValueError(f"cnn_ctc_v7 input missing: {path}")

    if sha256_path(TRAIN_MANIFEST) != TRAIN_MANIFEST_SHA256:
        raise ValueError("model-quality-v4 training manifest changed")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("model-quality-v4 validation manifest changed")

    package = load_spec(MODEL_SPEC)
    base = load_spec(V3_MODEL_SPEC)
    vocab = load_vocab(MODEL_SPEC.parent / "vocab.json")
    if canonical_sha256(package) != MODEL_SPEC_SHA256:
        raise ValueError("cnn_ctc_v7 model spec differs from reviewed design")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v7 vocabulary differs from frozen v3 vocabulary")
    if package["frontend"] != base["frontend"]:
        raise ValueError("cnn_ctc_v7 frontend differs from cnn_ctc_v3")
    if package["input_contract"] != base["input_contract"]:
        raise ValueError("cnn_ctc_v7 input contract differs from cnn_ctc_v3")
    if package["output_contract"] != base["output_contract"]:
        raise ValueError("cnn_ctc_v7 output contract differs from cnn_ctc_v3")
    if package["decoder"] != base["decoder"]:
        raise ValueError("cnn_ctc_v7 decoder differs from cnn_ctc_v3")
    if package["training"] != base["training"]:
        raise ValueError("cnn_ctc_v7 package training defaults differ from cnn_ctc_v3")

    network = package["network"]
    base_network = base["network"]
    if network["kind"] != base_network["kind"]:
        raise ValueError("cnn_ctc_v7 changed the v3 operator topology kind")
    if network["normalization"] != "none":
        raise ValueError("cnn_ctc_v7 must remain normalization-free")
    if network["activation"] != base_network["activation"]:
        raise ValueError("cnn_ctc_v7 activation differs from cnn_ctc_v3")
    if network["dropout"] != base_network["dropout"]:
        raise ValueError("cnn_ctc_v7 dropout differs from cnn_ctc_v3")
    if network["residual_projection_init"] != base_network["residual_projection_init"]:
        raise ValueError("cnn_ctc_v7 residual initialization differs from cnn_ctc_v3")
    if [layer["kernel"] for layer in network["stem"]] != [5, 5]:
        raise ValueError("cnn_ctc_v7 stem kernels changed")
    if [layer["stride"] for layer in network["stem"]] != [2, 2]:
        raise ValueError("cnn_ctc_v7 stem strides changed")
    if [layer["channels"] for layer in network["stem"]] != [64, 112]:
        raise ValueError("cnn_ctc_v7 stem width must be [64,112]")
    if [block["kernel"] for block in network["residual_blocks"]] != [11, 19, 27, 35, 43]:
        raise ValueError("cnn_ctc_v7 residual kernel schedule changed")
    if any(block["channels"] != 112 for block in network["residual_blocks"]):
        raise ValueError("cnn_ctc_v7 residual width must remain 112")

    estimate = model_resource_estimate(package)
    if estimate["parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v7 parameter estimate changed")
    if estimate["macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v7 MAC estimate changed")
    if estimate["receptive_field_feature_frames"] != EXPECTED_RECEPTIVE_FIELD:
        raise ValueError("cnn_ctc_v7 receptive field changed")
    if network["estimated_parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v7 declared parameter count changed")
    if network["estimated_macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v7 declared MAC count changed")

    qualification = load_json(QUALIFICATION)
    if qualification.get("status") != "structurally_valid":
        raise ValueError("model-quality-v4 qualification is not structurally_valid")
    if qualification.get("train", {}).get("manifest_sha256") != TRAIN_MANIFEST_SHA256:
        raise ValueError("model-quality-v4 qualified training hash changed")
    if qualification.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("model-quality-v4 qualified validation hash changed")
    if qualification.get("train", {}).get("stats", {}).get("records") != TRAIN_RECORDS:
        raise ValueError("model-quality-v4 training record count changed")
    if qualification.get("train", {}).get("stats", {}).get("audio_seconds") != TRAIN_AUDIO_SECONDS:
        raise ValueError("model-quality-v4 training duration changed")
    if qualification.get("validation", {}).get("stats", {}).get("records") != VALIDATION_RECORDS:
        raise ValueError("model-quality-v4 validation record count changed")
    if qualification.get("validation", {}).get("stats", {}).get("audio_seconds") != VALIDATION_AUDIO_SECONDS:
        raise ValueError("model-quality-v4 validation duration changed")
    boundary = qualification.get("boundary", {})
    if boundary.get("id") != "ami-model-quality-v4-edinburgh-full-corpus-asr-v1":
        raise ValueError("model-quality-v4 boundary id changed")
    if boundary.get("heldout_metrics_allowed_for_model_selection") is not False:
        raise ValueError("sealed held-out metrics must remain forbidden for selection")
    if qualification.get("partition") != {"record_overlap": 0, "meeting_overlap": 0}:
        raise ValueError("model-quality-v4 partition isolation changed")
    return qualification


def main() -> int:
    try:
        qualification = validate_inputs()
        parent = validate_parent()
        package = load_spec(MODEL_SPEC)
        head = git_head()

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v7 112-channel capacity ablation on model-quality-v4",
            "hypothesis": (
                "Widening only the v3 encoder from 96 to 112 residual channels will "
                "reduce ES2011 CER and under-emission while preserving the fixed "
                "OpenVINO/MYRIAD tensor contract and staying below 25 ms p95."
            ),
            "rationale": (
                "The fresh model-quality-v4 baseline has CER 0.758855, 69.07% blank "
                "frames, 15.79% empty hypotheses and 58.96% emitted/reference "
                "characters, while MA2450 p95 is only 16.78 ms. Previous controlled "
                "SpecAugment, valid-frame CMVN, blank-bias and InterCTC directions "
                "were negative on the earlier boundary. Capacity is therefore the "
                "next isolated variable."
            ),
            "changes": [
                f"branch from model-quality-v4 baseline {DEFAULT_PARENT}",
                "reuse exact frozen model-quality-v4 train/ES2011 manifests",
                "keep logmel-v1 frontend and fixed [1,64,512] input unchanged",
                "keep two stride-2 stems and residual kernels [11,19,27,35,43]",
                "widen second stem and all residual blocks from 96 to 112 channels",
                f"increase deployed parameters from 1346343 to {EXPECTED_PARAMETERS}",
                f"increase fixed-input MACs from 174804992 to {EXPECTED_MACS}",
                "keep 533-frame receptive field, standard CTC and greedy decoder",
                "keep 32 epochs, batch 1, Adam/cosine and validation-CER selection",
                "use no SpecAugment, blank-logit penalty or intermediate CTC",
                "retain exact ONNX/OpenVINO numerical and physical MYRIAD gates",
            ],
            "notes": (
                "This experiment may use only model-quality-v4 validation evidence. "
                "The sealed held-out corpus remains consumed and is forbidden for "
                "architecture selection."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v7",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V7_ARCHITECTURE),
            "export": {
                "format": "onnx",
                "onnx_opset": 11,
                "fixed_shapes": True,
            },
        }
        train_ref = {
            "id": "ami-model-quality-v4-train",
            "path": repo_relative(TRAIN_MANIFEST),
            "sha256": TRAIN_MANIFEST_SHA256,
        }
        validation_ref = {
            "id": "ami-model-quality-v4-es2011-validation",
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
                "max_cer": PARENT_CER,
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
            / f"cnn_ctc_v7-wide112-{head[:8]}-{VALIDATION_MANIFEST_SHA256[:8]}"
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
    output["comparison_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "cer": parent["metrics"]["cer"],
        "wer": parent["metrics"]["wer"],
        "latency_p95_ms": parent["metrics"]["inference_latency_p95_ms"],
    }
    output["qualification"] = {
        "train_records": qualification["train"]["stats"]["records"],
        "train_audio_seconds": qualification["train"]["stats"]["audio_seconds"],
        "validation_records": qualification["validation"]["stats"]["records"],
        "validation_audio_seconds": qualification["validation"]["stats"]["audio_seconds"],
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
