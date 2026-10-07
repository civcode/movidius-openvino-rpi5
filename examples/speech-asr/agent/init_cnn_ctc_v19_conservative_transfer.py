#!/usr/bin/env python3
"""Initialize cnn_ctc_v19 pretrained QuartzNet transfer/fine-tuning."""

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

from speech_asr.cnn_ctc import canonical_sha256, load_spec, load_vocab, model_resource_estimate  # noqa: E402
from speech_asr.orchestration import V19_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
VOCAB = MODEL_SPEC.parent / "vocab.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"
SCREEN_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1"
SCREEN_MANIFEST = SCREEN_DIR / "train.manifest.jsonl"
SCREEN_PROVENANCE = SCREEN_DIR / "screen.json"
VALIDATION_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"
PRETRAINED = ROOT / "work" / "speech-asr" / "pretrained" / "quartznet15x5-en-base-v2" / "QuartzNet15x5-En-Base.nemo"
PRETRAINED_IMPORT = PRETRAINED.parent / "import-verification.json"

MODEL_SPEC_SHA256 = "e8fceef8e30955782efeab6e732319c051998e061f58502bc66d39f71485d330"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
PRETRAINED_SIZE = 71083664
PRETRAINED_SHA384 = "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55"
SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
SCREEN_RECORDS = 3904
SCREEN_AUDIO_SECONDS = 6407.372
VALIDATION_RECORDS = 1273
VALIDATION_AUDIO_SECONDS = 2279.385
SCREEN_EPOCHS = 12

DEFAULT_PARENT = "exp-7c7ac81bc57f8f3b"
PARENT_ATTEMPT = "attempt-0001"
V18_BEST_VALIDATION_CER = 0.9003628405229511
V18_BEST_VALIDATION_CER_EPOCH = 3
V18_PHYSICAL_CER = 0.8975407475666648
V18_PHYSICAL_WER = 1.22219822932412
V18_PHYSICAL_P95_MS = 395.717213
V18_PRETRAINING_MYRIAD_AGREEMENT = 1.0
V18_INITIAL_VALIDATION_CER = 0.5776651500316765
V18_INITIAL_VALIDATION_WER = 0.7393651479162168
V18_PRETRAINING_MYRIAD_MAX_FRAME_TV = 0.0009230391151051188
EXPECTED_PARAMETERS = 18934631
EXPECTED_MACS = 4827463680
EXPECTED_RECEPTIVE_FIELD = 8057
EXPECTED_OUTPUT_FRAMES = 256


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha384_path(path: pathlib.Path) -> str:
    digest = hashlib.sha384()
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_screen() -> dict:
    for path in (SCREEN_MANIFEST, SCREEN_PROVENANCE, VALIDATION_MANIFEST):
        if not path.is_file():
            raise ValueError(f"architecture-screen input missing: {path}")
    if sha256_path(SCREEN_MANIFEST) != SCREEN_MANIFEST_SHA256:
        raise ValueError("architecture-screen training manifest changed")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen validation manifest changed")
    screen = load_json(SCREEN_PROVENANCE)
    if screen.get("schema") != "speech-asr/architecture-screen" or screen.get("status") != "valid":
        raise ValueError("architecture-screen provenance is not valid")
    if screen.get("train", {}).get("manifest_sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("architecture-screen training hash changed")
    if screen.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen validation hash changed")
    if screen.get("train", {}).get("stats", {}).get("records") != SCREEN_RECORDS:
        raise ValueError("architecture-screen training record count changed")
    if screen.get("train", {}).get("stats", {}).get("audio_seconds") != SCREEN_AUDIO_SECONDS:
        raise ValueError("architecture-screen training duration changed")
    if screen.get("validation", {}).get("stats", {}).get("records") != VALIDATION_RECORDS:
        raise ValueError("architecture-screen validation record count changed")
    if screen.get("validation", {}).get("stats", {}).get("audio_seconds") != VALIDATION_AUDIO_SECONDS:
        raise ValueError("architecture-screen validation duration changed")
    if screen.get("budget", {}).get("epochs") != SCREEN_EPOCHS:
        raise ValueError("architecture-screen epoch budget changed")
    if screen.get("roles", {}).get("sealed_heldout_allowed_for_selection") is not False:
        raise ValueError("sealed held-out evidence must remain forbidden")
    return screen


def validate_pretrained() -> dict:
    if not PRETRAINED.is_file():
        raise ValueError(
            f"pretrained QuartzNet archive missing: {PRETRAINED}; "
            "run ./scripts/prepare-cnn-ctc-v19-pretrained.sh first"
        )
    size = PRETRAINED.stat().st_size
    if size != PRETRAINED_SIZE:
        raise ValueError(f"pretrained QuartzNet size changed: {size}")
    digest = sha384_path(PRETRAINED)
    if digest != PRETRAINED_SHA384:
        raise ValueError("pretrained QuartzNet SHA-384 changed")

    if not PRETRAINED_IMPORT.is_file():
        raise ValueError(
            f"pretrained QuartzNet import verification missing: {PRETRAINED_IMPORT}; "
            "run ./scripts/prepare-cnn-ctc-v19-pretrained.sh first"
        )
    verified = load_json(PRETRAINED_IMPORT)
    if (
        verified.get("schema") != "speech-asr/pretrained-import-verification"
        or verified.get("version") != 1
        or verified.get("status") != "valid"
        or verified.get("model_id") != "cnn_ctc_v18"
    ):
        raise ValueError("pretrained QuartzNet import verification is not valid")
    source = verified.get("source")
    if not isinstance(source, dict):
        raise ValueError("pretrained QuartzNet import source evidence is missing")
    if source.get("size_bytes") != PRETRAINED_SIZE:
        raise ValueError("pretrained QuartzNet import source size changed")
    if source.get("sha384") != PRETRAINED_SHA384:
        raise ValueError("pretrained QuartzNet import source hash changed")
    if verified.get("shared_decoder_symbol_count") != 29:
        raise ValueError("pretrained QuartzNet shared decoder mapping changed")
    if verified.get("target_only_symbols_random_init") != list("0123456789"):
        raise ValueError("pretrained QuartzNet target-only decoder rows changed")
    if verified.get("encoder_source_tensors_used", 0) < 500:
        raise ValueError("pretrained QuartzNet encoder tensor import is incomplete")
    if verified.get("output_shape") != [1, 256, 39]:
        raise ValueError("pretrained QuartzNet import output shape changed")
    if verified.get("finite_output") is not True:
        raise ValueError("pretrained QuartzNet import produced non-finite output")
    fingerprint = verified.get("acoustic_state_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("pretrained QuartzNet acoustic-state fingerprint is missing")

    return {
        "path": repo_relative(PRETRAINED),
        "size_bytes": size,
        "sha384": digest,
        "import_verification": {
            "path": repo_relative(PRETRAINED_IMPORT),
            "acoustic_state_sha256": fingerprint,
            "encoder_source_tensors_used": verified["encoder_source_tensors_used"],
            "shared_decoder_symbol_count": verified["shared_decoder_symbol_count"],
        },
    }


def validate_parent() -> dict:
    parent = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
    attempt_dir = parent / "attempts" / PARENT_ATTEMPT
    paths = {
        "request": parent / "request" / "experiment.json",
        "model": parent / "request" / "model-spec.json",
        "train": parent / "request" / "train-config.json",
        "attempt": attempt_dir / "attempt.json",
        "training": attempt_dir / "results" / "training.json",
        "compatibility": attempt_dir / "results" / "compatibility.json",
        "summary": attempt_dir / "results" / "result.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"cnn_ctc_v18 parent evidence missing: {path}")
    docs = {name: load_json(path) for name, path in paths.items()}
    if docs["request"].get("experiment_id") != DEFAULT_PARENT:
        raise ValueError("cnn_ctc_v18 parent identity changed")
    if docs["request"].get("benchmark", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v18 parent benchmark changed")
    if docs["model"].get("model_id") != "cnn_ctc_v18":
        raise ValueError("pretrained-transfer parent is not cnn_ctc_v18")
    if docs["train"].get("training_manifest", {}).get("sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v18 parent training manifest changed")
    if docs["attempt"].get("state") != "AWAIT_REVIEW" or docs["attempt"].get("outcome") != "completed":
        raise ValueError("cnn_ctc_v18 parent is not completed")
    training = docs["training"]
    if training.get("status") != "completed":
        raise ValueError("cnn_ctc_v18 parent training is incomplete")
    if training.get("best_validation_cer") != V18_BEST_VALIDATION_CER:
        raise ValueError("cnn_ctc_v18 best validation CER changed")
    if training.get("best_validation_cer_epoch") != V18_BEST_VALIDATION_CER_EPOCH:
        raise ValueError("cnn_ctc_v18 best validation CER epoch changed")
    initial = training.get("initial_validation", {})
    if initial.get("cer") != V18_INITIAL_VALIDATION_CER:
        raise ValueError("cnn_ctc_v18 initial validation CER changed")
    if initial.get("wer") != V18_INITIAL_VALIDATION_WER:
        raise ValueError("cnn_ctc_v18 initial validation WER changed")
    metrics = docs["summary"].get("metrics", {})
    if metrics.get("cer") != V18_PHYSICAL_CER:
        raise ValueError("cnn_ctc_v18 physical CER changed")
    if metrics.get("wer") != V18_PHYSICAL_WER:
        raise ValueError("cnn_ctc_v18 physical WER changed")
    if metrics.get("inference_latency_p95_ms") != V18_PHYSICAL_P95_MS:
        raise ValueError("cnn_ctc_v18 physical p95 latency changed")
    probe = docs["compatibility"].get("pretraining_myriad_probe", {})
    if probe.get("comparison", {}).get("frame_argmax_agreement") != V18_PRETRAINING_MYRIAD_AGREEMENT:
        raise ValueError("cnn_ctc_v18 pretraining MYRIAD agreement changed")
    gate = probe.get("numerical_gate", {})
    if gate.get("status") != "accepted":
        raise ValueError("cnn_ctc_v18 pretraining MYRIAD gate changed")
    if gate.get("gate_kind") != "pretrained-semantic-parity-v1":
        raise ValueError("cnn_ctc_v18 pretraining MYRIAD gate kind changed")
    if gate.get("max_frame_total_variation") != V18_PRETRAINING_MYRIAD_MAX_FRAME_TV:
        raise ValueError("cnn_ctc_v18 pretraining MYRIAD probability drift changed")
    return docs


def validate_model() -> dict:
    package = load_spec(MODEL_SPEC)
    vocab = load_vocab(VOCAB)
    actual_hash = canonical_sha256(package)
    if actual_hash != MODEL_SPEC_SHA256:
        raise ValueError(
            f"cnn_ctc_v19 model spec differs from reviewed design: {actual_hash} != {MODEL_SPEC_SHA256}"
        )
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v19 vocabulary differs from frozen vocabulary")
    transfer = package.get("transfer", {})
    if transfer.get("source_size_bytes") != PRETRAINED_SIZE:
        raise ValueError("cnn_ctc_v19 pretrained source size contract changed")
    if transfer.get("source_sha384") != PRETRAINED_SHA384:
        raise ValueError("cnn_ctc_v19 pretrained source hash contract changed")
    if transfer.get("fine_tune") != "all_parameters":
        raise ValueError("cnn_ctc_v19 must fine-tune all parameters")
    training = package["training"]
    expected_training = {
        "seed": 1337,
        "batch_size": 8,
        "learning_rate": 0.0001,
        "epochs": 12,
        "optimizer": "novograd",
        "optimizer_betas": [0.95, 0.25],
        "optimizer_eps": 1e-8,
        "weight_decay": 0.001,
        "loss": "ctc",
        "zero_infinity": True,
        "gradient_clip_norm": 5,
        "lr_schedule": "cosine",
        "warmup_ratio": 0,
        "min_learning_rate": 1e-6,
        "checkpoint_selection": "best_validation_cer",
        "freeze_batchnorm_running_stats": True,
        "initial_validation_checkpoint_candidate": True,
    }
    if training != expected_training:
        raise ValueError("cnn_ctc_v19 fine-tuning recipe changed")
    estimate = model_resource_estimate(package)
    actual = (
        estimate["parameters"],
        estimate["macs_fixed_input"],
        estimate["receptive_field_feature_frames"],
        estimate["output_frames"],
    )
    expected = (EXPECTED_PARAMETERS, EXPECTED_MACS, EXPECTED_RECEPTIVE_FIELD, EXPECTED_OUTPUT_FRAMES)
    if actual != expected:
        raise ValueError(f"cnn_ctc_v19 resource estimate changed: {actual} != {expected}")
    return package


def main() -> int:
    try:
        screen = validate_screen()
        parent = validate_parent()
        pretrained = validate_pretrained()
        package = validate_model()
        head = git_head()

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v19 conservative pretrained QuartzNet fine-tuning",
            "hypothesis": (
                "cnn_ctc_v18 proved the pretrained QuartzNet start is strong on AMI "
                "(epoch-zero validation CER 0.5776651500316765) but batch-1, 1e-3 "
                "fine-tuning catastrophically forgot that representation. Freezing "
                "pretrained BatchNorm running statistics, increasing the batch to 8, "
                "using 1e-4 cosine fine-tuning, and allowing epoch zero to win "
                "checkpoint selection should preserve or improve the transfer baseline."
            ),
            "rationale": (
                "v18 started at validation CER 0.5776651500316765 / WER "
                "0.7393651479162168, but its best post-update validation CER was only "
                "0.9003628405229511 and the deployed checkpoint reached physical CER "
                "0.8975407475666648. v18 epoch 1 also observed a pre-clip gradient norm "
                "above 222k while BatchNorm running statistics were updated from "
                "single-sample batches."
            ),
            "changes": [
                f"parent to completed v18 evidence experiment {DEFAULT_PARENT}",
                "keep the exact pinned pretrained source, frontend, vocabulary and QuartzNet inference topology",
                "increase fine-tuning batch size from 1 to 8",
                "freeze BatchNorm running mean/variance statistics at pretrained values while keeping affine parameters trainable",
                "reduce NovoGrad peak learning rate from 1e-3 to 1e-4",
                "remove warmup and use cosine decay from 1e-4 to 1e-6",
                "treat zero-step pretrained validation as checkpoint epoch 0 so fine-tuning cannot replace it with a worse checkpoint",
                "keep the frozen 12-epoch AMI architecture-screen manifests and validation-CER selection",
                "re-prove ONNX, OpenVINO 2020.3 and physical MYRIAD numerical compatibility",
            ],
            "notes": (
                "No sealed held-out evidence is used. CER/WER/latency remain measured "
                "rather than hard-gated. The v18 semantic MYRIAD pretrained-model gate "
                "is retained unchanged."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v19",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v3"},
            "architecture": dict(V19_ARCHITECTURE),
            "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
        }
        train_config = {
            "schema": "speech-asr/train-config",
            "version": 1,
            "seed": int(package["training"]["seed"]),
            "device": "cuda",
            "epochs": SCREEN_EPOCHS,
            "batch_size": 8,
            "max_samples": None,
            "checkpoint_selection": "validation_cer",
            "optimizer": {"kind": "novograd", "learning_rate": 0.0001},
            "pretrained_source": {
                "path": repo_relative(PRETRAINED),
                "size_bytes": PRETRAINED_SIZE,
                "sha384": PRETRAINED_SHA384,
            },
            "training_manifest": {
                "id": "ami-model-quality-v4-architecture-screen-v1-train",
                "path": repo_relative(SCREEN_MANIFEST),
                "sha256": SCREEN_MANIFEST_SHA256,
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
                "training", "onnx_export", "openvino_conversion",
                "myriad_execution", "accuracy_evaluation",
            ],
            "thresholds": {
                "max_wer": None,
                "max_cer": None,
                "max_realtime_factor": None,
                "max_latency_p95_ms": None,
                "min_frame_argmax_agreement": 1.0,
            },
            "retry_policy": {
                "max_attempts": 3,
                "retryable_failure_classes": ["transport_preflight", "worker_busy", "hardware_transient"],
            },
        }

        draft = ROOT / "work" / "speech-asr" / "experiment-drafts" / (
            f"cnn_ctc_v19-conservative-transfer-{head[:8]}-{PRETRAINED_SHA384[:8]}"
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
            str(PYTHON), str(MANAGER), "init",
            "--proposal", str(paths["proposal"]),
            "--model-spec", str(paths["model"]),
            "--train-config", str(paths["train"]),
            "--acceptance", str(paths["acceptance"]),
            "--benchmark-id", "ami-model-quality-v4-architecture-screen-v1-es2011-validation",
            "--manifest", str(VALIDATION_MANIFEST),
            "--repo-commit", head,
            "--parent", DEFAULT_PARENT,
        ]
        proc = subprocess.run(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout.strip())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = json.loads(proc.stdout)
    output["conservative_transfer_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "v18_initial_validation_cer": parent["training"]["initial_validation"]["cer"],
        "v18_initial_validation_wer": parent["training"]["initial_validation"]["wer"],
        "v18_best_validation_cer": parent["training"]["best_validation_cer"],
        "v18_physical_cer": parent["summary"]["metrics"]["cer"],
        "v18_physical_p95_ms": parent["summary"]["metrics"]["inference_latency_p95_ms"],
        "pretrained": pretrained,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "optimizer": "novograd",
        "optimizer_betas": [0.95, 0.25],
        "peak_learning_rate": 0.0001,
        "warmup_ratio": 0.0,
        "batch_size": 8,
        "freeze_batchnorm_running_stats": True,
        "initial_validation_checkpoint_candidate": True,
        "epochs": SCREEN_EPOCHS,
        "train_records": screen["train"]["stats"]["records"],
        "validation_records": screen["validation"]["stats"]["records"],
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
