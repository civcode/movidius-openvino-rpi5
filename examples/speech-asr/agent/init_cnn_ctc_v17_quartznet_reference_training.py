#!/usr/bin/env python3
"""Initialize cnn_ctc_v17 QuartzNet reference-training alignment."""

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
from speech_asr.orchestration import V17_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v17" / "model_spec.json"
V16_MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v16" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

SCREEN_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1"
SCREEN_MANIFEST = SCREEN_DIR / "train.manifest.jsonl"
SCREEN_PROVENANCE = SCREEN_DIR / "screen.json"
VALIDATION_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"

MODEL_SPEC_SHA256 = "766a3851160bc4d6d1a5be8f27d45066accce9391683efc99ea5e9833dfb0fa1"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
SCREEN_RECORDS = 3904
SCREEN_AUDIO_SECONDS = 6407.372
VALIDATION_RECORDS = 1273
VALIDATION_AUDIO_SECONDS = 2279.385
SCREEN_EPOCHS = 12

DEFAULT_PARENT = "exp-4d7f92c301064c45"
PARENT_ATTEMPT = "attempt-0001"
V16_BEST_VALIDATION_CER = 0.9297356447618499
V16_BEST_VALIDATION_CER_EPOCH = 5
V16_BEST_VALIDATION_LOSS = 4.084610798893747
V16_BEST_VALIDATION_LOSS_EPOCH = 2
V16_LAST_VALIDATION_CER = 1.0
V16_LAST_VALIDATION_LOSS = 43.97023439682033
V16_PRETRAINING_MYRIAD_AGREEMENT = 1.0
V16_PRETRAINING_MYRIAD_MAX_ABS_ERROR = 7.482245564460754e-06

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
    if screen.get("id") != "ami-model-quality-v4-architecture-screen-v1":
        raise ValueError("architecture-screen identity changed")
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
            raise ValueError(f"cnn_ctc_v16 parent evidence missing: {path}")

    docs = {name: load_json(path) for name, path in paths.items()}
    request = docs["request"]
    model = docs["model"]
    train = docs["train"]
    attempt = docs["attempt"]
    training = docs["training"]
    compatibility = docs["compatibility"]
    summary = docs["summary"]

    if request.get("experiment_id") != DEFAULT_PARENT:
        raise ValueError("cnn_ctc_v16 parent identity changed")
    if request.get("benchmark", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v16 parent benchmark changed")
    if model.get("model_id") != "cnn_ctc_v16":
        raise ValueError("training-alignment parent is not cnn_ctc_v16")
    if train.get("training_manifest", {}).get("sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v16 parent training manifest changed")
    if train.get("validation_manifest", {}).get("sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v16 parent validation manifest changed")
    if train.get("epochs") != SCREEN_EPOCHS:
        raise ValueError("cnn_ctc_v16 parent epoch budget changed")
    if attempt.get("outcome") != "failed" or attempt.get("failure_class") != "hardware_execution":
        raise ValueError("cnn_ctc_v16 parent must be the recorded hardware-execution failure")
    if summary.get("outcome") != "failed":
        raise ValueError("cnn_ctc_v16 terminal summary changed")
    if training.get("status") != "completed" or training.get("model_id") != "cnn_ctc_v16":
        raise ValueError("cnn_ctc_v16 training evidence is incomplete")
    if training.get("best_validation_cer") != V16_BEST_VALIDATION_CER:
        raise ValueError("cnn_ctc_v16 best validation CER changed")
    if training.get("best_validation_cer_epoch") != V16_BEST_VALIDATION_CER_EPOCH:
        raise ValueError("cnn_ctc_v16 best validation CER epoch changed")
    if training.get("best_validation_loss") != V16_BEST_VALIDATION_LOSS:
        raise ValueError("cnn_ctc_v16 best validation loss changed")
    if training.get("best_validation_loss_epoch") != V16_BEST_VALIDATION_LOSS_EPOCH:
        raise ValueError("cnn_ctc_v16 best validation loss epoch changed")
    last = training.get("history", [])[-1]
    if last.get("validation_cer") != V16_LAST_VALIDATION_CER:
        raise ValueError("cnn_ctc_v16 final validation CER changed")
    if last.get("validation_loss") != V16_LAST_VALIDATION_LOSS:
        raise ValueError("cnn_ctc_v16 final validation loss changed")
    if compatibility.get("pretraining_myriad_frame_argmax_agreement") != V16_PRETRAINING_MYRIAD_AGREEMENT:
        raise ValueError("cnn_ctc_v16 pretraining MYRIAD agreement changed")
    if compatibility.get("pretraining_myriad_max_abs_error") != V16_PRETRAINING_MYRIAD_MAX_ABS_ERROR:
        raise ValueError("cnn_ctc_v16 pretraining MYRIAD max error changed")
    if compatibility.get("pretraining_myriad_numerical_gate", {}).get("status") != "accepted":
        raise ValueError("cnn_ctc_v16 pretraining MYRIAD gate is not accepted")
    return docs


def validate_model() -> dict:
    package = load_spec(MODEL_SPEC)
    base = load_spec(V16_MODEL_SPEC)
    vocab = load_vocab(MODEL_SPEC.parent / "vocab.json")

    actual_hash = canonical_sha256(package)
    if actual_hash != MODEL_SPEC_SHA256:
        raise ValueError(
            "cnn_ctc_v17 model spec differs from reviewed design: "
            f"{actual_hash} != {MODEL_SPEC_SHA256}"
        )
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v17 vocabulary differs from frozen vocabulary")

    for key in ("frontend", "input_contract", "output_contract", "decoder", "export", "deployment"):
        if package[key] != base[key]:
            raise ValueError(f"cnn_ctc_v17 {key} differs from cnn_ctc_v16")

    expected_network = dict(base["network"])
    expected_network["dropout"] = 0.0
    if package["network"] != expected_network:
        raise ValueError(
            "cnn_ctc_v17 must keep the v16 QuartzNet topology and change only "
            "training-time dropout"
        )

    training = package["training"]
    expected_training = {
        "seed": 1337,
        "batch_size": 1,
        "learning_rate": 0.01,
        "epochs": 32,
        "optimizer": "novograd",
        "optimizer_betas": [0.8, 0.5],
        "optimizer_eps": 1e-8,
        "weight_decay": 0.001,
        "loss": "ctc",
        "zero_infinity": True,
        "gradient_clip_norm": 5,
        "lr_schedule": "warmup_cosine",
        "warmup_ratio": 0.12,
        "min_learning_rate": 0.00001,
        "checkpoint_selection": "best_validation_loss",
    }
    if training != expected_training:
        raise ValueError("cnn_ctc_v17 training recipe differs from reviewed reference alignment")

    estimate = model_resource_estimate(package)
    actual = (
        estimate["parameters"],
        estimate["macs_fixed_input"],
        estimate["receptive_field_feature_frames"],
        estimate["output_frames"],
    )
    expected = (
        EXPECTED_PARAMETERS,
        EXPECTED_MACS,
        EXPECTED_RECEPTIVE_FIELD,
        EXPECTED_OUTPUT_FRAMES,
    )
    if actual != expected:
        raise ValueError(f"cnn_ctc_v17 resource estimate changed: {actual} != {expected}")
    return package


def main() -> int:
    try:
        screen = validate_screen()
        parent = validate_parent()
        package = validate_model()
        head = git_head()

        proposal = {
            "schema": "speech-asr/experiment-proposal",
            "version": 1,
            "title": "cnn_ctc_v17 QuartzNet reference-training alignment",
            "hypothesis": (
                "v16 proved that the QuartzNet-15x5 inference graph converts and executes "
                "numerically on MA2450, but the inherited Adam/dropout-0.2 recipe was "
                "unstable and never reached useful acoustic accuracy. Aligning the "
                "training-only recipe with QuartzNet/NeMo should test the architecture "
                "without another graph redesign."
            ),
            "rationale": (
                "v16 reached best validation CER 0.9297356447618499 at epoch 5, then "
                "collapsed back to CER 1.0 with validation loss 43.97. Its first epoch "
                "also observed an extreme pre-clip gradient norm. The published QuartzNet "
                "recipe uses NovoGrad, weight decay and cosine scheduling; the NeMo 15x5 "
                "configuration uses dropout 0.0 and NovoGrad lr 0.01 with betas 0.8/0.5. "
                "The v16 graph itself already passed the physical MYRIAD numerical gate."
            ),
            "changes": [
                f"parent to v16 evidence experiment {DEFAULT_PARENT}, including its failed full-corpus hardware session",
                "keep the exact QuartzNet-15x5 convolution/channel/stride/dilation topology",
                "keep the frozen 64-bin frontend, 39-token vocabulary and 256-frame output",
                "set training-time dropout from 0.2 to 0.0",
                "replace Adam with self-contained NovoGrad",
                "use peak learning rate 0.01, betas 0.8/0.5, epsilon 1e-8 and weight decay 0.001",
                "use step-based 12% linear warmup followed by cosine decay to 1e-5",
                "keep gradient clipping at 5.0",
                "keep batch size 1 and the exact 12-epoch architecture-screen budget",
                "keep validation-CER checkpoint selection and no augmentation/objective change",
                "reuse the physical compatibility gate even though the inference topology is unchanged",
                "use restart-capable persistent MYRIAD evaluation for long corpus execution",
            ],
            "notes": (
                "This experiment changes the QuartzNet training recipe, not its deployed "
                "operator topology. No numeric CER/latency threshold is imposed; compatibility "
                "and numerical agreement remain hard gates. Sealed held-out evidence is forbidden."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v17",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V17_ARCHITECTURE),
            "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
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
                "kind": "novograd",
                "learning_rate": float(package["training"]["learning_rate"]),
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
            / f"cnn_ctc_v17-quartznet-reference-training-{head[:8]}-{SCREEN_MANIFEST_SHA256[:8]}"
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
    training = parent["training"]
    compatibility = parent["compatibility"]
    output["quartznet_training_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "v16_outcome": parent["attempt"]["outcome"],
        "v16_failure_class": parent["attempt"]["failure_class"],
        "v16_best_validation_cer": training["best_validation_cer"],
        "v16_best_validation_cer_epoch": training["best_validation_cer_epoch"],
        "v16_best_validation_loss": training["best_validation_loss"],
        "v16_best_validation_loss_epoch": training["best_validation_loss_epoch"],
        "v16_pretraining_myriad_frame_argmax_agreement": compatibility["pretraining_myriad_frame_argmax_agreement"],
        "v16_pretraining_myriad_max_abs_error": compatibility["pretraining_myriad_max_abs_error"],
        "v15_physical_cer": 0.8428267004549905,
        "v14_physical_cer": 0.7898404653573691,
        "v11_physical_cer": 0.7865576225306686,
        "optimizer": "novograd",
        "optimizer_betas": [0.8, 0.5],
        "peak_learning_rate": 0.01,
        "weight_decay": 0.001,
        "warmup_ratio": 0.12,
        "dropout": 0.0,
        "train_records": screen["train"]["stats"]["records"],
        "train_audio_seconds": screen["train"]["stats"]["audio_seconds"],
        "train_manifest_sha256": SCREEN_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "epochs": SCREEN_EPOCHS,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
