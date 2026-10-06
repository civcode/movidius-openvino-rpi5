#!/usr/bin/env python3
"""Initialize cnn_ctc_v12 large-capacity probe on architecture-screen-v1 data."""

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
from speech_asr.orchestration import V12_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v12" / "model_spec.json"
V3_MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

SCREEN_DIR = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1"
)
SCREEN_MANIFEST = SCREEN_DIR / "train.manifest.jsonl"
SCREEN_PROVENANCE = SCREEN_DIR / "screen.json"
VALIDATION_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"
)

MODEL_SPEC_SHA256 = "cd534c32fe1090471e938e2f9cbab0865f07d3d7d65aa8de1b17f1f605985730"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
SCREEN_RECORDS = 3904
SCREEN_AUDIO_SECONDS = 6407.372
VALIDATION_RECORDS = 1273
VALIDATION_AUDIO_SECONDS = 2279.385
SCREEN_EPOCHS = 12
DEFAULT_PARENT = "exp-00e6b1e434d187d8"
PARENT_ATTEMPT = "attempt-0001"
PARENT_CER = 0.7984219316938317
PARENT_WER = 0.9930900453465774
PARENT_P95_MS = 16.7866342
EXPECTED_PARAMETERS = 19627687
EXPECTED_MACS = 2515673088
EXPECTED_RECEPTIVE_FIELD = 653
EXPECTED_OUTPUT_FRAMES = 128


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


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_screen() -> dict:
    for path in (SCREEN_MANIFEST, SCREEN_PROVENANCE, VALIDATION_MANIFEST):
        if not path.is_file():
            raise ValueError(f"architecture-screen input missing: {path}")
    if sha256_path(SCREEN_MANIFEST) != SCREEN_MANIFEST_SHA256:
        raise ValueError("architecture-screen training manifest changed")
    if sha256_path(VALIDATION_MANIFEST) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen validation manifest changed")

    screen = load_json(SCREEN_PROVENANCE)
    if screen.get("schema") != "speech-asr/architecture-screen":
        raise ValueError("architecture-screen provenance schema changed")
    if screen.get("status") != "valid":
        raise ValueError("architecture-screen provenance is not valid")
    if screen.get("id") != "ami-model-quality-v4-architecture-screen-v1":
        raise ValueError("architecture-screen identity changed")
    if screen.get("train", {}).get("manifest_sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("architecture-screen provenance training hash changed")
    if screen.get("validation", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen provenance validation hash changed")
    if screen.get("train", {}).get("stats", {}).get("records") != SCREEN_RECORDS:
        raise ValueError("architecture-screen training record count changed")
    if screen.get("train", {}).get("stats", {}).get("audio_seconds") != SCREEN_AUDIO_SECONDS:
        raise ValueError("architecture-screen training duration changed")
    if screen.get("validation", {}).get("stats", {}).get("records") != VALIDATION_RECORDS:
        raise ValueError("architecture-screen validation record count changed")
    if screen.get("validation", {}).get("stats", {}).get("audio_seconds") != VALIDATION_AUDIO_SECONDS:
        raise ValueError("architecture-screen validation duration changed")
    if len(screen.get("train", {}).get("stats", {}).get("meetings", [])) != 48:
        raise ValueError("architecture-screen must retain all 48 train meetings")
    if screen.get("budget", {}).get("epochs") != SCREEN_EPOCHS:
        raise ValueError("architecture-screen epoch budget changed")
    if screen.get("roles", {}).get("sealed_heldout_allowed_for_selection") is not False:
        raise ValueError("sealed held-out evidence must remain forbidden")
    return screen


def validate_parent() -> dict:
    parent = ROOT / "work" / "speech-asr" / "experiments" / DEFAULT_PARENT
    attempt_dir = parent / "attempts" / PARENT_ATTEMPT
    request_path = parent / "request" / "experiment.json"
    model_path = parent / "request" / "model-spec.json"
    train_path = parent / "request" / "train-config.json"
    attempt_path = attempt_dir / "attempt.json"
    acceptance_path = attempt_dir / "results" / "acceptance-evaluation.json"
    result_path = attempt_dir / "results" / "result.json"
    for path in (
        request_path,
        model_path,
        train_path,
        attempt_path,
        acceptance_path,
        result_path,
    ):
        if not path.is_file():
            raise ValueError(f"architecture-screen v3 parent evidence missing: {path}")

    request = load_json(request_path)
    model = load_json(model_path)
    train = load_json(train_path)
    attempt = load_json(attempt_path)
    acceptance = load_json(acceptance_path)
    result = load_json(result_path)

    if request.get("experiment_id") != DEFAULT_PARENT:
        raise ValueError("architecture-screen parent identity changed")
    if request.get("benchmark", {}).get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen parent benchmark changed")
    if model.get("model_id") != "cnn_ctc_v3":
        raise ValueError("architecture-screen parent is not cnn_ctc_v3")
    if train.get("training_manifest", {}).get("sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("architecture-screen parent training manifest changed")
    if train.get("validation_manifest", {}).get("sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("architecture-screen parent validation manifest changed")
    if train.get("epochs") != SCREEN_EPOCHS:
        raise ValueError("architecture-screen parent epoch budget changed")
    if attempt.get("outcome") != "completed":
        raise ValueError("architecture-screen parent did not complete")
    if acceptance.get("status") != "accepted":
        raise ValueError("architecture-screen parent acceptance did not pass")
    if result.get("metrics", {}).get("cer") != PARENT_CER:
        raise ValueError("architecture-screen parent CER changed")
    if result.get("metrics", {}).get("wer") != PARENT_WER:
        raise ValueError("architecture-screen parent WER changed")
    if result.get("metrics", {}).get("inference_latency_p95_ms") != PARENT_P95_MS:
        raise ValueError("architecture-screen parent p95 changed")
    return result


def validate_model() -> dict:
    package = load_spec(MODEL_SPEC)
    base = load_spec(V3_MODEL_SPEC)
    vocab = load_vocab(MODEL_SPEC.parent / "vocab.json")
    if canonical_sha256(package) != MODEL_SPEC_SHA256:
        raise ValueError("cnn_ctc_v12 model spec differs from reviewed design")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v12 vocabulary differs from frozen v3 vocabulary")
    if package["frontend"] != base["frontend"]:
        raise ValueError("cnn_ctc_v12 frontend differs from cnn_ctc_v3")
    if package["input_contract"] != base["input_contract"]:
        raise ValueError("cnn_ctc_v12 input contract differs from cnn_ctc_v3")
    if package["decoder"] != base["decoder"]:
        raise ValueError("cnn_ctc_v12 decoder differs from cnn_ctc_v3")
    base_training = base["training"]
    for key in (
        "seed",
        "batch_size",
        "epochs",
        "optimizer",
        "loss",
        "zero_infinity",
        "gradient_clip_norm",
        "checkpoint_selection",
    ):
        if package["training"].get(key) != base_training.get(key):
            raise ValueError(f"cnn_ctc_v12 training.{key} differs from cnn_ctc_v3")
    expected_schedule = {
        "learning_rate": 0.003,
        "lr_schedule": "onecycle",
        "min_learning_rate": 0.00003,
        "onecycle_pct_start": 0.1,
        "onecycle_div_factor": 10.0,
        "onecycle_final_div_factor": 10.0,
    }
    for key, expected in expected_schedule.items():
        if package["training"].get(key) != expected:
            raise ValueError(
                f"cnn_ctc_v12 training.{key} differs from reviewed high-LR policy"
            )

    estimate = model_resource_estimate(package)
    if estimate["parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v12 parameter estimate changed")
    if estimate["macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v12 MAC estimate changed")
    if estimate["receptive_field_feature_frames"] != EXPECTED_RECEPTIVE_FIELD:
        raise ValueError("cnn_ctc_v12 receptive field changed")
    if estimate["output_frames"] != EXPECTED_OUTPUT_FRAMES:
        raise ValueError("cnn_ctc_v12 output frame count changed")
    if package["network"]["estimated_parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v12 declared parameter count changed")
    if package["network"]["estimated_macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v12 declared MAC count changed")
    if package["output_contract"]["shape"] != [1, 128, 39]:
        raise ValueError("cnn_ctc_v12 output contract changed")
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
            "title": "cnn_ctc_v12 large-capacity MYRIAD probe",
            "hypothesis": (
                "The v9-v11 small-width family has reached diminishing returns while "
                "physical MA2450 inference remains far faster than the audio represented "
                "by each fixed input. A roughly 20M-parameter, 2.5B-MAC residual Conv1D "
                "encoder will test whether recognition quality improves materially when "
                "capacity, rather than latency preservation, becomes the priority."
            ),
            "rationale": (
                "v11 completed at CER 0.7865576225306686 and 14.9809824 ms p95 with "
                "1,117,287 parameters. The 112-to-128 width step only modestly improved "
                "CER, so further small width increments are retired. v12 deliberately "
                "jumps to 448 channels and 16 residual blocks using only the ordinary "
                "Conv1D/ReLU/residual operators already proven on OpenVINO 2020.3/MYRIAD."
            ),
            "changes": [
                f"retain architecture-screen v3 control {DEFAULT_PARENT} for review deltas",
                "reuse exact architecture-screen-v1 training manifest",
                "reuse full frozen ES2011 validation manifest",
                "keep logmel-v1 frontend, vocabulary, standard CTC and greedy decoder",
                "keep two stride-2 stems and 128 CTC output frames",
                "increase stem channels to [128,448]",
                "increase residual depth from 8 to 16 blocks",
                "use 448-channel kernel-5 residual blocks",
                "repeat dilation schedule [1,2,3,4,4,3,2,1] twice",
                "retain normalization-free ReLU/dropout/residual projection policy",
                "train for exactly 12 screen epochs with validation-CER selection",
                "use Adam with OneCycle LR: 3e-4 start, 3e-3 peak, 3e-5 finish",
                "use no augmentation, blank penalty, InterCTC or decoder change",
                "physically probe initialized ONNX/OpenVINO graph before training",
                "record physical latency but apply no latency rejection threshold",
            ],
            "notes": (
                "This is a capacity-boundary probe, not a latency-optimized candidate. "
                "There is no numeric CER, WER, RTF or p95 rejection threshold. Conversion, "
                "physical MYRIAD execution and numerical compatibility determine hardware "
                "feasibility; quality and measured latency are reviewed afterward. Sealed "
                "held-out evidence remains forbidden for architecture selection. "
                "The v12 training policy intentionally raises peak Adam LR by 10x "
                "versus v11, using OneCycle warmup/annealing."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v12",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V12_ARCHITECTURE),
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
            / f"cnn_ctc_v12-screen-{head[:8]}-{SCREEN_MANIFEST_SHA256[:8]}"
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
    output["capacity_probe_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "cer": parent["metrics"]["cer"],
        "wer": parent["metrics"]["wer"],
        "latency_p95_ms": parent["metrics"]["inference_latency_p95_ms"],
        "v11_cer": 0.7865576225306686,
        "v11_latency_p95_ms": 14.9809824,
        "latency_rejection_threshold_ms": None,
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
