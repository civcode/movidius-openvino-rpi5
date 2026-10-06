#!/usr/bin/env python3
"""Initialize cnn_ctc_v10 on architecture-screen-v1."""

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
from speech_asr.orchestration import V10_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v10" / "model_spec.json"
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

MODEL_SPEC_SHA256 = "007f9808b11bbf1e84600dd0e5c962486fb468ead81ef476ca8b32f1381e4315"
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
PROMOTION_RELATIVE_CER_RATIO = 0.97
PROMOTION_CER = 0.7744692737430167
EXPECTED_PARAMETERS = 865511
EXPECTED_MACS = 113149952
EXPECTED_RECEPTIVE_FIELD = 493
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
        raise ValueError("cnn_ctc_v10 model spec differs from reviewed design")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v10 vocabulary differs from frozen v3 vocabulary")
    if package["frontend"] != base["frontend"]:
        raise ValueError("cnn_ctc_v10 frontend differs from cnn_ctc_v3")
    if package["input_contract"] != base["input_contract"]:
        raise ValueError("cnn_ctc_v10 input contract differs from cnn_ctc_v3")
    if package["decoder"] != base["decoder"]:
        raise ValueError("cnn_ctc_v10 decoder differs from cnn_ctc_v3")
    if package["training"] != base["training"]:
        raise ValueError("cnn_ctc_v10 package training defaults differ from cnn_ctc_v3")

    estimate = model_resource_estimate(package)
    if estimate["parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v10 parameter estimate changed")
    if estimate["macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v10 MAC estimate changed")
    if estimate["receptive_field_feature_frames"] != EXPECTED_RECEPTIVE_FIELD:
        raise ValueError("cnn_ctc_v10 receptive field changed")
    if estimate["output_frames"] != EXPECTED_OUTPUT_FRAMES:
        raise ValueError("cnn_ctc_v10 output frame count changed")
    if package["network"]["estimated_parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("cnn_ctc_v10 declared parameter count changed")
    if package["network"]["estimated_macs_fixed_input"] != EXPECTED_MACS:
        raise ValueError("cnn_ctc_v10 declared MAC count changed")
    if package["output_contract"]["shape"] != [1, 128, 39]:
        raise ValueError("cnn_ctc_v10 output contract changed")
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
            "title": "cnn_ctc_v10 wider deep-dilated architecture screen",
            "hypothesis": (
                "v9's temporal geometry is efficient and restores healthy emission, "
                "but its higher training loss and slight CER regression indicate that "
                "96 channels may be capacity-limited under the fixed 12-epoch screen. "
                "Increasing only the v9-family width to 112 channels should recover "
                "quality while retaining substantial latency headroom."
            ),
            "rationale": (
                "v9 missed the v3-screen CER by only about 0.61% relative while cutting "
                "MA2450 p95 to about 12.09 ms and restoring emitted/reference characters "
                "to roughly 0.394. v10 keeps the same 128-frame, eight-block, kernel-7 "
                "dilated topology and changes only residual/stem width from 96 to 112."
            ),
            "changes": [
                f"branch from architecture-screen v3 control {DEFAULT_PARENT}",
                "reuse exact architecture-screen-v1 training manifest",
                "reuse full frozen ES2011 validation manifest",
                "keep logmel-v1 frontend, vocabulary, standard CTC and greedy decoder",
                "keep v3/v9 stem strides [2,2] and 128 CTC output frames",
                "keep eight kernel-7 residual blocks",
                "keep residual dilation schedule [1,2,3,4,4,3,2,1]",
                "increase second stem and residual width from 96 to 112 channels",
                "retain normalization-free ReLU/dropout/residual projection policy",
                "train for exactly 12 screen epochs with validation-CER selection",
                "use no augmentation, blank penalty, InterCTC or decoder change",
                "physically probe initialized ONNX/OpenVINO graph before training",
            ],
            "notes": (
                f"Screen control CER={PARENT_CER}. Acceptance requires no CER regression. "
                f"Promotion to a larger budget requires at least 3% relative CER "
                f"improvement (CER <= {PROMOTION_CER}). Sealed held-out evidence remains "
                "forbidden for architecture selection."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v10",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V10_ARCHITECTURE),
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
            / f"cnn_ctc_v10-screen-{head[:8]}-{SCREEN_MANIFEST_SHA256[:8]}"
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
    output["screen_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "cer": parent["metrics"]["cer"],
        "wer": parent["metrics"]["wer"],
        "latency_p95_ms": parent["metrics"]["inference_latency_p95_ms"],
        "promotion_cer": PROMOTION_CER,
        "promotion_relative_cer_ratio": PROMOTION_RELATIVE_CER_RATIO,
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
