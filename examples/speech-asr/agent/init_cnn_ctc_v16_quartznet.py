#!/usr/bin/env python3
"""Initialize cnn_ctc_v16 QuartzNet-15x5 architecture probe."""

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
from speech_asr.orchestration import V16_ARCHITECTURE  # noqa: E402

MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v16" / "model_spec.json"
V15_MODEL_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v15" / "model_spec.json"
MANAGER = SPEECH_ROOT / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"

SCREEN_DIR = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1"
SCREEN_MANIFEST = SCREEN_DIR / "train.manifest.jsonl"
SCREEN_PROVENANCE = SCREEN_DIR / "screen.json"
VALIDATION_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"

MODEL_SPEC_SHA256 = "70320bc878891f843312f0d8c1ae41ad7f414dbe2a0537e6766af8bf15f70a93"
VOCAB_SHA256 = "79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f"
SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"
VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
SCREEN_RECORDS = 3904
SCREEN_AUDIO_SECONDS = 6407.372
VALIDATION_RECORDS = 1273
VALIDATION_AUDIO_SECONDS = 2279.385
SCREEN_EPOCHS = 12
DEFAULT_PARENT = "exp-5d1209f5120388f8"
PARENT_ATTEMPT = "attempt-0001"
PARENT_CER = 0.8428267004549905
PARENT_WER = 0.9941697257611747
PARENT_P95_MS = 103.8926016
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
    request = load_json(parent / "request" / "experiment.json")
    model = load_json(parent / "request" / "model-spec.json")
    train = load_json(parent / "request" / "train-config.json")
    attempt = load_json(attempt_dir / "attempt.json")
    acceptance = load_json(attempt_dir / "results" / "acceptance-evaluation.json")
    result = load_json(attempt_dir / "results" / "result.json")
    if request.get("experiment_id") != DEFAULT_PARENT:
        raise ValueError("cnn_ctc_v15 parent identity changed")
    if model.get("model_id") != "cnn_ctc_v15":
        raise ValueError("architecture parent is not cnn_ctc_v15")
    if train.get("training_manifest", {}).get("sha256") != SCREEN_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v15 parent training manifest changed")
    if train.get("validation_manifest", {}).get("sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("cnn_ctc_v15 parent validation manifest changed")
    if train.get("epochs") != SCREEN_EPOCHS:
        raise ValueError("cnn_ctc_v15 parent epoch budget changed")
    if attempt.get("outcome") != "completed" or acceptance.get("status") != "accepted":
        raise ValueError("cnn_ctc_v15 parent did not complete cleanly")
    metrics = result.get("metrics", {})
    if metrics.get("cer") != PARENT_CER or metrics.get("wer") != PARENT_WER:
        raise ValueError("cnn_ctc_v15 parent accuracy changed")
    if metrics.get("inference_latency_p95_ms") != PARENT_P95_MS:
        raise ValueError("cnn_ctc_v15 parent latency changed")
    return result


def validate_model() -> dict:
    package = load_spec(MODEL_SPEC)
    parent = load_spec(V15_MODEL_SPEC)
    vocab = load_vocab(MODEL_SPEC.parent / "vocab.json")
    actual_hash = canonical_sha256(package)
    if actual_hash != MODEL_SPEC_SHA256:
        raise ValueError(f"cnn_ctc_v16 model spec differs from reviewed design: {actual_hash} != {MODEL_SPEC_SHA256}")
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise ValueError("cnn_ctc_v16 vocabulary differs from frozen vocabulary")
    if package["frontend"] != parent["frontend"] or package["input_contract"] != parent["input_contract"]:
        raise ValueError("cnn_ctc_v16 frontend/input contract changed")
    if package["decoder"] != parent["decoder"]:
        raise ValueError("cnn_ctc_v16 keeps greedy CTC for the acoustic screen")
    if package["export"] != parent["export"]:
        raise ValueError("cnn_ctc_v16 export contract changed")
    if package["network"]["kind"] != "quartznet-15x5-v1":
        raise ValueError("cnn_ctc_v16 is not the reviewed QuartzNet architecture")
    if package["network"]["block_groups"] != [
        {"name":"B1","channels":256,"kernel":33,"block_repeats":3,"module_repeats":5,"stride":1,"dilation":1,"padding":16},
        {"name":"B2","channels":256,"kernel":39,"block_repeats":3,"module_repeats":5,"stride":1,"dilation":1,"padding":19},
        {"name":"B3","channels":512,"kernel":51,"block_repeats":3,"module_repeats":5,"stride":1,"dilation":1,"padding":25},
        {"name":"B4","channels":512,"kernel":63,"block_repeats":3,"module_repeats":5,"stride":1,"dilation":1,"padding":31},
        {"name":"B5","channels":512,"kernel":75,"block_repeats":3,"module_repeats":5,"stride":1,"dilation":1,"padding":37},
    ]:
        raise ValueError("cnn_ctc_v16 QuartzNet 15x5 block schedule changed")
    estimate = model_resource_estimate(package)
    expected = (EXPECTED_PARAMETERS, EXPECTED_MACS, EXPECTED_RECEPTIVE_FIELD, EXPECTED_OUTPUT_FRAMES)
    actual = (estimate["parameters"], estimate["macs_fixed_input"], estimate["receptive_field_feature_frames"], estimate["output_frames"])
    if actual != expected:
        raise ValueError(f"cnn_ctc_v16 resource estimate changed: {actual} != {expected}")
    if package["output_contract"]["shape"] != [1, 256, 39]:
        raise ValueError("cnn_ctc_v16 output contract changed")
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
            "title": "cnn_ctc_v16 QuartzNet-15x5 architecture probe",
            "hypothesis": (
                "The v12-v15 home-grown 20M residual family failed to beat the 1.1M v11 Pareto point. "
                "A published QuartzNet-15x5 topology of comparable parameter size should provide a "
                "better-conditioned CTC acoustic encoder while preserving a clean VPU-logits / CPU-decoder split."
            ),
            "rationale": (
                "QuartzNet-15x5 is a published CTC acoustic architecture with 15 residual blocks, five "
                "time-channel separable modules per block, 64-bin input, and about 18.9M parameters. "
                "Open Model Zoo has an OpenVINO QuartzNet-15x5 conversion path. v15 regressed to CER "
                "0.8428267004549905 with 86.68% blank frames, so local tuning of that topology is closed."
            ),
            "changes": [
                f"parent transition to completed v15 experiment {DEFAULT_PARENT}",
                "replace the v12-v15 residual-temporal graph with QuartzNet-15x5",
                "use depthwise temporal plus pointwise channel convolutions",
                "use C1 256xk33 stride 2, then B1-B5 kernels 33/39/51/63/75",
                "repeat each QuartzNet block type three times and each block module five times",
                "use projected residuals and BatchNorm/ReLU/dropout 0.2",
                "use C2 512xk87 dilation 2 and C3 1024xk1",
                "adapt only the CTC head to the frozen 39-token project vocabulary",
                "retain the frozen logmel-v1 frontend and ES2011 screen data",
                "retain Adam cosine 3e-4 to 3e-5 for this architecture-only screen",
                "retain 12 screen epochs and validation-CER checkpoint selection",
                "run initialized ONNX/OpenVINO/MYRIAD compatibility before training",
                "keep greedy decoding for acoustic selection; CPU prefix-beam/LM is the next system layer after acoustic validation",
            ],
            "notes": (
                "No numeric CER or latency acceptance threshold is imposed. Hardware conversion/execution "
                "and numerical agreement remain hard gates. Sealed held-out evidence remains unavailable."
            ),
        }
        experiment_model = {
            "schema": "speech-asr/experiment-model-spec",
            "version": 1,
            "model_id": "cnn_ctc_v16",
            "family": "cnn_ctc",
            "frontend": {"kind": "logmel-v1"},
            "architecture": dict(V16_ARCHITECTURE),
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
            "optimizer": {"kind": "adam", "learning_rate": float(package["training"]["learning_rate"])},
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
            "required_gates": ["training","onnx_export","openvino_conversion","myriad_execution","accuracy_evaluation"],
            "thresholds": {
                "max_wer": None,
                "max_cer": None,
                "max_realtime_factor": None,
                "max_latency_p95_ms": None,
                "min_frame_argmax_agreement": 1.0,
            },
            "retry_policy": {
                "max_attempts": 3,
                "retryable_failure_classes": ["transport_preflight","worker_busy","hardware_transient"],
            },
        }
        draft = ROOT / "work" / "speech-asr" / "experiment-drafts" / f"cnn_ctc_v16-quartznet-{head[:8]}-{SCREEN_MANIFEST_SHA256[:8]}"
        paths = {name: draft / filename for name, filename in {
            "proposal":"proposal.json","model":"model-spec.json","train":"train-config.json","acceptance":"acceptance.json"
        }.items()}
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
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout.strip())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = json.loads(proc.stdout)
    output["quartznet_reference"] = {
        "parent": DEFAULT_PARENT,
        "attempt": PARENT_ATTEMPT,
        "parent_cer": parent["metrics"]["cer"],
        "parent_wer": parent["metrics"]["wer"],
        "parent_latency_p95_ms": parent["metrics"]["inference_latency_p95_ms"],
        "v15_blank_frame_fraction": 0.8668262967005881,
        "v14_cer": 0.7898404653573691,
        "v11_cer": 0.7865576225306686,
        "v11_latency_p95_ms": 14.9809824,
        "published_architecture": "QuartzNet-15x5",
        "published_parameter_scale_millions": 18.9,
        "openvino_reference_model": "Open Model Zoo quartznet-15x5-en",
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
