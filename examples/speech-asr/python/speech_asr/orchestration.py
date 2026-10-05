"""Deterministic Phase 10 execution/orchestration helpers."""

from __future__ import annotations

import json
import pathlib
import re
import shlex
from typing import Any, Mapping

SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
)
RSYNC_RSH = "ssh -o BatchMode=yes -o ConnectTimeout=10"
_WORKER_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
PRETRAINING_MYRIAD_MAX_ABS_ERROR = 0.01


V2_ARCHITECTURE = {
    "kind": "residual-temporal-v1",
    "stem_channels": [64, 96],
    "stem_kernels": [5, 5],
    "stem_strides": [2, 2],
    "residual_channels": 96,
    "residual_kernels": [11, 19, 27, 35, 43],
    "normalization": "batchnorm",
    "activation": "relu",
    "dropout": 0.1,
}


V3_ARCHITECTURE = {
    "kind": "residual-temporal-v2",
    "stem_channels": [64, 96],
    "stem_kernels": [5, 5],
    "stem_strides": [2, 2],
    "residual_channels": 96,
    "residual_kernels": [11, 19, 27, 35, 43],
    "normalization": "none",
    "activation": "relu",
    "dropout": 0.1,
    "residual_projection_init": "kaiming_scaled_0.01",
}


def executor_model_basename(model_id: str) -> str:
    if model_id in {"cnn_ctc_v1", "cnn_ctc_v2", "cnn_ctc_v3"}:
        return model_id
    raise ValueError(f"unsupported Phase 11 model executor: {model_id!r}")


def validate_model_executor_request(
    model_spec: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> None:
    model_id = str(model_spec.get("model_id"))
    if model_id not in {"cnn_ctc_v1", "cnn_ctc_v2", "cnn_ctc_v3"}:
        raise ValueError(f"unsupported Phase 11 model executor: {model_id!r}")
    if model_spec.get("family") != "cnn_ctc":
        raise ValueError(f"{model_id} executor requires family 'cnn_ctc'")
    if model_spec.get("frontend") != {"kind": "logmel-v1"}:
        raise ValueError(
            f"{model_id} executor requires exact frontend declaration "
            "{'kind': 'logmel-v1'}"
        )

    expected_architecture = {
        "cnn_ctc_v1": {"kind": "cnn_ctc_v1"},
        "cnn_ctc_v2": V2_ARCHITECTURE,
        "cnn_ctc_v3": V3_ARCHITECTURE,
    }[model_id]
    if model_spec.get("architecture") != expected_architecture:
        raise ValueError(
            f"{model_id} executor requires its exact frozen architecture; "
            "Phase 11 must not ignore architecture fields"
        )
    export = model_spec.get("export")
    if export != {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True}:
        raise ValueError(f"{model_id} executor requires frozen ONNX opset-11 export")

    optimizer = train_config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("training optimizer declaration is missing")
    if dict(optimizer) != {
        "kind": "adam",
        "learning_rate": optimizer.get("learning_rate"),
    }:
        raise ValueError(
            f"{model_id} executor supports only optimizer kind + learning_rate"
        )
    if optimizer.get("kind") != "adam":
        raise ValueError(f"{model_id} executor supports only Adam")


def validate_cnn_ctc_v1_executor_request(
    model_spec: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> None:
    validate_model_executor_request(model_spec, train_config)


def compatibility_probe_command(
    *,
    root: pathlib.Path,
    build_dir: pathlib.Path,
    model_spec: Mapping[str, Any],
) -> list[str] | None:
    model_id = str(model_spec["model_id"])
    if model_id == "cnn_ctc_v1":
        return None
    if model_id in {"cnn_ctc_v2", "cnn_ctc_v3"}:
        return [
            str(root / "scripts" / f"probe-{model_id.replace('_', '-')}.sh"),
            "--work-dir",
            str(build_dir.parent / "compatibility-probe"),
        ]
    raise ValueError(f"unsupported Phase 11 model executor: {model_id!r}")


def training_command(
    *,
    root: pathlib.Path,
    build_dir: pathlib.Path,
    train_config: Mapping[str, Any],
    model_spec: Mapping[str, Any] | None = None,
) -> list[str]:
    model_id = (
        "cnn_ctc_v1"
        if model_spec is None
        else str(model_spec["model_id"])
    )
    executable = {
        "cnn_ctc_v1": "train-cnn-ctc-v1.sh",
        "cnn_ctc_v2": "train-cnn-ctc-v2.sh",
        "cnn_ctc_v3": "train-cnn-ctc-v3.sh",
    }.get(model_id)
    if executable is None:
        raise ValueError(f"unsupported Phase 11 model executor: {model_id!r}")
    command = [
        str(root / "scripts" / executable),
        "--manifest",
        str(root / train_config["training_manifest"]["path"]),
        "--validation-manifest",
        str(root / train_config["validation_manifest"]["path"]),
        "--device",
        str(train_config["device"]),
        "--epochs",
        str(train_config["epochs"]),
        "--batch-size",
        str(train_config["batch_size"]),
        "--seed",
        str(train_config["seed"]),
        "--learning-rate",
        str(train_config["optimizer"]["learning_rate"]),
        "--work-dir",
        str(build_dir),
    ]
    max_samples = train_config.get("max_samples")
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    checkpoint_selection = train_config.get("checkpoint_selection")
    if checkpoint_selection is not None:
        command.extend([
            "--checkpoint-selection",
            str(checkpoint_selection),
        ])
    augmentation = train_config.get("augmentation")
    if augmentation is not None:
        if model_id != "cnn_ctc_v3":
            raise ValueError(
                "training augmentation is supported only by cnn_ctc_v3"
            )
        command.extend([
            "--augmentation-kind",
            str(augmentation["kind"]),
            "--frequency-masks",
            str(augmentation["frequency_masks"]),
            "--frequency-max-width",
            str(augmentation["frequency_max_width"]),
            "--time-masks",
            str(augmentation["time_masks"]),
            "--time-max-width",
            str(augmentation["time_max_width"]),
            "--time-max-fraction",
            str(augmentation["time_max_fraction"]),
            "--augmentation-mask-value",
            str(augmentation["mask_value"]),
            "--augmentation-seed-offset",
            str(augmentation["seed_offset"]),
        ])
    return command


def validate_training_device_evidence(
    *,
    train_config: Mapping[str, Any],
    training_result: Mapping[str, Any],
) -> dict[str, Any]:
    requested = str(train_config.get("device"))
    device_used = training_result.get("device_used")
    model_device = training_result.get("model_device", device_used)
    logits_device = training_result.get("observed_logits_device")
    ctc_loss_device = training_result.get("ctc_loss_device")
    cuda_available = training_result.get("cuda_available")
    cuda_name = training_result.get("cuda_device_name")
    peak_allocated = training_result.get(
        "cuda_peak_memory_allocated_bytes",
        0,
    )
    peak_reserved = training_result.get(
        "cuda_peak_memory_reserved_bytes",
        0,
    )

    if requested == "cuda":
        for label, value in (
            ("device_used", device_used),
            ("model_device", model_device),
            ("observed_logits_device", logits_device),
        ):
            if not isinstance(value, str) or not value.startswith("cuda"):
                raise ValueError(
                    f"CUDA training requested but {label}={value!r}"
                )
        if cuda_available is not True:
            raise ValueError(
                "CUDA training requested but cuda_available is not true"
            )
        if (
            not isinstance(peak_allocated, int)
            or isinstance(peak_allocated, bool)
            or peak_allocated <= 0
        ):
            raise ValueError(
                "CUDA training requested but peak allocated VRAM is not positive"
            )

    return {
        "requested": requested,
        "device_used": device_used,
        "model_device": model_device,
        "observed_logits_device": logits_device,
        "ctc_loss_device": ctc_loss_device,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_name,
        "cuda_peak_memory_allocated_bytes": peak_allocated,
        "cuda_peak_memory_reserved_bytes": peak_reserved,
    }


def load_compatibility_result(build_dir: pathlib.Path) -> dict[str, Any]:
    paths = {
        "onnx_comparison": build_dir / "export" / "onnx-comparison.json",
        "ir_validation": build_dir / "openvino" / "fp16" / "ir-validation.json",
        "openvino_artifacts": build_dir / "openvino" / "fp16" / "artifacts.json",
    }
    documents: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"compatibility evidence missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"compatibility evidence must be an object: {path}")
        documents[name] = value

    if documents["ir_validation"].get("status") != "valid":
        raise ValueError("OpenVINO IR compatibility validation is not valid")
    comparison = documents["onnx_comparison"].get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("ONNX comparison evidence is missing comparison metrics")

    return {
        "schema": "speech-asr/execution-compatibility",
        "version": 1,
        "status": "completed",
        **documents,
    }


def pretraining_myriad_compatibility(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    max_abs_error = comparison.get("max_abs_error")
    reasons: list[str] = []
    if (
        not isinstance(max_abs_error, (int, float))
        or isinstance(max_abs_error, bool)
        or max_abs_error < 0
    ):
        reasons.append("max_abs_error is missing or invalid")
    elif max_abs_error > PRETRAINING_MYRIAD_MAX_ABS_ERROR:
        reasons.append(
            f"max_abs_error {max_abs_error!r} exceeds "
            f"{PRETRAINING_MYRIAD_MAX_ABS_ERROR!r}"
        )

    return {
        "status": "accepted" if not reasons else "rejected",
        "max_abs_error_limit": PRETRAINING_MYRIAD_MAX_ABS_ERROR,
        "max_abs_error": max_abs_error,
        "frame_argmax_agreement": comparison.get("frame_argmax_agreement"),
        "frame_argmax_mismatches": comparison.get("frame_argmax_mismatches"),
        "min_reference_top2_margin": comparison.get(
            "min_reference_top2_margin"
        ),
        "reasons": reasons,
    }


def acceptance_evaluation(
    *,
    policy: Mapping[str, Any],
    training_present: bool,
    compatibility: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = compatibility.get("onnx_comparison", {}).get("comparison", {})
    hardware_metrics = hardware.get("metrics", {})
    gates = {
        "training": bool(training_present),
        "onnx_export": compatibility.get("status") == "completed",
        "openvino_conversion": (
            compatibility.get("ir_validation", {}).get("status") == "valid"
        ),
        "myriad_execution": (
            hardware.get("status") == "completed"
            and hardware.get("runtime", {}).get("backend") == "MYRIAD"
        ),
        "accuracy_evaluation": (
            "wer" in hardware_metrics and "cer" in hardware_metrics
        ),
    }

    reasons: list[str] = []
    required = list(policy["required_gates"])
    for gate in required:
        if not gates.get(gate, False):
            reasons.append(f"required gate failed: {gate}")

    thresholds = policy["thresholds"]
    checks: dict[str, dict[str, Any]] = {}

    def maximum(name: str, metric: str, actual: Any) -> None:
        limit = thresholds.get(name)
        if limit is None:
            checks[name] = {"applied": False, "limit": None, "actual": actual, "passed": True}
            return
        passed = isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual <= limit
        checks[name] = {"applied": True, "limit": limit, "actual": actual, "passed": passed}
        if not passed:
            reasons.append(f"{metric} {actual!r} exceeds maximum {limit!r}")

    maximum("max_wer", "WER", hardware_metrics.get("wer"))
    maximum("max_cer", "CER", hardware_metrics.get("cer"))
    maximum(
        "max_realtime_factor",
        "realtime_factor",
        hardware_metrics.get("realtime_factor"),
    )
    maximum(
        "max_latency_p95_ms",
        "inference_latency_p95_ms",
        hardware_metrics.get("inference_latency_p95_ms"),
    )

    minimum_name = "min_frame_argmax_agreement"
    minimum_limit = thresholds.get(minimum_name)
    actual_agreement = comparison.get("frame_argmax_agreement")
    if minimum_limit is None:
        checks[minimum_name] = {
            "applied": False,
            "limit": None,
            "actual": actual_agreement,
            "passed": True,
        }
    else:
        passed = (
            isinstance(actual_agreement, (int, float))
            and not isinstance(actual_agreement, bool)
            and actual_agreement >= minimum_limit
        )
        checks[minimum_name] = {
            "applied": True,
            "limit": minimum_limit,
            "actual": actual_agreement,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"frame_argmax_agreement {actual_agreement!r} is below minimum "
                f"{minimum_limit!r}"
            )

    return {
        "schema": "speech-asr/acceptance-evaluation",
        "version": 1,
        "status": "accepted" if not reasons else "rejected",
        "gates": gates,
        "required_gates": required,
        "threshold_checks": checks,
        "reasons": reasons,
    }


def validate_worker_alias(worker: str) -> None:
    if (
        not worker
        or worker.startswith("-")
        or _WORKER_RE.fullmatch(worker) is None
    ):
        raise ValueError("invalid SSH worker alias")


def ssh_command(worker: str, remote_argv: list[str]) -> list[str]:
    validate_worker_alias(worker)
    return ["ssh", *SSH_OPTIONS, worker, shlex.join(remote_argv)]


def rsync_push_command(
    *,
    worker: str,
    sources: list[pathlib.Path],
    remote_dir: str,
) -> list[str]:
    if not sources:
        raise ValueError("rsync push requires at least one source")
    validate_worker_alias(worker)
    return [
        "rsync",
        "-a",
        "--checksum",
        "-e",
        RSYNC_RSH,
        "--",
        *[str(path) for path in sources],
        f"{worker}:{remote_dir.rstrip('/')}/",
    ]


def rsync_pull_command(
    *,
    worker: str,
    remote_dir: str,
    local_dir: pathlib.Path,
) -> list[str]:
    validate_worker_alias(worker)
    return [
        "rsync",
        "-a",
        "--checksum",
        "-e",
        RSYNC_RSH,
        "--",
        f"{worker}:{remote_dir.rstrip('/')}/",
        str(local_dir) + "/",
    ]


def remote_failure_class(returncode: int) -> tuple[str, str]:
    if returncode == 75:
        return "blocked", "worker_busy"
    if returncode == 20:
        return "blocked", "transport_preflight"
    if returncode == 21:
        return "failed", "hardware_execution"
    if returncode == 22:
        return "failed", "result_contract"
    return "failed", "hardware_execution"
