"""Deterministic Phase 10 execution/orchestration helpers."""

from __future__ import annotations

import json
import pathlib
import shlex
from typing import Any, Mapping

SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
)
RSYNC_RSH = "ssh -o BatchMode=yes -o ConnectTimeout=10"


def validate_cnn_ctc_v1_executor_request(
    model_spec: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> None:
    if model_spec.get("model_id") != "cnn_ctc_v1":
        raise ValueError(
            f"unsupported Phase 10 model executor: {model_spec.get('model_id')!r}"
        )
    if model_spec.get("family") != "cnn_ctc":
        raise ValueError("cnn_ctc_v1 executor requires family 'cnn_ctc'")
    if model_spec.get("frontend") != {"kind": "logmel-v1"}:
        raise ValueError(
            "cnn_ctc_v1 executor requires exact frontend declaration "
            "{'kind': 'logmel-v1'}"
        )
    if model_spec.get("architecture") != {"kind": "cnn_ctc_v1"}:
        raise ValueError(
            "cnn_ctc_v1 executor requires exact architecture declaration "
            "{'kind': 'cnn_ctc_v1'}; Phase 10 must not ignore architecture fields"
        )
    export = model_spec.get("export")
    if export != {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True}:
        raise ValueError("cnn_ctc_v1 executor requires frozen ONNX opset-11 export")

    optimizer = train_config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("training optimizer declaration is missing")
    if dict(optimizer) != {
        "kind": "adam",
        "learning_rate": optimizer.get("learning_rate"),
    }:
        raise ValueError(
            "cnn_ctc_v1 executor supports only optimizer kind + learning_rate"
        )
    if optimizer.get("kind") != "adam":
        raise ValueError("cnn_ctc_v1 executor supports only Adam")


def training_command(
    *,
    root: pathlib.Path,
    build_dir: pathlib.Path,
    train_config: Mapping[str, Any],
) -> list[str]:
    command = [
        str(root / "scripts" / "train-cnn-ctc-v1.sh"),
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
    return command


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


def ssh_command(worker: str, remote_argv: list[str]) -> list[str]:
    if not worker or any(ch in worker for ch in "\r\n"):
        raise ValueError("invalid SSH worker alias")
    return ["ssh", *SSH_OPTIONS, worker, shlex.join(remote_argv)]


def rsync_push_command(
    *,
    worker: str,
    sources: list[pathlib.Path],
    remote_dir: str,
) -> list[str]:
    if not sources:
        raise ValueError("rsync push requires at least one source")
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
