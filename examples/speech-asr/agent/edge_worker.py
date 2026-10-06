#!/usr/bin/env python3
"""Validate and execute one hash-bound speech-ASR deployment on edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import (  # noqa: E402
    validate_deployment_manifest,
    validate_edge_worker_result,
    validate_experiment_result,
)

EXIT_PREFLIGHT = 20
EXIT_EXECUTION = 21
EXIT_RESULT = 22


@dataclass
class WorkerError(Exception):
    message: str
    failure_class: str
    exit_code: int

    def __str__(self) -> str:
        return self.message


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise WorkerError(
            "could not resolve worker Git revision",
            "transport_preflight",
            EXIT_PREFLIGHT,
        )
    return proc.stdout.strip()


def safe_git_head() -> str | None:
    try:
        return git_head()
    except Exception:
        return None


def safe_repo_path(relative: str) -> pathlib.Path:
    value = pathlib.PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise WorkerError(
            f"unsafe repository-relative path: {relative!r}",
            "request_config",
            EXIT_PREFLIGHT,
        )
    return ROOT.joinpath(*value.parts)


def write_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def bundle_artifact_path(bundle: pathlib.Path, ref: dict[str, str]) -> pathlib.Path:
    name = pathlib.PurePosixPath(ref["path"]).name
    if not name:
        raise WorkerError(
            f"deployment artifact has invalid path: {ref['path']!r}",
            "request_config",
            EXIT_PREFLIGHT,
        )
    return bundle / name


def verify_bundle(manifest: dict[str, Any], bundle: pathlib.Path) -> dict[str, pathlib.Path]:
    resolved: dict[str, pathlib.Path] = {}
    for name, ref in manifest["artifacts"].items():
        path = bundle_artifact_path(bundle, ref)
        if not path.is_file():
            raise WorkerError(
                f"deployment artifact missing from bundle: {name}: {path}",
                "transport_preflight",
                EXIT_PREFLIGHT,
            )
        actual = sha256_path(path)
        if actual != ref["sha256"]:
            raise WorkerError(
                f"deployment artifact hash mismatch for {name}: "
                f"{actual} != {ref['sha256']}",
                "transport_preflight",
                EXIT_PREFLIGHT,
            )
        resolved[name] = path
    return resolved


def validate_environment(manifest: dict[str, Any]) -> pathlib.Path:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise WorkerError(
            f"worker host must be arm64/aarch64, got {platform.machine()!r}",
            "transport_preflight",
            EXIT_PREFLIGHT,
        )

    head = git_head()
    if head != manifest["worker_commit"] or head != manifest["controller_commit"]:
        raise WorkerError(
            f"worker revision mismatch: HEAD={head}, "
            f"required={manifest['worker_commit']}",
            "transport_preflight",
            EXIT_PREFLIGHT,
        )

    manifest_path = safe_repo_path(manifest["benchmark"]["manifest_path"])
    if not manifest_path.is_file():
        raise WorkerError(
            f"dataset manifest missing: {manifest_path}",
            "transport_preflight",
            EXIT_PREFLIGHT,
        )
    actual_manifest_sha = sha256_path(manifest_path)
    expected_manifest_sha = manifest["benchmark"]["manifest_sha256"]
    if actual_manifest_sha != expected_manifest_sha:
        raise WorkerError(
            f"dataset manifest hash mismatch: {actual_manifest_sha} != "
            f"{expected_manifest_sha}",
            "transport_preflight",
            EXIT_PREFLIGHT,
        )
    return manifest_path


def require_supported_request(
    manifest: dict[str, Any],
    artifacts: dict[str, pathlib.Path],
) -> tuple[pathlib.Path, pathlib.Path]:
    model_id = manifest["model"]["id"]
    if model_id not in {
        "cnn_ctc_v1",
        "cnn_ctc_v2",
        "cnn_ctc_v3",
        "cnn_ctc_v4",
        "cnn_ctc_v5",
        "cnn_ctc_v6",
        "cnn_ctc_v7",
        "cnn_ctc_v8",
        "cnn_ctc_v9",
        "cnn_ctc_v10",
        "cnn_ctc_v11",
        "cnn_ctc_v12",
        "cnn_ctc_v13",
        "cnn_ctc_v14",
    }:
        raise WorkerError(
            f"unsupported Phase 11 model executor: {model_id!r}",
            "request_config",
            EXIT_PREFLIGHT,
        )
    if manifest["evaluator"]["id"] != "cnn-ctc-myriad-v1":
        raise WorkerError(
            f"unsupported edge evaluator: {manifest['evaluator']['id']!r}",
            "request_config",
            EXIT_PREFLIGHT,
        )
    try:
        xml = artifacts["openvino_xml"]
        binary = artifacts["openvino_bin"]
    except KeyError as exc:
        raise WorkerError(
            f"{model_id} deployment requires openvino_xml and openvino_bin",
            "request_config",
            EXIT_PREFLIGHT,
        ) from exc
    if xml.name != f"{model_id}.xml" or binary.name != f"{model_id}.bin":
        raise WorkerError(
            f"{model_id} deployment artifact filenames must remain "
            f"{model_id}.xml and {model_id}.bin",
            "request_config",
            EXIT_PREFLIGHT,
        )
    return xml, binary


def run_evaluator(
    manifest: dict[str, Any],
    dataset_manifest: pathlib.Path,
    bundle: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "hardware.json"
    evaluation_work = output_dir / "evaluation"
    model_id = manifest["model"]["id"]
    evaluator = {
        "cnn_ctc_v1": "evaluate-cnn-ctc-v1.sh",
        "cnn_ctc_v2": "evaluate-cnn-ctc-v2.sh",
        "cnn_ctc_v3": "evaluate-cnn-ctc-v3.sh",
        "cnn_ctc_v4": "evaluate-cnn-ctc-v4.sh",
        "cnn_ctc_v5": "evaluate-cnn-ctc-v5.sh",
        "cnn_ctc_v6": "evaluate-cnn-ctc-v6.sh",
        "cnn_ctc_v7": "evaluate-cnn-ctc-v7.sh",
        "cnn_ctc_v8": "evaluate-cnn-ctc-v8.sh",
        "cnn_ctc_v9": "evaluate-cnn-ctc-v9.sh",
        "cnn_ctc_v10": "evaluate-cnn-ctc-v10.sh",
        "cnn_ctc_v11": "evaluate-cnn-ctc-v11.sh",
        "cnn_ctc_v12": "evaluate-cnn-ctc-v12.sh",
        "cnn_ctc_v13": "evaluate-cnn-ctc-v13.sh",
        "cnn_ctc_v14": "evaluate-cnn-ctc-v14.sh",
    }.get(model_id)
    if evaluator is None:
        raise WorkerError(
            f"unsupported Phase 11 model evaluator: {model_id!r}",
            "request_config",
            EXIT_PREFLIGHT,
        )
    command = [
        str(ROOT / "scripts" / evaluator),
        "--platform",
        "arm64",
        "--manifest",
        str(dataset_manifest),
        "--benchmark-id",
        manifest["benchmark"]["id"],
        "--ir-dir",
        str(bundle),
        "--experiment-id",
        manifest["experiment_id"],
        "--work-dir",
        str(evaluation_work),
        "--output",
        str(result_path),
    ]
    log_path = output_dir / "evaluator.log"
    output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc_live = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc_live.stdout is None:
            raise WorkerError(
                "edge evaluator stdout is unavailable",
                "hardware_execution",
                EXIT_EXECUTION,
            )
        for line in proc_live.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            output.append(line)
        returncode = proc_live.wait()
    proc = subprocess.CompletedProcess(
        command,
        returncode,
        "".join(output),
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-30:])
        raise WorkerError(
            f"edge evaluator failed with status {proc.returncode}\n{tail}",
            "hardware_execution",
            EXIT_EXECUTION,
        )
    if not result_path.is_file():
        raise WorkerError(
            "edge evaluator completed without hardware result",
            "result_contract",
            EXIT_RESULT,
        )
    return result_path


def validate_hardware_result(
    manifest: dict[str, Any],
    result_path: pathlib.Path,
) -> dict[str, Any]:
    try:
        result = validate_experiment_result(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise WorkerError(
            f"invalid hardware result: {exc}",
            "result_contract",
            EXIT_RESULT,
        ) from exc

    errors: list[str] = []
    if result["experiment_id"] != manifest["experiment_id"]:
        errors.append("experiment_id mismatch")
    if result["status"] != "completed":
        errors.append("hardware result is not completed")
    if result["benchmark"]["id"] != manifest["benchmark"]["id"]:
        errors.append("benchmark id mismatch")
    if result["benchmark"]["manifest_sha256"] != manifest["benchmark"]["manifest_sha256"]:
        errors.append("benchmark manifest hash mismatch")
    runtime = result["runtime"]
    if runtime["repo_commit"] != manifest["worker_commit"]:
        errors.append("runtime repository commit mismatch")
    if runtime["backend"] != "MYRIAD" or runtime["target"] != "arm64":
        errors.append("runtime backend/target mismatch")

    result_artifacts = result["model"]["artifact_sha256"]
    for name, ref in manifest["artifacts"].items():
        basename = pathlib.PurePosixPath(ref["path"]).name
        if result_artifacts.get(basename) != ref["sha256"]:
            errors.append(f"model artifact hash mismatch for {basename}")

    if errors:
        raise WorkerError(
            "hardware result provenance mismatch: " + "; ".join(errors),
            "result_contract",
            EXIT_RESULT,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=pathlib.Path, required=True)
    parser.add_argument("--bundle-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    worker_result_path = args.output_dir / "worker-result.json"
    deployment_sha = None
    experiment_id = None
    attempt_id = None

    try:
        if not args.deployment.is_file():
            raise WorkerError(
                f"deployment manifest missing: {args.deployment}",
                "transport_preflight",
                EXIT_PREFLIGHT,
            )
        deployment_sha = sha256_path(args.deployment)
        manifest = validate_deployment_manifest(
            json.loads(args.deployment.read_text(encoding="utf-8"))
        )
        experiment_id = manifest["experiment_id"]
        attempt_id = manifest["attempt_id"]
        dataset_manifest = validate_environment(manifest)
        artifacts = verify_bundle(manifest, args.bundle_dir)
        require_supported_request(manifest, artifacts)
        hardware_path = run_evaluator(
            manifest,
            dataset_manifest,
            args.bundle_dir,
            args.output_dir,
        )
        result = validate_hardware_result(manifest, hardware_path)
        worker_result = {
            "schema": "speech-asr/edge-worker-result",
            "version": 1,
            "status": "completed",
            "failure_class": None,
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "worker_commit": git_head(),
            "deployment_sha256": deployment_sha,
            "hardware_result": {
                "path": str(hardware_path),
                "sha256": sha256_path(hardware_path),
            },
            "evaluator_log": {
                "path": str(args.output_dir / "evaluator.log"),
                "sha256": sha256_path(args.output_dir / "evaluator.log"),
            },
            "metrics": result["metrics"],
        }
        validate_edge_worker_result(worker_result)
        write_json(worker_result_path, worker_result)
    except WorkerError as exc:
        worker_result = {
            "schema": "speech-asr/edge-worker-result",
            "version": 1,
            "status": "failed",
            "failure_class": exc.failure_class,
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "worker_commit": safe_git_head(),
            "deployment_sha256": deployment_sha,
            "diagnostics": {"summary": exc.message},
        }
        try:
            validate_edge_worker_result(worker_result)
            write_json(worker_result_path, worker_result)
        except (OSError, ValueError):
            pass
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        worker_result = {
            "schema": "speech-asr/edge-worker-result",
            "version": 1,
            "status": "failed",
            "failure_class": "result_contract",
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "worker_commit": None,
            "deployment_sha256": deployment_sha,
            "diagnostics": {"summary": str(exc)},
        }
        try:
            validate_edge_worker_result(worker_result)
            write_json(worker_result_path, worker_result)
        except (OSError, ValueError):
            pass
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RESULT

    print(json.dumps(worker_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
