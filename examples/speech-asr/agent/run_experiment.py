#!/usr/bin/env python3
"""Execute one approved Phase 9 speech-ASR experiment from oberon to edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import (  # noqa: E402
    canonical_json_sha256,
    validate_acceptance_policy,
    validate_edge_worker_result,
    validate_experiment_model_spec,
    validate_experiment_result,
    validate_train_config,
)
from speech_asr.experiment import (  # noqa: E402
    load_json,
    sha256_path,
    validate_request_identity,
    write_json,
)
from speech_asr.orchestration import (  # noqa: E402
    SSH_OPTIONS,
    acceptance_evaluation,
    compatibility_probe_command,
    executor_model_basename,
    load_compatibility_result,
    pretraining_myriad_compatibility,
    remote_failure_class,
    rsync_pull_command,
    rsync_push_command,
    ssh_command,
    training_command,
    validate_model_executor_request,
    validate_training_device_evidence,
)

MANAGER = ROOT / "examples" / "speech-asr" / "tools" / "manage_experiment.py"
PYTHON = ROOT / "scripts" / "python.sh"
BOOTSTRAP = ROOT / "scripts" / "edge-speech-bootstrap.sh"


class ExecutionFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        outcome: str,
        failure_class: str,
    ):
        super().__init__(message)
        self.outcome = outcome
        self.failure_class = failure_class


def run_process(
    command: list[str],
    *,
    log_path: pathlib.Path | None = None,
    input_text: str | None = None,
    cwd: pathlib.Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def manager(*args: str) -> dict[str, Any]:
    proc = run_process([str(PYTHON), str(MANAGER), *args])
    if proc.returncode != 0:
        raise RuntimeError(
            "experiment manager failed:\n"
            + "\n".join(proc.stdout.splitlines()[-30:])
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("experiment manager returned no JSON result")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError("experiment manager result must be an object")
    return value


def git_head() -> str:
    proc = run_process(["git", "rev-parse", "HEAD"])
    if proc.returncode != 0:
        raise ExecutionFailure(
            "could not resolve controller Git revision",
            outcome="failed",
            failure_class="request_config",
        )
    return proc.stdout.strip()


def tracked_tree_is_clean() -> bool:
    proc = run_process(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    )
    return proc.returncode == 0 and not proc.stdout.strip()


def read_request_bundle(
    experiment_dir: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_path = experiment_dir / "request" / "experiment.json"
    if not request_path.is_file():
        raise ExecutionFailure(
            f"experiment request missing: {request_path}",
            outcome="failed",
            failure_class="request_config",
        )
    try:
        request = validate_request_identity(load_json(request_path))
        model_spec = validate_experiment_model_spec(
            load_json(experiment_dir / "request" / "model-spec.json")
        )
        train_config = validate_train_config(
            load_json(experiment_dir / "request" / "train-config.json")
        )
        acceptance = validate_acceptance_policy(
            load_json(experiment_dir / "request" / "acceptance.json")
        )
    except Exception as exc:
        raise ExecutionFailure(
            f"invalid approved experiment bundle: {exc}",
            outcome="failed",
            failure_class="request_config",
        ) from exc

    documents = {
        "model_spec": model_spec,
        "train_config": train_config,
        "acceptance": acceptance,
    }
    for name, document in documents.items():
        expected = request["documents"][name]["sha256"]
        actual = canonical_json_sha256(document)
        if expected != actual:
            raise ExecutionFailure(
                f"{name} hash mismatch: {actual} != {expected}",
                outcome="failed",
                failure_class="request_config",
            )
    return request, model_spec, train_config, acceptance


def verify_declared_manifest(ref: dict[str, Any], label: str) -> pathlib.Path:
    path = ROOT / ref["path"]
    if not path.is_file():
        raise ExecutionFailure(
            f"{label} missing: {path}",
            outcome="failed",
            failure_class="request_config",
        )
    actual = sha256_path(path)
    if actual != ref["sha256"]:
        raise ExecutionFailure(
            f"{label} hash mismatch: {actual} != {ref['sha256']}",
            outcome="failed",
            failure_class="request_config",
        )
    return path


def remote_home(worker: str, log: pathlib.Path) -> str:
    proc = run_process(
        ssh_command(worker, ["sh", "-c", 'printf "%s\\n" "$HOME"']),
        log_path=log,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        raise ExecutionFailure(
            "SSH connectivity/home preflight failed",
            outcome="blocked",
            failure_class="transport_preflight",
        )
    home = lines[-1]
    if not home.startswith("/"):
        raise ExecutionFailure(
            f"edge HOME is not an absolute path: {home!r}",
            outcome="blocked",
            failure_class="transport_preflight",
        )
    return home


def remote_run(
    worker: str,
    argv: list[str],
    *,
    log_path: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    return run_process(ssh_command(worker, argv), log_path=log_path)


def terminalize(
    *,
    experiment_dir: pathlib.Path,
    attempt_id: str,
    failure: ExecutionFailure,
    diagnostics_path: pathlib.Path,
) -> None:
    diagnostics = {
        "summary": str(failure),
        "failure_class": failure.failure_class,
    }
    write_json(diagnostics_path, diagnostics)
    manager(
        "transition",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--state",
        "AWAIT_REVIEW",
        "--outcome",
        failure.outcome,
        "--failure-class",
        failure.failure_class,
        "--diagnostics-json",
        str(diagnostics_path),
    )


def worker_paths(
    *,
    home: str,
    base_repo: str | None,
    worker_repo: str | None,
) -> tuple[str, str]:
    base = base_repo or f"{home}/workspace/movidius-openvino-rpi5"
    worker = worker_repo or f"{home}/workspace/movidius-openvino-rpi5-worker"
    for value in (base, worker):
        if (
            not value.startswith("/")
            or any(ch in value for ch in "\r\n")
            or any(ch.isspace() for ch in value)
        ):
            raise ExecutionFailure(
                f"edge repository path must be absolute and whitespace-free: {value!r}",
                outcome="failed",
                failure_class="request_config",
            )
    return base, worker


def bootstrap_edge(
    *,
    worker: str,
    base_repo: str,
    worker_repo: str,
    commit: str,
    log_path: pathlib.Path,
) -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    command = ssh_command(
        worker,
        ["bash", "-s", "--", base_repo, worker_repo, commit],
    )
    proc = run_process(command, log_path=log_path, input_text=script)
    if proc.returncode != 0:
        raise ExecutionFailure(
            "edge worker bootstrap failed:\n"
            + "\n".join(proc.stdout.splitlines()[-20:]),
            outcome="blocked",
            failure_class="transport_preflight",
        )


def preflight_edge(
    *,
    worker: str,
    worker_repo: str,
    benchmark: dict[str, Any],
    log_path: pathlib.Path,
) -> None:
    if shutil.which("rsync") is None:
        raise ExecutionFailure(
            "rsync is required on the controller",
            outcome="blocked",
            failure_class="transport_preflight",
        )
    rsync_check = remote_run(
        worker,
        ["sh", "-c", "command -v rsync >/dev/null"],
        log_path=log_path.with_name("edge-rsync-preflight.log"),
    )
    if rsync_check.returncode != 0:
        raise ExecutionFailure(
            "rsync is required on the edge worker",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    command = [
        "bash",
        f"{worker_repo}/scripts/edge-speech-preflight.sh",
        "--manifest",
        benchmark["manifest_path"],
        "--manifest-sha256",
        benchmark["manifest_sha256"],
    ]
    proc = remote_run(worker, command, log_path=log_path)
    if proc.returncode != 0:
        raise ExecutionFailure(
            "edge dataset/runtime preflight failed:\n"
            + "\n".join(proc.stdout.splitlines()[-30:]),
            outcome="blocked",
            failure_class="transport_preflight",
        )


def physical_compatibility_probe(
    *,
    experiment_dir: pathlib.Path,
    attempt_id: str,
    worker: str,
    worker_repo: str,
    request: dict[str, Any],
    model_spec: dict[str, Any],
    logs_dir: pathlib.Path,
) -> pathlib.Path | None:
    model_id = str(model_spec["model_id"])
    if model_id not in {
        "cnn_ctc_v2",
        "cnn_ctc_v3",
        "cnn_ctc_v4",
        "cnn_ctc_v5",
        "cnn_ctc_v6",
    }:
        return None

    attempt_dir = experiment_dir / "attempts" / attempt_id
    probe_dir = attempt_dir / "build" / "compatibility-probe"
    export_dir = probe_dir / "export"
    ir_dir = probe_dir / "openvino" / "fp16"
    local_files = [
        ir_dir / f"{model_id}.xml",
        ir_dir / f"{model_id}.bin",
        export_dir / "golden-input.f32",
    ]
    for path in local_files:
        if not path.is_file():
            raise ExecutionFailure(
                f"pretraining compatibility artifact missing: {path}",
                outcome="failed",
                failure_class="controller_execution",
            )

    remote_stage = (
        f"{worker_repo}/work/speech-asr/remote-probe/"
        f"{request['experiment_id']}/{attempt_id}"
    )
    mkdir = remote_run(
        worker,
        ["mkdir", "-p", remote_stage],
        log_path=logs_dir / "edge-model-probe-stage.log",
    )
    if mkdir.returncode != 0:
        raise ExecutionFailure(
            "could not create edge model-probe staging directory",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    push = run_process(
        rsync_push_command(
            worker=worker,
            sources=local_files,
            remote_dir=remote_stage,
        ),
        log_path=logs_dir / "edge-model-probe-rsync-push.log",
    )
    if push.returncode != 0:
        raise ExecutionFailure(
            "pretraining model-probe rsync failed",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    remote_output = f"{remote_stage}/myriad-output.f32"
    probe = remote_run(
        worker,
        [
            "bash",
            f"{worker_repo}/scripts/edge-speech-model-probe.sh",
            "--model",
            f"{remote_stage}/{model_id}.xml",
            "--weights",
            f"{remote_stage}/{model_id}.bin",
            "--tensor",
            f"{remote_stage}/golden-input.f32",
            "--output",
            remote_output,
        ],
        log_path=logs_dir / "edge-model-probe.log",
    )
    if probe.returncode == 75:
        raise ExecutionFailure(
            "pretraining MYRIAD graph probe blocked: worker busy",
            outcome="blocked",
            failure_class="worker_busy",
        )
    if probe.returncode != 0:
        raise ExecutionFailure(
            "pretraining MYRIAD graph probe failed:\n"
            + "\n".join(probe.stdout.splitlines()[-30:]),
            outcome="failed",
            failure_class="hardware_execution",
        )

    collected = attempt_dir / "generated" / "pretraining-probe"
    collected.mkdir(parents=True, exist_ok=True)
    pull = run_process(
        [
            "rsync",
            "-a",
            "--checksum",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10",
            "--",
            f"{worker}:{remote_output}",
            str(collected / "myriad-output.f32"),
        ],
        log_path=logs_dir / "edge-model-probe-rsync-pull.log",
    )
    if pull.returncode != 0:
        raise ExecutionFailure(
            "pretraining MYRIAD probe output collection failed",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    comparison_path = collected / "comparison.json"
    comparison = run_process(
        [
            str(ROOT / "scripts" / "python-apps.sh"),
            str(
                ROOT
                / "examples"
                / "speech-asr"
                / "evaluation"
                / "compare_cnn_ctc_v1_tensor.py"
            ),
            str(export_dir / "golden-output.f32"),
            str(collected / "myriad-output.f32"),
            "--spec",
            str(
                ROOT
                / "examples"
                / "speech-asr"
                / "models"
                / model_id
                / "model_spec.json"
            ),
            "--candidate-name",
            "MYRIAD-arm64-pretraining",
            "--output",
            str(comparison_path),
        ],
        log_path=logs_dir / "controller-model-probe-compare.log",
    )
    if comparison.returncode != 0:
        raise ExecutionFailure(
            "pretraining MYRIAD numerical comparison failed",
            outcome="failed",
            failure_class="result_contract",
        )

    comparison_doc = load_json(comparison_path)
    comparison_metrics = comparison_doc.get("comparison", {})
    numerical_gate = pretraining_myriad_compatibility(comparison_metrics)
    if numerical_gate["status"] != "accepted":
        raise ExecutionFailure(
            "pretraining MYRIAD numerical compatibility failed: "
            + "; ".join(numerical_gate["reasons"]),
            outcome="failed",
            failure_class="hardware_execution",
        )

    evidence_path = collected / "probe.json"
    write_json(
        evidence_path,
        {
            "schema": "speech-asr/pretraining-myriad-probe",
            "version": 1,
            "status": "completed",
            "model_id": model_id,
            "output_sha256": sha256_file(collected / "myriad-output.f32"),
            "comparison": comparison_doc["comparison"],
            "numerical_gate": numerical_gate,
            "log_sha256": sha256_file(logs_dir / "edge-model-probe.log"),
        },
    )
    return evidence_path


def execute_local(
    *,
    experiment_dir: pathlib.Path,
    attempt_id: str,
    model_spec: dict[str, Any],
    train_config: dict[str, Any],
    logs_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    attempt_dir = experiment_dir / "attempts" / attempt_id
    model_id = executor_model_basename(str(model_spec["model_id"]))
    build_dir = attempt_dir / "build" / model_id
    command = training_command(
        root=ROOT,
        build_dir=build_dir,
        train_config=train_config,
        model_spec=model_spec,
    )
    proc = run_process(
        command,
        log_path=logs_dir / "controller-training.log",
    )
    if proc.returncode != 0:
        raise ExecutionFailure(
            "local training/export/conversion failed:\n"
            + "\n".join(proc.stdout.splitlines()[-40:]),
            outcome="failed",
            failure_class="controller_execution",
        )

    training_result_path = build_dir / "training" / "training-result.json"
    if not training_result_path.is_file():
        raise ExecutionFailure(
            f"local executor did not produce training result: {training_result_path}",
            outcome="failed",
            failure_class="controller_execution",
        )
    try:
        training_result_doc = load_json(training_result_path)
        device_evidence = validate_training_device_evidence(
            train_config=train_config,
            training_result=training_result_doc,
        )
    except Exception as exc:
        raise ExecutionFailure(
            f"training device evidence invalid: {exc}",
            outcome="failed",
            failure_class="controller_execution",
        ) from exc
    print(
        json.dumps(
            {
                "event": "training_device_evidence",
                **device_evidence,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    compatibility = load_compatibility_result(build_dir)
    probe_evidence = (
        attempt_dir
        / "generated"
        / "pretraining-probe"
        / "probe.json"
    )
    if probe_evidence.is_file():
        compatibility["pretraining_myriad_probe"] = load_json(probe_evidence)
    compatibility_path = attempt_dir / "generated" / "compatibility.json"
    write_json(compatibility_path, compatibility)

    training_result = training_result_path
    manager(
        "record-stage",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--stage",
        "training",
        "--source",
        str(training_result),
    )
    manager(
        "record-stage",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--stage",
        "compatibility",
        "--source",
        str(compatibility_path),
    )

    artifacts = {
        "checkpoint": build_dir / "training" / "checkpoint.pt",
        "onnx": build_dir / "export" / f"{model_id}.onnx",
        "openvino_xml": build_dir / "openvino" / "fp16" / f"{model_id}.xml",
        "openvino_bin": build_dir / "openvino" / "fp16" / f"{model_id}.bin",
    }
    for name, path in artifacts.items():
        if not path.is_file():
            raise ExecutionFailure(
                f"local executor did not produce artifact {name}: {path}",
                outcome="failed",
                failure_class="controller_execution",
            )
        manager(
            "record-artifact",
            "--experiment",
            str(experiment_dir),
            "--attempt",
            attempt_id,
            "--name",
            name,
            "--source",
            str(path),
        )
    return build_dir, compatibility


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_remote(
    *,
    experiment_dir: pathlib.Path,
    attempt_id: str,
    worker: str,
    worker_repo: str,
    request: dict[str, Any],
    model_spec: dict[str, Any],
    build_dir: pathlib.Path,
    logs_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    deployment = experiment_dir / "attempts" / attempt_id / "deployment-manifest.json"
    manager(
        "deployment-manifest",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--worker-commit",
        request["source"]["repo_commit"],
        "--evaluator",
        "cnn-ctc-myriad-v1",
        "--result-path",
        f"attempts/{attempt_id}/results/hardware.json",
    )
    if not deployment.is_file():
        raise ExecutionFailure(
            "deployment manifest was not created",
            outcome="failed",
            failure_class="result_contract",
        )

    remote_stage = (
        f"{worker_repo}/work/speech-asr/remote-worker/"
        f"{request['experiment_id']}/{attempt_id}"
    )
    incoming = f"{remote_stage}/incoming"
    output = f"{remote_stage}/output"
    mkdir = remote_run(
        worker,
        ["mkdir", "-p", incoming, output],
        log_path=logs_dir / "edge-stage.log",
    )
    if mkdir.returncode != 0:
        raise ExecutionFailure(
            "could not create edge staging directory",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    model_id = executor_model_basename(str(model_spec["model_id"]))
    sources = [
        deployment,
        build_dir / "openvino" / "fp16" / f"{model_id}.xml",
        build_dir / "openvino" / "fp16" / f"{model_id}.bin",
    ]
    push = run_process(
        rsync_push_command(
            worker=worker,
            sources=sources,
            remote_dir=incoming,
        ),
        log_path=logs_dir / "edge-rsync-push.log",
    )
    if push.returncode != 0:
        raise ExecutionFailure(
            "deployment rsync to edge failed",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    worker_proc = remote_run(
        worker,
        [
            "bash",
            f"{worker_repo}/scripts/edge-speech-worker.sh",
            "--deployment",
            f"{incoming}/deployment-manifest.json",
            "--bundle-dir",
            incoming,
            "--output-dir",
            output,
        ],
        log_path=logs_dir / "edge-worker-ssh.log",
    )

    collected = experiment_dir / "attempts" / attempt_id / "edge"
    collected.mkdir(parents=True, exist_ok=True)
    pull = run_process(
        rsync_pull_command(
            worker=worker,
            remote_dir=output,
            local_dir=collected,
        ),
        log_path=logs_dir / "edge-rsync-pull.log",
    )

    if worker_proc.returncode != 0:
        outcome, failure_class = remote_failure_class(worker_proc.returncode)
        details = "\n".join(worker_proc.stdout.splitlines()[-40:])
        if pull.returncode == 0 and (collected / "worker-result.json").is_file():
            try:
                worker_result = validate_edge_worker_result(
                    load_json(collected / "worker-result.json")
                )
                failure_class = worker_result.get("failure_class") or failure_class
                if failure_class == "request_config":
                    outcome = "failed"
                details = worker_result.get("diagnostics", {}).get("summary", details)
            except Exception:
                pass
        raise ExecutionFailure(
            f"edge worker failed ({worker_proc.returncode}): {details}",
            outcome=outcome,
            failure_class=failure_class,
        )

    if pull.returncode != 0:
        raise ExecutionFailure(
            "edge execution completed but evidence rsync failed",
            outcome="blocked",
            failure_class="transport_preflight",
        )

    worker_result_path = collected / "worker-result.json"
    hardware_path = collected / "hardware.json"
    if not worker_result_path.is_file() or not hardware_path.is_file():
        raise ExecutionFailure(
            "edge worker evidence bundle is incomplete",
            outcome="failed",
            failure_class="result_contract",
        )

    worker_result = validate_edge_worker_result(load_json(worker_result_path))
    if worker_result.get("status") != "completed":
        raise ExecutionFailure(
            "edge worker result is not completed",
            outcome="failed",
            failure_class="result_contract",
        )
    if worker_result.get("experiment_id") != request["experiment_id"]:
        raise ExecutionFailure(
            "edge worker experiment identity mismatch",
            outcome="failed",
            failure_class="result_contract",
        )
    if worker_result.get("attempt_id") != attempt_id:
        raise ExecutionFailure(
            "edge worker attempt identity mismatch",
            outcome="failed",
            failure_class="result_contract",
        )
    if worker_result.get("deployment_sha256") != sha256_file(deployment):
        raise ExecutionFailure(
            "edge worker deployment hash mismatch",
            outcome="failed",
            failure_class="result_contract",
        )
    if worker_result.get("hardware_result", {}).get("sha256") != sha256_file(hardware_path):
        raise ExecutionFailure(
            "edge worker hardware result hash mismatch",
            outcome="failed",
            failure_class="result_contract",
        )

    hardware = validate_experiment_result(load_json(hardware_path))
    if hardware["experiment_id"] != request["experiment_id"]:
        raise ExecutionFailure(
            "collected hardware result experiment identity mismatch",
            outcome="failed",
            failure_class="result_contract",
        )
    if hardware["runtime"]["repo_commit"] != request["source"]["repo_commit"]:
        raise ExecutionFailure(
            "collected hardware result source revision mismatch",
            outcome="failed",
            failure_class="result_contract",
        )

    manager(
        "record-stage",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--stage",
        "hardware",
        "--source",
        str(hardware_path),
    )
    manager(
        "record-stage",
        "--experiment",
        str(experiment_dir),
        "--attempt",
        attempt_id,
        "--stage",
        "accuracy",
        "--source",
        str(hardware_path),
    )
    return worker_result_path, hardware


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=pathlib.Path, required=True)
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-base-repo")
    parser.add_argument("--edge-worker-repo")
    args = parser.parse_args()

    experiment_dir = args.experiment.resolve()
    attempt_id: str | None = None
    diagnostics_path: pathlib.Path | None = None

    try:
        request, model_spec, train_config, acceptance = read_request_bundle(
            experiment_dir
        )
        manager("validate", "--experiment", str(experiment_dir))
        start = manager("start-attempt", "--experiment", str(experiment_dir))
        attempt_id = start["attempt_id"]
        attempt_dir = experiment_dir / "attempts" / attempt_id
        logs_dir = attempt_dir / "logs"
        diagnostics_path = attempt_dir / "generated" / "terminal-diagnostics.json"

        head = git_head()
        if head != request["source"]["repo_commit"]:
            raise ExecutionFailure(
                f"controller HEAD {head} does not match approved commit "
                f"{request['source']['repo_commit']}",
                outcome="failed",
                failure_class="request_config",
            )
        if not tracked_tree_is_clean():
            raise ExecutionFailure(
                "controller checkout has tracked local modifications",
                outcome="failed",
                failure_class="request_config",
            )

        verify_declared_manifest(
            train_config["training_manifest"], "training manifest"
        )
        verify_declared_manifest(
            train_config["validation_manifest"], "validation manifest"
        )
        benchmark_path = verify_declared_manifest(
            {
                "path": request["benchmark"]["manifest_path"],
                "sha256": request["benchmark"]["manifest_sha256"],
            },
            "benchmark manifest",
        )
        del benchmark_path
        try:
            validate_model_executor_request(model_spec, train_config)
        except ValueError as exc:
            raise ExecutionFailure(
                str(exc),
                outcome="failed",
                failure_class="request_config",
            ) from exc

        home = remote_home(args.worker, logs_dir / "edge-connectivity.log")
        base_repo, worker_repo = worker_paths(
            home=home,
            base_repo=args.edge_base_repo,
            worker_repo=args.edge_worker_repo,
        )
        bootstrap_edge(
            worker=args.worker,
            base_repo=base_repo,
            worker_repo=worker_repo,
            commit=head,
            log_path=logs_dir / "edge-bootstrap.log",
        )
        preflight_edge(
            worker=args.worker,
            worker_repo=worker_repo,
            benchmark=request["benchmark"],
            log_path=logs_dir / "edge-preflight.log",
        )

        probe_build = (
            attempt_dir / "build" / executor_model_basename(str(model_spec["model_id"]))
        )
        probe_command = compatibility_probe_command(
            root=ROOT,
            build_dir=probe_build,
            model_spec=model_spec,
        )
        if probe_command is not None:
            probe = run_process(
                probe_command,
                log_path=logs_dir / "controller-compatibility-probe.log",
            )
            if probe.returncode != 0:
                raise ExecutionFailure(
                    "pre-training compatibility gate failed:\n"
                    + "\n".join(probe.stdout.splitlines()[-40:]),
                    outcome="failed",
                    failure_class="controller_execution",
                )
            physical_compatibility_probe(
                experiment_dir=experiment_dir,
                attempt_id=attempt_id,
                worker=args.worker,
                worker_repo=worker_repo,
                request=request,
                model_spec=model_spec,
                logs_dir=logs_dir,
            )

        manager(
            "transition",
            "--experiment",
            str(experiment_dir),
            "--attempt",
            attempt_id,
            "--state",
            "EXECUTE",
        )
        build_dir, compatibility = execute_local(
            experiment_dir=experiment_dir,
            attempt_id=attempt_id,
            model_spec=model_spec,
            train_config=train_config,
            logs_dir=logs_dir,
        )

        manager(
            "transition",
            "--experiment",
            str(experiment_dir),
            "--attempt",
            attempt_id,
            "--state",
            "EVALUATE",
        )
        worker_result_path, hardware = execute_remote(
            experiment_dir=experiment_dir,
            attempt_id=attempt_id,
            worker=args.worker,
            worker_repo=worker_repo,
            request=request,
            model_spec=model_spec,
            build_dir=build_dir,
            logs_dir=logs_dir,
        )

        acceptance_result = acceptance_evaluation(
            policy=acceptance,
            training_present=True,
            compatibility=compatibility,
            hardware=hardware,
        )
        acceptance_path = attempt_dir / "results" / "acceptance-evaluation.json"
        write_json(acceptance_path, acceptance_result)

        metrics_path = attempt_dir / "generated" / "summary-metrics.json"
        headline_metrics = {
            name: hardware["metrics"][name]
            for name in (
                "wer",
                "cer",
                "realtime_factor",
                "inference_latency_p50_ms",
                "inference_latency_p95_ms",
                "failures",
            )
        }
        write_json(metrics_path, headline_metrics)
        diagnostics = {
            "summary": (
                "execution completed; acceptance policy "
                + acceptance_result["status"]
            ),
            "acceptance": acceptance_result,
            "worker": {
                "alias": args.worker,
                "worker_result_sha256": sha256_file(worker_result_path),
                "edge_worker_repo": worker_repo,
            },
        }
        diagnostics_path = attempt_dir / "generated" / "summary-diagnostics.json"
        write_json(diagnostics_path, diagnostics)

        manager(
            "transition",
            "--experiment",
            str(experiment_dir),
            "--attempt",
            attempt_id,
            "--state",
            "AWAIT_REVIEW",
            "--outcome",
            "completed",
            "--metrics-json",
            str(metrics_path),
            "--diagnostics-json",
            str(diagnostics_path),
        )
        manager("validate", "--experiment", str(experiment_dir))

        print(
            json.dumps(
                {
                    "status": "completed",
                    "experiment_id": request["experiment_id"],
                    "attempt_id": attempt_id,
                    "review_state": "AWAIT_REVIEW",
                    "acceptance": acceptance_result["status"],
                    "acceptance_reasons": acceptance_result["reasons"],
                    "metrics": headline_metrics,
                    "threshold_checks": acceptance_result["threshold_checks"],
                    "result": str(
                        attempt_dir / "results" / "result.json"
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except ExecutionFailure as exc:
        if attempt_id is not None and diagnostics_path is not None:
            try:
                terminalize(
                    experiment_dir=experiment_dir,
                    attempt_id=attempt_id,
                    failure=exc,
                    diagnostics_path=diagnostics_path,
                )
            except Exception as terminal_exc:
                print(
                    f"error: {exc}\nadditional lifecycle error: {terminal_exc}",
                    file=sys.stderr,
                )
                return 3
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if attempt_id is not None and diagnostics_path is not None:
            failure = ExecutionFailure(
                f"unexpected controller failure: {exc}",
                outcome="failed",
                failure_class="controller_execution",
            )
            try:
                terminalize(
                    experiment_dir=experiment_dir,
                    attempt_id=attempt_id,
                    failure=failure,
                    diagnostics_path=diagnostics_path,
                )
            except Exception:
                pass
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
