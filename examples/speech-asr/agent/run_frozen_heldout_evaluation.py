#!/usr/bin/env python3
"""Run one sealed held-out evaluation of a previously trained speech experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
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
from speech_asr.orchestration import (  # noqa: E402
    rsync_pull_command,
    rsync_push_command,
    ssh_command,
    validate_worker_alias,
)

SOURCE_EXPERIMENT_ID = "exp-87538823d2bf1562"
SOURCE_ATTEMPT_ID = "attempt-0001"
BENCHMARK_ID = "ami-full-corpus-asr-sc-heldout-v1"
DEFAULT_EXPERIMENT = (
    ROOT / "work" / "speech-asr" / "experiments" / SOURCE_EXPERIMENT_ID
)
DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "heldout-eval-v1" / "manifest.jsonl"
)
DEFAULT_QUALIFICATION = (
    ROOT / "work" / "speech-asr" / "ami" / "heldout-eval-v1" / "qualification.json"
)
REFERENCE_EVALUATOR = (
    SPEECH_ROOT / "evaluation" / "evaluate_frozen_cnn_ctc_v3_reference.py"
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def run(
    command: list[str],
    *,
    log_path: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    return proc


def run_streaming(
    command: list[str],
    *,
    log_path: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout is None:
            raise ValueError("streaming subprocess stdout is unavailable")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            output.append(line)
        returncode = proc.wait()
    return subprocess.CompletedProcess(
        command,
        returncode,
        "".join(output),
    )


def ssh_run(
    worker: str,
    command: list[str],
    *,
    log_path: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(
        ssh_command(worker, command),
        log_path=log_path,
    )


def bootstrap_edge(
    *,
    worker: str,
    base_repo: str,
    worker_repo: str,
    commit: str,
    log_path: pathlib.Path,
) -> None:
    script_path = ROOT / "scripts" / "edge-speech-bootstrap.sh"
    script = script_path.read_text(encoding="utf-8")
    proc = subprocess.run(
        ssh_command(
            worker,
            ["bash", "-s", "--", base_repo, worker_repo, commit],
        ),
        cwd=ROOT,
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise ValueError(
            "could not pin held-out edge worker to evaluation commit:\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
        )


def resolve_ref(ref: dict[str, str]) -> pathlib.Path:
    raw = pathlib.Path(ref["path"])
    path = raw if raw.is_absolute() else ROOT / raw
    if not path.is_file():
        raise ValueError(f"recorded source artifact is missing: {path}")
    actual = sha256_path(path)
    if actual != ref["sha256"]:
        raise ValueError(
            f"recorded source artifact hash mismatch: {path}: "
            f"{actual} != {ref['sha256']}"
        )
    return path


def source_state(
    experiment_dir: pathlib.Path,
) -> dict[str, Any]:
    if experiment_dir.name != SOURCE_EXPERIMENT_ID:
        raise ValueError(
            "held-out evaluation is frozen to "
            f"{SOURCE_EXPERIMENT_ID}, got {experiment_dir.name}"
        )
    request = load_json(experiment_dir / "request" / "experiment.json")
    model_spec = load_json(experiment_dir / "request" / "model-spec.json")
    attempt_dir = (
        experiment_dir / "attempts" / SOURCE_ATTEMPT_ID
    )
    attempt = load_json(attempt_dir / "attempt.json")
    acceptance = load_json(
        attempt_dir / "results" / "acceptance-evaluation.json"
    )
    result = load_json(attempt_dir / "results" / "result.json")

    if request.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source experiment request identity mismatch")
    if model_spec.get("model_id") != "cnn_ctc_v3":
        raise ValueError("held-out source model must be cnn_ctc_v3")
    if attempt.get("attempt_id") != SOURCE_ATTEMPT_ID:
        raise ValueError("source attempt identity mismatch")
    if attempt.get("state") != "AWAIT_REVIEW":
        raise ValueError("source attempt must be in AWAIT_REVIEW")
    if attempt.get("outcome") != "completed":
        raise ValueError("source attempt did not complete")
    if acceptance.get("status") != "accepted":
        raise ValueError("source attempt was not accepted")
    if result.get("outcome") != "completed":
        raise ValueError("source result is not completed")

    artifacts = attempt.get("artifacts", {})
    required = ("checkpoint", "onnx", "openvino_xml", "openvino_bin")
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(f"source attempt lacks frozen artifacts: {missing}")

    resolved = {name: resolve_ref(artifacts[name]) for name in required}
    return {
        "request": request,
        "model_spec": model_spec,
        "attempt": attempt,
        "acceptance": acceptance,
        "result": result,
        "artifacts": artifacts,
        "paths": resolved,
    }


def validate_cached_reference(
    reference: dict[str, Any],
    *,
    source: dict[str, Any],
    manifest_path: pathlib.Path,
    reference_device: str,
) -> None:
    if reference.get("schema") != "speech-asr/frozen-reference-evaluation":
        raise ValueError("cached reference schema mismatch")
    if reference.get("status") != "completed":
        raise ValueError("cached reference is not completed")
    if reference.get("role") != "heldout_test":
        raise ValueError("cached reference role mismatch")

    ref_source = reference.get("source", {})
    if ref_source.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("cached reference source experiment mismatch")
    if ref_source.get("attempt_id") != SOURCE_ATTEMPT_ID:
        raise ValueError("cached reference source attempt mismatch")
    if ref_source.get("checkpoint_sha256") != source["artifacts"]["checkpoint"]["sha256"]:
        raise ValueError("cached reference checkpoint hash mismatch")
    if ref_source.get("onnx_sha256") != source["artifacts"]["onnx"]["sha256"]:
        raise ValueError("cached reference ONNX hash mismatch")

    benchmark = reference.get("benchmark", {})
    if benchmark.get("id") != BENCHMARK_ID:
        raise ValueError("cached reference benchmark mismatch")
    if benchmark.get("manifest_sha256") != sha256_path(manifest_path):
        raise ValueError("cached reference manifest hash mismatch")

    runtime = reference.get("runtime", {})
    expected_device = "cpu" if reference_device == "cpu" else "cuda:0"
    if runtime.get("pytorch_device") != expected_device:
        raise ValueError(
            "cached reference PyTorch device mismatch: "
            f"{runtime.get('pytorch_device')!r} != {expected_device!r}"
        )
    if runtime.get("onnx_provider") != "CPUExecutionProvider":
        raise ValueError("cached reference ONNX provider mismatch")
    if reference.get("agreement", {}).get("frame_argmax_agreement") != 1.0:
        raise ValueError("cached reference agreement is not exact")


def validate_qualification(
    qualification_path: pathlib.Path,
    manifest_path: pathlib.Path,
) -> dict[str, Any]:
    qualification = load_json(qualification_path)
    if qualification.get("schema") != "speech-asr/heldout-evaluation-manifest":
        raise ValueError("held-out qualification schema mismatch")
    if qualification.get("status") != "structurally_valid":
        raise ValueError("held-out qualification is not structurally_valid")
    if qualification.get("role") != "test_only":
        raise ValueError("held-out qualification must remain test_only")
    partition = qualification.get("partition", {})
    if partition.get("training_allowed") is not False:
        raise ValueError("held-out partition must forbid training")
    if partition.get("checkpoint_selection_allowed") is not False:
        raise ValueError("held-out partition must forbid checkpoint selection")
    if qualification.get("overlap") != {
        "training_records": 0,
        "selection_records": 0,
        "training_meetings": 0,
        "selection_meetings": 0,
    }:
        raise ValueError("held-out qualification overlap is not zero")

    expected = qualification.get("test", {}).get("manifest_sha256")
    actual = sha256_path(manifest_path)
    if expected != actual:
        raise ValueError(
            f"held-out manifest hash mismatch: {actual} != {expected}"
        )
    return qualification


def make_deployment(
    *,
    source: dict[str, Any],
    manifest_path: pathlib.Path,
    repo_commit: str,
) -> dict[str, Any]:
    artifacts = source["artifacts"]
    request = source["request"]
    deployment = {
        "schema": "speech-asr/deployment-manifest",
        "version": 1,
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "attempt_id": SOURCE_ATTEMPT_ID,
        "controller_commit": repo_commit,
        "worker_commit": repo_commit,
        "model": {
            "id": "cnn_ctc_v3",
            "spec_sha256": request["documents"]["model_spec"]["sha256"],
        },
        "benchmark": {
            "id": BENCHMARK_ID,
            "manifest_path": manifest_path.resolve().relative_to(
                ROOT.resolve()
            ).as_posix(),
            "manifest_sha256": sha256_path(manifest_path),
        },
        "runtime": {
            "target": "arm64",
            "openvino_version": "2020.3.2",
        },
        "evaluator": {
            "id": "cnn-ctc-myriad-v1",
            "result_path": "heldout-evaluation/hardware.json",
        },
        "artifacts": {
            "openvino_xml": dict(artifacts["openvino_xml"]),
            "openvino_bin": dict(artifacts["openvino_bin"]),
        },
    }
    return validate_deployment_manifest(deployment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=pathlib.Path,
        default=DEFAULT_EXPERIMENT,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--qualification",
        type=pathlib.Path,
        default=DEFAULT_QUALIFICATION,
    )
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-base-repo")
    parser.add_argument("--worker-repo")
    parser.add_argument(
        "--reference-device",
        choices=("cuda", "cpu"),
        default="cpu",
    )
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="reuse an existing validated reference.json after a retryable edge failure",
    )
    args = parser.parse_args()

    try:
        validate_worker_alias(args.worker)
        if not args.manifest.is_file():
            raise ValueError(
                f"held-out manifest missing: {args.manifest}"
            )
        expected_qualification = args.manifest.parent / "qualification.json"
        if args.qualification.resolve() != expected_qualification.resolve():
            raise ValueError(
                "held-out qualification must be the sealed sibling of "
                "the held-out manifest"
            )

        manifest_sha = sha256_path(args.manifest)
        output_dir = (
            ROOT
            / "work"
            / "speech-asr"
            / "heldout-evaluations"
            / SOURCE_EXPERIMENT_ID
            / manifest_sha[:16]
        )
        final_path = output_dir / "result.json"
        if final_path.exists():
            raise ValueError(
                "held-out evaluation is already sealed at "
                f"{final_path}; do not rerun it for model selection"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"

        verify_proc = run(
            [
                str(ROOT / "scripts" / "qualify-speech-heldout-eval.sh"),
                "--output-dir",
                str(args.manifest.parent),
                "--verify-only",
            ],
            log_path=logs_dir / "heldout-qualification-verify.log",
        )
        if verify_proc.returncode != 0:
            raise ValueError(
                "held-out qualification verification failed:\n"
                + "\n".join(verify_proc.stdout.splitlines()[-40:])
            )

        qualification = validate_qualification(
            args.qualification,
            args.manifest,
        )
        source = source_state(args.experiment)
        repo_commit = git_head()

        reference_path = output_dir / "reference.json"
        if args.reuse_reference:
            if not reference_path.is_file():
                raise ValueError(
                    "--reuse-reference requested but reference.json is missing"
                )
            reference = load_json(reference_path)
            validate_cached_reference(
                reference,
                source=source,
                manifest_path=args.manifest,
                reference_device=args.reference_device,
            )
            print(
                "held-out reference: reusing validated "
                f"{reference_path}",
                flush=True,
            )
        else:
            reference_proc = run_streaming(
                [
                    str(ROOT / "scripts" / "python-training.sh"),
                    str(REFERENCE_EVALUATOR),
                    "--checkpoint",
                    str(source["paths"]["checkpoint"]),
                    "--onnx",
                    str(source["paths"]["onnx"]),
                    "--manifest",
                    str(args.manifest),
                    "--benchmark-id",
                    BENCHMARK_ID,
                    "--source-experiment-id",
                    SOURCE_EXPERIMENT_ID,
                    "--source-attempt-id",
                    SOURCE_ATTEMPT_ID,
                    "--device",
                    args.reference_device,
                    "--progress-every",
                    "250",
                    "--quiet-result",
                    "--output",
                    str(reference_path),
                ],
                log_path=logs_dir / "reference.log",
            )
            if reference_proc.returncode != 0 or not reference_path.is_file():
                raise ValueError(
                    "frozen PyTorch/ONNX reference evaluation failed:\n"
                    + "\n".join(reference_proc.stdout.splitlines()[-40:])
                )
            reference = load_json(reference_path)
            validate_cached_reference(
                reference,
                source=source,
                manifest_path=args.manifest,
                reference_device=args.reference_device,
            )

        home_proc = ssh_run(
            args.worker,
            ["sh", "-lc", 'printf "%s\\n" "$HOME"'],
            log_path=logs_dir / "edge-home.log",
        )
        if home_proc.returncode != 0:
            raise ValueError("could not resolve edge HOME")
        remote_home = home_proc.stdout.strip().splitlines()[-1]
        if not remote_home.startswith("/"):
            raise ValueError("edge HOME is not absolute")
        base_repo = (
            args.edge_base_repo
            or f"{remote_home}/workspace/movidius-openvino-rpi5"
        )
        worker_repo = (
            args.worker_repo
            or f"{remote_home}/workspace/movidius-openvino-rpi5-worker"
        )
        for path in (base_repo, worker_repo):
            if not path.startswith("/") or any(
                char.isspace() for char in path
            ):
                raise ValueError(
                    "edge repo paths must be absolute and whitespace-free"
                )

        bootstrap_edge(
            worker=args.worker,
            base_repo=base_repo,
            worker_repo=worker_repo,
            commit=repo_commit,
            log_path=logs_dir / "edge-bootstrap.log",
        )

        deployment = make_deployment(
            source=source,
            manifest_path=args.manifest,
            repo_commit=repo_commit,
        )
        deployment_path = output_dir / "deployment-manifest.json"
        write_json(deployment_path, deployment)

        remote_stage = (
            f"{worker_repo}/work/speech-asr/heldout-evaluations/"
            f"{SOURCE_EXPERIMENT_ID}/{manifest_sha[:16]}"
        )
        incoming = f"{remote_stage}/incoming"
        remote_output = f"{remote_stage}/output"
        mkdir = ssh_run(
            args.worker,
            ["mkdir", "-p", incoming, remote_output],
            log_path=logs_dir / "edge-stage.log",
        )
        if mkdir.returncode != 0:
            raise ValueError("could not create edge held-out staging directory")

        push = run(
            rsync_push_command(
                worker=args.worker,
                sources=[
                    deployment_path,
                    source["paths"]["openvino_xml"],
                    source["paths"]["openvino_bin"],
                ],
                remote_dir=incoming,
            ),
            log_path=logs_dir / "edge-rsync-push.log",
        )
        if push.returncode != 0:
            raise ValueError("held-out deployment rsync to edge failed")

        worker_proc = ssh_run(
            args.worker,
            [
                "bash",
                f"{worker_repo}/scripts/edge-speech-worker.sh",
                "--deployment",
                f"{incoming}/deployment-manifest.json",
                "--bundle-dir",
                incoming,
                "--output-dir",
                remote_output,
            ],
            log_path=logs_dir / "edge-worker.log",
        )

        edge_dir = output_dir / "edge"
        edge_dir.mkdir(parents=True, exist_ok=True)
        pull = run(
            rsync_pull_command(
                worker=args.worker,
                remote_dir=remote_output,
                local_dir=edge_dir,
            ),
            log_path=logs_dir / "edge-rsync-pull.log",
        )
        if pull.returncode != 0:
            raise ValueError("could not collect held-out edge result")
        if worker_proc.returncode != 0:
            raise ValueError(
                "held-out edge worker failed:\n"
                + "\n".join(worker_proc.stdout.splitlines()[-40:])
            )

        worker_result = validate_edge_worker_result(
            load_json(edge_dir / "worker-result.json")
        )
        if worker_result.get("status") != "completed":
            raise ValueError("held-out edge worker did not complete")
        hardware = validate_experiment_result(
            load_json(edge_dir / "hardware.json")
        )
        if hardware["benchmark"]["manifest_sha256"] != manifest_sha:
            raise ValueError("held-out hardware benchmark hash mismatch")
        if hardware["model"]["artifact_sha256"].get("cnn_ctc_v3.xml") != (
            source["artifacts"]["openvino_xml"]["sha256"]
        ):
            raise ValueError("held-out hardware XML provenance mismatch")
        if hardware["model"]["artifact_sha256"].get("cnn_ctc_v3.bin") != (
            source["artifacts"]["openvino_bin"]["sha256"]
        ):
            raise ValueError("held-out hardware BIN provenance mismatch")

        pytorch_metrics = reference["metrics"]["pytorch"]
        hardware_metrics = hardware["metrics"]
        result = {
            "schema": "speech-asr/heldout-evaluation-result",
            "version": 1,
            "status": "completed",
            "role": "test_only",
            "sealed": True,
            "source": {
                "experiment_id": SOURCE_EXPERIMENT_ID,
                "attempt_id": SOURCE_ATTEMPT_ID,
                "training_repo_commit": source["request"]["source"][
                    "repo_commit"
                ],
                "artifact_sha256": {
                    name: source["artifacts"][name]["sha256"]
                    for name in (
                        "checkpoint",
                        "onnx",
                        "openvino_xml",
                        "openvino_bin",
                    )
                },
            },
            "evaluation": {
                "repo_commit": repo_commit,
                "benchmark_id": BENCHMARK_ID,
                "manifest_sha256": manifest_sha,
                "qualification_sha256": sha256_path(args.qualification),
                "checkpoint_selection_allowed": False,
                "training_allowed": False,
                "repeat_for_model_selection": False,
            },
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_path(reference_path),
                "agreement": reference["agreement"],
                "metrics": pytorch_metrics,
                "onnx_metrics": reference["metrics"]["onnx"],
            },
            "hardware": {
                "path": str(edge_dir / "hardware.json"),
                "sha256": sha256_path(edge_dir / "hardware.json"),
                "runtime": hardware["runtime"],
                "metrics": hardware_metrics,
            },
            "cross_runtime_delta": {
                "wer": hardware_metrics["wer"] - pytorch_metrics["wer"],
                "cer": hardware_metrics["cer"] - pytorch_metrics["cer"],
                "blank_frame_fraction": (
                    hardware_metrics["decoder"]["blank_frame_fraction"]
                    - pytorch_metrics["decoder"]["blank_frame_fraction"]
                ),
                "empty_hypothesis_fraction": (
                    hardware_metrics["decoder"]["empty_hypothesis_fraction"]
                    - pytorch_metrics["decoder"]["empty_hypothesis_fraction"]
                ),
            },
            "qualification": qualification,
        }
        write_json(final_path, result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "status": result["status"],
        "role": result["role"],
        "sealed": result["sealed"],
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_attempt_id": SOURCE_ATTEMPT_ID,
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": result["evaluation"]["manifest_sha256"],
        "reference": {
            "wer": result["reference"]["metrics"]["wer"],
            "cer": result["reference"]["metrics"]["cer"],
            "decoder": result["reference"]["metrics"]["decoder"],
            "onnx_frame_argmax_agreement": result["reference"][
                "agreement"
            ]["frame_argmax_agreement"],
        },
        "hardware": {
            "wer": result["hardware"]["metrics"]["wer"],
            "cer": result["hardware"]["metrics"]["cer"],
            "decoder": result["hardware"]["metrics"]["decoder"],
            "inference_latency_p50_ms": result["hardware"]["metrics"][
                "inference_latency_p50_ms"
            ],
            "inference_latency_p95_ms": result["hardware"]["metrics"][
                "inference_latency_p95_ms"
            ],
            "realtime_factor": result["hardware"]["metrics"][
                "realtime_factor"
            ],
        },
        "result": str(final_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
