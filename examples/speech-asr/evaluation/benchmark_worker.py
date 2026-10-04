#!/usr/bin/env python3
"""Run the frozen benchmark-v1 measurement policy for rm_cnn4a."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform as host_platform
import statistics
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    canonical_json_sha256,
    validate_acoustic_benchmark_result,
    validate_acoustic_regression_result,
    validate_benchmark_contract,
)

EXIT_OK = 0
EXIT_REQUEST = 2
EXIT_EXECUTION = 3
EXIT_RESULT = 4

BENCHMARK_ID = "rm_cnn4a-vendor-regression-v1"
BENCHMARK_CONTRACT = SPEECH_ROOT / "contracts" / "benchmark-v1.yaml"
RUNNER = HERE / "benchmark_rm_cnn4a.py"

METRIC_NAMES = (
    "weighted_mean_infer_ms_per_frame",
    "utterance_avg_infer_ms_per_frame_p50",
    "utterance_avg_infer_ms_per_frame_p95",
    "max_error_max",
    "avg_error_mean",
    "rms_error_mean",
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def native_target(machine: str | None = None) -> str | None:
    value = (machine or host_platform.machine()).lower()
    if value in {"x86_64", "amd64"}:
        return "amd64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    if value in {"armv7l", "armv7"}:
        return "armv7"
    return None


def load_measurement_policy() -> tuple[dict[str, Any], str]:
    document = json.loads(BENCHMARK_CONTRACT.read_text(encoding="utf-8"))
    validated = validate_benchmark_contract(document)
    policy = validated["measurement_policy"]
    return {
        "warmup_count": policy["warmup_count"],
        "measured_iterations": policy["measured_iterations"],
        "within_run_aggregation": policy["aggregation"],
        "across_runs_aggregation": "unweighted_run_statistics",
        "warmup_included_in_aggregate": False,
        "iteration_scope": "independent_full_fixture_process",
    }, canonical_json_sha256(validated)


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement set")
    mean = statistics.fmean(values)
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    return {
        "mean": mean,
        "median": statistics.median(values),
        "min": minimum,
        "max": maximum,
        "range": spread,
        "relative_range": spread / mean if mean != 0 else 0.0,
        "cv_population": statistics.pstdev(values) / mean if mean != 0 else 0.0,
    }


def identity(result: dict[str, Any]) -> dict[str, Any]:
    model = result["model"]
    graph_key = (
        "canonical_graph_sha256"
        if model.get("canonical_graph_sha256")
        else "xml_sha256"
    )
    return {
        "model.id": model["id"],
        "model.family": model["family"],
        "model.bin_sha256": model["bin_sha256"],
        f"model.{graph_key}": model[graph_key],
        "fixture.features_sha256": result["fixture"]["features_sha256"],
        "fixture.reference_scores_sha256": result["fixture"]["reference_scores_sha256"],
        "runtime.backend": result["runtime"]["backend"],
        "runtime.target": result["runtime"]["target"],
        "runtime.openvino_version": result["runtime"]["openvino_version"],
        "runtime.repo_commit": result["runtime"]["repo_commit"],
        "metrics.utterances": result["metrics"]["utterances"],
        "metrics.total_frames": result["metrics"]["total_frames"],
    }


def require_same_identity(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    left = identity(reference)
    right = identity(candidate)
    if left != right:
        differences = [
            f"{key}: {left.get(key)!r} != {right.get(key)!r}"
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        ]
        raise ValueError("benchmark run identity changed: " + "; ".join(differences))


def run_record(
    *,
    index: int,
    result: dict[str, Any],
    result_path: pathlib.Path,
    log_path: pathlib.Path,
    returncode: int,
) -> dict[str, Any]:
    record = {
        "index": index,
        "status": result["status"],
        "result_path": relative_path(result_path),
        "log_path": relative_path(log_path),
        "returncode": returncode,
        "raw_log_sha256": result["provenance"].get("raw_log_sha256"),
    }
    if result["status"] == "completed":
        record["metrics"] = {
            name: result["metrics"][name]
            for name in METRIC_NAMES
        }
        record["metrics"]["model_load_ms"] = result["provenance"]["model_load_ms"]
    elif "diagnostics" in result:
        record["diagnostics"] = result["diagnostics"]
    return record


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 5:
        raise ValueError("benchmark-v1 requires exactly five measured results")
    reference = results[0]
    for candidate in results[1:]:
        require_same_identity(reference, candidate)
    if any(item["status"] != "completed" for item in results):
        raise ValueError("cannot aggregate failed measurement runs")
    if any(item["metrics"]["failures"] != 0 for item in results):
        raise ValueError("cannot aggregate runs with inference failures")

    metrics: dict[str, Any] = {}
    for name in METRIC_NAMES:
        metrics[name] = summary_stats(
            [float(item["metrics"][name]) for item in results]
        )
    metrics["model_load_ms"] = summary_stats(
        [float(item["provenance"]["model_load_ms"]) for item in results]
    )

    return {
        "measured_runs": len(results),
        "counts": {
            "utterances": reference["metrics"]["utterances"],
            "total_frames": reference["metrics"]["total_frames"],
            "failures": sum(item["metrics"]["failures"] for item in results),
        },
        "metrics": metrics,
    }


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        f"status: {result['status']}",
        f"benchmark: {result['benchmark']['id']}",
        (
            "runtime: "
            f"{result['runtime']['backend']} / {result['runtime']['target']} / "
            f"OpenVINO {result['runtime']['openvino_version']}"
        ),
        (
            "policy: "
            f"{result['benchmark']['measurement_policy']['warmup_count']} warmup "
            "(excluded), "
            f"{result['benchmark']['measurement_policy']['measured_iterations']} "
            "measured independent fixture runs"
        ),
    ]
    if result["status"] != "completed":
        lines.append(f"failure: {result['diagnostics']['summary']}")
        return "\n".join(lines) + "\n"

    aggregate = result["aggregate"]
    counts = aggregate["counts"]
    lines.append(
        f"fixture: {counts['utterances']} utterances / "
        f"{counts['total_frames']} frames / {counts['failures']} failures"
    )
    labels = (
        ("weighted_mean_infer_ms_per_frame", "weighted infer ms/frame"),
        ("utterance_avg_infer_ms_per_frame_p50", "utterance p50 ms/frame"),
        ("utterance_avg_infer_ms_per_frame_p95", "utterance p95 ms/frame"),
        ("model_load_ms", "model load ms"),
        ("rms_error_mean", "vendor RMS error"),
    )
    for key, label in labels:
        stat = aggregate["metrics"][key]
        lines.append(
            f"{label}: mean={stat['mean']:.9g} median={stat['median']:.9g} "
            f"min={stat['min']:.9g} max={stat['max']:.9g} "
            f"range={stat['range']:.9g} "
            f"relative_range={stat['relative_range']:.6%} "
            f"cv={stat['cv_population']:.6%}"
        )
    return "\n".join(lines) + "\n"


def write_worker_result(
    result: dict[str, Any],
    output_path: pathlib.Path,
    summary_path: pathlib.Path,
) -> None:
    validate_acoustic_benchmark_result(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary(result), encoding="utf-8")


def invoke_runner(
    *,
    backend: str,
    platform: str,
    result_path: pathlib.Path,
    log_path: pathlib.Path,
) -> tuple[int, dict[str, Any] | None, str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--backend",
        backend,
        "--platform",
        platform,
        "--output",
        str(result_path),
        "--log",
        str(log_path),
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = None
    if result_path.is_file():
        try:
            result = validate_acoustic_regression_result(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ContractValidationError):
            result = None
    tail = "\n".join(proc.stdout.splitlines()[-20:])
    return proc.returncode, result, tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("rm_cnn4a",))
    parser.add_argument("--backend", required=True, choices=("cpu", "myriad"))
    parser.add_argument(
        "--platform",
        required=True,
        choices=("armv7", "arm64", "amd64"),
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=None,
        help="default: work/speech-asr/rm_cnn4a/benchmark-<backend>-<platform>",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()

    selected_native = native_target()
    if selected_native is None:
        print(
            f"unsupported native host architecture: {host_platform.machine()}",
            file=sys.stderr,
        )
        return EXIT_REQUEST
    if args.platform != selected_native:
        print(
            f"benchmark worker requires native target {selected_native}; "
            f"requested {args.platform}",
            file=sys.stderr,
        )
        return EXIT_REQUEST
    if args.backend == "cpu" and args.platform != "amd64":
        print("CPU benchmark requires --platform amd64", file=sys.stderr)
        return EXIT_REQUEST

    try:
        policy, contract_sha = load_measurement_policy()
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        print(f"benchmark contract error: {exc}", file=sys.stderr)
        return EXIT_REQUEST

    default_work = (
        ROOT
        / "work"
        / "speech-asr"
        / "rm_cnn4a"
        / f"benchmark-{args.backend}-{args.platform}"
    )
    work_dir = (args.work_dir or default_work)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    output_path = args.output or (work_dir / "result.json")
    summary_path = args.summary or (work_dir / "summary.txt")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    runs_dir = work_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "benchmark_contract_sha256": contract_sha,
        "runner_sha256": sha256_path(RUNNER),
        "worker_sha256": sha256_path(pathlib.Path(__file__)),
        "python_version": host_platform.python_version(),
    }
    request = {
        "model": args.model,
        "backend": args.backend,
        "platform": args.platform,
    }

    warmup_records: list[dict[str, Any]] = []
    measured_records: list[dict[str, Any]] = []
    measured_results: list[dict[str, Any]] = []
    identity_reference: dict[str, Any] | None = None

    total = policy["warmup_count"] + policy["measured_iterations"]
    for ordinal in range(total):
        is_warmup = ordinal < policy["warmup_count"]
        phase = "warmup" if is_warmup else "measured"
        index = ordinal if is_warmup else ordinal - policy["warmup_count"]
        result_path = runs_dir / f"{phase}-{index:02d}.json"
        log_path = runs_dir / f"{phase}-{index:02d}.log"
        returncode, single, tail = invoke_runner(
            backend=args.backend,
            platform=args.platform,
            result_path=result_path,
            log_path=log_path,
        )

        if single is None:
            print(
                f"{phase} run {index} did not produce a valid result\n{tail}",
                file=sys.stderr,
            )
            return EXIT_EXECUTION

        record = run_record(
            index=index,
            result=single,
            result_path=result_path,
            log_path=log_path,
            returncode=returncode,
        )
        (warmup_records if is_warmup else measured_records).append(record)

        if identity_reference is None:
            identity_reference = single
        else:
            try:
                require_same_identity(identity_reference, single)
            except ValueError as exc:
                worker = {
                    "schema": "speech-asr/acoustic-benchmark-result",
                    "version": 1,
                    "status": "failed",
                    "benchmark": {
                        "id": BENCHMARK_ID,
                        "contract_sha256": contract_sha,
                        "measurement_policy": policy,
                    },
                    "request": request,
                    "model": identity_reference["model"],
                    "fixture": identity_reference["fixture"],
                    "runtime": {
                        **identity_reference["runtime"],
                        "host_machine": host_platform.machine(),
                    },
                    "runs": {
                        "warmup": warmup_records,
                        "measured": measured_records,
                    },
                    "provenance": provenance,
                    "diagnostics": {"summary": str(exc)},
                }
                try:
                    write_worker_result(worker, output_path, summary_path)
                except ContractValidationError:
                    pass
                print(str(exc), file=sys.stderr)
                return EXIT_RESULT

        if returncode != 0 or single["status"] != "completed":
            summary = (
                single.get("diagnostics", {}).get("summary")
                or f"{phase} run {index} failed with status {returncode}"
            )
            worker = {
                "schema": "speech-asr/acoustic-benchmark-result",
                "version": 1,
                "status": "failed",
                "benchmark": {
                    "id": BENCHMARK_ID,
                    "contract_sha256": contract_sha,
                    "measurement_policy": policy,
                },
                "request": request,
                "model": single["model"],
                "fixture": single["fixture"],
                "runtime": {
                    **single["runtime"],
                    "host_machine": host_platform.machine(),
                },
                "runs": {
                    "warmup": warmup_records,
                    "measured": measured_records,
                },
                "provenance": provenance,
                "diagnostics": {"summary": summary},
            }
            try:
                write_worker_result(worker, output_path, summary_path)
            except ContractValidationError as exc:
                print(str(exc), file=sys.stderr)
                return EXIT_RESULT
            print(render_summary(worker), end="")
            print(f"result: {relative_path(output_path)}")
            print(f"summary: {relative_path(summary_path)}")
            return EXIT_EXECUTION

        if not is_warmup:
            measured_results.append(single)

    assert identity_reference is not None
    try:
        aggregate = aggregate_results(measured_results)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_RESULT

    worker = {
        "schema": "speech-asr/acoustic-benchmark-result",
        "version": 1,
        "status": "completed",
        "benchmark": {
            "id": BENCHMARK_ID,
            "contract_sha256": contract_sha,
            "measurement_policy": policy,
        },
        "request": request,
        "model": identity_reference["model"],
        "fixture": identity_reference["fixture"],
        "runtime": {
            **identity_reference["runtime"],
            "host_machine": host_platform.machine(),
        },
        "runs": {
            "warmup": warmup_records,
            "measured": measured_records,
        },
        "aggregate": aggregate,
        "provenance": provenance,
    }

    try:
        write_worker_result(worker, output_path, summary_path)
    except ContractValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_RESULT

    print(render_summary(worker), end="")
    print(f"result: {relative_path(output_path)}")
    print(f"summary: {relative_path(summary_path)}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
