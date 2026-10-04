#!/usr/bin/env python3
"""Run rm_cnn4a vendor-score regression on CPU reference or MYRIAD."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    validate_acoustic_regression_result,
)
from speech_asr.regression import parse_speech_sample_output  # noqa: E402


def sha256(path: pathlib.Path) -> str:
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
        check=True,
    )
    return proc.stdout.strip()


def repo_path(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def build_command(backend: str, platform: str) -> list[str]:
    backend = backend.lower()
    if backend == "cpu":
        if platform != "amd64":
            raise ValueError("CPU reference regression requires platform amd64")
        mode = "speech-reference"
    elif backend == "myriad":
        mode = "speech-regress"
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return [str(ROOT / "run.sh"), "--platform", platform, mode]


def output_tail(text: str, lines: int = 12) -> str:
    selected = [line for line in text.splitlines() if line.strip()]
    return "\n".join(selected[-lines:])


def failed_metrics() -> dict:
    return {
        "utterances": 0,
        "total_frames": 0,
        "weighted_mean_infer_ms_per_frame": 0.0,
        "utterance_avg_infer_ms_per_frame_p50": 0.0,
        "utterance_avg_infer_ms_per_frame_p95": 0.0,
        "max_error_max": 0.0,
        "avg_error_mean": 0.0,
        "rms_error_mean": 0.0,
        "failures": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "myriad"), default="myriad")
    parser.add_argument(
        "--platform",
        default=None,
        choices=("armv7", "arm64", "amd64"),
        help="defaults to amd64 for CPU and arm64 for MYRIAD",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--log", type=pathlib.Path)
    args = parser.parse_args()

    platform = args.platform or ("amd64" if args.backend == "cpu" else "arm64")
    try:
        command = build_command(args.backend, platform)
    except ValueError as exc:
        parser.error(str(exc))

    suffix = "cpu-reference" if args.backend == "cpu" else f"myriad-{platform}"
    output_path = repo_path(
        args.output
        or pathlib.Path(f"work/speech-asr/rm_cnn4a/result-{suffix}.json")
    )
    log_path = repo_path(
        args.log
        or pathlib.Path(f"work/speech-asr/rm_cnn4a/speech_sample-{suffix}.log")
    )

    model_root = ROOT / "vendor" / "models" / "rm_cnn4a_smbr"
    xml = model_root / "openvino" / "fp16" / "rm_cnn4a_fp16.xml"
    binary = model_root / "openvino" / "fp16" / "rm_cnn4a_fp16.bin"
    model_spec = model_root / "openvino" / "model-spec.json"
    features = model_root / "source" / "feat1_10.ark"
    scores = model_root / "source" / "score1_10.ark"
    required = (xml, binary, model_spec, features, scores)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(
            "missing rm_cnn4a artifacts; run ./scripts/prepare-rm-cnn4a.sh first",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    raw_log_sha = sha256(log_path)

    runtime_backend = "CPU" if args.backend == "cpu" else "MYRIAD"
    base = {
        "schema": "speech-asr/acoustic-regression-result",
        "version": 1,
        "model": {
            "id": "rm_cnn4a-fp16",
            "family": "rm_cnn4a",
            "xml_sha256": sha256(xml),
            "bin_sha256": sha256(binary),
        },
        "fixture": {
            "features_sha256": sha256(features),
            "reference_scores_sha256": sha256(scores),
        },
        "runtime": {
            "repo_commit": git_head(),
            "backend": runtime_backend,
            "target": platform,
            "openvino_version": "2020.3.2",
        },
        "provenance": {
            "raw_log_sha256": raw_log_sha,
            "model_spec_sha256": sha256(model_spec),
            "command": command,
        },
    }

    exit_code = proc.returncode
    if proc.returncode == 0:
        try:
            parsed = parse_speech_sample_output(proc.stdout)
            result = {
                **base,
                "status": "completed",
                "metrics": parsed["metrics"],
                "provenance": {
                    **base["provenance"],
                    "model_load_ms": parsed["model_load_ms"],
                },
            }
        except ValueError as exc:
            exit_code = 1
            result = {
                **base,
                "status": "failed",
                "metrics": failed_metrics(),
                "diagnostics": {
                    "summary": str(exc),
                    "log_tail": output_tail(proc.stdout),
                    "log_path": str(log_path.relative_to(ROOT)),
                },
            }
    else:
        result = {
            **base,
            "status": "failed",
            "metrics": failed_metrics(),
            "diagnostics": {
                "summary": f"{command[-1]} exited with status {proc.returncode}",
                "log_tail": output_tail(proc.stdout),
                "log_path": str(log_path.relative_to(ROOT)),
            },
        }

    try:
        validate_acoustic_regression_result(result)
    except ContractValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
