#!/usr/bin/env python3
"""Run the rm_cnn4a MYRIAD regression and emit a machine-readable result."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="arm64", choices=("armv7", "arm64", "amd64"))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("work/speech-asr/rm_cnn4a/result.json"),
    )
    parser.add_argument(
        "--log",
        type=pathlib.Path,
        default=pathlib.Path("work/speech-asr/rm_cnn4a/speech_sample.log"),
    )
    args = parser.parse_args()

    model_root = ROOT / "vendor" / "models" / "rm_cnn4a_smbr"
    xml = model_root / "openvino" / "fp16" / "rm_cnn4a_fp16.xml"
    binary = model_root / "openvino" / "fp16" / "rm_cnn4a_fp16.bin"
    features = model_root / "source" / "feat1_10.ark"
    scores = model_root / "source" / "score1_10.ark"
    required = (xml, binary, features, scores)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("missing rm_cnn4a artifacts; run ./scripts/prepare-rm-cnn4a.sh first", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    command = [str(ROOT / "run.sh"), "--platform", args.platform, "speech-regress"]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(proc.stdout, encoding="utf-8")
    raw_log_sha = sha256(args.log)

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
            "backend": "MYRIAD",
            "target": args.platform,
            "openvino_version": "2020.3.2",
        },
        "provenance": {
            "raw_log_sha256": raw_log_sha,
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
                "metrics": {
                    "utterances": 0,
                    "total_frames": 0,
                    "weighted_mean_infer_ms_per_frame": 0.0,
                    "utterance_avg_infer_ms_per_frame_p50": 0.0,
                    "utterance_avg_infer_ms_per_frame_p95": 0.0,
                    "max_error_max": 0.0,
                    "avg_error_mean": 0.0,
                    "rms_error_mean": 0.0,
                    "failures": 1,
                },
                "diagnostics": {"summary": str(exc)},
            }
    else:
        result = {
            **base,
            "status": "failed",
            "metrics": {
                "utterances": 0,
                "total_frames": 0,
                "weighted_mean_infer_ms_per_frame": 0.0,
                "utterance_avg_infer_ms_per_frame_p50": 0.0,
                "utterance_avg_infer_ms_per_frame_p95": 0.0,
                "max_error_max": 0.0,
                "avg_error_mean": 0.0,
                "rms_error_mean": 0.0,
                "failures": 1,
            },
            "diagnostics": {
                "summary": f"speech-regress exited with status {proc.returncode}",
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
