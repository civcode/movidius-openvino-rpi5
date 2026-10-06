#!/usr/bin/env python3
"""Read the single sealed held-out evaluation result."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]

SOURCE_EXPERIMENT_ID = "exp-87538823d2bf1562"
DEFAULT_ROOT = (
    ROOT
    / "work"
    / "speech-asr"
    / "heldout-evaluations"
    / SOURCE_EXPERIMENT_ID
)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def discover_result(root: pathlib.Path) -> pathlib.Path:
    results = sorted(root.glob("*/result.json"))
    if not results:
        raise ValueError(
            f"no sealed held-out evaluation result found below {root}"
        )
    if len(results) != 1:
        raise ValueError(
            "expected exactly one sealed held-out evaluation result; "
            f"found {len(results)}: {results}"
        )
    return results[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=pathlib.Path)
    args = parser.parse_args()

    try:
        path = args.result or discover_result(DEFAULT_ROOT)
        if not path.is_file():
            raise ValueError(f"held-out result missing: {path}")
        result = load_json(path)
        if result.get("schema") != "speech-asr/heldout-evaluation-result":
            raise ValueError("held-out result schema mismatch")
        if result.get("status") != "completed":
            raise ValueError("held-out result is not completed")
        if result.get("role") != "test_only":
            raise ValueError("held-out result role is not test_only")
        if result.get("sealed") is not True:
            raise ValueError("held-out result is not sealed")

        reference = result["reference"]
        hardware = result["hardware"]
        summary = {
            "status": result["status"],
            "role": result["role"],
            "sealed": result["sealed"],
            "source": result["source"],
            "evaluation": result["evaluation"],
            "reference": {
                "metrics": reference["metrics"],
                "agreement": reference["agreement"],
            },
            "hardware": {
                "runtime": hardware["runtime"],
                "metrics": {
                    key: hardware["metrics"][key]
                    for key in (
                        "wer",
                        "cer",
                        "realtime_factor",
                        "inference_latency_p50_ms",
                        "inference_latency_p95_ms",
                        "failures",
                        "manifest_samples",
                        "evaluated_samples",
                        "skipped",
                        "decoder",
                    )
                    if key in hardware["metrics"]
                },
            },
            "cross_runtime_delta": result["cross_runtime_delta"],
            "supersedes": result.get("supersedes"),
            "result": str(path),
        }
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
