#!/usr/bin/env python3
"""Validate a versioned speech-ASR contract or result document."""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from speech_asr.contracts import (  # noqa: E402
    ContractValidationError,
    canonical_json_sha256,
    validate_acoustic_benchmark_result,
    validate_acoustic_regression_result,
    validate_audio_contract,
    validate_benchmark_contract,
    validate_acceptance_policy,
    validate_deployment_manifest,
    validate_edge_worker_result,
    validate_experiment_attempt,
    validate_experiment_history,
    validate_experiment_model_spec,
    validate_experiment_proposal,
    validate_experiment_request,
    validate_experiment_result,
    validate_experiment_summary,
    validate_model_contract,
    validate_speech_sample,
    validate_streaming_contract,
    validate_streaming_replay_result,
    validate_text_contract,
    validate_train_config,
)


VALIDATORS = {
    "audio": validate_audio_contract,
    "benchmark": validate_benchmark_contract,
    "model": validate_model_contract,
    "text": validate_text_contract,
    "sample": validate_speech_sample,
    "result": validate_experiment_result,
    "regression": validate_acoustic_regression_result,
    "acoustic-benchmark": validate_acoustic_benchmark_result,
    "streaming": validate_streaming_contract,
    "streaming-replay": validate_streaming_replay_result,
    "experiment-proposal": validate_experiment_proposal,
    "experiment-model-spec": validate_experiment_model_spec,
    "train-config": validate_train_config,
    "acceptance-policy": validate_acceptance_policy,
    "experiment-request": validate_experiment_request,
    "experiment-attempt": validate_experiment_attempt,
    "experiment-summary": validate_experiment_summary,
    "experiment-history": validate_experiment_history,
    "deployment-manifest": validate_deployment_manifest,
    "edge-worker-result": validate_edge_worker_result,
}

SCHEMA_TO_KIND = {
    "speech-asr/audio": "audio",
    "speech-asr/benchmark": "benchmark",
    "speech-asr/model": "model",
    "speech-asr/text-normalization": "text",
    "speech-asr/sample": "sample",
    "speech-asr/experiment-result": "result",
    "speech-asr/acoustic-regression-result": "regression",
    "speech-asr/acoustic-benchmark-result": "acoustic-benchmark",
    "speech-asr/streaming": "streaming",
    "speech-asr/streaming-replay-result": "streaming-replay",
    "speech-asr/experiment-proposal": "experiment-proposal",
    "speech-asr/experiment-model-spec": "experiment-model-spec",
    "speech-asr/train-config": "train-config",
    "speech-asr/acceptance-policy": "acceptance-policy",
    "speech-asr/experiment-request": "experiment-request",
    "speech-asr/experiment-attempt": "experiment-attempt",
    "speech-asr/experiment-summary": "experiment-summary",
    "speech-asr/experiment-history": "experiment-history",
    "speech-asr/deployment-manifest": "deployment-manifest",
    "speech-asr/edge-worker-result": "edge-worker-result",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=pathlib.Path)
    parser.add_argument(
        "--kind",
        choices=("auto", *VALIDATORS),
        default="auto",
    )
    args = parser.parse_args()

    try:
        with args.document.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    kind = args.kind
    if kind == "auto":
        schema = document.get("schema") if isinstance(document, dict) else None
        kind = SCHEMA_TO_KIND.get(schema)
        if kind is None:
            print("error: cannot infer contract kind from $.schema", file=sys.stderr)
            return 2

    try:
        VALIDATORS[kind](document)
    except ContractValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"valid {kind} sha256={canonical_json_sha256(document)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
