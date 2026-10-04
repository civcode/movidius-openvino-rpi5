"""Semantic validation for versioned speech-ASR contracts.

The JSON schema files in ``contracts/`` are the portable contract description.
The three historical `.yaml` root contracts intentionally use JSON-compatible
YAML 1.2 syntax so they can be parsed and validated with the Python standard
library. This module adds the cross-field invariants that matter to the
benchmark (for example canonical 16 kHz audio and sample-bound timing).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Mapping, MutableSequence, Sequence

_CANONICAL_SAMPLE_RATE = 16000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class ContractValidationError(ValueError):
    """Raised when a speech-ASR contract document is invalid."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("contract validation failed:\n- " + "\n- ".join(self.errors))


def canonical_json_sha256(document: Any) -> str:
    """Return SHA-256 of the canonical UTF-8 JSON representation of *document*."""

    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, path: str, errors: MutableSequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected object")
        return {}
    return value


def _sequence(value: Any, path: str, errors: MutableSequence[str]) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        errors.append(f"{path}: expected array")
        return ()
    return value


def _nonempty_string(value: Any, path: str, errors: MutableSequence[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected non-empty string")


def _integer(
    value: Any,
    path: str,
    errors: MutableSequence[str],
    minimum: int = 0,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: expected integer")
    elif value < minimum:
        errors.append(f"{path}: must be >= {minimum}")


def _number(
    value: Any,
    path: str,
    errors: MutableSequence[str],
    minimum: float = 0.0,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number")
    elif value < minimum:
        errors.append(f"{path}: must be >= {minimum}")


def _sha256(value: Any, path: str, errors: MutableSequence[str]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected lowercase 64-character SHA-256 hex digest")


def _git_sha(value: Any, path: str, errors: MutableSequence[str]) -> None:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected 7-40 lowercase hexadecimal Git commit id")


def _require_values(
    values: Any,
    required: Iterable[str],
    path: str,
    errors: MutableSequence[str],
) -> None:
    sequence = _sequence(values, path, errors)
    present = {item for item in sequence if isinstance(item, str)}
    for item in required:
        if item not in present:
            errors.append(f"{path}: missing required value {item!r}")


def _raise_if_errors(errors: MutableSequence[str]) -> None:
    if errors:
        raise ContractValidationError(errors)


def validate_audio_contract(document: Any) -> Dict[str, Any]:
    """Validate the global canonical-audio contract."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/audio":
        errors.append("$.schema: expected 'speech-asr/audio'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    if root.get("normalized_sample_schema") != "speech-sample-v1.schema.json":
        errors.append(
            "$.normalized_sample_schema: expected 'speech-sample-v1.schema.json'"
        )

    audio = _mapping(root.get("canonical_audio"), "$.canonical_audio", errors)
    if audio.get("sample_rate_hz") != _CANONICAL_SAMPLE_RATE:
        errors.append("$.canonical_audio.sample_rate_hz: expected 16000")
    if audio.get("channels") != 1:
        errors.append("$.canonical_audio.channels: expected 1")
    if audio.get("sample_type") != "float32":
        errors.append("$.canonical_audio.sample_type: expected 'float32'")
    if audio.get("encoding") != "f32le":
        errors.append("$.canonical_audio.encoding: expected 'f32le'")

    timing = _mapping(root.get("timing"), "$.timing", errors)
    if timing.get("canonical_unit") != "sample_index":
        errors.append("$.timing.canonical_unit: expected 'sample_index'")
    if timing.get("origin") != "normalized_audio_start":
        errors.append("$.timing.origin: expected 'normalized_audio_start'")
    if timing.get("integer_only") is not True:
        errors.append("$.timing.integer_only: expected true")

    source = _mapping(root.get("source_policy"), "$.source_policy", errors)
    if source.get("action") != "normalize_before_feature_extraction":
        errors.append(
            "$.source_policy.action: expected 'normalize_before_feature_extraction'"
        )
    rules = _sequence(root.get("rules"), "$.rules", errors)
    if not rules:
        errors.append("$.rules: at least one rule is required")
    for index, rule in enumerate(rules):
        _nonempty_string(rule, f"$.rules[{index}]", errors)

    _raise_if_errors(errors)
    return dict(root)


def validate_benchmark_contract(document: Any) -> Dict[str, Any]:
    """Validate benchmark-v1 and its frozen measurement policy."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/benchmark":
        errors.append("$.schema: expected 'speech-asr/benchmark'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    if root.get("result_schema") != "experiment-result-v1.schema.json":
        errors.append("$.result_schema: expected 'experiment-result-v1.schema.json'")

    dataset = _mapping(root.get("dataset"), "$.dataset", errors)
    if dataset.get("canonical_corpus") != "AMI":
        errors.append("$.dataset.canonical_corpus: expected 'AMI'")
    if dataset.get("normalized_manifest_format") != "jsonl":
        errors.append("$.dataset.normalized_manifest_format: expected 'jsonl'")
    if dataset.get("timing_unit") != "sample_index":
        errors.append("$.dataset.timing_unit: expected 'sample_index'")
    splits = _mapping(dataset.get("splits"), "$.dataset.splits", errors)
    for split in ("smoke", "benchmark", "validation"):
        _nonempty_string(splits.get(split), f"$.dataset.splits.{split}", errors)

    if root.get("audio_contract") != "audio-v1":
        errors.append("$.audio_contract: expected 'audio-v1'")

    scoring = _mapping(root.get("scoring"), "$.scoring", errors)
    if scoring.get("text_normalization_version") != "text-v1":
        errors.append("$.scoring.text_normalization_version: expected 'text-v1'")
    if scoring.get("text_normalization_contract") != "text-v1.json":
        errors.append("$.scoring.text_normalization_contract: expected 'text-v1.json'")
    _require_values(scoring.get("metrics"), ("wer", "cer"), "$.scoring.metrics", errors)

    _require_values(
        root.get("hardware_metrics"),
        (
            "realtime_factor",
            "inference_latency_p50_ms",
            "inference_latency_p95_ms",
            "model_load_success",
            "inference_failures",
        ),
        "$.hardware_metrics",
        errors,
    )

    policy = _mapping(root.get("measurement_policy"), "$.measurement_policy", errors)
    _integer(policy.get("warmup_count"), "$.measurement_policy.warmup_count", errors)
    if policy.get("warmup_count") != 1:
        errors.append("$.measurement_policy.warmup_count: expected frozen value 1")
    _integer(
        policy.get("measured_iterations"),
        "$.measurement_policy.measured_iterations",
        errors,
        minimum=1,
    )
    if policy.get("measured_iterations") != 5:
        errors.append("$.measurement_policy.measured_iterations: expected frozen value 5")
    percentiles = _sequence(
        policy.get("latency_percentiles"),
        "$.measurement_policy.latency_percentiles",
        errors,
    )
    if list(percentiles) != [50, 95]:
        errors.append("$.measurement_policy.latency_percentiles: expected [50, 95]")
    if policy.get("aggregation") != "corpus_weighted":
        errors.append("$.measurement_policy.aggregation: expected 'corpus_weighted'")

    _raise_if_errors(errors)
    return dict(root)


def validate_model_contract(document: Any) -> Dict[str, Any]:
    """Validate the global model-package boundary."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/model":
        errors.append("$.schema: expected 'speech-asr/model'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")

    package = _mapping(root.get("package"), "$.package", errors)
    _require_values(
        package.get("required_fields"),
        (
            "id",
            "family",
            "frontend_profile",
            "input_contract",
            "output_contract",
            "decoder_kind",
            "deployment",
            "provenance",
        ),
        "$.package.required_fields",
        errors,
    )

    frontend = _mapping(root.get("frontend"), "$.frontend", errors)
    if frontend.get("policy") != "model_declared":
        errors.append("$.frontend.policy: expected 'model_declared'")
    if frontend.get("canonical_audio_contract") != "audio-v1":
        errors.append("$.frontend.canonical_audio_contract: expected 'audio-v1'")

    deployment = _mapping(root.get("deployment"), "$.deployment", errors)
    if deployment.get("target") != "MYRIAD":
        errors.append("$.deployment.target: expected 'MYRIAD'")
    if deployment.get("openvino_version") != "2020.3.2":
        errors.append("$.deployment.openvino_version: expected '2020.3.2'")
    if deployment.get("precision") != "FP16":
        errors.append("$.deployment.precision: expected 'FP16'")

    provenance = _mapping(root.get("provenance"), "$.provenance", errors)
    _require_values(
        provenance.get("required_fields"),
        ("source_id", "source_sha256", "spec_sha256", "artifact_sha256"),
        "$.provenance.required_fields",
        errors,
    )

    _raise_if_errors(errors)
    return dict(root)


def validate_text_contract(document: Any) -> Dict[str, Any]:
    """Validate the frozen text-v1 normalization contract."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/text-normalization":
        errors.append("$.schema: expected 'speech-asr/text-normalization'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    if root.get("id") != "text-v1":
        errors.append("$.id: expected 'text-v1'")
    if root.get("language") != "en":
        errors.append("$.language: expected 'en'")
    if root.get("implementation") != "python/speech_asr/text.py::normalize_text_v1":
        errors.append(
            "$.implementation: expected 'python/speech_asr/text.py::normalize_text_v1'"
        )
    rules = _mapping(root.get("rules"), "$.rules", errors)
    expected = {
        "unicode_normalization": "NFKC",
        "case": "lower",
        "apostrophe": "normalize_curly_to_ascii_and_preserve_inside_tokens",
        "punctuation": "replace_with_space_except_internal_apostrophe",
        "numbers": "preserve_digits_without_verbalization",
        "whitespace": "trim_and_collapse",
        "special_tokens": (
            "do_not_drop_unless_dataset_adapter_explicitly_maps_them_before_text-v1"
        ),
    }
    for key, value in expected.items():
        if rules.get(key) != value:
            errors.append(f"$.rules.{key}: expected {value!r}")

    _raise_if_errors(errors)
    return dict(root)


def validate_speech_sample(document: Any) -> Dict[str, Any]:
    """Validate and return a shallow copy of a normalized speech sample."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)

    if root.get("schema") != "speech-asr/sample":
        errors.append("$.schema: expected 'speech-asr/sample'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    _nonempty_string(root.get("id"), "$.id", errors)

    audio = _mapping(root.get("audio"), "$.audio", errors)
    _nonempty_string(audio.get("path"), "$.audio.path", errors)
    if audio.get("sample_rate_hz") != _CANONICAL_SAMPLE_RATE:
        errors.append(f"$.audio.sample_rate_hz: expected {_CANONICAL_SAMPLE_RATE}")
    if audio.get("channels") != 1:
        errors.append("$.audio.channels: expected 1")
    if audio.get("sample_type") != "float32":
        errors.append("$.audio.sample_type: expected 'float32'")
    if audio.get("encoding") != "f32le":
        errors.append("$.audio.encoding: expected 'f32le'")
    _integer(audio.get("start_sample"), "$.audio.start_sample", errors)
    _integer(audio.get("end_sample"), "$.audio.end_sample", errors)

    start = audio.get("start_sample")
    end = audio.get("end_sample")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and end <= start
    ):
        errors.append("$.audio.end_sample: must be greater than start_sample")

    transcript = _mapping(root.get("transcript"), "$.transcript", errors)
    if not isinstance(transcript.get("text"), str):
        errors.append("$.transcript.text: expected string")
    if transcript.get("normalization") != "text-v1":
        errors.append("$.transcript.normalization: expected 'text-v1'")

    words = transcript.get("words", [])
    if not isinstance(words, list):
        errors.append("$.transcript.words: expected array")
        words = []

    previous_start = None
    for index, word_value in enumerate(words):
        path = f"$.transcript.words[{index}]"
        word = _mapping(word_value, path, errors)
        _nonempty_string(word.get("text"), path + ".text", errors)
        _integer(word.get("start_sample"), path + ".start_sample", errors)
        _integer(word.get("end_sample"), path + ".end_sample", errors)
        word_start = word.get("start_sample")
        word_end = word.get("end_sample")
        if (
            isinstance(word_start, int)
            and not isinstance(word_start, bool)
            and isinstance(word_end, int)
            and not isinstance(word_end, bool)
        ):
            if word_end <= word_start:
                errors.append(path + ".end_sample: must be greater than start_sample")
            if isinstance(start, int) and word_start < start:
                errors.append(path + ".start_sample: precedes sample audio range")
            if isinstance(end, int) and word_end > end:
                errors.append(path + ".end_sample: exceeds sample audio range")
            if previous_start is not None and word_start < previous_start:
                errors.append(
                    path + ".start_sample: word timings must be ordered by start_sample"
                )
            previous_start = word_start

    metadata = root.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        errors.append("$.metadata: expected object when present")

    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_result(document: Any) -> Dict[str, Any]:
    """Validate and return a shallow copy of an experiment result document."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)

    if root.get("schema") != "speech-asr/experiment-result":
        errors.append("$.schema: expected 'speech-asr/experiment-result'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    _nonempty_string(root.get("experiment_id"), "$.experiment_id", errors)

    status = root.get("status")
    if status not in {"completed", "failed", "blocked"}:
        errors.append("$.status: expected one of completed, failed, blocked")

    benchmark = _mapping(root.get("benchmark"), "$.benchmark", errors)
    _nonempty_string(benchmark.get("id"), "$.benchmark.id", errors)
    if benchmark.get("contract_version") != 1:
        errors.append("$.benchmark.contract_version: expected 1")
    _sha256(benchmark.get("manifest_sha256"), "$.benchmark.manifest_sha256", errors)
    if benchmark.get("text_normalization_version") != "text-v1":
        errors.append("$.benchmark.text_normalization_version: expected 'text-v1'")

    model = _mapping(root.get("model"), "$.model", errors)
    _nonempty_string(model.get("id"), "$.model.id", errors)
    _nonempty_string(model.get("family"), "$.model.family", errors)
    _sha256(model.get("spec_sha256"), "$.model.spec_sha256", errors)
    artifact_hashes = _mapping(
        model.get("artifact_sha256"),
        "$.model.artifact_sha256",
        errors,
    )
    if not artifact_hashes:
        errors.append("$.model.artifact_sha256: at least one artifact hash is required")
    for name, digest in artifact_hashes.items():
        _nonempty_string(name, "$.model.artifact_sha256 key", errors)
        _sha256(digest, f"$.model.artifact_sha256[{name!r}]", errors)

    runtime = _mapping(root.get("runtime"), "$.runtime", errors)
    _git_sha(runtime.get("repo_commit"), "$.runtime.repo_commit", errors)
    _nonempty_string(runtime.get("backend"), "$.runtime.backend", errors)
    if runtime.get("openvino_version") != "2020.3.2":
        errors.append("$.runtime.openvino_version: expected '2020.3.2'")
    _nonempty_string(runtime.get("target"), "$.runtime.target", errors)
    firmware_sha = runtime.get("firmware_sha256")
    if firmware_sha is not None:
        _sha256(firmware_sha, "$.runtime.firmware_sha256", errors)

    metrics = root.get("metrics")
    if status == "completed":
        metrics = _mapping(metrics, "$.metrics", errors)
        for key in (
            "wer",
            "cer",
            "realtime_factor",
            "inference_latency_p50_ms",
            "inference_latency_p95_ms",
        ):
            _number(metrics.get(key), f"$.metrics.{key}", errors)
        _integer(metrics.get("failures"), "$.metrics.failures", errors)
    elif metrics is not None and not isinstance(metrics, Mapping):
        errors.append("$.metrics: expected object when present")

    diagnostics = root.get("diagnostics")
    if status in {"failed", "blocked"}:
        diagnostics = _mapping(diagnostics, "$.diagnostics", errors)
        _nonempty_string(diagnostics.get("summary"), "$.diagnostics.summary", errors)
    elif diagnostics is not None and not isinstance(diagnostics, Mapping):
        errors.append("$.diagnostics: expected object when present")

    provenance = _mapping(root.get("provenance"), "$.provenance", errors)
    _sha256(
        provenance.get("dataset_manifest_sha256"),
        "$.provenance.dataset_manifest_sha256",
        errors,
    )
    _sha256(
        provenance.get("model_spec_sha256"),
        "$.provenance.model_spec_sha256",
        errors,
    )

    _raise_if_errors(errors)
    return dict(root)


def validate_acoustic_regression_result(document: Any) -> Dict[str, Any]:
    """Validate a vendor acoustic-score regression result."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/acoustic-regression-result":
        errors.append("$.schema: expected 'speech-asr/acoustic-regression-result'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    status = root.get("status")
    if status not in {"completed", "failed"}:
        errors.append("$.status: expected one of completed, failed")

    model = _mapping(root.get("model"), "$.model", errors)
    _nonempty_string(model.get("id"), "$.model.id", errors)
    _nonempty_string(model.get("family"), "$.model.family", errors)
    _sha256(model.get("xml_sha256"), "$.model.xml_sha256", errors)
    _sha256(model.get("bin_sha256"), "$.model.bin_sha256", errors)

    fixture = _mapping(root.get("fixture"), "$.fixture", errors)
    _sha256(fixture.get("features_sha256"), "$.fixture.features_sha256", errors)
    _sha256(
        fixture.get("reference_scores_sha256"),
        "$.fixture.reference_scores_sha256",
        errors,
    )

    runtime = _mapping(root.get("runtime"), "$.runtime", errors)
    _git_sha(runtime.get("repo_commit"), "$.runtime.repo_commit", errors)
    backend = runtime.get("backend")
    if backend not in {"CPU", "MYRIAD"}:
        errors.append("$.runtime.backend: expected one of CPU, MYRIAD")
    _nonempty_string(runtime.get("target"), "$.runtime.target", errors)
    if backend == "CPU" and runtime.get("target") != "amd64":
        errors.append("$.runtime.target: CPU reference backend requires 'amd64'")
    if runtime.get("openvino_version") != "2020.3.2":
        errors.append("$.runtime.openvino_version: expected '2020.3.2'")

    metrics = _mapping(root.get("metrics"), "$.metrics", errors)
    _integer(metrics.get("utterances"), "$.metrics.utterances", errors)
    _integer(metrics.get("total_frames"), "$.metrics.total_frames", errors)
    for key in (
        "weighted_mean_infer_ms_per_frame",
        "utterance_avg_infer_ms_per_frame_p50",
        "utterance_avg_infer_ms_per_frame_p95",
        "max_error_max",
        "avg_error_mean",
        "rms_error_mean",
    ):
        _number(metrics.get(key), f"$.metrics.{key}", errors)
    _integer(metrics.get("failures"), "$.metrics.failures", errors)

    provenance = _mapping(root.get("provenance"), "$.provenance", errors)
    _sha256(provenance.get("raw_log_sha256"), "$.provenance.raw_log_sha256", errors)

    diagnostics = root.get("diagnostics")
    if status == "failed":
        diagnostics = _mapping(diagnostics, "$.diagnostics", errors)
        _nonempty_string(diagnostics.get("summary"), "$.diagnostics.summary", errors)

    _raise_if_errors(errors)
    return dict(root)


def validate_acoustic_benchmark_result(document: Any) -> Dict[str, Any]:
    """Validate a repeated rm_cnn4a benchmark-worker result."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/acoustic-benchmark-result":
        errors.append("$.schema: expected 'speech-asr/acoustic-benchmark-result'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")

    status = root.get("status")
    if status not in {"completed", "failed"}:
        errors.append("$.status: expected one of completed, failed")

    benchmark = _mapping(root.get("benchmark"), "$.benchmark", errors)
    if benchmark.get("id") != "rm_cnn4a-vendor-regression-v1":
        errors.append(
            "$.benchmark.id: expected 'rm_cnn4a-vendor-regression-v1'"
        )
    _sha256(benchmark.get("contract_sha256"), "$.benchmark.contract_sha256", errors)
    policy = _mapping(benchmark.get("measurement_policy"), "$.benchmark.measurement_policy", errors)
    if policy.get("warmup_count") != 1:
        errors.append("$.benchmark.measurement_policy.warmup_count: expected 1")
    if policy.get("measured_iterations") != 5:
        errors.append("$.benchmark.measurement_policy.measured_iterations: expected 5")
    if policy.get("within_run_aggregation") != "corpus_weighted":
        errors.append(
            "$.benchmark.measurement_policy.within_run_aggregation: "
            "expected 'corpus_weighted'"
        )
    if policy.get("warmup_included_in_aggregate") is not False:
        errors.append(
            "$.benchmark.measurement_policy.warmup_included_in_aggregate: expected false"
        )

    request = _mapping(root.get("request"), "$.request", errors)
    if request.get("model") != "rm_cnn4a":
        errors.append("$.request.model: expected 'rm_cnn4a'")
    if request.get("backend") not in {"cpu", "myriad"}:
        errors.append("$.request.backend: expected one of cpu, myriad")
    if request.get("platform") not in {"armv7", "arm64", "amd64"}:
        errors.append("$.request.platform: expected one of armv7, arm64, amd64")

    model = _mapping(root.get("model"), "$.model", errors)
    _nonempty_string(model.get("id"), "$.model.id", errors)
    _nonempty_string(model.get("family"), "$.model.family", errors)
    _sha256(model.get("xml_sha256"), "$.model.xml_sha256", errors)
    _sha256(model.get("bin_sha256"), "$.model.bin_sha256", errors)
    if model.get("canonical_graph_sha256") is not None:
        _sha256(
            model.get("canonical_graph_sha256"),
            "$.model.canonical_graph_sha256",
            errors,
        )

    fixture = _mapping(root.get("fixture"), "$.fixture", errors)
    _sha256(fixture.get("features_sha256"), "$.fixture.features_sha256", errors)
    _sha256(
        fixture.get("reference_scores_sha256"),
        "$.fixture.reference_scores_sha256",
        errors,
    )

    runtime = _mapping(root.get("runtime"), "$.runtime", errors)
    _git_sha(runtime.get("repo_commit"), "$.runtime.repo_commit", errors)
    if runtime.get("backend") not in {"CPU", "MYRIAD"}:
        errors.append("$.runtime.backend: expected one of CPU, MYRIAD")
    _nonempty_string(runtime.get("target"), "$.runtime.target", errors)
    if runtime.get("openvino_version") != "2020.3.2":
        errors.append("$.runtime.openvino_version: expected '2020.3.2'")
    _nonempty_string(runtime.get("host_machine"), "$.runtime.host_machine", errors)

    runs = _mapping(root.get("runs"), "$.runs", errors)
    warmup = _sequence(runs.get("warmup"), "$.runs.warmup", errors)
    measured = _sequence(runs.get("measured"), "$.runs.measured", errors)
    if len(warmup) > 1:
        errors.append("$.runs.warmup: at most one run is allowed")
    if len(measured) > 5:
        errors.append("$.runs.measured: at most five runs are allowed")

    for group_name, values in (("warmup", warmup), ("measured", measured)):
        for index, value in enumerate(values):
            path = f"$.runs.{group_name}[{index}]"
            item = _mapping(value, path, errors)
            _integer(item.get("index"), path + ".index", errors)
            if item.get("status") not in {"completed", "failed"}:
                errors.append(path + ".status: expected completed or failed")
            _nonempty_string(item.get("result_path"), path + ".result_path", errors)
            _nonempty_string(item.get("log_path"), path + ".log_path", errors)
            _integer(item.get("returncode"), path + ".returncode", errors)
            _sha256(item.get("raw_log_sha256"), path + ".raw_log_sha256", errors)
            if item.get("status") == "completed":
                run_metrics = _mapping(item.get("metrics"), path + ".metrics", errors)
                for metric_name in (
                    "weighted_mean_infer_ms_per_frame",
                    "utterance_avg_infer_ms_per_frame_p50",
                    "utterance_avg_infer_ms_per_frame_p95",
                    "model_load_ms",
                    "max_error_max",
                    "avg_error_mean",
                    "rms_error_mean",
                ):
                    _number(
                        run_metrics.get(metric_name),
                        path + f".metrics.{metric_name}",
                        errors,
                    )

    provenance = _mapping(root.get("provenance"), "$.provenance", errors)
    _sha256(
        provenance.get("benchmark_contract_sha256"),
        "$.provenance.benchmark_contract_sha256",
        errors,
    )
    _sha256(provenance.get("runner_sha256"), "$.provenance.runner_sha256", errors)
    _sha256(provenance.get("worker_sha256"), "$.provenance.worker_sha256", errors)
    _nonempty_string(provenance.get("python_version"), "$.provenance.python_version", errors)
    if (
        benchmark.get("contract_sha256") is not None
        and provenance.get("benchmark_contract_sha256") is not None
        and benchmark.get("contract_sha256") != provenance.get("benchmark_contract_sha256")
    ):
        errors.append(
            "$.provenance.benchmark_contract_sha256: must match $.benchmark.contract_sha256"
        )

    expected_backend = "CPU" if request.get("backend") == "cpu" else "MYRIAD"
    if runtime.get("backend") != expected_backend:
        errors.append(
            "$.runtime.backend: does not match $.request.backend"
        )
    if runtime.get("target") != request.get("platform"):
        errors.append(
            "$.runtime.target: does not match $.request.platform"
        )

    aggregate = root.get("aggregate")
    diagnostics = root.get("diagnostics")
    if status == "completed":
        if len(warmup) != 1:
            errors.append("$.runs.warmup: completed result requires exactly one warmup")
        if len(measured) != 5:
            errors.append("$.runs.measured: completed result requires exactly five measured runs")
        for path, values in (("$.runs.warmup", warmup), ("$.runs.measured", measured)):
            for index, item in enumerate(values):
                if isinstance(item, Mapping) and item.get("status") != "completed":
                    errors.append(f"{path}[{index}].status: completed worker requires completed run")

        aggregate = _mapping(aggregate, "$.aggregate", errors)
        if aggregate.get("measured_runs") != 5:
            errors.append("$.aggregate.measured_runs: expected 5")
        counts = _mapping(aggregate.get("counts"), "$.aggregate.counts", errors)
        _integer(counts.get("utterances"), "$.aggregate.counts.utterances", errors)
        _integer(counts.get("total_frames"), "$.aggregate.counts.total_frames", errors)
        if counts.get("failures") != 0:
            errors.append("$.aggregate.counts.failures: expected 0")

        metrics = _mapping(aggregate.get("metrics"), "$.aggregate.metrics", errors)
        for metric_name in (
            "weighted_mean_infer_ms_per_frame",
            "utterance_avg_infer_ms_per_frame_p50",
            "utterance_avg_infer_ms_per_frame_p95",
            "model_load_ms",
            "max_error_max",
            "avg_error_mean",
            "rms_error_mean",
        ):
            stats = _mapping(metrics.get(metric_name), f"$.aggregate.metrics.{metric_name}", errors)
            for stat_name in ("mean", "median", "min", "max", "range", "relative_range", "cv_population"):
                _number(
                    stats.get(stat_name),
                    f"$.aggregate.metrics.{metric_name}.{stat_name}",
                    errors,
                )
    else:
        diagnostics = _mapping(diagnostics, "$.diagnostics", errors)
        _nonempty_string(diagnostics.get("summary"), "$.diagnostics.summary", errors)
        if aggregate is not None and not isinstance(aggregate, Mapping):
            errors.append("$.aggregate: expected object when present")

    _raise_if_errors(errors)
    return dict(root)
