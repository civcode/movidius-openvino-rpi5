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
_EXPERIMENT_ID_RE = re.compile(r"^exp-[0-9a-f]{16}$")
_ATTEMPT_ID_RE = re.compile(r"^attempt-[0-9]{4}$")
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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



def _artifact_ref(value: Any, path: str, errors: MutableSequence[str]) -> Mapping[str, Any]:
    ref = _mapping(value, path, errors)
    _nonempty_string(ref.get("path"), path + ".path", errors)
    _sha256(ref.get("sha256"), path + ".sha256", errors)
    return ref


def _manifest_ref(value: Any, path: str, errors: MutableSequence[str]) -> Mapping[str, Any]:
    ref = _mapping(value, path, errors)
    _nonempty_string(ref.get("id"), path + ".id", errors)
    _nonempty_string(ref.get("path"), path + ".path", errors)
    _sha256(ref.get("sha256"), path + ".sha256", errors)
    return ref


def validate_experiment_proposal(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-proposal":
        errors.append("$.schema: expected 'speech-asr/experiment-proposal'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    for key in ("title", "hypothesis", "rationale"):
        _nonempty_string(root.get(key), f"$.{key}", errors)
    changes = _sequence(root.get("changes"), "$.changes", errors)
    if not changes:
        errors.append("$.changes: at least one declared change is required")
    seen: set[str] = set()
    for index, value in enumerate(changes):
        _nonempty_string(value, f"$.changes[{index}]", errors)
        if isinstance(value, str):
            if value in seen:
                errors.append(f"$.changes[{index}]: duplicate declared change")
            seen.add(value)
    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_model_spec(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-model-spec":
        errors.append("$.schema: expected 'speech-asr/experiment-model-spec'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    model_id = root.get("model_id")
    if not isinstance(model_id, str) or _MODEL_ID_RE.fullmatch(model_id) is None:
        errors.append("$.model_id: expected lowercase model identifier")
    _nonempty_string(root.get("family"), "$.family", errors)
    for key in ("frontend", "architecture"):
        value = _mapping(root.get(key), f"$.{key}", errors)
        _nonempty_string(value.get("kind"), f"$.{key}.kind", errors)
    export = _mapping(root.get("export"), "$.export", errors)
    if export.get("format") != "onnx":
        errors.append("$.export.format: expected 'onnx'")
    if export.get("onnx_opset") != 11:
        errors.append("$.export.onnx_opset: expected 11")
    if export.get("fixed_shapes") is not True:
        errors.append("$.export.fixed_shapes: expected true")
    _raise_if_errors(errors)
    return dict(root)


def validate_train_config(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/train-config":
        errors.append("$.schema: expected 'speech-asr/train-config'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    _integer(root.get("seed"), "$.seed", errors)
    if root.get("device") not in {"cuda", "cpu"}:
        errors.append("$.device: expected one of cuda, cpu")
    _integer(root.get("epochs"), "$.epochs", errors)
    _integer(root.get("batch_size"), "$.batch_size", errors, minimum=1)
    max_samples = root.get("max_samples")
    if max_samples is not None:
        _integer(max_samples, "$.max_samples", errors, minimum=1)
    checkpoint_selection = root.get("checkpoint_selection")
    if checkpoint_selection is not None and checkpoint_selection not in {
        "validation_loss",
        "validation_cer",
    }:
        errors.append(
            "$.checkpoint_selection: expected validation_loss or validation_cer"
        )
    ctc_objective_value = root.get("ctc_objective")
    if ctc_objective_value is not None:
        ctc_objective = _mapping(
            ctc_objective_value,
            "$.ctc_objective",
            errors,
        )
        if ctc_objective.get("kind") != "blank-logit-penalty-v1":
            errors.append(
                "$.ctc_objective.kind: expected 'blank-logit-penalty-v1'"
            )
        blank_logit_penalty = ctc_objective.get("blank_logit_penalty")
        _number(
            blank_logit_penalty,
            "$.ctc_objective.blank_logit_penalty",
            errors,
        )
        if (
            isinstance(blank_logit_penalty, (int, float))
            and not isinstance(blank_logit_penalty, bool)
            and not (0 < blank_logit_penalty <= 1.0)
        ):
            errors.append(
                "$.ctc_objective.blank_logit_penalty: must be > 0 and <= 1"
            )
    augmentation_value = root.get("augmentation")
    if augmentation_value is not None:
        augmentation = _mapping(
            augmentation_value,
            "$.augmentation",
            errors,
        )
        if augmentation.get("kind") != "specaugment-v1":
            errors.append(
                "$.augmentation.kind: expected 'specaugment-v1'"
            )
        for key, maximum in (
            ("frequency_masks", 4),
            ("frequency_max_width", 32),
            ("time_masks", 4),
            ("time_max_width", 128),
        ):
            _integer(
                augmentation.get(key),
                f"$.augmentation.{key}",
                errors,
                minimum=1,
            )
            value = augmentation.get(key)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > maximum
            ):
                errors.append(
                    f"$.augmentation.{key}: must be <= {maximum}"
                )
        time_fraction = augmentation.get("time_max_fraction")
        _number(
            time_fraction,
            "$.augmentation.time_max_fraction",
            errors,
        )
        if (
            isinstance(time_fraction, (int, float))
            and not isinstance(time_fraction, bool)
            and not (0 < time_fraction <= 0.5)
        ):
            errors.append(
                "$.augmentation.time_max_fraction: must be > 0 and <= 0.5"
            )
        if augmentation.get("mask_value") != 0:
            errors.append("$.augmentation.mask_value: expected 0")
        _integer(
            augmentation.get("seed_offset"),
            "$.augmentation.seed_offset",
            errors,
            minimum=1,
        )
    optimizer = _mapping(root.get("optimizer"), "$.optimizer", errors)
    _nonempty_string(optimizer.get("kind"), "$.optimizer.kind", errors)
    learning_rate = optimizer.get("learning_rate")
    _number(learning_rate, "$.optimizer.learning_rate", errors)
    if isinstance(learning_rate, (int, float)) and not isinstance(learning_rate, bool) and learning_rate <= 0:
        errors.append("$.optimizer.learning_rate: must be > 0")
    _manifest_ref(root.get("training_manifest"), "$.training_manifest", errors)
    _manifest_ref(root.get("validation_manifest"), "$.validation_manifest", errors)
    _raise_if_errors(errors)
    return dict(root)


def validate_acceptance_policy(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/acceptance-policy":
        errors.append("$.schema: expected 'speech-asr/acceptance-policy'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    allowed_gates = {
        "training",
        "onnx_export",
        "openvino_conversion",
        "myriad_execution",
        "accuracy_evaluation",
    }
    gates = _sequence(root.get("required_gates"), "$.required_gates", errors)
    if not gates:
        errors.append("$.required_gates: at least one gate is required")
    seen: set[str] = set()
    for index, gate in enumerate(gates):
        if gate not in allowed_gates:
            errors.append(f"$.required_gates[{index}]: unknown gate {gate!r}")
        elif gate in seen:
            errors.append(f"$.required_gates[{index}]: duplicate gate")
        else:
            seen.add(gate)

    thresholds = _mapping(root.get("thresholds"), "$.thresholds", errors)
    for key in ("max_wer", "max_cer", "max_realtime_factor", "max_latency_p95_ms"):
        value = thresholds.get(key)
        if value is not None:
            _number(value, f"$.thresholds.{key}", errors)
    agreement = thresholds.get("min_frame_argmax_agreement")
    if agreement is not None:
        _number(agreement, "$.thresholds.min_frame_argmax_agreement", errors)
        if isinstance(agreement, (int, float)) and not isinstance(agreement, bool) and agreement > 1:
            errors.append("$.thresholds.min_frame_argmax_agreement: must be <= 1")

    retry = _mapping(root.get("retry_policy"), "$.retry_policy", errors)
    _integer(retry.get("max_attempts"), "$.retry_policy.max_attempts", errors, minimum=1)
    retryable = _sequence(
        retry.get("retryable_failure_classes"),
        "$.retry_policy.retryable_failure_classes",
        errors,
    )
    allowed_retry = {"transport_preflight", "worker_busy", "hardware_transient"}
    for index, value in enumerate(retryable):
        if value not in allowed_retry:
            errors.append(
                f"$.retry_policy.retryable_failure_classes[{index}]: "
                f"unknown retryable class {value!r}"
            )
    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_request(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-request":
        errors.append("$.schema: expected 'speech-asr/experiment-request'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        errors.append("$.experiment_id: expected exp- followed by 16 lowercase hex digits")
    parent_id = root.get("parent_experiment_id")
    if parent_id is not None and (
        not isinstance(parent_id, str) or _EXPERIMENT_ID_RE.fullmatch(parent_id) is None
    ):
        errors.append("$.parent_experiment_id: expected null or experiment id")
    if parent_id is not None and parent_id == experiment_id:
        errors.append("$.parent_experiment_id: experiment cannot be its own parent")
    _sha256(root.get("identity_sha256"), "$.identity_sha256", errors)
    if root.get("state") != "APPROVED":
        errors.append("$.state: immutable request must start at APPROVED")

    source = _mapping(root.get("source"), "$.source", errors)
    commit = source.get("repo_commit")
    if not isinstance(commit, str) or len(commit) != 40 or _GIT_SHA_RE.fullmatch(commit) is None:
        errors.append("$.source.repo_commit: expected full 40-character Git commit id")

    benchmark = _mapping(root.get("benchmark"), "$.benchmark", errors)
    _nonempty_string(benchmark.get("id"), "$.benchmark.id", errors)
    _nonempty_string(benchmark.get("manifest_path"), "$.benchmark.manifest_path", errors)
    _sha256(benchmark.get("manifest_sha256"), "$.benchmark.manifest_sha256", errors)

    documents = _mapping(root.get("documents"), "$.documents", errors)
    expected_document_paths = {
        "proposal": "request/proposal.json",
        "model_spec": "request/model-spec.json",
        "train_config": "request/train-config.json",
        "acceptance": "request/acceptance.json",
    }
    for name in ("proposal", "model_spec", "train_config", "acceptance"):
        ref = _artifact_ref(documents.get(name), f"$.documents.{name}", errors)
        if ref.get("path") != expected_document_paths[name]:
            errors.append(
                f"$.documents.{name}.path: expected {expected_document_paths[name]!r}"
            )

    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_attempt(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-attempt":
        errors.append("$.schema: expected 'speech-asr/experiment-attempt'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        errors.append("$.experiment_id: invalid experiment id")
    attempt_id = root.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        errors.append("$.attempt_id: expected attempt-NNNN")
    attempt_index = root.get("attempt_index")
    _integer(attempt_index, "$.attempt_index", errors, minimum=1)
    if isinstance(attempt_index, int) and 1 <= attempt_index <= 9999:
        expected = f"attempt-{attempt_index:04d}"
        if attempt_id != expected:
            errors.append(f"$.attempt_id: expected {expected!r} for attempt_index")
    _sha256(root.get("request_sha256"), "$.request_sha256", errors)

    allowed_states = ("APPROVED", "EXECUTE", "EVALUATE", "AWAIT_REVIEW")
    state = root.get("state")
    if state not in allowed_states:
        errors.append("$.state: unknown lifecycle state")
    outcome = root.get("outcome")
    if outcome not in {"pending", "completed", "failed", "blocked"}:
        errors.append("$.outcome: expected pending, completed, failed or blocked")
    failure_class = root.get("failure_class")
    allowed_failures = {
        None,
        "request_config",
        "controller_execution",
        "transport_preflight",
        "worker_busy",
        "hardware_execution",
        "hardware_transient",
        "result_contract",
    }
    if failure_class not in allowed_failures:
        errors.append("$.failure_class: unknown failure class")
    if outcome in {"pending", "completed"} and failure_class is not None:
        errors.append("$.failure_class: must be null for pending/completed attempts")
    if outcome in {"failed", "blocked"} and failure_class is None:
        errors.append("$.failure_class: failed/blocked attempts require a failure class")
    if state == "AWAIT_REVIEW" and outcome == "pending":
        errors.append("$.outcome: AWAIT_REVIEW cannot remain pending")
    if state != "AWAIT_REVIEW" and outcome != "pending":
        errors.append("$.outcome: terminal outcome requires AWAIT_REVIEW state")

    events = _sequence(root.get("events"), "$.events", errors)
    if not events:
        errors.append("$.events: at least the APPROVED event is required")
    event_states: list[str] = []
    for index, value in enumerate(events):
        event = _mapping(value, f"$.events[{index}]", errors)
        event_state = event.get("state")
        if event_state not in allowed_states:
            errors.append(f"$.events[{index}].state: unknown lifecycle state")
        else:
            event_states.append(event_state)
        _nonempty_string(event.get("at_utc"), f"$.events[{index}].at_utc", errors)
    if event_states:
        if event_states[0] != "APPROVED":
            errors.append("$.events[0].state: first event must be APPROVED")
        if state in allowed_states and event_states[-1] != state:
            errors.append("$.events: final event state must equal $.state")
        positions = {name: index for index, name in enumerate(allowed_states)}
        for previous, current in zip(event_states, event_states[1:]):
            if current == "AWAIT_REVIEW":
                if previous not in {"APPROVED", "EXECUTE", "EVALUATE"}:
                    errors.append("$.events: invalid transition to AWAIT_REVIEW")
            elif positions.get(current, -1) != positions.get(previous, -1) + 1:
                errors.append(f"$.events: invalid transition {previous} -> {current}")

    artifacts = _mapping(root.get("artifacts"), "$.artifacts", errors)
    for name, ref in artifacts.items():
        _nonempty_string(name, "$.artifacts key", errors)
        _artifact_ref(ref, f"$.artifacts.{name}", errors)

    stage_results = _mapping(root.get("stage_results"), "$.stage_results", errors)
    allowed_stages = {"training", "compatibility", "hardware", "accuracy", "result"}
    for name, ref in stage_results.items():
        if name not in allowed_stages:
            errors.append(f"$.stage_results.{name}: unknown stage")
        else:
            _artifact_ref(ref, f"$.stage_results.{name}", errors)

    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_summary(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-summary":
        errors.append("$.schema: expected 'speech-asr/experiment-summary'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        errors.append("$.experiment_id: invalid experiment id")
    attempt_id = root.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        errors.append("$.attempt_id: invalid attempt id")
    _sha256(root.get("request_sha256"), "$.request_sha256", errors)
    if root.get("outcome") not in {"completed", "failed", "blocked"}:
        errors.append("$.outcome: expected completed, failed or blocked")
    if root.get("review_state") != "AWAIT_REVIEW":
        errors.append("$.review_state: expected AWAIT_REVIEW")
    for field in ("stage_results", "artifacts"):
        refs = _mapping(root.get(field), f"$.{field}", errors)
        for name, ref in refs.items():
            _artifact_ref(ref, f"$.{field}.{name}", errors)
    _raise_if_errors(errors)
    return dict(root)


def validate_experiment_history(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/experiment-history":
        errors.append("$.schema: expected 'speech-asr/experiment-history'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    experiments = _sequence(root.get("experiments"), "$.experiments", errors)
    ids: list[str] = []
    for index, value in enumerate(experiments):
        path = f"$.experiments[{index}]"
        item = _mapping(value, path, errors)
        experiment_id = item.get("experiment_id")
        if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
            errors.append(path + ".experiment_id: invalid experiment id")
        else:
            ids.append(experiment_id)
        parent = item.get("parent_experiment_id")
        if parent is not None and (
            not isinstance(parent, str) or _EXPERIMENT_ID_RE.fullmatch(parent) is None
        ):
            errors.append(path + ".parent_experiment_id: invalid parent experiment id")
        _sha256(item.get("request_sha256"), path + ".request_sha256", errors)
        _nonempty_string(item.get("path"), path + ".path", errors)
        latest_attempt = item.get("latest_attempt_id")
        if latest_attempt is not None and (
            not isinstance(latest_attempt, str) or _ATTEMPT_ID_RE.fullmatch(latest_attempt) is None
        ):
            errors.append(path + ".latest_attempt_id: invalid attempt id")
        if item.get("latest_outcome") not in {None, "pending", "completed", "failed", "blocked"}:
            errors.append(path + ".latest_outcome: invalid outcome")
    if ids != sorted(ids):
        errors.append("$.experiments: entries must be sorted by experiment_id")
    if len(ids) != len(set(ids)):
        errors.append("$.experiments: duplicate experiment_id")
    known = set(ids)
    for index, value in enumerate(experiments):
        if isinstance(value, Mapping):
            parent = value.get("parent_experiment_id")
            if parent is not None and parent not in known:
                errors.append(
                    f"$.experiments[{index}].parent_experiment_id: "
                    "parent is not present in history index"
                )
    _raise_if_errors(errors)
    return dict(root)


def validate_edge_worker_result(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/edge-worker-result":
        errors.append("$.schema: expected 'speech-asr/edge-worker-result'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")

    status = root.get("status")
    if status not in {"completed", "failed"}:
        errors.append("$.status: expected completed or failed")

    failure_class = root.get("failure_class")
    allowed_failures = {
        None,
        "request_config",
        "transport_preflight",
        "hardware_execution",
        "hardware_transient",
        "result_contract",
    }
    if failure_class not in allowed_failures:
        errors.append("$.failure_class: unknown edge-worker failure class")

    experiment_id = root.get("experiment_id")
    attempt_id = root.get("attempt_id")
    worker_commit = root.get("worker_commit")
    deployment_sha = root.get("deployment_sha256")

    if experiment_id is not None and (
        not isinstance(experiment_id, str)
        or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None
    ):
        errors.append("$.experiment_id: invalid experiment id")
    if attempt_id is not None and (
        not isinstance(attempt_id, str)
        or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None
    ):
        errors.append("$.attempt_id: invalid attempt id")
    if worker_commit is not None and (
        not isinstance(worker_commit, str)
        or len(worker_commit) != 40
        or _GIT_SHA_RE.fullmatch(worker_commit) is None
    ):
        errors.append("$.worker_commit: expected null or full Git commit id")
    if deployment_sha is not None:
        _sha256(deployment_sha, "$.deployment_sha256", errors)

    if status == "completed":
        if failure_class is not None:
            errors.append("$.failure_class: completed worker result must be null")
        for key, value in (
            ("experiment_id", experiment_id),
            ("attempt_id", attempt_id),
            ("worker_commit", worker_commit),
            ("deployment_sha256", deployment_sha),
        ):
            if value is None:
                errors.append(f"$.{key}: completed worker result requires value")
        _artifact_ref(root.get("hardware_result"), "$.hardware_result", errors)
        _artifact_ref(root.get("evaluator_log"), "$.evaluator_log", errors)
        _mapping(root.get("metrics"), "$.metrics", errors)

    if status == "failed":
        if failure_class is None:
            errors.append("$.failure_class: failed worker result requires class")
        diagnostics = _mapping(root.get("diagnostics"), "$.diagnostics", errors)
        _nonempty_string(
            diagnostics.get("summary"),
            "$.diagnostics.summary",
            errors,
        )

    _raise_if_errors(errors)
    return dict(root)


def validate_deployment_manifest(document: Any) -> Dict[str, Any]:
    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/deployment-manifest":
        errors.append("$.schema: expected 'speech-asr/deployment-manifest'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        errors.append("$.experiment_id: invalid experiment id")
    attempt_id = root.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        errors.append("$.attempt_id: invalid attempt id")
    for key in ("controller_commit", "worker_commit"):
        commit = root.get(key)
        if not isinstance(commit, str) or len(commit) != 40 or _GIT_SHA_RE.fullmatch(commit) is None:
            errors.append(f"$.{key}: expected full 40-character Git commit id")
    if (
        isinstance(root.get("controller_commit"), str)
        and isinstance(root.get("worker_commit"), str)
        and root.get("controller_commit") != root.get("worker_commit")
    ):
        errors.append("$.worker_commit: must match $.controller_commit")
    model = _mapping(root.get("model"), "$.model", errors)
    _nonempty_string(model.get("id"), "$.model.id", errors)
    _sha256(model.get("spec_sha256"), "$.model.spec_sha256", errors)
    benchmark = _mapping(root.get("benchmark"), "$.benchmark", errors)
    _nonempty_string(benchmark.get("id"), "$.benchmark.id", errors)
    _nonempty_string(benchmark.get("manifest_path"), "$.benchmark.manifest_path", errors)
    _sha256(benchmark.get("manifest_sha256"), "$.benchmark.manifest_sha256", errors)
    runtime = _mapping(root.get("runtime"), "$.runtime", errors)
    if runtime.get("target") != "arm64":
        errors.append("$.runtime.target: expected 'arm64'")
    if runtime.get("openvino_version") != "2020.3.2":
        errors.append("$.runtime.openvino_version: expected '2020.3.2'")
    evaluator = _mapping(root.get("evaluator"), "$.evaluator", errors)
    _nonempty_string(evaluator.get("id"), "$.evaluator.id", errors)
    _nonempty_string(evaluator.get("result_path"), "$.evaluator.result_path", errors)
    artifacts = _mapping(root.get("artifacts"), "$.artifacts", errors)
    if len(artifacts) < 2:
        errors.append("$.artifacts: at least two deployment artifacts are required")
    for name, ref in artifacts.items():
        _nonempty_string(name, "$.artifacts key", errors)
        _artifact_ref(ref, f"$.artifacts.{name}", errors)
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


def validate_streaming_contract(document: Any) -> Dict[str, Any]:
    """Validate the recorded-audio streaming-v1 runtime contract."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/streaming":
        errors.append("$.schema: expected 'speech-asr/streaming'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    if root.get("audio_contract") != "audio-v1":
        errors.append("$.audio_contract: expected 'audio-v1'")

    timing = _mapping(root.get("timing"), "$.timing", errors)
    if timing.get("canonical_unit") != "sample_index":
        errors.append("$.timing.canonical_unit: expected 'sample_index'")
    if timing.get("origin") != "normalized_audio_start":
        errors.append("$.timing.origin: expected 'normalized_audio_start'")
    if timing.get("sample_rate_hz") != _CANONICAL_SAMPLE_RATE:
        errors.append("$.timing.sample_rate_hz: expected 16000")

    chunking = _mapping(root.get("chunking"), "$.chunking", errors)
    if chunking.get("authoritative_unit") != "ms":
        errors.append("$.chunking.authoritative_unit: expected 'ms'")
    if chunking.get("overlap_ownership") != "midpoint_partition":
        errors.append("$.chunking.overlap_ownership: expected 'midpoint_partition'")
    if chunking.get("context_policy") != "left_right_expand_inference_only":
        errors.append(
            "$.chunking.context_policy: expected 'left_right_expand_inference_only'"
        )
    defaults = _mapping(
        chunking.get("default_profile"),
        "$.chunking.default_profile",
        errors,
    )
    for key in ("chunk_ms", "overlap_ms", "left_context_ms", "right_context_ms"):
        _integer(defaults.get(key), f"$.chunking.default_profile.{key}", errors)
    if (
        isinstance(defaults.get("chunk_ms"), int)
        and isinstance(defaults.get("overlap_ms"), int)
        and defaults.get("overlap_ms") >= defaults.get("chunk_ms")
    ):
        errors.append("$.chunking.default_profile.overlap_ms: must be smaller than chunk_ms")

    events = _mapping(root.get("events"), "$.events", errors)
    _require_values(events.get("kinds"), ("partial", "final"), "$.events.kinds", errors)
    if events.get("hypothesis_semantics") != "cumulative":
        errors.append("$.events.hypothesis_semantics: expected 'cumulative'")
    if events.get("text_normalization") != "text-v1":
        errors.append("$.events.text_normalization: expected 'text-v1'")
    stabilization = _mapping(events.get("stabilization"), "$.events.stabilization", errors)
    if stabilization.get("kind") != "consecutive_cumulative_prefix":
        errors.append(
            "$.events.stabilization.kind: expected 'consecutive_cumulative_prefix'"
        )
    _integer(
        stabilization.get("default_repeats"),
        "$.events.stabilization.default_repeats",
        errors,
        minimum=1,
    )
    if stabilization.get("stabilized_tokens_must_not_be_revised") is not True:
        errors.append(
            "$.events.stabilization.stabilized_tokens_must_not_be_revised: expected true"
        )

    _raise_if_errors(errors)
    return dict(root)


def validate_streaming_replay_result(document: Any) -> Dict[str, Any]:
    """Validate a deterministic recorded-audio streaming replay result."""

    errors: list[str] = []
    root = _mapping(document, "$", errors)
    if root.get("schema") != "speech-asr/streaming-replay-result":
        errors.append("$.schema: expected 'speech-asr/streaming-replay-result'")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    status = root.get("status")
    if status not in {"completed", "failed"}:
        errors.append("$.status: expected completed or failed")

    source = _mapping(root.get("source"), "$.source", errors)
    _nonempty_string(source.get("path"), "$.source.path", errors)
    _sha256(source.get("source_sha256"), "$.source.source_sha256", errors)
    _sha256(
        source.get("canonical_audio_sha256"),
        "$.source.canonical_audio_sha256",
        errors,
    )
    _integer(source.get("sample_count"), "$.source.sample_count", errors, minimum=1)

    config = _mapping(root.get("config"), "$.config", errors)
    if config.get("sample_rate_hz") != _CANONICAL_SAMPLE_RATE:
        errors.append("$.config.sample_rate_hz: expected 16000")
    for key in ("chunk_ms", "overlap_ms", "left_context_ms", "right_context_ms"):
        _integer(config.get(key), f"$.config.{key}", errors)
    if (
        isinstance(config.get("chunk_ms"), int)
        and isinstance(config.get("overlap_ms"), int)
        and config.get("overlap_ms") >= config.get("chunk_ms")
    ):
        errors.append("$.config.overlap_ms: must be smaller than chunk_ms")
    _integer(
        config.get("stabilization_repeats"),
        "$.config.stabilization_repeats",
        errors,
        minimum=1,
    )

    decoder = _mapping(root.get("decoder"), "$.decoder", errors)
    if decoder.get("kind") != "scripted-cumulative-v1":
        errors.append("$.decoder.kind: expected 'scripted-cumulative-v1'")
    _sha256(decoder.get("script_sha256"), "$.decoder.script_sha256", errors)

    events = _sequence(root.get("events"), "$.events", errors)
    final_count = 0
    previous_observed = -1
    for index, value in enumerate(events):
        path = f"$.events[{index}]"
        event = _mapping(value, path, errors)
        if event.get("kind") not in {"partial", "final"}:
            errors.append(path + ".kind: expected partial or final")
        final_count += int(event.get("kind") == "final")
        if not isinstance(event.get("text"), str):
            errors.append(path + ".text: expected string")
        _integer(event.get("observed_sample"), path + ".observed_sample", errors)
        _integer(event.get("source_end_sample"), path + ".source_end_sample", errors)
        _number(event.get("latency_ms"), path + ".latency_ms", errors)
        _integer(event.get("stable_token_count"), path + ".stable_token_count", errors)
        observed = event.get("observed_sample")
        source_end = event.get("source_end_sample")
        if isinstance(observed, int) and isinstance(source_end, int):
            if source_end > observed:
                errors.append(path + ".source_end_sample: cannot exceed observed_sample")
            if observed < previous_observed:
                errors.append(path + ".observed_sample: events must be ordered")
            previous_observed = observed

    metrics = _mapping(root.get("metrics"), "$.metrics", errors)
    _integer(metrics.get("audio_samples"), "$.metrics.audio_samples", errors, minimum=1)
    _integer(metrics.get("chunk_count"), "$.metrics.chunk_count", errors, minimum=1)
    _number(
        metrics.get("final_event_latency_ms"),
        "$.metrics.final_event_latency_ms",
        errors,
    )
    _integer(
        metrics.get("stabilized_token_count"),
        "$.metrics.stabilized_token_count",
        errors,
    )

    comparison = root.get("offline_comparison")
    if comparison is not None:
        comparison = _mapping(comparison, "$.offline_comparison", errors)
        if not isinstance(comparison.get("exact_match"), bool):
            errors.append("$.offline_comparison.exact_match: expected boolean")
        _number(comparison.get("wer"), "$.offline_comparison.wer", errors)
        _number(comparison.get("cer"), "$.offline_comparison.cer", errors)

    provenance = _mapping(root.get("provenance"), "$.provenance", errors)
    _git_sha(provenance.get("repo_commit"), "$.provenance.repo_commit", errors)
    _sha256(
        provenance.get("streaming_contract_sha256"),
        "$.provenance.streaming_contract_sha256",
        errors,
    )
    _sha256(
        provenance.get("implementation_sha256"),
        "$.provenance.implementation_sha256",
        errors,
    )
    _nonempty_string(provenance.get("python_version"), "$.provenance.python_version", errors)

    if status == "completed":
        if final_count != 1:
            errors.append("$.events: completed replay requires exactly one final event")
        if source.get("sample_count") != metrics.get("audio_samples"):
            errors.append("$.metrics.audio_samples: must match $.source.sample_count")
    else:
        diagnostics = _mapping(root.get("diagnostics"), "$.diagnostics", errors)
        _nonempty_string(diagnostics.get("summary"), "$.diagnostics.summary", errors)

    _raise_if_errors(errors)
    return dict(root)
