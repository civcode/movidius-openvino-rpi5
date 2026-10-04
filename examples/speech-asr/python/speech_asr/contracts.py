"""Semantic validation for versioned speech-ASR contracts.

The JSON schema files in ``contracts/`` are the portable contract description.
This module intentionally uses only the Python standard library and adds the
cross-field invariants that matter to the benchmark (for example canonical
16 kHz audio and sample-bound timing).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Mapping, MutableSequence

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


def _nonempty_string(value: Any, path: str, errors: MutableSequence[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected non-empty string")


def _integer(value: Any, path: str, errors: MutableSequence[str], minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: expected integer")
    elif value < minimum:
        errors.append(f"{path}: must be >= {minimum}")


def _number(value: Any, path: str, errors: MutableSequence[str], minimum: float = 0.0) -> None:
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


def _raise_if_errors(errors: MutableSequence[str]) -> None:
    if errors:
        raise ContractValidationError(errors)


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
    if isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool):
        if end <= start:
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
        if isinstance(word_start, int) and not isinstance(word_start, bool) and isinstance(word_end, int) and not isinstance(word_end, bool):
            if word_end <= word_start:
                errors.append(path + ".end_sample: must be greater than start_sample")
            if isinstance(start, int) and word_start < start:
                errors.append(path + ".start_sample: precedes sample audio range")
            if isinstance(end, int) and word_end > end:
                errors.append(path + ".end_sample: exceeds sample audio range")
            if previous_start is not None and word_start < previous_start:
                errors.append(path + ".start_sample: word timings must be ordered by start_sample")
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
    artifact_hashes = _mapping(model.get("artifact_sha256"), "$.model.artifact_sha256", errors)
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
        for key in ("wer", "cer", "realtime_factor", "inference_latency_p50_ms", "inference_latency_p95_ms"):
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
    _sha256(provenance.get("dataset_manifest_sha256"), "$.provenance.dataset_manifest_sha256", errors)
    _sha256(provenance.get("model_spec_sha256"), "$.provenance.model_spec_sha256", errors)

    _raise_if_errors(errors)
    return dict(root)
