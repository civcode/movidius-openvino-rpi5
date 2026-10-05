"""Phase 9 experiment identity, lifecycle and artifact helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .contracts import (
    canonical_json_sha256,
    validate_acceptance_policy,
    validate_deployment_manifest,
    validate_experiment_attempt,
    validate_experiment_history,
    validate_experiment_model_spec,
    validate_experiment_proposal,
    validate_experiment_request,
    validate_experiment_summary,
    validate_train_config,
)

DOCUMENT_FILENAMES = {
    "proposal": "request/proposal.json",
    "model_spec": "request/model-spec.json",
    "train_config": "request/train-config.json",
    "acceptance": "request/acceptance.json",
}

DOCUMENT_VALIDATORS = {
    "proposal": validate_experiment_proposal,
    "model_spec": validate_experiment_model_spec,
    "train_config": validate_train_config,
    "acceptance": validate_acceptance_policy,
}

STAGES = ("training", "compatibility", "hardware", "accuracy", "result")
TERMINAL_OUTCOMES = ("completed", "failed", "blocked")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request_document(name: str, document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        validator = DOCUMENT_VALIDATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown experiment request document: {name}") from exc
    return validator(document)


def document_ref(name: str, document: Mapping[str, Any]) -> dict[str, str]:
    validate_request_document(name, document)
    return {
        "path": DOCUMENT_FILENAMES[name],
        "sha256": canonical_json_sha256(document),
    }


def request_identity_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_experiment_id": request.get("parent_experiment_id"),
        "source": copy.deepcopy(request.get("source")),
        "benchmark": copy.deepcopy(request.get("benchmark")),
        "documents": copy.deepcopy(request.get("documents")),
    }


def experiment_id_from_identity(identity_sha256: str) -> str:
    if len(identity_sha256) != 64 or any(c not in "0123456789abcdef" for c in identity_sha256):
        raise ValueError("identity_sha256 must be lowercase 64-character hex")
    return "exp-" + identity_sha256[:16]


def make_experiment_request(
    *,
    proposal: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    train_config: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    repo_commit: str,
    benchmark_id: str,
    manifest_path: str,
    manifest_sha256: str,
    parent_experiment_id: str | None = None,
) -> dict[str, Any]:
    documents = {
        "proposal": document_ref("proposal", proposal),
        "model_spec": document_ref("model_spec", model_spec),
        "train_config": document_ref("train_config", train_config),
        "acceptance": document_ref("acceptance", acceptance),
    }
    request: dict[str, Any] = {
        "schema": "speech-asr/experiment-request",
        "version": 1,
        "experiment_id": "exp-" + "0" * 16,
        "parent_experiment_id": parent_experiment_id,
        "identity_sha256": "0" * 64,
        "state": "APPROVED",
        "source": {"repo_commit": repo_commit},
        "benchmark": {
            "id": benchmark_id,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
        },
        "documents": documents,
    }
    identity_sha = canonical_json_sha256(request_identity_payload(request))
    request["identity_sha256"] = identity_sha
    request["experiment_id"] = experiment_id_from_identity(identity_sha)
    validate_experiment_request(request)
    validate_request_identity(request)
    return request


def validate_request_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_experiment_request(request)
    expected_sha = canonical_json_sha256(request_identity_payload(validated))
    if validated["identity_sha256"] != expected_sha:
        raise ValueError(
            "experiment request identity_sha256 mismatch: "
            f"declared={validated['identity_sha256']} expected={expected_sha}"
        )
    expected_id = experiment_id_from_identity(expected_sha)
    if validated["experiment_id"] != expected_id:
        raise ValueError(
            "experiment request id mismatch: "
            f"declared={validated['experiment_id']} expected={expected_id}"
        )
    return validated


def request_sha256(request: Mapping[str, Any]) -> str:
    validate_request_identity(request)
    return canonical_json_sha256(request)


def attempt_id(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= 9999:
        raise ValueError("attempt index must be an integer from 1 through 9999")
    return f"attempt-{index:04d}"


def make_attempt(
    request: Mapping[str, Any],
    *,
    index: int,
    at_utc: str | None = None,
) -> dict[str, Any]:
    validate_request_identity(request)
    timestamp = at_utc or utc_now()
    attempt = {
        "schema": "speech-asr/experiment-attempt",
        "version": 1,
        "experiment_id": request["experiment_id"],
        "attempt_id": attempt_id(index),
        "attempt_index": index,
        "request_sha256": request_sha256(request),
        "state": "APPROVED",
        "outcome": "pending",
        "failure_class": None,
        "events": [{"state": "APPROVED", "at_utc": timestamp}],
        "stage_results": {},
    }
    validate_experiment_attempt(attempt)
    return attempt


def transition_attempt(
    attempt: Mapping[str, Any],
    state: str,
    *,
    outcome: str | None = None,
    failure_class: str | None = None,
    at_utc: str | None = None,
) -> dict[str, Any]:
    current = validate_experiment_attempt(attempt)
    previous = current["state"]
    if previous == "AWAIT_REVIEW":
        raise ValueError("attempt is already terminal at AWAIT_REVIEW")

    if state == "EXECUTE":
        if previous != "APPROVED":
            raise ValueError(f"invalid transition {previous} -> EXECUTE")
        if outcome is not None or failure_class is not None:
            raise ValueError("non-terminal transition cannot set outcome/failure_class")
        terminal_outcome = "pending"
    elif state == "EVALUATE":
        if previous != "EXECUTE":
            raise ValueError(f"invalid transition {previous} -> EVALUATE")
        if outcome is not None or failure_class is not None:
            raise ValueError("non-terminal transition cannot set outcome/failure_class")
        terminal_outcome = "pending"
    elif state == "AWAIT_REVIEW":
        if previous not in {"APPROVED", "EXECUTE", "EVALUATE"}:
            raise ValueError(f"invalid transition {previous} -> AWAIT_REVIEW")
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError("AWAIT_REVIEW requires completed, failed or blocked outcome")
        if outcome == "completed":
            if previous != "EVALUATE":
                raise ValueError("completed attempt must pass through EVALUATE")
            if failure_class is not None:
                raise ValueError("completed attempt cannot have failure_class")
        elif failure_class is None:
            raise ValueError("failed/blocked attempt requires failure_class")
        terminal_outcome = outcome
    else:
        raise ValueError(f"unsupported lifecycle state: {state}")

    updated = copy.deepcopy(current)
    updated["state"] = state
    updated["outcome"] = terminal_outcome
    updated["failure_class"] = failure_class
    updated["events"].append({"state": state, "at_utc": at_utc or utc_now()})
    validate_experiment_attempt(updated)
    return updated


def set_stage_result(
    attempt: Mapping[str, Any],
    stage: str,
    *,
    path: str,
    sha256: str,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown experiment stage: {stage}")
    updated = copy.deepcopy(validate_experiment_attempt(attempt))
    if stage in updated["stage_results"]:
        existing = updated["stage_results"][stage]
        if existing != {"path": path, "sha256": sha256}:
            raise ValueError(f"stage result is immutable once recorded: {stage}")
        return updated
    updated["stage_results"][stage] = {"path": path, "sha256": sha256}
    validate_experiment_attempt(updated)
    return updated


def make_summary(
    attempt: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, str]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_experiment_attempt(attempt)
    if validated["state"] != "AWAIT_REVIEW":
        raise ValueError("summary requires terminal AWAIT_REVIEW attempt")
    summary: dict[str, Any] = {
        "schema": "speech-asr/experiment-summary",
        "version": 1,
        "experiment_id": validated["experiment_id"],
        "attempt_id": validated["attempt_id"],
        "request_sha256": validated["request_sha256"],
        "outcome": validated["outcome"],
        "review_state": "AWAIT_REVIEW",
        "stage_results": {
            name: copy.deepcopy(ref)
            for name, ref in validated["stage_results"].items()
            if name != "result"
        },
        "artifacts": copy.deepcopy(dict(artifacts or {})),
    }
    if metrics is not None:
        summary["metrics"] = copy.deepcopy(dict(metrics))
    if diagnostics is not None:
        summary["diagnostics"] = copy.deepcopy(dict(diagnostics))
    validate_experiment_summary(summary)
    return summary


def make_deployment_manifest(
    *,
    request: Mapping[str, Any],
    attempt: Mapping[str, Any],
    worker_commit: str,
    model_id: str,
    model_spec_sha256: str,
    evaluator_id: str,
    result_path: str,
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    validated_request = validate_request_identity(request)
    validated_attempt = validate_experiment_attempt(attempt)
    if validated_attempt["experiment_id"] != validated_request["experiment_id"]:
        raise ValueError("attempt does not belong to deployment request")
    if validated_attempt["request_sha256"] != request_sha256(validated_request):
        raise ValueError("attempt request hash does not match deployment request")
    manifest = {
        "schema": "speech-asr/deployment-manifest",
        "version": 1,
        "experiment_id": validated_request["experiment_id"],
        "attempt_id": validated_attempt["attempt_id"],
        "controller_commit": validated_request["source"]["repo_commit"],
        "worker_commit": worker_commit,
        "model": {"id": model_id, "spec_sha256": model_spec_sha256},
        "benchmark": copy.deepcopy(validated_request["benchmark"]),
        "runtime": {"target": "arm64", "openvino_version": "2020.3.2"},
        "evaluator": {"id": evaluator_id, "result_path": result_path},
        "artifacts": copy.deepcopy(dict(artifacts)),
    }
    validate_deployment_manifest(manifest)
    return manifest


def history_document(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [copy.deepcopy(dict(value)) for value in entries]
    values.sort(key=lambda value: value["experiment_id"])
    history = {
        "schema": "speech-asr/experiment-history",
        "version": 1,
        "experiments": values,
    }
    validate_experiment_history(history)
    return history
