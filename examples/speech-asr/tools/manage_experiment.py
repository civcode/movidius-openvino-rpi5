#!/usr/bin/env python3
"""Create and manage immutable Phase 9 speech-ASR experiment records."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.contracts import validate_experiment_attempt  # noqa: E402
from speech_asr.experiment import (  # noqa: E402
    DOCUMENT_FILENAMES,
    DOCUMENT_VALIDATORS,
    history_document,
    load_json,
    make_attempt,
    make_deployment_manifest,
    make_experiment_request,
    make_summary,
    request_sha256,
    set_artifact,
    set_stage_result,
    sha256_path,
    transition_attempt,
    validate_request_identity,
    write_json,
)

DEFAULT_ROOT = ROOT / "work" / "speech-asr" / "experiments"


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    value = proc.stdout.strip()
    if len(value) != 40:
        raise ValueError(f"expected full Git commit id, got {value!r}")
    return value


def repo_relative(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be below repository root: {path}") from exc


def experiment_relative(path: pathlib.Path, experiment_dir: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(experiment_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be below experiment directory: {path}") from exc


def attempt_path(experiment_dir: pathlib.Path, attempt_id: str) -> pathlib.Path:
    return experiment_dir / "attempts" / attempt_id / "attempt.json"


def load_experiment(experiment_dir: pathlib.Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    request_path = experiment_dir / "request" / "experiment.json"
    if not request_path.is_file():
        raise ValueError(f"experiment request missing: {request_path}")
    request = validate_request_identity(load_json(request_path))
    if experiment_dir.name != request["experiment_id"]:
        raise ValueError(
            f"experiment directory {experiment_dir.name!r} does not match "
            f"{request['experiment_id']!r}"
        )

    documents: dict[str, dict[str, Any]] = {}
    for name, relative in DOCUMENT_FILENAMES.items():
        path = experiment_dir / relative
        if not path.is_file():
            raise ValueError(f"request document missing: {path}")
        document = load_json(path)
        DOCUMENT_VALIDATORS[name](document)
        from speech_asr.contracts import canonical_json_sha256

        digest = canonical_json_sha256(document)
        expected = request["documents"][name]["sha256"]
        if digest != expected:
            raise ValueError(
                f"{name} hash mismatch: declared={expected} actual={digest}"
            )
        documents[name] = document
    return request, documents


def list_attempts(experiment_dir: pathlib.Path) -> list[tuple[int, pathlib.Path, dict[str, Any]]]:
    values = []
    attempts_dir = experiment_dir / "attempts"
    if not attempts_dir.is_dir():
        return values
    for path in sorted(attempts_dir.glob("attempt-*/attempt.json")):
        value = validate_experiment_attempt(load_json(path))
        values.append((int(value["attempt_index"]), path, value))
    values.sort(key=lambda item: item[0])
    return values


def rebuild_history(root: pathlib.Path) -> dict[str, Any]:
    entries = []
    if root.is_dir():
        for experiment_dir in sorted(root.glob("exp-*")):
            if not experiment_dir.is_dir():
                continue
            request, _documents = load_experiment(experiment_dir)
            attempts = list_attempts(experiment_dir)
            latest_id = None
            latest_outcome = None
            if attempts:
                _index, _path, latest = attempts[-1]
                latest_id = latest["attempt_id"]
                latest_outcome = latest["outcome"]
            entries.append(
                {
                    "experiment_id": request["experiment_id"],
                    "parent_experiment_id": request["parent_experiment_id"],
                    "request_sha256": request_sha256(request),
                    "path": experiment_dir.name,
                    "latest_attempt_id": latest_id,
                    "latest_outcome": latest_outcome,
                }
            )
    history = history_document(entries)
    write_json(root / "history.json", history)
    return history


def command_init(args: argparse.Namespace) -> None:
    proposal = load_json(args.proposal)
    model_spec = load_json(args.model_spec)
    train_config = load_json(args.train_config)
    acceptance = load_json(args.acceptance)
    for name, document in (
        ("proposal", proposal),
        ("model_spec", model_spec),
        ("train_config", train_config),
        ("acceptance", acceptance),
    ):
        DOCUMENT_VALIDATORS[name](document)

    if not args.manifest.is_file():
        raise ValueError(f"benchmark manifest missing: {args.manifest}")
    manifest_sha = sha256_path(args.manifest)

    for field in ("training_manifest", "validation_manifest"):
        declared = train_config[field]
        declared_path = pathlib.Path(declared["path"])
        if not declared_path.is_absolute():
            declared_path = ROOT / declared_path
        if not declared_path.is_file():
            raise ValueError(f"{field} file missing: {declared_path}")
        actual = sha256_path(declared_path)
        if actual != declared["sha256"]:
            raise ValueError(
                f"{field} hash mismatch: declared={declared['sha256']} actual={actual}"
            )

    if args.parent is not None:
        parent_dir = args.root / args.parent
        parent_request, _parent_documents = load_experiment(parent_dir)
        if parent_request["experiment_id"] != args.parent:
            raise ValueError("parent experiment identity mismatch")

    commit = args.repo_commit or git_head()
    request = make_experiment_request(
        proposal=proposal,
        model_spec=model_spec,
        train_config=train_config,
        acceptance=acceptance,
        repo_commit=commit,
        benchmark_id=args.benchmark_id,
        manifest_path=repo_relative(args.manifest),
        manifest_sha256=manifest_sha,
        parent_experiment_id=args.parent,
    )

    experiment_dir = args.root / request["experiment_id"]
    request_path = experiment_dir / "request" / "experiment.json"
    if request_path.exists():
        existing = validate_request_identity(load_json(request_path))
        if existing != request:
            raise ValueError(
                f"experiment id collision with different request: {experiment_dir}"
            )
        load_experiment(experiment_dir)
        rebuild_history(args.root)
        print(json.dumps({"status": "existing", "experiment_id": request["experiment_id"], "path": str(experiment_dir)}, sort_keys=True))
        return

    experiment_dir.mkdir(parents=True, exist_ok=False)
    write_json(experiment_dir / DOCUMENT_FILENAMES["proposal"], proposal)
    write_json(experiment_dir / DOCUMENT_FILENAMES["model_spec"], model_spec)
    write_json(experiment_dir / DOCUMENT_FILENAMES["train_config"], train_config)
    write_json(experiment_dir / DOCUMENT_FILENAMES["acceptance"], acceptance)
    write_json(request_path, request)
    (experiment_dir / "attempts").mkdir()
    rebuild_history(args.root)
    print(json.dumps({"status": "created", "experiment_id": request["experiment_id"], "path": str(experiment_dir)}, sort_keys=True))


def command_start_attempt(args: argparse.Namespace) -> None:
    request, documents = load_experiment(args.experiment)
    attempts = list_attempts(args.experiment)
    retry = documents["acceptance"]["retry_policy"]
    max_attempts = int(retry["max_attempts"])

    if attempts:
        _index, _path, previous = attempts[-1]
        if previous["state"] != "AWAIT_REVIEW":
            raise ValueError(
                f"latest attempt is not terminal: {previous['attempt_id']} "
                f"state={previous['state']}"
            )
        if previous["outcome"] == "completed":
            raise ValueError("completed experiment is idempotent; create a new experiment request")
        retryable = set(retry["retryable_failure_classes"])
        if previous["failure_class"] not in retryable:
            raise ValueError(
                f"failure class {previous['failure_class']!r} is not retryable by policy"
            )

    index = len(attempts) + 1
    if index > max_attempts:
        raise ValueError(
            f"retry policy allows {max_attempts} attempts; next would be {index}"
        )
    attempt = make_attempt(request, index=index)
    directory = args.experiment / "attempts" / attempt["attempt_id"]
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("results", "logs"):
        (directory / name).mkdir()
    write_json(directory / "attempt.json", attempt)
    rebuild_history(args.experiment.parent)
    print(json.dumps({"status": "created", "attempt_id": attempt["attempt_id"], "state": attempt["state"]}, sort_keys=True))


def load_attempt(experiment_dir: pathlib.Path, attempt_id: str) -> tuple[pathlib.Path, dict[str, Any]]:
    path = attempt_path(experiment_dir, attempt_id)
    if not path.is_file():
        raise ValueError(f"attempt does not exist: {path}")
    request, _documents = load_experiment(experiment_dir)
    attempt = validate_experiment_attempt(load_json(path))
    if attempt["experiment_id"] != request["experiment_id"]:
        raise ValueError("attempt experiment_id does not match request")
    if attempt["request_sha256"] != request_sha256(request):
        raise ValueError("attempt request_sha256 does not match immutable request")
    return path, attempt


def command_transition(args: argparse.Namespace) -> None:
    path, attempt = load_attempt(args.experiment, args.attempt)
    updated = transition_attempt(
        attempt,
        args.state,
        outcome=args.outcome,
        failure_class=args.failure_class,
    )

    if args.state == "AWAIT_REVIEW":
        summary = make_summary(
            updated,
            diagnostics=(
                {"summary": args.diagnostic}
                if args.diagnostic
                else None
            ),
        )
        result_path = path.parent / "results" / "result.json"
        if result_path.exists():
            raise ValueError(f"terminal result already exists: {result_path}")
        write_json(result_path, summary)
        updated = set_stage_result(
            updated,
            "result",
            path=experiment_relative(result_path, args.experiment),
            sha256=sha256_path(result_path),
        )

    write_json(path, updated)
    rebuild_history(args.experiment.parent)
    print(json.dumps({"attempt_id": updated["attempt_id"], "state": updated["state"], "outcome": updated["outcome"]}, sort_keys=True))


def command_record_stage(args: argparse.Namespace) -> None:
    path, attempt = load_attempt(args.experiment, args.attempt)
    required_state = {
        "training": "EXECUTE",
        "compatibility": "EXECUTE",
        "hardware": "EVALUATE",
        "accuracy": "EVALUATE",
    }
    if args.stage == "result":
        raise ValueError("result stage is generated by terminal transition")
    wanted = required_state[args.stage]
    if attempt["state"] != wanted:
        raise ValueError(
            f"{args.stage} result requires {wanted} state, got {attempt['state']}"
        )
    if not args.source.is_file():
        raise ValueError(f"stage result missing: {args.source}")

    destination = path.parent / "results" / f"{args.stage}.json"
    if destination.exists():
        if sha256_path(destination) != sha256_path(args.source):
            raise ValueError(f"stage result is immutable once copied: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source, destination)

    updated = set_stage_result(
        attempt,
        args.stage,
        path=experiment_relative(destination, args.experiment),
        sha256=sha256_path(destination),
    )
    write_json(path, updated)
    print(json.dumps({"attempt_id": updated["attempt_id"], "stage": args.stage, "sha256": updated["stage_results"][args.stage]["sha256"]}, sort_keys=True))


def command_record_artifact(args: argparse.Namespace) -> None:
    path, attempt = load_attempt(args.experiment, args.attempt)
    if attempt["state"] not in {"EXECUTE", "EVALUATE"}:
        raise ValueError("artifacts may only be recorded during EXECUTE/EVALUATE")
    if not args.source.is_file():
        raise ValueError(f"artifact missing: {args.source}")
    updated = set_artifact(
        attempt,
        args.name,
        path=repo_relative(args.source),
        sha256=sha256_path(args.source),
    )
    write_json(path, updated)
    print(json.dumps({"attempt_id": updated["attempt_id"], "artifact": args.name, "sha256": updated["artifacts"][args.name]["sha256"]}, sort_keys=True))


def command_deployment(args: argparse.Namespace) -> None:
    request, documents = load_experiment(args.experiment)
    _path, attempt = load_attempt(args.experiment, args.attempt)
    if attempt["state"] != "EVALUATE":
        raise ValueError("deployment manifest requires EVALUATE state")
    artifact_names = args.artifact or ["openvino_xml", "openvino_bin"]
    missing = [name for name in artifact_names if name not in attempt["artifacts"]]
    if missing:
        raise ValueError(f"deployment artifacts not recorded: {missing}")
    artifacts = {name: attempt["artifacts"][name] for name in artifact_names}
    manifest = make_deployment_manifest(
        request=request,
        attempt=attempt,
        worker_commit=args.worker_commit,
        model_id=documents["model_spec"]["model_id"],
        model_spec_sha256=request["documents"]["model_spec"]["sha256"],
        evaluator_id=args.evaluator,
        result_path=args.result_path,
        artifacts=artifacts,
    )
    destination = (
        args.experiment
        / "attempts"
        / args.attempt
        / "deployment-manifest.json"
    )
    if destination.exists() and load_json(destination) != manifest:
        raise ValueError(f"deployment manifest is immutable once written: {destination}")
    write_json(destination, manifest)
    print(json.dumps({"deployment_manifest": str(destination), "sha256": sha256_path(destination)}, sort_keys=True))


def verify_ref(
    *,
    name: str,
    ref: dict[str, str],
    base: pathlib.Path,
) -> None:
    path = pathlib.Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise ValueError(f"{name} referenced file missing: {path}")
    actual = sha256_path(path)
    if actual != ref["sha256"]:
        raise ValueError(
            f"{name} hash mismatch: declared={ref['sha256']} actual={actual}"
        )


def command_validate(args: argparse.Namespace) -> None:
    request, _documents = load_experiment(args.experiment)
    attempts = list_attempts(args.experiment)
    for _index, path, attempt in attempts:
        if attempt["experiment_id"] != request["experiment_id"]:
            raise ValueError(f"{path}: experiment_id mismatch")
        if attempt["request_sha256"] != request_sha256(request):
            raise ValueError(f"{path}: request_sha256 mismatch")
        for name, ref in attempt["stage_results"].items():
            verify_ref(name=f"{attempt['attempt_id']} stage {name}", ref=ref, base=args.experiment)
        for name, ref in attempt["artifacts"].items():
            verify_ref(name=f"{attempt['attempt_id']} artifact {name}", ref=ref, base=ROOT)
    print(json.dumps({"status": "valid", "experiment_id": request["experiment_id"], "attempts": len(attempts)}, sort_keys=True))


def command_index(args: argparse.Namespace) -> None:
    history = rebuild_history(args.root)
    print(json.dumps({"status": "valid", "experiments": len(history["experiments"]), "path": str(args.root / "history.json")}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("init")
    create.add_argument("--proposal", type=pathlib.Path, required=True)
    create.add_argument("--model-spec", type=pathlib.Path, required=True)
    create.add_argument("--train-config", type=pathlib.Path, required=True)
    create.add_argument("--acceptance", type=pathlib.Path, required=True)
    create.add_argument("--benchmark-id", required=True)
    create.add_argument("--manifest", type=pathlib.Path, required=True)
    create.add_argument("--parent")
    create.add_argument("--repo-commit")
    create.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    create.set_defaults(func=command_init)

    start = sub.add_parser("start-attempt")
    start.add_argument("--experiment", type=pathlib.Path, required=True)
    start.set_defaults(func=command_start_attempt)

    transition = sub.add_parser("transition")
    transition.add_argument("--experiment", type=pathlib.Path, required=True)
    transition.add_argument("--attempt", required=True)
    transition.add_argument("--state", required=True, choices=("EXECUTE", "EVALUATE", "AWAIT_REVIEW"))
    transition.add_argument("--outcome", choices=("completed", "failed", "blocked"))
    transition.add_argument(
        "--failure-class",
        choices=(
            "request_config",
            "controller_execution",
            "transport_preflight",
            "worker_busy",
            "hardware_execution",
            "hardware_transient",
            "result_contract",
        ),
    )
    transition.add_argument("--diagnostic")
    transition.set_defaults(func=command_transition)

    stage = sub.add_parser("record-stage")
    stage.add_argument("--experiment", type=pathlib.Path, required=True)
    stage.add_argument("--attempt", required=True)
    stage.add_argument("--stage", required=True, choices=("training", "compatibility", "hardware", "accuracy", "result"))
    stage.add_argument("--source", type=pathlib.Path, required=True)
    stage.set_defaults(func=command_record_stage)

    artifact = sub.add_parser("record-artifact")
    artifact.add_argument("--experiment", type=pathlib.Path, required=True)
    artifact.add_argument("--attempt", required=True)
    artifact.add_argument("--name", required=True)
    artifact.add_argument("--source", type=pathlib.Path, required=True)
    artifact.set_defaults(func=command_record_artifact)

    deployment = sub.add_parser("deployment-manifest")
    deployment.add_argument("--experiment", type=pathlib.Path, required=True)
    deployment.add_argument("--attempt", required=True)
    deployment.add_argument("--worker-commit", required=True)
    deployment.add_argument("--evaluator", required=True)
    deployment.add_argument("--result-path", required=True)
    deployment.add_argument(
        "--artifact",
        action="append",
        help="recorded artifact name to include; defaults to openvino_xml/openvino_bin",
    )
    deployment.set_defaults(func=command_deployment)

    validate = sub.add_parser("validate")
    validate.add_argument("--experiment", type=pathlib.Path, required=True)
    validate.set_defaults(func=command_validate)

    index = sub.add_parser("index")
    index.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    index.set_defaults(func=command_index)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
