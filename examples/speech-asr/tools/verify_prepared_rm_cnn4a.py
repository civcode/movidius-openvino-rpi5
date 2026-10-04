#!/usr/bin/env python3
"""Verify a prepared rm_cnn4a source/IR bundle without network or hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.openvino_ir import inspect_ir  # noqa: E402


class VerificationError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object: {path}")
    return value


def load_lock(path: pathlib.Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise VerificationError(f"missing source lock: {path}") from exc
    result: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise VerificationError(f"{path}:{number}: invalid SHA-256 lock line")
        name = parts[1].lstrip("*").strip()
        result[name] = parts[0].lower()
    return result


def verify_bundle(model_root: pathlib.Path, source_spec_path: pathlib.Path) -> dict:
    source_dir = model_root / "source"
    ir_dir = model_root / "openvino" / "fp16"
    model_spec_path = model_root / "openvino" / "model-spec.json"
    lock_path = source_dir / "SOURCE-LOCK.sha256"

    source_spec = load_json(source_spec_path)
    model_spec = load_json(model_spec_path)
    lock = load_lock(lock_path)

    expected_names = [entry["name"] for entry in source_spec.get("files", [])]
    expected_names.append("LICENSE.txt")
    missing_lock = [name for name in expected_names if name not in lock]
    if missing_lock:
        raise VerificationError(
            "source lock is missing entries: " + ", ".join(sorted(missing_lock))
        )

    actual_source_hashes: dict[str, str] = {}
    for name in expected_names:
        path = source_dir / name
        if not path.is_file():
            raise VerificationError(f"missing source artifact: {path}")
        digest = sha256(path)
        actual_source_hashes[name] = digest
        if lock[name] != digest:
            raise VerificationError(
                f"source hash mismatch for {name}: expected {lock[name]}, got {digest}"
            )

    source_spec_hash = sha256(source_spec_path)
    if model_spec.get("source_spec_sha256") != source_spec_hash:
        raise VerificationError("model-spec source_spec_sha256 does not match source-v1.json")

    model_source_hashes = model_spec.get("source_artifact_sha256")
    if not isinstance(model_source_hashes, dict):
        raise VerificationError("model-spec source_artifact_sha256 is missing")
    for name in [entry["name"] for entry in source_spec.get("files", [])]:
        if model_source_hashes.get(name) != actual_source_hashes[name]:
            raise VerificationError(
                f"model-spec source hash does not match prepared artifact: {name}"
            )

    xml = ir_dir / "rm_cnn4a_fp16.xml"
    binary = ir_dir / "rm_cnn4a_fp16.bin"
    ir_contract_path = ir_dir / "ir-contract.json"
    for path in (xml, binary, ir_contract_path):
        if not path.is_file():
            raise VerificationError(f"missing prepared IR artifact: {path}")

    actual_contract = inspect_ir(xml, binary)
    stored_contract = load_json(ir_contract_path)
    if stored_contract != actual_contract:
        raise VerificationError("ir-contract.json does not match the actual XML/BIN")

    openvino = model_spec.get("openvino")
    if not isinstance(openvino, dict):
        raise VerificationError("model-spec openvino section is missing")
    if openvino.get("version") != "2020.3.2" or openvino.get("precision") != "FP16":
        raise VerificationError("model-spec OpenVINO version/precision is not 2020.3.2 FP16")
    if openvino.get("xml_sha256") != actual_contract["xml_sha256"]:
        raise VerificationError("model-spec XML hash does not match prepared IR")
    if openvino.get("bin_sha256") != actual_contract["bin_sha256"]:
        raise VerificationError("model-spec BIN hash does not match prepared IR")
    if openvino.get("ir_contract") != actual_contract:
        raise VerificationError("model-spec IR contract does not match prepared IR")

    fixture = model_spec.get("reference_fixture")
    if not isinstance(fixture, dict):
        raise VerificationError("model-spec reference_fixture is missing")
    if fixture.get("features") != "source/feat1_10.ark":
        raise VerificationError("unexpected reference feature fixture path")
    if fixture.get("scores") != "source/score1_10.ark":
        raise VerificationError("unexpected reference score fixture path")

    return {
        "schema": "speech-asr/prepared-model-verification",
        "version": 1,
        "id": model_spec.get("id"),
        "source_spec_sha256": source_spec_hash,
        "source_lock_sha256": sha256(lock_path),
        "xml_sha256": actual_contract["xml_sha256"],
        "bin_sha256": actual_contract["bin_sha256"],
        "features_sha256": actual_source_hashes["feat1_10.ark"],
        "reference_scores_sha256": actual_source_hashes["score1_10.ark"],
        "inputs": actual_contract["inputs"],
        "outputs": actual_contract["outputs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=pathlib.Path,
        default=ROOT / "vendor" / "models" / "rm_cnn4a_smbr",
    )
    parser.add_argument(
        "--source-spec",
        type=pathlib.Path,
        default=SPEECH_ROOT / "models" / "rm_cnn4a" / "source-v1.json",
    )
    args = parser.parse_args()
    try:
        result = verify_bundle(args.model_root, args.source_spec)
    except VerificationError as exc:
        print(f"rm_cnn4a verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
