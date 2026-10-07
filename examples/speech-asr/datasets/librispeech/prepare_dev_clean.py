#!/usr/bin/env python3
"""Download, verify and manifest OpenSLR LibriSpeech dev-clean."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parents[1]
ROOT = SPEECH_ROOT.parents[1]

URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME = "dev-clean.tar.gz"
ARCHIVE_MD5 = "42e2234ba48799c1f50f24a7926300a1"
EXPECTED_UTTERANCES = 2703
SAMPLE_RATE = 16000


def md5_path(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".part",
        delete=False,
    ) as tmp:
        temp = pathlib.Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, tmp, length=1024 * 1024)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(destination)


def safe_extract(archive: pathlib.Path, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (output / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.name}") from exc
        tar.extractall(output)


def transcript_records(output: pathlib.Path) -> list[dict]:
    corpus = output / "LibriSpeech" / "dev-clean"
    if not corpus.is_dir():
        raise ValueError(f"LibriSpeech dev-clean tree missing: {corpus}")
    records = []
    for transcript in sorted(corpus.glob("*/*/*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample_id, text = line.split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"malformed transcript line in {transcript}: {line!r}") from exc
            speaker, chapter, _ = sample_id.split("-", 2)
            audio = corpus / speaker / chapter / f"{sample_id}.flac"
            if not audio.is_file():
                raise ValueError(f"audio missing for transcript {sample_id}: {audio}")
            info = sf.info(str(audio))
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                raise ValueError(
                    f"{sample_id}: expected 16 kHz mono FLAC, got "
                    f"{info.samplerate} Hz / {info.channels} channels"
                )
            records.append({
                "id": sample_id,
                "audio_path": audio.relative_to(output).as_posix(),
                "sample_rate_hz": info.samplerate,
                "sample_count": int(info.frames),
                "text": text,
            })
    records.sort(key=lambda item: item["id"])
    if len(records) != EXPECTED_UTTERANCES:
        raise ValueError(
            f"LibriSpeech dev-clean utterance count changed: "
            f"{len(records)} != {EXPECTED_UTTERANCES}"
        )
    return records


def write_manifest(output: pathlib.Path, records: list[dict], archive: pathlib.Path) -> dict:
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    total_samples = sum(int(record["sample_count"]) for record in records)
    result = {
        "schema": "speech-asr/librispeech-reference-dataset",
        "version": 1,
        "id": "librispeech-dev-clean",
        "status": "valid",
        "source_url": URL,
        "archive": str(archive),
        "archive_md5": md5_path(archive),
        "manifest": str(manifest),
        "utterances": len(records),
        "sample_rate_hz": SAMPLE_RATE,
        "audio_samples": total_samples,
        "audio_seconds": total_samples / SAMPLE_RATE,
    }
    (output / "dataset.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def verify(output: pathlib.Path, archive: pathlib.Path | None) -> dict:
    if archive is not None and archive.is_file():
        actual = md5_path(archive)
        if actual != ARCHIVE_MD5:
            raise ValueError(f"LibriSpeech archive MD5 mismatch: {actual} != {ARCHIVE_MD5}")
    records = transcript_records(output)
    manifest = output / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"manifest missing: {manifest}")
    manifest_records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest_records != records:
        raise ValueError("prepared LibriSpeech manifest does not match extracted corpus")
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("utterances") != EXPECTED_UTTERANCES:
        raise ValueError("LibriSpeech dataset metadata utterance count changed")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "librispeech" / "downloads",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        archive = args.cache / ARCHIVE_NAME
        if args.verify_only:
            print(json.dumps(verify(args.output, archive if archive.is_file() else None), sort_keys=True))
            return 0
        if not archive.is_file():
            download(URL, archive)
        actual = md5_path(archive)
        if actual != ARCHIVE_MD5:
            raise ValueError(f"LibriSpeech archive MD5 mismatch: {actual} != {ARCHIVE_MD5}")
        corpus = args.output / "LibriSpeech" / "dev-clean"
        if not corpus.is_dir():
            safe_extract(archive, args.output)
        records = transcript_records(args.output)
        result = write_manifest(args.output, records, archive)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
