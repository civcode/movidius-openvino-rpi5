"""Deterministic AMI NXT-to-normalized-manifest preparation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import urllib.request
import wave
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from xml.etree import ElementTree as ET

from .contracts import canonical_json_sha256, validate_speech_sample
from .text import normalize_text_v1

NITE_NS = "http://nite.sourceforge.net/"
NITE_ID = "{%s}id" % NITE_NS
_CHILD = "{%s}child" % NITE_NS
_HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")
SAMPLE_RATE = 16000


class AmiPreparationError(RuntimeError):
    """Raised when pinned AMI data cannot be prepared deterministically."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds_to_samples(value: str) -> int:
    samples = Decimal(value) * SAMPLE_RATE
    return int(samples.to_integral_value(rounding=ROUND_HALF_UP))


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return destination
        raise AmiPreparationError(
            f"cached source hash mismatch for {destination}: expected {expected_sha256}, got {actual}"
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "movidius-openvino-rpi5-speech-asr/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(tmp)
        if actual != expected_sha256:
            raise AmiPreparationError(
                f"download hash mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_word_items(xml_bytes: bytes) -> Sequence[Dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    items = []
    for element in list(root):
        item_id = element.attrib.get(NITE_ID)
        if not item_id:
            continue
        items.append(
            {
                "id": item_id,
                "kind": _local_name(element.tag),
                "start": element.attrib.get("starttime"),
                "end": element.attrib.get("endtime"),
                "punc": element.attrib.get("punc") == "true",
                "text": (element.text or "").strip(),
            }
        )
    return items


def _selected_segment(segment_xml: bytes, segment_id: str) -> Dict[str, str]:
    root = ET.fromstring(segment_xml)
    for segment in root.findall("segment"):
        if segment.attrib.get(NITE_ID) != segment_id:
            continue
        start = segment.attrib.get("transcriber_start")
        end = segment.attrib.get("transcriber_end")
        channel = segment.attrib.get("channel")
        children = segment.findall(_CHILD)
        if not start or not end or channel is None or len(children) != 1:
            raise AmiPreparationError(
                f"segment {segment_id} has an unsupported/incomplete NXT shape"
            )
        href = children[0].attrib.get("href", "")
        match = _HREF_RE.search(href)
        if not match:
            raise AmiPreparationError(
                f"segment {segment_id} has unrecognized word href: {href}"
            )
        return {
            "start": start,
            "end": end,
            "channel": channel,
            "first_word_id": match.group(1),
            "last_word_id": match.group(2) or match.group(1),
        }
    raise AmiPreparationError(f"segment not found: {segment_id}")


def _slice_items(
    items: Sequence[Mapping[str, Any]], first_id: str, last_id: str
) -> Sequence[Mapping[str, Any]]:
    indices = {item["id"]: index for index, item in enumerate(items)}
    if first_id not in indices or last_id not in indices:
        raise AmiPreparationError(f"word range not found: {first_id}..{last_id}")
    first = indices[first_id]
    last = indices[last_id]
    if last < first:
        raise AmiPreparationError(f"reversed word range: {first_id}..{last_id}")
    return items[first : last + 1]


def _read_pcm16_mono_clip(source: Path, start_sample: int, end_sample: int) -> bytes:
    with wave.open(str(source), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise AmiPreparationError(
                f"{source}: expected 16000 Hz, got {wav.getframerate()}"
            )
        if wav.getnchannels() != 1:
            raise AmiPreparationError(
                f"{source}: expected mono, got {wav.getnchannels()} channels"
            )
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise AmiPreparationError(
                f"{source}: expected uncompressed signed 16-bit PCM WAV"
            )
        if (
            start_sample < 0
            or end_sample <= start_sample
            or end_sample > wav.getnframes()
        ):
            raise AmiPreparationError(
                f"{source}: invalid clip range {start_sample}:{end_sample} "
                f"for {wav.getnframes()} frames"
            )
        wav.setpos(start_sample)
        data = wav.readframes(end_sample - start_sample)
    expected = (end_sample - start_sample) * 2
    if len(data) != expected:
        raise AmiPreparationError(
            f"{source}: short read: expected {expected} bytes, got {len(data)}"
        )
    return data


def _pcm16le_to_f32le(data: bytes) -> bytes:
    if len(data) % 2:
        raise AmiPreparationError("PCM16 payload has odd byte length")
    out = bytearray((len(data) // 2) * 4)
    offset = 0
    for (sample,) in struct.iter_unpack("<h", data):
        struct.pack_into("<f", out, offset, sample / 32768.0)
        offset += 4
    return bytes(out)


def render_segment(
    *,
    annotation_zip: zipfile.ZipFile,
    source_audio: Path,
    output_audio: Path,
    meeting: str,
    speaker: str,
    segment_id: str,
    source_audio_sha256: str,
    annotation_sha256: str,
    audio_stream: str,
) -> Dict[str, Any]:
    words_member = f"words/{meeting}.{speaker}.words.xml"
    segments_member = f"segments/{meeting}.{speaker}.segments.xml"
    try:
        words_xml = annotation_zip.read(words_member)
        segments_xml = annotation_zip.read(segments_member)
    except KeyError as exc:
        raise AmiPreparationError(f"missing AMI annotation member: {exc}") from exc

    items = _parse_word_items(words_xml)
    segment = _selected_segment(segments_xml, segment_id)
    selected = _slice_items(
        items, segment["first_word_id"], segment["last_word_id"]
    )

    source_start = seconds_to_samples(segment["start"])
    source_end = seconds_to_samples(segment["end"])
    pcm16 = _read_pcm16_mono_clip(source_audio, source_start, source_end)
    f32 = _pcm16le_to_f32le(pcm16)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    output_audio.write_bytes(f32)

    words = []
    lexical_text = []
    for item in selected:
        if item["kind"] != "w" or item["punc"]:
            continue
        text = normalize_text_v1(item["text"])
        if not text:
            continue
        if item["start"] is None or item["end"] is None:
            raise AmiPreparationError(f"word {item['id']} is missing timing")
        word_start = seconds_to_samples(item["start"]) - source_start
        word_end = seconds_to_samples(item["end"]) - source_start
        if word_end <= word_start:
            raise AmiPreparationError(
                f"lexical word {item['id']} has non-positive duration"
            )
        lexical_text.append(text)
        words.append(
            {
                "text": text,
                "start_sample": word_start,
                "end_sample": word_end,
            }
        )

    normalized_text = normalize_text_v1(" ".join(lexical_text))
    sample_id = f"ami-{meeting}-{speaker}-{segment_id.rsplit('.', 1)[-1]}"
    document = {
        "schema": "speech-asr/sample",
        "version": 1,
        "id": sample_id,
        "audio": {
            "path": output_audio.as_posix(),
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "sample_type": "float32",
            "encoding": "f32le",
            "start_sample": 0,
            "end_sample": source_end - source_start,
        },
        "transcript": {
            "text": normalized_text,
            "normalization": "text-v1",
            "words": words,
        },
        "metadata": {
            "corpus": "AMI",
            "meeting": meeting,
            "speaker": speaker,
            "segment_id": segment_id,
            "channel": segment["channel"],
            "audio_stream": audio_stream,
            "source_start_sample": source_start,
            "source_end_sample": source_end,
            "source_audio_sha256": source_audio_sha256,
            "annotation_sha256": annotation_sha256,
            "normalized_audio_sha256": hashlib.sha256(f32).hexdigest(),
        },
    }
    validate_speech_sample(document)
    return document


def prepare_from_spec(
    *,
    spec: Mapping[str, Any],
    annotation_zip_path: Path,
    audio_paths: Mapping[str, Path],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.jsonl"
    records = []

    annotation_sha = sha256_file(annotation_zip_path)
    expected_annotation = spec["annotations"]["sha256"]
    if annotation_sha != expected_annotation:
        raise AmiPreparationError(
            f"annotation archive hash mismatch: expected {expected_annotation}, "
            f"got {annotation_sha}"
        )

    with zipfile.ZipFile(annotation_zip_path) as archive:
        for source in spec["sources"]:
            meeting = source["meeting"]
            source_audio = audio_paths[meeting]
            source_hash = sha256_file(source_audio)
            if source_hash != source["audio"]["sha256"]:
                raise AmiPreparationError(
                    f"audio hash mismatch for {meeting}: expected "
                    f"{source['audio']['sha256']}, got {source_hash}"
                )
            for selection in source["selections"]:
                speaker = selection["speaker"]
                for segment_id in selection["segments"]:
                    suffix = segment_id.rsplit(".", 1)[-1]
                    filename = f"{meeting}.{speaker}.{suffix}.f32"
                    record = render_segment(
                        annotation_zip=archive,
                        source_audio=source_audio,
                        output_audio=audio_dir / filename,
                        meeting=meeting,
                        speaker=speaker,
                        segment_id=segment_id,
                        source_audio_sha256=source_hash,
                        annotation_sha256=annotation_sha,
                        audio_stream=source["audio"]["stream"],
                    )
                    record["audio"]["path"] = f"audio/{filename}"
                    records.append(record)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

    manifest_sha = sha256_file(manifest_path)
    provenance = {
        "schema": "speech-asr/ami-preparation",
        "version": 1,
        "split_id": spec["id"],
        "split_spec_sha256": canonical_json_sha256(spec),
        "annotation_sha256": annotation_sha,
        "manifest_sha256": manifest_sha,
        "records": len(records),
        "audio_sha256": {
            source["meeting"]: source["audio"]["sha256"]
            for source in spec["sources"]
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(
            provenance, ensure_ascii=False, sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return provenance


def acquire_sources(
    spec: Mapping[str, Any], cache_dir: Path
) -> tuple[Path, Dict[str, Path]]:
    annotations = spec["annotations"]
    annotation_path = download_verified(
        annotations["url"],
        cache_dir / annotations["filename"],
        annotations["sha256"],
    )
    audio_paths = {}
    for source in spec["sources"]:
        audio = source["audio"]
        audio_paths[source["meeting"]] = download_verified(
            audio["url"], cache_dir / audio["filename"], audio["sha256"]
        )
    return annotation_path, audio_paths
