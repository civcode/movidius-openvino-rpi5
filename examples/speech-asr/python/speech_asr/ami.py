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


def _require_string(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected non-empty string")


def _require_sha256(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        errors.append(f"{path}: expected lowercase 64-character SHA-256")


def validate_split_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a frozen AMI split selection before network or filesystem work."""

    errors: list[str] = []
    if spec.get("schema") != "speech-asr/ami-split":
        errors.append("$.schema: expected 'speech-asr/ami-split'")
    if spec.get("version") != 1:
        errors.append("$.version: expected 1")
    _require_string(spec.get("id"), "$.id", errors)
    if spec.get("license") != "CC-BY-4.0":
        errors.append("$.license: expected 'CC-BY-4.0'")

    annotations = spec.get("annotations")
    if not isinstance(annotations, Mapping):
        errors.append("$.annotations: expected object")
        annotations = {}
    for key in ("version", "filename", "url"):
        _require_string(annotations.get(key), f"$.annotations.{key}", errors)
    _require_sha256(annotations.get("sha256"), "$.annotations.sha256", errors)

    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("$.sources: expected non-empty array")
        sources = []

    meetings: set[str] = set()
    declared_samples: set[tuple[str, str, str]] = set()
    for source_index, source_value in enumerate(sources):
        path = f"$.sources[{source_index}]"
        if not isinstance(source_value, Mapping):
            errors.append(f"{path}: expected object")
            continue
        meeting = source_value.get("meeting")
        _require_string(meeting, path + ".meeting", errors)
        if isinstance(meeting, str):
            if meeting in meetings:
                errors.append(f"{path}.meeting: duplicate meeting {meeting!r}")
            meetings.add(meeting)

        audio = source_value.get("audio")
        if not isinstance(audio, Mapping):
            errors.append(path + ".audio: expected object")
            audio = {}
        for key in ("stream", "filename", "url"):
            _require_string(audio.get(key), path + f".audio.{key}", errors)
        _require_sha256(audio.get("sha256"), path + ".audio.sha256", errors)

        selections = source_value.get("selections")
        if not isinstance(selections, list) or not selections:
            errors.append(path + ".selections: expected non-empty array")
            continue
        speakers: set[str] = set()
        for selection_index, selection_value in enumerate(selections):
            selection_path = path + f".selections[{selection_index}]"
            if not isinstance(selection_value, Mapping):
                errors.append(selection_path + ": expected object")
                continue
            speaker = selection_value.get("speaker")
            _require_string(speaker, selection_path + ".speaker", errors)
            if isinstance(speaker, str):
                if speaker in speakers:
                    errors.append(
                        selection_path + f".speaker: duplicate speaker {speaker!r}"
                    )
                speakers.add(speaker)

            has_all = selection_value.get("all_segments") is True
            segments = selection_value.get("segments")
            has_list = isinstance(segments, list) and bool(segments)
            if has_all == has_list:
                errors.append(
                    selection_path
                    + ": declare exactly one of all_segments=true or a non-empty segments array"
                )
                continue
            if has_list:
                seen: set[str] = set()
                for segment_index, segment_id in enumerate(segments):
                    segment_path = selection_path + f".segments[{segment_index}]"
                    _require_string(segment_id, segment_path, errors)
                    if not isinstance(segment_id, str):
                        continue
                    expected_prefix = f"{meeting}.sync." if isinstance(meeting, str) else ""
                    if expected_prefix and not segment_id.startswith(expected_prefix):
                        errors.append(
                            segment_path + f": expected prefix {expected_prefix!r}"
                        )
                    if segment_id in seen:
                        errors.append(segment_path + f": duplicate segment {segment_id!r}")
                    seen.add(segment_id)
                    key = (str(meeting), str(speaker), segment_id)
                    if key in declared_samples:
                        errors.append(segment_path + ": duplicate sample selection")
                    declared_samples.add(key)

    expected = spec.get("expected")
    if expected is not None:
        if not isinstance(expected, Mapping):
            errors.append("$.expected: expected object")
        else:
            records = expected.get("records")
            if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
                errors.append("$.expected.records: expected positive integer")
            _require_sha256(
                expected.get("manifest_sha256"),
                "$.expected.manifest_sha256",
                errors,
            )
            _require_sha256(
                expected.get("logical_tree_sha256"),
                "$.expected.logical_tree_sha256",
                errors,
            )

    if errors:
        raise AmiPreparationError("invalid AMI split spec:\n- " + "\n- ".join(errors))
    return dict(spec)


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


def _segment_ids(segment_xml: bytes) -> list[str]:
    root = ET.fromstring(segment_xml)
    return [
        segment.attrib[NITE_ID]
        for segment in root.findall("segment")
        if segment.attrib.get(NITE_ID)
    ]


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


def _read_pcm16_clip(
    source: Path,
    start_sample: int,
    end_sample: int,
) -> tuple[bytes, int]:
    with wave.open(str(source), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise AmiPreparationError(
                f"{source}: expected 16000 Hz, got {wav.getframerate()}"
            )
        channels = wav.getnchannels()
        if channels not in (1, 2):
            raise AmiPreparationError(
                f"{source}: expected mono or stereo, got {channels} channels"
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
    expected = (end_sample - start_sample) * channels * 2
    if len(data) != expected:
        raise AmiPreparationError(
            f"{source}: short read: expected {expected} bytes, got {len(data)}"
        )
    return data, channels


def _pcm16le_to_f32le_mono(data: bytes, *, channels: int) -> bytes:
    if channels not in (1, 2):
        raise AmiPreparationError(
            f"PCM16 normalization supports only 1 or 2 channels, got {channels}"
        )
    frame_bytes = channels * 2
    if len(data) % frame_bytes:
        raise AmiPreparationError(
            f"PCM16 payload size {len(data)} is not a whole {channels}-channel frame"
        )

    frames = len(data) // frame_bytes
    out = bytearray(frames * 4)
    offset = 0
    if channels == 1:
        for (sample,) in struct.iter_unpack("<h", data):
            struct.pack_into("<f", out, offset, sample / 32768.0)
            offset += 4
    else:
        for left, right in struct.iter_unpack("<hh", data):
            # Deterministic equal-power-neutral arithmetic mean. This keeps the
            # normalized mono contract at one output sample per input WAV frame
            # without clipping either full-scale equal-channel endpoint.
            sample = (left + right) / 65536.0
            struct.pack_into("<f", out, offset, sample)
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
    pcm16, source_channels = _read_pcm16_clip(
        source_audio,
        source_start,
        source_end,
    )
    f32 = _pcm16le_to_f32le_mono(pcm16, channels=source_channels)
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
            **(
                {
                    "source_audio_channels": source_channels,
                    "channel_normalization": "stereo-average-v1",
                }
                if source_channels == 2
                else {}
            ),
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
    validate_split_spec(spec)
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
                segments_member = f"segments/{meeting}.{speaker}.segments.xml"
                try:
                    segment_xml = archive.read(segments_member)
                except KeyError as exc:
                    raise AmiPreparationError(
                        f"missing AMI annotation member: {segments_member}"
                    ) from exc
                segment_ids = (
                    _segment_ids(segment_xml)
                    if selection.get("all_segments") is True
                    else list(selection["segments"])
                )
                for segment_id in segment_ids:
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
    normalized_audio = {
        record["id"]: record["metadata"]["normalized_audio_sha256"]
        for record in records
    }
    logical_tree_hash = canonical_json_sha256(
        {"manifest_sha256": manifest_sha, "normalized_audio_sha256": normalized_audio}
    )
    provenance = {
        "schema": "speech-asr/ami-preparation",
        "version": 1,
        "split_id": spec["id"],
        "split_spec_sha256": canonical_json_sha256(spec),
        "annotation_sha256": annotation_sha,
        "manifest_sha256": manifest_sha,
        "records": len(records),
        "normalized_audio_sha256": normalized_audio,
        "logical_tree_sha256": logical_tree_hash,
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


def verify_prepared_dataset(
    *, spec: Mapping[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """Verify manifest, clip hashes/sizes and provenance without network access."""

    validate_split_spec(spec)
    manifest_path = output_dir / "manifest.jsonl"
    provenance_path = output_dir / "provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise AmiPreparationError(
            f"{output_dir}: expected manifest.jsonl and provenance.json"
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("split_id") != spec["id"]:
        raise AmiPreparationError("prepared split id does not match split spec")
    expected_spec_hash = canonical_json_sha256(spec)
    if provenance.get("split_spec_sha256") != expected_spec_hash:
        raise AmiPreparationError("prepared split spec hash does not match current spec")

    manifest_hash = sha256_file(manifest_path)
    if provenance.get("manifest_sha256") != manifest_hash:
        raise AmiPreparationError("manifest SHA-256 does not match provenance")

    records = []
    ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AmiPreparationError(
                    f"manifest line {line_number}: invalid JSON: {exc}"
                ) from exc
            validate_speech_sample(record)
            sample_id = record["id"]
            if sample_id in ids:
                raise AmiPreparationError(f"duplicate manifest sample id: {sample_id}")
            ids.add(sample_id)
            clip = output_dir / record["audio"]["path"]
            if not clip.is_file():
                raise AmiPreparationError(f"missing normalized audio: {clip}")
            expected_bytes = (
                record["audio"]["end_sample"] - record["audio"]["start_sample"]
            ) * 4
            actual_bytes = clip.stat().st_size
            if actual_bytes != expected_bytes:
                raise AmiPreparationError(
                    f"{clip}: expected {expected_bytes} bytes, got {actual_bytes}"
                )
            clip_hash = sha256_file(clip)
            if clip_hash != record["metadata"].get("normalized_audio_sha256"):
                raise AmiPreparationError(f"{clip}: SHA-256 differs from manifest metadata")
            records.append(record)

    if provenance.get("records") != len(records):
        raise AmiPreparationError("record count does not match provenance")

    expected_clip_paths = {
        (output_dir / record["audio"]["path"]).resolve() for record in records
    }
    actual_clip_paths = {
        path.resolve() for path in (output_dir / "audio").glob("*.f32")
    } if (output_dir / "audio").is_dir() else set()
    extras = sorted(str(path) for path in actual_clip_paths - expected_clip_paths)
    if extras:
        raise AmiPreparationError(
            "prepared audio directory contains unreferenced clips: " + ", ".join(extras)
        )

    normalized_audio = {
        record["id"]: record["metadata"]["normalized_audio_sha256"]
        for record in records
    }
    if provenance.get("normalized_audio_sha256") != normalized_audio:
        raise AmiPreparationError("normalized audio hash map does not match provenance")
    logical_tree_hash = canonical_json_sha256(
        {"manifest_sha256": manifest_hash, "normalized_audio_sha256": normalized_audio}
    )
    if provenance.get("logical_tree_sha256") != logical_tree_hash:
        raise AmiPreparationError("logical tree hash does not match provenance")

    expected = spec.get("expected")
    if isinstance(expected, Mapping):
        checks = {
            "records": len(records),
            "manifest_sha256": manifest_hash,
            "logical_tree_sha256": logical_tree_hash,
        }
        for key, actual in checks.items():
            if expected.get(key) != actual:
                raise AmiPreparationError(
                    f"prepared {key} does not match frozen split expectation: "
                    f"expected {expected.get(key)!r}, got {actual!r}"
                )

    return {
        "schema": "speech-asr/ami-verification",
        "version": 1,
        "split_id": spec["id"],
        "records": len(records),
        "manifest_sha256": manifest_hash,
        "logical_tree_sha256": logical_tree_hash,
    }
