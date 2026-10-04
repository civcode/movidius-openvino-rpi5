import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.ami import (
    AmiPreparationError,
    prepare_from_spec,
    seconds_to_samples,
    sha256_file,
    validate_split_spec,
    verify_prepared_dataset,
)
from speech_asr.contracts import validate_speech_sample


SEGMENTS = b'''<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="TEST.A.segs" xmlns:nite="http://nite.sourceforge.net/">
  <segment nite:id="TEST.sync.1" channel="0" transcriber_start="0.100" transcriber_end="0.600">
    <nite:child href="TEST.A.words.xml#id(TEST.A.words0)..id(TEST.A.words4)"/>
  </segment>
  <segment nite:id="TEST.sync.2" channel="0" transcriber_start="0.650" transcriber_end="0.900">
    <nite:child href="TEST.A.words.xml#id(TEST.A.words5)..id(TEST.A.words6)"/>
  </segment>
</nite:root>'''

WORDS = b'''<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="TEST.A.words" xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="TEST.A.words0" starttime="0.150" endtime="0.250">Hello</w>
  <w nite:id="TEST.A.words1" starttime="0.250" endtime="0.250" punc="true">,</w>
  <vocalsound nite:id="TEST.A.words2" starttime="0.260" endtime="0.300" type="laugh"/>
  <w nite:id="TEST.A.words3" starttime="0.350" endtime="0.500">WORLD</w>
  <w nite:id="TEST.A.words4" starttime="0.500" endtime="0.500" punc="true">.</w>
  <w nite:id="TEST.A.words5" starttime="0.700" endtime="0.780">second</w>
  <w nite:id="TEST.A.words6" starttime="0.800" endtime="0.880">segment</w>
</nite:root>'''


def write_test_wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        frames = bytearray()
        for i in range(16000):
            frames += struct.pack("<h", (i % 2000) - 1000)
        wav.writeframes(bytes(frames))


def make_fixture(tmp):
    audio = tmp / "TEST.wav"
    write_test_wav(audio)
    archive = tmp / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("segments/TEST.A.segments.xml", SEGMENTS)
        zf.writestr("words/TEST.A.words.xml", WORDS)
    return audio, archive


def make_spec(audio, archive, selection):
    return {
        "schema": "speech-asr/ami-split",
        "version": 1,
        "id": "test-smoke-v1",
        "license": "CC-BY-4.0",
        "annotations": {
            "version": "test",
            "filename": "annotations.zip",
            "url": "https://example.invalid/annotations.zip",
            "sha256": sha256_file(archive),
        },
        "sources": [
            {
                "meeting": "TEST",
                "audio": {
                    "stream": "Mix-Headset",
                    "filename": "TEST.wav",
                    "url": "https://example.invalid/TEST.wav",
                    "sha256": sha256_file(audio),
                },
                "selections": [selection],
            }
        ],
    }


class AmiPreparationTests(unittest.TestCase):
    def test_checked_in_split_specs_validate(self):
        split_dir = SPEECH / "datasets" / "ami" / "splits"
        for name in ("smoke-v1.json", "benchmark-v1.json"):
            with self.subTest(name=name):
                spec = json.loads((split_dir / name).read_text(encoding="utf-8"))
                validated = validate_split_spec(spec)
                self.assertEqual(validated["version"], 1)
                self.assertIn("expected", validated)
                self.assertGreater(validated["expected"]["records"], 0)

    def test_seconds_to_samples_is_decimal_deterministic(self):
        self.assertEqual(seconds_to_samples("77.408"), 1238528)

    def test_split_spec_requires_explicit_selection_mode(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio, archive = make_fixture(tmp)
            spec = make_spec(audio, archive, {"speaker": "A"})
            with self.assertRaisesRegex(AmiPreparationError, "exactly one"):
                validate_split_spec(spec)

    def test_prepare_is_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio, archive = make_fixture(tmp)
            spec = make_spec(
                audio,
                archive,
                {"speaker": "A", "segments": ["TEST.sync.1"]},
            )

            out1 = tmp / "out1"
            out2 = tmp / "out2"
            p1 = prepare_from_spec(
                spec=spec,
                annotation_zip_path=archive,
                audio_paths={"TEST": audio},
                output_dir=out1,
            )
            p2 = prepare_from_spec(
                spec=spec,
                annotation_zip_path=archive,
                audio_paths={"TEST": audio},
                output_dir=out2,
            )
            self.assertEqual(p1["manifest_sha256"], p2["manifest_sha256"])
            self.assertEqual(p1["logical_tree_sha256"], p2["logical_tree_sha256"])
            self.assertEqual(
                (out1 / "manifest.jsonl").read_bytes(),
                (out2 / "manifest.jsonl").read_bytes(),
            )

            verified = verify_prepared_dataset(spec=spec, output_dir=out1)
            self.assertEqual(verified["records"], 1)
            record = json.loads((out1 / "manifest.jsonl").read_text().strip())
            validate_speech_sample(record)
            self.assertEqual(record["transcript"]["text"], "hello world")
            self.assertEqual(record["audio"]["end_sample"], 8000)
            self.assertEqual(record["transcript"]["words"][0]["start_sample"], 800)
            self.assertEqual(record["transcript"]["words"][1]["end_sample"], 6400)

            f32 = (out1 / record["audio"]["path"]).read_bytes()
            self.assertEqual(len(f32), 8000 * 4)
            self.assertEqual(
                hashlib.sha256(f32).hexdigest(),
                record["metadata"]["normalized_audio_sha256"],
            )

    def test_all_segments_selection_uses_annotation_order(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio, archive = make_fixture(tmp)
            spec = make_spec(audio, archive, {"speaker": "A", "all_segments": True})
            output = tmp / "out"
            prepare_from_spec(
                spec=spec,
                annotation_zip_path=archive,
                audio_paths={"TEST": audio},
                output_dir=output,
            )
            records = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["id"] for record in records], [
                "ami-TEST-A-1",
                "ami-TEST-A-2",
            ])
            self.assertEqual(records[1]["transcript"]["text"], "second segment")

    def test_verifier_rejects_unreferenced_clip(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio, archive = make_fixture(tmp)
            spec = make_spec(
                audio,
                archive,
                {"speaker": "A", "segments": ["TEST.sync.1"]},
            )
            output = tmp / "out"
            prepare_from_spec(
                spec=spec,
                annotation_zip_path=archive,
                audio_paths={"TEST": audio},
                output_dir=output,
            )
            (output / "audio" / "stale.f32").write_bytes(b"stale")
            with self.assertRaisesRegex(AmiPreparationError, "unreferenced clips"):
                verify_prepared_dataset(spec=spec, output_dir=output)

    def test_verifier_detects_corrupted_audio(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio, archive = make_fixture(tmp)
            spec = make_spec(
                audio,
                archive,
                {"speaker": "A", "segments": ["TEST.sync.1"]},
            )
            output = tmp / "out"
            prepare_from_spec(
                spec=spec,
                annotation_zip_path=archive,
                audio_paths={"TEST": audio},
                output_dir=output,
            )
            record = json.loads((output / "manifest.jsonl").read_text().strip())
            clip = output / record["audio"]["path"]
            clip.write_bytes(clip.read_bytes()[:-4])
            with self.assertRaisesRegex(AmiPreparationError, "expected .* bytes"):
                verify_prepared_dataset(spec=spec, output_dir=output)


if __name__ == "__main__":
    unittest.main()
