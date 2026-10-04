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

from speech_asr.ami import prepare_from_spec, seconds_to_samples, sha256_file
from speech_asr.contracts import validate_speech_sample


SEGMENTS = b'''<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="TEST.A.segs" xmlns:nite="http://nite.sourceforge.net/">
  <segment nite:id="TEST.sync.1" channel="0" transcriber_start="0.100" transcriber_end="0.600">
    <nite:child href="TEST.A.words.xml#id(TEST.A.words0)..id(TEST.A.words4)"/>
  </segment>
</nite:root>'''

WORDS = b'''<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="TEST.A.words" xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="TEST.A.words0" starttime="0.150" endtime="0.250">Hello</w>
  <w nite:id="TEST.A.words1" starttime="0.250" endtime="0.250" punc="true">,</w>
  <vocalsound nite:id="TEST.A.words2" starttime="0.260" endtime="0.300" type="laugh"/>
  <w nite:id="TEST.A.words3" starttime="0.350" endtime="0.500">WORLD</w>
  <w nite:id="TEST.A.words4" starttime="0.500" endtime="0.500" punc="true">.</w>
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


class AmiPreparationTests(unittest.TestCase):
    def test_seconds_to_samples_is_decimal_deterministic(self):
        self.assertEqual(seconds_to_samples("77.408"), 1238528)

    def test_prepare_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = pathlib.Path(tmp_name)
            audio = tmp / "TEST.wav"
            write_test_wav(audio)
            archive = tmp / "annotations.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("segments/TEST.A.segments.xml", SEGMENTS)
                zf.writestr("words/TEST.A.words.xml", WORDS)

            annotation_sha = sha256_file(archive)
            audio_sha = sha256_file(audio)
            spec = {
                "schema": "speech-asr/ami-split",
                "version": 1,
                "id": "test-smoke-v1",
                "annotations": {"sha256": annotation_sha},
                "sources": [
                    {
                        "meeting": "TEST",
                        "audio": {
                            "stream": "Mix-Headset",
                            "sha256": audio_sha,
                        },
                        "selections": [
                            {
                                "speaker": "A",
                                "segments": ["TEST.sync.1"],
                            }
                        ],
                    }
                ],
            }

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
            self.assertEqual(
                p1["manifest_sha256"], p2["manifest_sha256"]
            )
            self.assertEqual(
                (out1 / "manifest.jsonl").read_bytes(),
                (out2 / "manifest.jsonl").read_bytes(),
            )

            record = json.loads(
                (out1 / "manifest.jsonl").read_text().strip()
            )
            validate_speech_sample(record)
            self.assertEqual(record["transcript"]["text"], "hello world")
            self.assertEqual(record["audio"]["end_sample"], 8000)
            self.assertEqual(
                record["transcript"]["words"][0]["start_sample"], 800
            )
            self.assertEqual(
                record["transcript"]["words"][1]["end_sample"], 6400
            )

            f32 = (out1 / record["audio"]["path"]).read_bytes()
            self.assertEqual(len(f32), 8000 * 4)
            self.assertEqual(
                hashlib.sha256(f32).hexdigest(),
                record["metadata"]["normalized_audio_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
