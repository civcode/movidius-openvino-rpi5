import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = ROOT / "examples" / "speech-asr" / "evaluation" / "replay_streaming.py"


class StreamingReplayCliTests(unittest.TestCase):
    def test_f32le_recorded_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            audio = root / "audio.f32"
            audio.write_bytes(b"".join(struct.pack("<f", 0.05) for _ in range(6400)))
            script = root / "updates.json"
            script.write_text(
                json.dumps(
                    {
                        "schema": "speech-asr/scripted-streaming-updates",
                        "version": 1,
                        "offline_text": "hello world again",
                        "updates": [
                            {
                                "available_sample": 1600,
                                "source_end_sample": 1200,
                                "text": "hello",
                            },
                            {
                                "available_sample": 3200,
                                "source_end_sample": 2800,
                                "text": "hello world",
                            },
                            {
                                "available_sample": 5000,
                                "source_end_sample": 4400,
                                "text": "hello world again",
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output = root / "result.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    str(audio),
                    "--updates",
                    str(script),
                    "--audio-format",
                    "f32le",
                    "--chunk-ms",
                    "100",
                    "--overlap-ms",
                    "20",
                    "--left-context-ms",
                    "20",
                    "--right-context-ms",
                    "20",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["offline_comparison"]["exact_match"])
            self.assertEqual(result["events"][-1]["kind"], "final")
            self.assertEqual(result["events"][-1]["text"], "hello world again")
            self.assertIn("offline comparison: exact=True", proc.stdout)


if __name__ == "__main__":
    unittest.main()
