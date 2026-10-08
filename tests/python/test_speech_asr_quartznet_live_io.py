"""Hardware-free tests for live QuartzNet WAV and microphone I/O."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/speech-asr/runtime/quartznet_fixed512_live.py"
spec = importlib.util.spec_from_file_location("quartznet_live_io_under_test", SOURCE)
live = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live)


class FakeRecognizer:
    def __init__(self):
        self.chunks = []
        self.finalizations = 0

    def push_audio(self, samples):
        self.chunks.append(np.array(samples, copy=True))
        return ()

    def finalize(self):
        self.finalizations += 1
        return "finished"


class FakeConsole:
    def __init__(self):
        self.messages = []
        self.updates = []

    def status(self, text):
        self.messages.append(text)

    def update(self, value):
        self.updates.append(value)


class QuartzNetLiveInputTests(unittest.TestCase):
    def test_pcm16_decoder_preserves_odd_byte_across_reads(self):
        decoder = live.Pcm16ChunkDecoder()
        self.assertEqual(decoder.decode(b"\x01").size, 0)
        result = decoder.decode(b"\x00\xff\x7f")
        np.testing.assert_allclose(
            result, np.array([1, 32767], dtype=np.float32) / 32768.0
        )
        self.assertEqual(decoder.total_samples, 2)
        decoder.finish()

    def test_pcm16_decoder_rejects_trailing_half_sample(self):
        decoder = live.Pcm16ChunkDecoder()
        decoder.decode(b"\xff")
        with self.assertRaisesRegex(RuntimeError, "truncated PCM16"):
            decoder.finish()

    def test_microphone_feeds_samples_and_finalizes_without_hardware(self):
        python = sys.executable
        script = (
            "import os,time;"
            "os.write(1,b'\\x01');"
            "time.sleep(0.03);"
            "os.write(1,b'\\x00\\xff\\x7f')"
        )
        recognizer = FakeRecognizer()
        console = FakeConsole()
        with mock.patch.object(live.shutil, "which", return_value="/usr/bin/arecord"):
            with mock.patch.object(live, "microphone_command", return_value=[python, "-c", script]):
                live.feed_microphone(recognizer, console, device="fake", read_frames=1)
        captured = np.concatenate(recognizer.chunks)
        np.testing.assert_allclose(
            captured, np.array([1, 32767], dtype=np.float32) / 32768.0
        )
        self.assertEqual(recognizer.finalizations, 1)
        self.assertEqual(console.updates, ["finished"])

    def test_microphone_reports_alsa_process_failure(self):
        python = sys.executable
        script = "import sys;sys.stderr.write('ALSA device busy\\n');sys.exit(3)"
        recognizer = FakeRecognizer()
        with mock.patch.object(live.shutil, "which", return_value="/usr/bin/arecord"):
            with mock.patch.object(live, "microphone_command", return_value=[python, "-c", script]):
                with self.assertRaisesRegex(RuntimeError, "status 3: ALSA device busy"):
                    live.feed_microphone(recognizer, FakeConsole(), device="fake", read_frames=32)
        self.assertEqual(recognizer.finalizations, 0)

    def test_microphone_rejects_empty_capture(self):
        recognizer = FakeRecognizer()
        with mock.patch.object(live.shutil, "which", return_value="/usr/bin/arecord"):
            with mock.patch.object(live, "microphone_command", return_value=[sys.executable, "-c", "pass"]):
                with self.assertRaisesRegex(RuntimeError, "without recording audio"):
                    live.feed_microphone(recognizer, FakeConsole(), device="fake", read_frames=32)
        self.assertEqual(recognizer.finalizations, 0)

    def test_control_c_stops_capture_and_finalizes(self):
        recognizer = FakeRecognizer()
        console = FakeConsole()

        def request_stop_on_wait(*_args):
            handler = live.signal.getsignal(live.signal.SIGINT)
            handler(live.signal.SIGINT, None)
            return ([], [], [])

        with mock.patch.object(live.shutil, "which", return_value="/usr/bin/arecord"):
            with mock.patch.object(
                live, "microphone_command",
                return_value=[sys.executable, "-c", "import time;time.sleep(30)"],
            ):
                with mock.patch.object(live.select, "select", side_effect=request_stop_on_wait):
                    live.feed_microphone(recognizer, console, device="fake", read_frames=32)
        self.assertEqual(recognizer.finalizations, 1)
        self.assertEqual(console.updates, ["finished"])
        self.assertTrue(any("stop requested" in text for text in console.messages))

    def test_missing_microphone_tool_has_clear_error(self):
        with mock.patch.object(live.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "arecord is required"):
                live.feed_microphone(FakeRecognizer(), FakeConsole(), device="fake", read_frames=32)

    def test_wav_feed_finalizes_and_delivers_all_samples(self):
        recognizer = FakeRecognizer()
        console = FakeConsole()
        audio = types.SimpleNamespace(
            samples=tuple(np.arange(11, dtype=np.float32) / 100),
            duration_seconds=11 / 16000.0,
        )
        with mock.patch.object(live, "read_wav_canonical", return_value=audio):
            live.feed_wav(
                recognizer, console, pathlib.Path("fixture.wav"),
                realtime=False, feed_samples=4,
            )
        self.assertEqual([len(chunk) for chunk in recognizer.chunks], [4, 4, 3])
        np.testing.assert_allclose(
            np.concatenate(recognizer.chunks), audio.samples
        )
        self.assertEqual(recognizer.finalizations, 1)
        self.assertEqual(console.updates, ["finished"])

    def test_list_microphones_does_not_start_inference(self):
        with mock.patch.object(live.shutil, "which", return_value="/usr/bin/arecord"):
            with mock.patch.object(live.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                self.assertEqual(live.list_microphones(), 0)
        run.assert_called_once_with(["arecord", "-L"], check=False)


if __name__ == "__main__":
    unittest.main()
