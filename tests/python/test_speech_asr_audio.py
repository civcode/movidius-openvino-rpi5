import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.audio import (
    CanonicalAudio,
    decode_pcm_to_mono_float32,
    read_f32le,
    read_wav_canonical,
    resample_linear,
    write_f32le,
)
from speech_asr.features import RawAudioFrontend, dump_feature_tensor


class CanonicalAudioTests(unittest.TestCase):
    def test_pcm16_stereo_downmix(self):
        payload = struct.pack("<hhhh", 32767, -32768, 16384, 16384)
        mono = decode_pcm_to_mono_float32(payload, sample_width=2, channels=2)
        self.assertAlmostEqual(mono[0], (-1 / 65536), places=6)
        self.assertAlmostEqual(mono[1], 0.5, places=6)

    def test_linear_resample_has_deterministic_length(self):
        values = tuple(float(i) for i in range(8))
        upsampled = resample_linear(values, 8000, 16000)
        self.assertEqual(len(upsampled), 16)
        self.assertEqual(upsampled[0], 0.0)
        self.assertEqual(upsampled[2], 1.0)

    def test_wav_to_canonical(self):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / "input.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(struct.pack("<hhhh", 0, 0, 16384, 16384))
            audio = read_wav_canonical(path)
            self.assertEqual(audio.sample_rate_hz, 16000)
            self.assertEqual(audio.channels, 1)
            self.assertEqual(audio.sample_count, 4)

    def test_f32le_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / "audio.f32"
            audio = CanonicalAudio((0.0, 0.25, -0.5, 1.0))
            digest = write_f32le(path, audio)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            loaded = read_f32le(path)
            self.assertEqual(loaded.samples, audio.samples)

    def test_slice(self):
        audio = CanonicalAudio((0.0, 0.1, 0.2, 0.3))
        self.assertEqual(audio.slice(1, 3).samples, (0.1, 0.2))


class FeatureTensorTests(unittest.TestCase):
    def test_raw_frontend_contract(self):
        audio = CanonicalAudio((0.0, 0.5, -0.5))
        tensor = RawAudioFrontend().extract(audio)
        self.assertEqual(tensor.layout, "BT")
        self.assertEqual(tensor.shape, (1, 3))

    def test_dump_is_deterministic(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            tensor = RawAudioFrontend().extract(CanonicalAudio((0.0, 0.5, -0.5)))
            first = dump_feature_tensor(root / "a" / "tensor", tensor, "test-v1")
            second = dump_feature_tensor(root / "b" / "tensor", tensor, "test-v1")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(
                (root / "a" / "tensor.f32").read_bytes(),
                (root / "b" / "tensor.f32").read_bytes(),
            )
            metadata = json.loads((root / "a" / "tensor.json").read_text())
            self.assertEqual(metadata["shape"], [1, 3])


if __name__ == "__main__":
    unittest.main()
