import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "speech-asr" / "python"))

from speech_asr.audio import CanonicalAudio
from speech_asr.streaming import (
    EnergyVad,
    SampleRingBuffer,
    ScriptedCumulativeDecoder,
    ScriptedUpdate,
    StreamingConfig,
    TranscriptStabilizer,
    compare_offline_streaming,
    plan_chunks,
    replay_recorded_audio,
)


class StreamingConfigTests(unittest.TestCase):
    def test_derived_sample_counts(self):
        config = StreamingConfig(
            chunk_ms=100,
            overlap_ms=20,
            left_context_ms=10,
            right_context_ms=30,
        )
        self.assertEqual(config.chunk_samples, 1600)
        self.assertEqual(config.overlap_samples, 320)
        self.assertEqual(config.hop_samples, 1280)
        self.assertEqual(config.left_context_samples, 160)
        self.assertEqual(config.right_context_samples, 480)

    def test_rejects_overlap_not_smaller_than_chunk(self):
        with self.assertRaisesRegex(ValueError, "overlap_ms"):
            StreamingConfig(chunk_ms=100, overlap_ms=100)


class RingBufferTests(unittest.TestCase):
    def test_absolute_indexing_and_eviction(self):
        ring = SampleRingBuffer(4)
        ring.append([0, 1, 2])
        self.assertEqual(ring.slice(0, 3), (0.0, 1.0, 2.0))
        ring.append([3, 4, 5])
        self.assertEqual(ring.start_sample, 2)
        self.assertEqual(ring.end_sample, 6)
        self.assertEqual(ring.slice(2, 6), (2.0, 3.0, 4.0, 5.0))
        with self.assertRaises(ValueError):
            ring.slice(1, 2)


class ChunkPlanningTests(unittest.TestCase):
    def test_emission_regions_cover_audio_exactly_once(self):
        audio = CanonicalAudio(tuple(0.0 for _ in range(5000)))
        config = StreamingConfig(
            chunk_ms=100,
            overlap_ms=25,
            left_context_ms=20,
            right_context_ms=10,
        )
        chunks = plan_chunks(audio, config)
        self.assertEqual(chunks[0].emit_start_sample, 0)
        self.assertEqual(chunks[-1].emit_end_sample, audio.sample_count)
        self.assertEqual(
            sum(chunk.emitted_sample_count for chunk in chunks),
            audio.sample_count,
        )
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left.emit_end_sample, right.emit_start_sample)

    def test_context_does_not_change_emission_ownership(self):
        audio = CanonicalAudio(tuple(0.0 for _ in range(4000)))
        plain = plan_chunks(
            audio,
            StreamingConfig(
                chunk_ms=100,
                overlap_ms=20,
                left_context_ms=0,
                right_context_ms=0,
            ),
        )
        context = plan_chunks(
            audio,
            StreamingConfig(
                chunk_ms=100,
                overlap_ms=20,
                left_context_ms=50,
                right_context_ms=40,
            ),
        )
        self.assertEqual(
            [(c.emit_start_sample, c.emit_end_sample) for c in plain],
            [(c.emit_start_sample, c.emit_end_sample) for c in context],
        )
        self.assertLessEqual(context[1].inference_start_sample, plain[1].inference_start_sample)
        self.assertGreaterEqual(context[0].inference_end_sample, plain[0].inference_end_sample)


class VadTests(unittest.TestCase):
    def test_energy_vad(self):
        vad = EnergyVad(0.1)
        self.assertFalse(vad.is_speech([0.01, -0.01]))
        self.assertTrue(vad.is_speech([0.2, -0.2]))


class StabilizationTests(unittest.TestCase):
    def test_tokens_stabilize_after_repeated_prefix(self):
        tracker = TranscriptStabilizer(repeats=2)
        self.assertEqual(tracker.update("hello", 100), ())
        stable = tracker.update("hello world", 200)
        self.assertEqual([item.token for item in stable], ["hello"])
        stable = tracker.update("hello world again", 300)
        self.assertEqual([item.token for item in stable], ["world"])
        self.assertEqual(tracker.stable_tokens, ("hello", "world"))

    def test_rejects_revision_after_stability(self):
        tracker = TranscriptStabilizer(repeats=1)
        tracker.update("hello", 100)
        with self.assertRaisesRegex(ValueError, "revised"):
            tracker.update("yellow", 200)


class ReplayTests(unittest.TestCase):
    def test_recorded_replay_is_deterministic_and_matches_offline(self):
        audio = CanonicalAudio(tuple(0.05 for _ in range(6400)))
        config = StreamingConfig(
            chunk_ms=100,
            overlap_ms=20,
            left_context_ms=20,
            right_context_ms=20,
            stabilization_repeats=2,
        )
        updates = [
            ScriptedUpdate(available_sample=1600, source_end_sample=1200, text="hello"),
            ScriptedUpdate(available_sample=3200, source_end_sample=2800, text="hello world"),
            ScriptedUpdate(available_sample=5000, source_end_sample=4400, text="hello world again"),
        ]
        first = replay_recorded_audio(
            audio,
            ScriptedCumulativeDecoder(updates),
            config=config,
            offline_text="hello world again",
            vad=EnergyVad(0.01),
        )
        second = replay_recorded_audio(
            audio,
            ScriptedCumulativeDecoder(updates),
            config=config,
            offline_text="hello world again",
            vad=EnergyVad(0.01),
        )
        self.assertEqual(first, second)
        self.assertTrue(first["offline_comparison"]["exact_match"])
        self.assertEqual(first["offline_comparison"]["wer"], 0.0)
        self.assertEqual(first["events"][-1]["kind"], "final")
        self.assertEqual(first["events"][-1]["text"], "hello world again")
        self.assertGreater(first["metrics"]["chunk_count"], 1)
        self.assertIsNotNone(first["metrics"]["word_stabilization_delay_ms"])

    def test_offline_comparison_reports_errors(self):
        result = compare_offline_streaming("hello world", "hello there")
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["wer"], 0.5)


if __name__ == "__main__":
    unittest.main()
