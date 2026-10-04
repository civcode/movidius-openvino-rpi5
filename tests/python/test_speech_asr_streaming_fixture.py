import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "examples" / "speech-asr" / "tools" / "make_streaming_updates.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("make_streaming_updates", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load make_streaming_updates")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamingFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_build_script_uses_word_sample_timing(self):
        record = {
            "id": "sample-1",
            "audio": {"start_sample": 1000, "end_sample": 5000},
            "transcript": {
                "text": "Hello WORLD",
                "words": [
                    {"text": "Hello", "start_sample": 1100, "end_sample": 2000},
                    {"text": "WORLD", "start_sample": 2500, "end_sample": 4500},
                ],
            },
        }
        script = self.tool.build_script(record, decoder_delay_ms=10)
        self.assertEqual(script["offline_text"], "hello world")
        self.assertEqual(
            script["updates"],
            [
                {
                    "available_sample": 1160,
                    "source_end_sample": 1000,
                    "text": "hello",
                },
                {
                    "available_sample": 3660,
                    "source_end_sample": 3500,
                    "text": "hello world",
                },
            ],
        )

    def test_delay_is_capped_at_clip_end_and_coalesces(self):
        record = {
            "id": "sample-1",
            "audio": {"start_sample": 0, "end_sample": 1000},
            "transcript": {
                "text": "a b",
                "words": [
                    {"text": "a", "start_sample": 800, "end_sample": 900},
                    {"text": "b", "start_sample": 900, "end_sample": 1000},
                ],
            },
        }
        script = self.tool.build_script(record, decoder_delay_ms=100)
        self.assertEqual(len(script["updates"]), 1)
        self.assertEqual(script["updates"][0]["available_sample"], 1000)
        self.assertEqual(script["updates"][0]["text"], "a b")


if __name__ == "__main__":
    unittest.main()
