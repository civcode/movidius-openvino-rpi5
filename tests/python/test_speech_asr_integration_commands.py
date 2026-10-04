import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class SpeechIntegrationCommandTests(unittest.TestCase):
    def test_test_runner_discovers_only_speech_suite(self):
        text = (ROOT / "scripts" / "test-speech-asr.sh").read_text(encoding="utf-8")
        self.assertIn("test_speech_asr_*.py", text)
        self.assertNotIn("docker", text.lower())

    def test_readme_does_not_claim_rm_cnn4a_raw_audio_frontend(self):
        text = (ROOT / "examples" / "speech-asr" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Kaldi feature ARK", text)
        self.assertIn("AMI/custom-model path", text)


if __name__ == "__main__":
    unittest.main()
