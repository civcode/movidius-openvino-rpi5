import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class SpeechIntegrationCommandTests(unittest.TestCase):
    def test_test_runner_discovers_only_speech_suite(self):
        text = (ROOT / "scripts" / "test-speech-asr.sh").read_text(encoding="utf-8")
        self.assertIn("test_speech_asr_*.py", text)
        self.assertNotIn("docker", text.lower())

    def test_run_script_exposes_separate_cpu_reference_mode(self):
        text = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("speech-reference", text)
        self.assertIn("DEVICE=CPU", text)
        self.assertIn("DEVICE=MYRIAD", text)
        self.assertIn("requires --platform amd64", text)

    def test_benchmark_wrapper_exposes_cpu_and_myriad_backends(self):
        text = (
            ROOT
            / "examples"
            / "speech-asr"
            / "evaluation"
            / "benchmark_rm_cnn4a.py"
        ).read_text(encoding="utf-8")
        self.assertIn('choices=("cpu", "myriad")', text)
        self.assertIn('"speech-reference"', text)
        self.assertIn('"speech-regress"', text)

    def test_readme_does_not_claim_rm_cnn4a_raw_audio_frontend(self):
        text = (ROOT / "examples" / "speech-asr" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Kaldi feature ARK", text)
        self.assertIn("AMI/custom-model path", text)


if __name__ == "__main__":
    unittest.main()
