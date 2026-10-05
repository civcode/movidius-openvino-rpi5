import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class SpeechRuntimeWiringTests(unittest.TestCase):
    def test_docker_builds_only_speech_sample_from_upstream_samples(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("-DENABLE_SAMPLES=ON", text)
        self.assertIn("-DBUILD_SAMPLE_NAME=speech_sample", text)
        self.assertIn("/bin/speech_sample", text)

    def test_run_wrapper_exposes_regression_mode(self):
        text = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("speech-regress", text)
        self.assertIn("DEVICE=MYRIAD", text)
        self.assertIn('-d "${DEVICE}"', text)
        self.assertIn("-bs 1", text)
        self.assertIn("score1_10.ark", text)


if __name__ == "__main__":
    unittest.main()
