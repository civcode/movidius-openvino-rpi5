import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "examples"
    / "speech-asr"
    / "evaluation"
    / "phase7-ami-smoke-acceptance-v1.json"
)


class Phase7AcceptanceEvidenceTests(unittest.TestCase):
    def test_frozen_acceptance_evidence(self):
        data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], "speech-asr/phase-acceptance")
        self.assertEqual(data["phase"], 7)
        self.assertEqual(data["status"], "accepted")
        self.assertEqual(data["source"]["split_id"], "ami-smoke-v1")
        self.assertEqual(data["source"]["sample_id"], "ami-ES2002a-A-4")
        self.assertEqual(data["source"]["audio_samples"], 56752)
        self.assertTrue(data["replay"]["byte_identical"])
        self.assertEqual(
            data["replay"]["result_sha256"]["replay_1"],
            data["replay"]["result_sha256"]["replay_2"],
        )
        self.assertTrue(data["replay"]["contract_validation"]["status"] == "valid")
        self.assertEqual(data["metrics"]["chunks"], 15)
        self.assertEqual(data["metrics"]["events"], 10)
        self.assertEqual(data["metrics"]["stabilized_tokens"], 11)
        self.assertEqual(data["metrics"]["final_latency_ms"], 85.0)
        self.assertTrue(data["metrics"]["offline_comparison"]["exact_match"])
        self.assertEqual(data["metrics"]["offline_comparison"]["wer"], 0.0)
        self.assertEqual(data["metrics"]["offline_comparison"]["cer"], 0.0)
        self.assertTrue(data["acceptance"]["requirements_met"])


if __name__ == "__main__":
    unittest.main()
