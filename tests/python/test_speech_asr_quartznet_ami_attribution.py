import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
EVALUATOR = SPEECH / "evaluation" / "evaluate_quartznet15x5_ami_attribution.py"
RUNNER = ROOT / "scripts" / "evaluate-quartznet15x5-ami-attribution.sh"


class QuartzNetAmiAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")

    def test_frozen_ami_validation_identity(self):
        self.assertIn(
            'EXPECTED_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"',
            self.source,
        )
        self.assertIn("EXPECTED_SAMPLES = 1273", self.source)

    def test_stage_order_is_single_variable_attribution(self):
        expected = (
            "s0-source-head-source-frontend",
            "s1-source-head-fixed-frontend",
            "s2-ami-head-fixed-frontend-epoch0",
            "s3-v19-finetuned",
        )
        positions = [self.source.index(stage) for stage in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('delta(stages[0], stages[1], "fixed_frontend")', self.source)
        self.assertIn('delta(stages[1], stages[2], "39_class_head")', self.source)
        self.assertIn('delta(stages[2], stages[3], "v19_fine_tuning")', self.source)

    def test_source_and_ami_heads_are_distinct(self):
        self.assertIn("load_reference_model(", self.source)
        self.assertIn("CnnCtcV19(v19_spec, len(v19_vocab", self.source)
        self.assertIn(
            'epoch0_import["target_only_symbols_random_init"] != list("0123456789")',
            self.source,
        )

    def test_frozen_v19_evidence_self_checks(self):
        self.assertIn("EXPECTED_EPOCH0_WER = 0.7393651479162168", self.source)
        self.assertIn("EXPECTED_EPOCH0_CER = 0.5776651500316765", self.source)
        self.assertIn("EXPECTED_FINETUNED_WER = 0.5983588857698121", self.source)
        self.assertIn("EXPECTED_FINETUNED_CER = 0.46201693255773774", self.source)
        self.assertIn('checkpoint.get("best_epoch", -1)', self.source)
        self.assertIn("FROZEN_METRIC_TOLERANCE = 1e-12", self.source)

    def test_analysis_performs_no_new_training_or_hardware_work(self):
        self.assertIn('"training_performed": False', self.source)
        self.assertIn("PyTorch reference attribution only", self.source)
        forbidden = (
            "run-myriad-tensor.sh",
            "run-speech-experiment.sh",
            "edge-speech",
            "docker run",
            "ssh ",
            "openvino",
        )
        combined = self.runner.lower()
        for value in forbidden:
            self.assertNotIn(value, combined)

    def test_entrypoint_exists(self):
        self.assertTrue(RUNNER.is_file())


if __name__ == "__main__":
    unittest.main()
