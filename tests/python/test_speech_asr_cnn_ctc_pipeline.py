import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class CnnCtcPipelineSourceTests(unittest.TestCase):
    def test_training_environment_is_uv_managed(self):
        prepare = (ROOT / "scripts" / "prepare-python-env.sh").read_text(encoding="utf-8")
        self.assertIn("work/venv-training", prepare)
        self.assertIn("requirements/training.txt", prepare)

    def test_probe_orders_compatibility_before_device(self):
        probe = (ROOT / "scripts" / "probe-cnn-ctc-v1.sh").read_text(encoding="utf-8")
        self.assertLess(
            probe.index("compare ONNX Runtime"),
            probe.index("MYRIAD load/compile/infer"),
        )
        self.assertIn("--init-only", probe)
        self.assertIn("--no-device", probe)

    def test_full_pipeline_exports_before_mo(self):
        pipeline = (ROOT / "scripts" / "train-cnn-ctc-v1.sh").read_text(encoding="utf-8")
        self.assertLess(
            pipeline.index("export fixed-shape ONNX"),
            pipeline.index("convert OpenVINO"),
        )
        self.assertIn("--device", pipeline)

    def test_myriad_evaluator_uses_standard_result_contract(self):
        source = (
            ROOT / "examples" / "speech-asr" / "evaluation" / "evaluate_cnn_ctc_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"speech-asr/experiment-result"', source)
        self.assertIn("validate_experiment_result", source)
        self.assertIn('"backend": "MYRIAD"', source)


if __name__ == "__main__":
    unittest.main()
