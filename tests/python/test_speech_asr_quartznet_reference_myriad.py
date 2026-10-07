import json
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
TRAINING = SPEECH / "training"
PYTHON = SPEECH / "python"
sys.path.insert(0, str(PYTHON))
sys.path.insert(0, str(TRAINING))

from quartznet15x5_reference import nemo_reference_features
from speech_asr.quartznet_reference_frontend import (
    nemo_reference_features_numpy,
    reference_feature_lengths,
)

SPEC_PATH = SPEECH / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
EVALUATOR = SPEECH / "evaluation" / "evaluate_quartznet15x5_reference_myriad.py"
CONTROLLER = SPEECH / "evaluation" / "run_quartznet15x5_reference_myriad_edge.py"
EDGE_RUNNER = ROOT / "scripts" / "evaluate-quartznet15x5-reference-myriad.sh"

# NumPy and torch use different FFT implementations across host architectures.
# Keep this as a strict numerical smoke bound; the authoritative semantic guard
# is the full 2,703-utterance WER requalification in the deployment controller.
MAX_NUMPY_TORCH_FEATURE_ABS_ERROR = 2e-4


class QuartzNetReferenceMyriadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.evaluator = EVALUATOR.read_text(encoding="utf-8")
        cls.controller = CONTROLLER.read_text(encoding="utf-8")
        cls.edge_runner = EDGE_RUNNER.read_text(encoding="utf-8")

    def test_numpy_frontend_matches_qualified_torch_frontend(self):
        generator = np.random.default_rng(1337)
        for sample_count in (16000, 82160, 123456):
            samples = generator.normal(0.0, 0.08, sample_count).astype(np.float32)
            samples = np.clip(samples, -0.9, 0.9)
            torch_features, torch_valid = nemo_reference_features(
                samples,
                self.spec,
                device=torch.device("cpu"),
            )
            numpy_features, numpy_valid = nemo_reference_features_numpy(
                samples,
                self.spec,
            )
            self.assertEqual(numpy_valid, torch_valid)
            self.assertEqual(tuple(numpy_features.shape), tuple(torch_features.shape))
            max_abs = float(
                np.max(
                    np.abs(
                        numpy_features
                        - torch_features.detach().cpu().numpy()
                    )
                )
            )
            self.assertLessEqual(
                max_abs,
                MAX_NUMPY_TORCH_FEATURE_ABS_ERROR,
                f"sample_count={sample_count} max_abs={max_abs}",
            )

    def test_reference_feature_lengths_match_centered_stft_padding(self):
        valid, padded = reference_feature_lengths(82160, self.spec)
        self.assertEqual(valid, 513)
        self.assertEqual(padded, 528)
        self.assertEqual(padded % 16, 0)

    def test_myriad_qualification_keeps_source_contract(self):
        self.assertIn("EXPECTED_SAMPLES = 2703", self.evaluator)
        self.assertIn("QUALIFIED_WER = 0.037939781625675524", self.evaluator)
        self.assertIn("MAX_WER = 0.05", self.evaluator)
        self.assertIn("MAX_ABS_WER_DELTA = 0.005", self.evaluator)
        self.assertIn("MAX_FRAME_TOTAL_VARIATION = 0.002", self.evaluator)
        self.assertIn('EXECUTION_MODE = "exact-time-runtime-reshape-v1"', self.evaluator)
        self.assertIn("output_elements = output_frames * 29", self.evaluator)
        self.assertIn('"--reshape-time"', self.evaluator)
        self.assertIn('"training_performed": False', self.evaluator)

    def test_edge_runner_is_host_only_and_reshape_guarded(self):
        self.assertIn("--backend host", self.edge_runner)
        self.assertIn("check-reshape", self.edge_runner)
        self.assertIn("--runtime-backend host", self.edge_runner)
        self.assertIn("OMP_NUM_THREADS=1", self.edge_runner)
        self.assertNotIn("--backend docker", self.edge_runner)

    def test_controller_stages_and_validates_evidence(self):
        self.assertIn('"--refresh-runtime"', self.controller)
        self.assertIn("rsync_push_command", self.controller)
        self.assertIn("rsync_pull_command", self.controller)
        self.assertIn("qualify-quartznet15x5-reference-numpy.sh", self.controller)
        self.assertIn("prepare-quartznet15x5-reference-myriad.sh", self.controller)
        self.assertIn('myriad.get("runtime_backend") != "host"', self.controller)
        self.assertIn('"valid_comparison"', self.controller)
        self.assertIn("frame_argmax_agreement", self.controller)
        self.assertIn("max_frame_total_variation", self.controller)
        self.assertIn('"full_tensor_comparison"', self.evaluator)
        self.assertIn('"valid_comparison"', self.evaluator)


if __name__ == "__main__":
    unittest.main()
