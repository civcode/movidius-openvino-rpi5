import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
SPEC = SPEECH / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
VOCAB = SPEECH / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"


class QuartzNetReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC.read_text(encoding="utf-8"))
        cls.vocab = json.loads(VOCAB.read_text(encoding="utf-8"))

    def test_source_output_contract_is_exact_29_class_order(self):
        expected = [" "] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "<blank>"]
        self.assertEqual(self.vocab["tokens"], expected)
        self.assertEqual(self.vocab["blank_index"], 28)
        self.assertEqual(self.spec["output_contract"]["classes"], 29)
        self.assertEqual(self.spec["output_contract"]["blank_index"], 28)

    def test_reference_source_and_librispeech_target_are_frozen(self):
        self.assertEqual(
            self.spec["source"]["sha384"],
            "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55",
        )
        self.assertEqual(
            self.spec["dataset"]["md5"],
            "42e2234ba48799c1f50f24a7926300a1",
        )
        self.assertEqual(self.spec["dataset"]["expected_utterances"], 2703)
        self.assertEqual(
            self.spec["qualification"]["published_dev_clean_wer"],
            0.0379,
        )
        self.assertEqual(
            self.spec["qualification"]["reproduction_max_wer"],
            0.05,
        )

    def test_historical_nemo_frontend_semantics_are_explicit(self):
        frontend = self.spec["frontend"]
        self.assertTrue(frontend["stft_center"])
        self.assertFalse(frontend["hann_periodic"])
        self.assertEqual(frontend["stft_pad_mode"], "constant")
        self.assertEqual(frontend["normalization"], "per_feature")
        self.assertEqual(frontend["normalization_ddof"], 1)
        self.assertEqual(frontend["normalization_epsilon"], 1e-5)
        self.assertEqual(frontend["evaluation_dither"], 0)
        self.assertEqual(frontend["pad_to"], 16)

        source = (
            SPEECH / "training" / "quartznet15x5_reference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("periodic=bool(frontend[\"hann_periodic\"])", source)
        self.assertIn("center=bool(frontend[\"stft_center\"])", source)
        self.assertIn("pad_mode=str(frontend[\"stft_pad_mode\"])", source)
        self.assertIn("valid_frames = sample_count // hop", source)
        self.assertIn("/ float(valid_frames - 1)", source)
        self.assertIn("torch.sqrt(variance) +", source)

    def test_reference_import_has_no_random_target_only_head(self):
        source = (
            SPEECH / "training" / "quartznet15x5_reference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_pretrained_quartznet(", source)
        self.assertIn('imported["shared_decoder_symbol_count"] != 29', source)
        self.assertIn('imported["target_only_symbol_count"] != 0', source)

    def test_qualification_is_zero_training_and_cpu_reference_only(self):
        qualification = self.spec["qualification"]
        self.assertFalse(qualification["training_allowed"])
        self.assertFalse(qualification["openvino_allowed"])
        self.assertFalse(qualification["myriad_allowed"])

        evaluator = (
            SPEECH / "evaluation" / "evaluate_quartznet15x5_reference.py"
        ).read_text(encoding="utf-8")
        runner = (
            ROOT / "scripts" / "qualify-quartznet15x5-reference.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"training_performed": False', evaluator)
        self.assertNotIn("optimizer", evaluator.lower())
        for forbidden in (
            "run-myriad-tensor.sh",
            "evaluate-cnn-ctc-v19-deployed.sh",
            "run-speech-experiment.sh",
            "prepare-cnn-ctc-v18.sh",
        ):
            self.assertNotIn(forbidden, runner)
        self.assertIn("eval_status=$?", runner)
        self.assertIn('"$eval_status" -ne 0 && "$eval_status" -ne 3', runner)
        self.assertIn('exit "$eval_status"', runner)

    def test_entrypoints_exist(self):
        for name in (
            "prepare-librispeech-dev-clean.sh",
            "evaluate-quartznet15x5-reference.sh",
            "export-quartznet15x5-reference.sh",
            "compare-quartznet15x5-reference-onnx.sh",
            "qualify-quartznet15x5-reference.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
