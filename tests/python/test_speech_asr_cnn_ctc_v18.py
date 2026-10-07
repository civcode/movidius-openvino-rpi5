import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V18_ARCHITECTURE,
    compatibility_probe_command,
    pretraining_myriad_compatibility,
    training_command,
    validate_model_executor_request,
)

SPEC = SPEECH / "models" / "cnn_ctc_v18" / "model_spec.json"


def experiment_model():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v18",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v3"},
        "architecture": dict(V18_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 12,
        "batch_size": 1,
        "max_samples": None,
        "checkpoint_selection": "validation_cer",
        "optimizer": {"kind": "novograd", "learning_rate": 0.001},
        "pretrained_source": {
            "path": "work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo",
            "size_bytes": 71083664,
            "sha384": "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55",
        },
        "training_manifest": {
            "id": "ami-model-quality-v4-architecture-screen-v1-train",
            "path": "work/speech-asr/ami/model-quality-v4-architecture-screen-v1/train.manifest.jsonl",
            "sha256": "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b",
        },
        "validation_manifest": {
            "id": "ami-model-quality-v4-es2011-validation",
            "path": "work/speech-asr/ami/model-quality-v4/validation.manifest.jsonl",
            "sha256": "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088",
        },
    }


class CnnCtcV18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC)

    def test_pretrained_quartznet_contract(self):
        network = self.spec["network"]
        frontend = self.spec["frontend"]
        transfer = self.spec["transfer"]
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(network["kind"], "quartznet-15x5-v1")
        self.assertEqual(network["dropout"], 0)
        self.assertEqual(network["batchnorm_eps"], 0.001)
        self.assertEqual(frontend["kind"], "logmel-v3")
        self.assertEqual(frontend["window_samples"], 320)
        self.assertEqual(frontend["preemphasis"], 0.97)
        self.assertEqual(frontend["mel_scale"], "slaney")
        self.assertEqual(
            frontend["normalization"],
            "per_mel_bin_mean_std_valid_zero_pad",
        )
        self.assertEqual(
            transfer["source_sha384"],
            "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55",
        )
        self.assertEqual(transfer["source_size_bytes"], 71083664)
        self.assertEqual(transfer["fine_tune"], "all_parameters")
        self.assertEqual(len(transfer["source_vocab"]), 29)
        self.assertEqual(estimate["parameters"], 18934631)
        self.assertEqual(estimate["macs_fixed_input"], 4827463680)
        self.assertEqual(estimate["receptive_field_feature_frames"], 8057)
        self.assertEqual(estimate["output_frames"], 256)

    def test_transfer_training_recipe_is_frozen(self):
        training = self.spec["training"]
        self.assertEqual(training["optimizer"], "novograd")
        self.assertEqual(training["learning_rate"], 0.001)
        self.assertEqual(training["optimizer_betas"], [0.95, 0.25])
        self.assertEqual(training["weight_decay"], 0.001)
        self.assertEqual(training["warmup_ratio"], 0.12)
        self.assertEqual(training["min_learning_rate"], 1e-6)
        self.assertEqual(training["checkpoint_selection"], "best_validation_cer")
        self.assertEqual(training["epochs"], 12)

    def test_request_and_commands_are_registered(self):
        model = experiment_model()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v18"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v18.sh")
        self.assertIn("--pretrained", train)
        self.assertIn("/repo/work/speech-asr/pretrained/quartznet15x5-en-base-v2/QuartzNet15x5-En-Base.nemo", train)
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v18"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v18.sh")

    def test_initializer_freezes_v17_and_pretrained_source(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v18_pretrained_transfer.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-dbcda9a6f7ae7d7f"', source)
        self.assertIn(
            'MODEL_SPEC_SHA256 = "008416f73aa2f90e1eb7b60cb2021175cbc0826b09810aed9aebbe41fe6f422c"',
            source,
        )
        self.assertIn("V17_PHYSICAL_CER = 0.9365892990842596", source)
        self.assertIn("PRETRAINED_SIZE = 71083664", source)
        self.assertIn("PRETRAINED_SHA384 =", source)
        self.assertIn("sha384_path(PRETRAINED)", source)
        self.assertIn("PRETRAINED_IMPORT", source)
        self.assertIn('"speech-asr/pretrained-import-verification"', source)
        self.assertIn('"kind": "novograd"', source)
        self.assertIn('"learning_rate": 0.001', source)
        self.assertIn('"max_cer": None', source)
        self.assertIn('"min_frame_argmax_agreement": 1.0', source)

    def test_trainer_requires_and_records_pretrained_initialization(self):
        trainer = (
            SPEECH / "training" / "train_cnn_ctc_v18.py"
        ).read_text(encoding="utf-8")
        importer = (
            SPEECH / "training" / "pretrained_cnn_ctc_v18.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            SPEECH / "evaluation" / "evaluate_cnn_ctc_v18.py"
        ).read_text(encoding="utf-8")
        controller = (
            SPEECH / "agent" / "run_experiment.py"
        ).read_text(encoding="utf-8")
        verifier = (
            SPEECH / "training" / "verify_cnn_ctc_v18_pretrained.py"
        ).read_text(encoding="utf-8")
        prepare = (
            ROOT / "scripts" / "prepare-cnn-ctc-v18-pretrained.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("load_pretrained_quartznet(", trainer)
        self.assertIn('"pretrained_initialization"', trainer)
        self.assertIn('"initial_validation"', trainer)
        self.assertIn("verify_source(archive)", importer)
        self.assertIn("SOURCE_SHA384", importer)
        self.assertIn("shared_decoder_symbols", importer)
        self.assertIn("shared_decoder_symbol_count", verifier)
        self.assertIn("encoder_source_tensors_used", verifier)
        self.assertIn("torch.isfinite(logits).all()", verifier)
        self.assertIn("verify_cnn_ctc_v18_pretrained.py", prepare)
        self.assertIn("write_json(\n        evidence_path", controller)
        self.assertIn("frame_argmax_mismatches=", controller)
        self.assertIn("max_mismatched_reference_top2_margin=", controller)
        self.assertIn("MAX_SERVER_RESTARTS = 4", evaluator)
        self.assertIn("infer_with_restart(", evaluator)

    def test_pretrained_myriad_gate_accepts_observed_semantic_parity(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.7036104202270508,
                "frame_argmax_agreement": 1.0,
                "frame_argmax_mismatches": 0,
                "min_reference_top2_margin": 5.668647766113281,
                "max_softmax_abs_error": 0.0008063222413164928,
                "max_frame_total_variation": 0.0009230391151051188,
            },
            model_id="cnn_ctc_v18",
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(
            result["gate_kind"],
            "pretrained-semantic-parity-v1",
        )
        self.assertIsNone(result["max_abs_error_limit"])
        self.assertEqual(
            result["max_frame_total_variation_limit"],
            0.002,
        )

    def test_pretrained_myriad_gate_rejects_any_argmax_change(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.001,
                "frame_argmax_agreement": 255 / 256,
                "frame_argmax_mismatches": 1,
                "min_reference_top2_margin": 0.01,
                "max_softmax_abs_error": 0.0001,
                "max_frame_total_variation": 0.0002,
            },
            model_id="cnn_ctc_v18",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(
            any("frame_argmax" in reason for reason in result["reasons"])
        )

    def test_pretrained_myriad_gate_rejects_excess_probability_drift(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.001,
                "frame_argmax_agreement": 1.0,
                "frame_argmax_mismatches": 0,
                "min_reference_top2_margin": 1.0,
                "max_softmax_abs_error": 0.001,
                "max_frame_total_variation": 0.0021,
            },
            model_id="cnn_ctc_v18",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(
            any("total_variation" in reason for reason in result["reasons"])
        )

    def test_entrypoints_exist(self):
        for name in (
            "prepare-cnn-ctc-v18-pretrained.sh",
            "init-cnn-ctc-v18-pretrained-transfer.sh",
            "train-cnn-ctc-v18.sh",
            "probe-cnn-ctc-v18.sh",
            "prepare-cnn-ctc-v18.sh",
            "evaluate-cnn-ctc-v18.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
