import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.cnn_ctc import load_spec, model_resource_estimate
from speech_asr.orchestration import (
    V19_ARCHITECTURE,
    compatibility_probe_command,
    pretraining_myriad_compatibility,
    training_command,
    validate_model_executor_request,
)

SPEC = SPEECH / "models" / "cnn_ctc_v19" / "model_spec.json"


def experiment_model():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v19",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v3"},
        "architecture": dict(V19_ARCHITECTURE),
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 12,
        "batch_size": 8,
        "max_samples": None,
        "checkpoint_selection": "validation_cer",
        "optimizer": {"kind": "novograd", "learning_rate": 0.0001},
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


class CnnCtcV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(SPEC)

    def test_inference_contract_is_unchanged_from_v18(self):
        network = self.spec["network"]
        estimate = model_resource_estimate(self.spec)
        self.assertEqual(self.spec["input_contract"]["shape"], [1, 64, 512])
        self.assertEqual(self.spec["output_contract"]["shape"], [1, 256, 39])
        self.assertEqual(network["kind"], "quartznet-15x5-v1")
        self.assertEqual(network["batchnorm_eps"], 0.001)
        self.assertEqual(network["dropout"], 0)
        self.assertEqual(estimate["parameters"], 18934631)
        self.assertEqual(estimate["macs_fixed_input"], 4827463680)
        self.assertEqual(estimate["receptive_field_feature_frames"], 8057)
        self.assertEqual(estimate["output_frames"], 256)

    def test_conservative_fine_tuning_recipe_is_frozen(self):
        training = self.spec["training"]
        self.assertEqual(training["batch_size"], 8)
        self.assertEqual(training["learning_rate"], 0.0001)
        self.assertEqual(training["optimizer"], "novograd")
        self.assertEqual(training["optimizer_betas"], [0.95, 0.25])
        self.assertEqual(training["weight_decay"], 0.001)
        self.assertEqual(training["lr_schedule"], "cosine")
        self.assertEqual(training["warmup_ratio"], 0)
        self.assertEqual(training["min_learning_rate"], 1e-6)
        self.assertTrue(training["freeze_batchnorm_running_stats"])
        self.assertTrue(training["initial_validation_checkpoint_candidate"])
        self.assertEqual(training["checkpoint_selection"], "best_validation_cer")

    def test_request_and_commands_are_registered(self):
        model = experiment_model()
        config = train_config()
        validate_model_executor_request(model, config)
        train = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v19"),
            train_config=config,
            model_spec=model,
        )
        self.assertEqual(train[0], "/repo/scripts/train-cnn-ctc-v19.sh")
        self.assertIn("--pretrained", train)
        probe = compatibility_probe_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/cnn_ctc_v19"),
            model_spec=model,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe[0], "/repo/scripts/probe-cnn-ctc-v19.sh")

    def test_initializer_freezes_v18_failure_and_epoch_zero_evidence(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v19_conservative_transfer.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-7c7ac81bc57f8f3b"', source)
        self.assertIn(
            'MODEL_SPEC_SHA256 = "e8fceef8e30955782efeab6e732319c051998e061f58502bc66d39f71485d330"',
            source,
        )
        self.assertIn("V18_INITIAL_VALIDATION_CER = 0.5776651500316765", source)
        self.assertIn("V18_INITIAL_VALIDATION_WER = 0.7393651479162168", source)
        self.assertIn("V18_BEST_VALIDATION_CER = 0.9003628405229511", source)
        self.assertIn("V18_PHYSICAL_CER = 0.8975407475666648", source)
        self.assertIn("V18_PRETRAINING_MYRIAD_MAX_FRAME_TV = 0.0009230391151051188", source)
        self.assertIn('"batch_size": 8', source)
        self.assertIn('"learning_rate": 0.0001', source)

    def test_trainer_freezes_bn_stats_and_allows_epoch_zero_selection(self):
        source = (
            SPEECH / "training" / "train_cnn_ctc_v19.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def freeze_batchnorm_running_stats(", source)
        self.assertIn("module.eval()", source)
        self.assertIn("model.train()", source)
        self.assertIn(
            "frozen_batchnorm_modules = freeze_batchnorm_running_stats(model)",
            source,
        )
        self.assertNotIn("requires_grad_(False)", source)
        self.assertIn("best_epoch = 0", source)
        self.assertIn(
            'best_validation_cer = float(initial_validation["cer"])',
            source,
        )
        self.assertIn(
            'selected_metric_value = (',
            source,
        )
        self.assertIn('"freeze_batchnorm_running_stats": freeze_bn_stats', source)
        self.assertIn('"initial_validation_checkpoint_candidate": True', source)

    def test_myriad_evaluator_restarts_on_pipe_io_failures(self):
        source = (
            SPEECH / "evaluation" / "evaluate_cnn_ctc_v19.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "except (BrokenPipeError, OSError, ValueError) as exc:",
            source,
        )
        self.assertIn(
            "persistent MYRIAD server input pipe write failed:",
            source,
        )
        self.assertIn(
            "persistent MYRIAD server output pipe read failed:",
            source,
        )
        self.assertIn("except RuntimeError as exc:", source)
        self.assertIn("MAX_SERVER_RESTARTS = 4", source)
        self.assertIn('"--progress-interval"', source)
        self.assertIn("default=25", source)
        self.assertIn("percent=", source)
        self.assertIn("rate=", source)
        self.assertIn("eta=", source)
        self.assertIn("restarts=", source)
        self.assertIn("def write_f32_file_exact(", source)
        self.assertIn("os.replace(temporary, path)", source)
        self.assertNotIn(".tofile(", source)

    def test_controller_streams_edge_worker_output_live(self):
        source = (
            SPEECH / "agent" / "run_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("stream_output: bool = False", source)
        self.assertIn("stream_output=stream_output", source)
        self.assertIn(
            'log_path=logs_dir / "edge-worker-ssh.log",\n'
            '        stream_output=True,',
            source,
        )

    def test_v19_uses_pretrained_semantic_myriad_gate(self):
        result = pretraining_myriad_compatibility(
            {
                "max_abs_error": 0.7036104202270508,
                "frame_argmax_agreement": 1.0,
                "frame_argmax_mismatches": 0,
                "min_reference_top2_margin": 5.668647766113281,
                "max_softmax_abs_error": 0.0008063222413164928,
                "max_frame_total_variation": 0.0009230391151051188,
            },
            model_id="cnn_ctc_v19",
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["gate_kind"], "pretrained-semantic-parity-v1")
        self.assertEqual(result["max_frame_total_variation_limit"], 0.002)

    def test_entrypoints_exist(self):
        for name in (
            "prepare-cnn-ctc-v19-pretrained.sh",
            "init-cnn-ctc-v19-conservative-transfer.sh",
            "train-cnn-ctc-v19.sh",
            "probe-cnn-ctc-v19.sh",
            "prepare-cnn-ctc-v19.sh",
            "evaluate-cnn-ctc-v19.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
