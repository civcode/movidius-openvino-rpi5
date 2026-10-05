import hashlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.contracts import ContractValidationError, validate_edge_worker_result

from speech_asr.orchestration import (
    acceptance_evaluation,
    remote_failure_class,
    rsync_pull_command,
    rsync_push_command,
    ssh_command,
    training_command,
    validate_cnn_ctc_v1_executor_request,
)


def model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v1",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1"},
        "architecture": {"kind": "cnn_ctc_v1"},
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    sha = "a" * 64
    ref = {"id": "ami-smoke-v1", "path": "work/manifest.jsonl", "sha256": sha}
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 2,
        "batch_size": 3,
        "max_samples": 7,
        "optimizer": {"kind": "adam", "learning_rate": 0.0005},
        "training_manifest": dict(ref),
        "validation_manifest": dict(ref),
    }


class Phase10ExecutorTests(unittest.TestCase):
    def test_frozen_cnn_ctc_v1_request_is_supported(self):
        validate_cnn_ctc_v1_executor_request(model_spec(), train_config())

    def test_executor_rejects_ignored_architecture_fields(self):
        spec = model_spec()
        spec["architecture"]["channels"] = [999]
        with self.assertRaisesRegex(ValueError, "must not ignore architecture"):
            validate_cnn_ctc_v1_executor_request(spec, train_config())

    def test_training_command_propagates_declared_config(self):
        command = training_command(
            root=pathlib.Path("/repo"),
            build_dir=pathlib.Path("/repo/work/attempt/build"),
            train_config=train_config(),
        )
        text = " ".join(command)
        self.assertIn("--seed 1337", text)
        self.assertIn("--epochs 2", text)
        self.assertIn("--batch-size 3", text)
        self.assertIn("--learning-rate 0.0005", text)
        self.assertIn("--max-samples 7", text)
        self.assertIn("--work-dir /repo/work/attempt/build", text)

    def test_acceptance_rejection_does_not_change_execution_status(self):
        policy = {
            "required_gates": [
                "training",
                "onnx_export",
                "openvino_conversion",
                "myriad_execution",
                "accuracy_evaluation",
            ],
            "thresholds": {
                "max_wer": 0.5,
                "max_cer": None,
                "max_realtime_factor": 1.0,
                "max_latency_p95_ms": None,
                "min_frame_argmax_agreement": 1.0,
            },
        }
        compatibility = {
            "status": "completed",
            "ir_validation": {"status": "valid"},
            "onnx_comparison": {
                "comparison": {"frame_argmax_agreement": 1.0}
            },
        }
        hardware = {
            "status": "completed",
            "runtime": {"backend": "MYRIAD"},
            "metrics": {
                "wer": 1.0,
                "cer": 0.9,
                "realtime_factor": 0.1,
                "inference_latency_p95_ms": 5.0,
            },
        }
        result = acceptance_evaluation(
            policy=policy,
            training_present=True,
            compatibility=compatibility,
            hardware=hardware,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(any("WER" in reason for reason in result["reasons"]))

    def test_acceptance_passes_when_declared_thresholds_pass(self):
        policy = {
            "required_gates": ["training", "myriad_execution"],
            "thresholds": {
                "max_wer": None,
                "max_cer": None,
                "max_realtime_factor": 1.0,
                "max_latency_p95_ms": None,
                "min_frame_argmax_agreement": 1.0,
            },
        }
        compatibility = {
            "status": "completed",
            "ir_validation": {"status": "valid"},
            "onnx_comparison": {
                "comparison": {"frame_argmax_agreement": 1.0}
            },
        }
        hardware = {
            "status": "completed",
            "runtime": {"backend": "MYRIAD"},
            "metrics": {
                "wer": 1.0,
                "cer": 0.9,
                "realtime_factor": 0.01,
                "inference_latency_p95_ms": 5.0,
            },
        }
        result = acceptance_evaluation(
            policy=policy,
            training_present=True,
            compatibility=compatibility,
            hardware=hardware,
        )
        self.assertEqual(result["status"], "accepted")


class Phase10TransportTests(unittest.TestCase):
    def test_ssh_is_noninteractive_without_disabling_host_key_checks(self):
        command = ssh_command("edge", ["true"])
        text = " ".join(command)
        self.assertIn("BatchMode=yes", text)
        self.assertIn("ConnectTimeout=10", text)
        self.assertNotIn("StrictHostKeyChecking=no", text)

    def test_worker_alias_cannot_be_an_ssh_option(self):
        with self.assertRaisesRegex(ValueError, "worker alias"):
            ssh_command("-oProxyCommand=bad", ["true"])
        with self.assertRaisesRegex(ValueError, "worker alias"):
            ssh_command("edge;touch-bad", ["true"])

    def test_rsync_uses_same_noninteractive_ssh_policy(self):
        push = rsync_push_command(
            worker="edge",
            sources=[pathlib.Path("/tmp/model.xml")],
            remote_dir="/tmp/incoming",
        )
        pull = rsync_pull_command(
            worker="edge",
            remote_dir="/tmp/output",
            local_dir=pathlib.Path("/tmp/local"),
        )
        for command in (push, pull):
            text = " ".join(command)
            self.assertIn("BatchMode=yes", text)
            self.assertIn("ConnectTimeout=10", text)
            self.assertNotIn("StrictHostKeyChecking=no", text)

    def test_worker_exit_codes_have_stable_failure_classes(self):
        self.assertEqual(remote_failure_class(75), ("blocked", "worker_busy"))
        self.assertEqual(
            remote_failure_class(20),
            ("blocked", "transport_preflight"),
        )
        self.assertEqual(
            remote_failure_class(21),
            ("failed", "hardware_execution"),
        )
        self.assertEqual(
            remote_failure_class(22),
            ("failed", "result_contract"),
        )


class EdgeWorkerResultContractTests(unittest.TestCase):
    def test_completed_worker_result_validates(self):
        sha = "a" * 64
        document = {
            "schema": "speech-asr/edge-worker-result",
            "version": 1,
            "status": "completed",
            "failure_class": None,
            "experiment_id": "exp-0123456789abcdef",
            "attempt_id": "attempt-0001",
            "worker_commit": "b" * 40,
            "deployment_sha256": sha,
            "hardware_result": {"path": "hardware.json", "sha256": sha},
            "evaluator_log": {"path": "evaluator.log", "sha256": sha},
            "metrics": {"wer": 1.0},
        }
        self.assertEqual(
            validate_edge_worker_result(document)["status"],
            "completed",
        )

    def test_failed_worker_result_requires_failure_class(self):
        document = {
            "schema": "speech-asr/edge-worker-result",
            "version": 1,
            "status": "failed",
            "failure_class": None,
            "experiment_id": None,
            "attempt_id": None,
            "worker_commit": None,
            "deployment_sha256": None,
            "diagnostics": {"summary": "preflight failed"},
        }
        with self.assertRaisesRegex(ContractValidationError, "requires class"):
            validate_edge_worker_result(document)


class EdgeWorkerBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SPEECH / "agent" / "edge_worker.py"
        spec = importlib.util.spec_from_file_location("edge_worker", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load edge worker")
        module = importlib.util.module_from_spec(spec)
        sys.modules["edge_worker"] = module
        spec.loader.exec_module(module)
        cls.worker = module

    def test_bundle_verification_accepts_exact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            artifact = root / "cnn_ctc_v1.xml"
            artifact.write_bytes(b"xml")
            digest = hashlib.sha256(b"xml").hexdigest()
            manifest = {
                "artifacts": {
                    "openvino_xml": {
                        "path": "work/model/cnn_ctc_v1.xml",
                        "sha256": digest,
                    }
                }
            }
            result = self.worker.verify_bundle(manifest, root)
            self.assertEqual(result["openvino_xml"], artifact)

    def test_bundle_verification_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            artifact = root / "cnn_ctc_v1.xml"
            artifact.write_bytes(b"xml")
            manifest = {
                "artifacts": {
                    "openvino_xml": {
                        "path": "work/model/cnn_ctc_v1.xml",
                        "sha256": "0" * 64,
                    }
                }
            }
            with self.assertRaisesRegex(Exception, "hash mismatch"):
                self.worker.verify_bundle(manifest, root)


class Phase10SourceInvariantTests(unittest.TestCase):
    def test_bootstrap_uses_dedicated_worktree_without_git_pull(self):
        text = (ROOT / "scripts" / "edge-speech-bootstrap.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("worktree add --detach", text)
        self.assertIn("fetch --quiet origin", text)
        self.assertNotIn("git pull", text)

    def test_worker_lock_is_nonblocking(self):
        text = (ROOT / "scripts" / "edge-speech-worker.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("flock -n", text)
        self.assertIn("worker_busy", text)

    def test_controller_does_not_rebuild_runtime_or_disable_host_key_checks(self):
        text = (
            SPEECH / "agent" / "run_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("build.sh --platform", text)
        self.assertNotIn("StrictHostKeyChecking=no", text)
        self.assertIn("AWAIT_REVIEW", text)

    def test_training_wrapper_supports_attempt_local_outputs(self):
        text = (ROOT / "scripts" / "train-cnn-ctc-v1.sh").read_text(
            encoding="utf-8"
        )
        for value in (
            "--work-dir",
            "--seed",
            "--learning-rate",
            "--batch-size",
        ):
            self.assertIn(value, text)

    def test_edge_evaluator_can_bind_phase9_identity(self):
        text = (
            SPEECH / "evaluation" / "evaluate_cnn_ctc_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--experiment-id", text)
        self.assertIn("--work-dir", text)


if __name__ == "__main__":
    unittest.main()
