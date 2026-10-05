import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.contracts import (
    ContractValidationError,
    validate_acceptance_policy,
    validate_deployment_manifest,
    validate_experiment_attempt,
    validate_experiment_model_spec,
    validate_experiment_proposal,
    validate_experiment_request,
    validate_experiment_summary,
    validate_train_config,
)
from speech_asr.experiment import (
    history_document,
    make_attempt,
    make_deployment_manifest,
    make_experiment_request,
    make_summary,
    request_sha256,
    set_artifact,
    set_stage_result,
    transition_attempt,
    validate_request_identity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40


def proposal():
    return {
        "schema": "speech-asr/experiment-proposal",
        "version": 1,
        "title": "wider temporal encoder",
        "hypothesis": "more channels improve CTC fit without breaking MYRIAD",
        "rationale": "Phase 8 established the fixed deployment lifecycle.",
        "changes": ["increase encoder channels"],
    }


def model_spec():
    return {
        "schema": "speech-asr/experiment-model-spec",
        "version": 1,
        "model_id": "cnn_ctc_v2",
        "family": "cnn_ctc",
        "frontend": {"kind": "logmel-v1", "mel_bins": 64},
        "architecture": {"kind": "temporal-conv", "channels": [64, 64, 80]},
        "export": {"format": "onnx", "onnx_opset": 11, "fixed_shapes": True},
    }


def train_config():
    return {
        "schema": "speech-asr/train-config",
        "version": 1,
        "seed": 1337,
        "device": "cuda",
        "epochs": 1,
        "batch_size": 2,
        "max_samples": None,
        "optimizer": {"kind": "adam", "learning_rate": 0.001},
        "training_manifest": {
            "id": "ami-smoke-v1",
            "path": "work/speech-asr/ami/ami-smoke-v1/manifest.jsonl",
            "sha256": SHA_A,
        },
        "validation_manifest": {
            "id": "ami-smoke-v1",
            "path": "work/speech-asr/ami/ami-smoke-v1/manifest.jsonl",
            "sha256": SHA_A,
        },
    }


def acceptance():
    return {
        "schema": "speech-asr/acceptance-policy",
        "version": 1,
        "required_gates": [
            "training",
            "onnx_export",
            "openvino_conversion",
            "myriad_execution",
            "accuracy_evaluation",
        ],
        "thresholds": {
            "max_wer": None,
            "max_cer": None,
            "max_realtime_factor": 1.0,
            "max_latency_p95_ms": None,
            "min_frame_argmax_agreement": None,
        },
        "retry_policy": {
            "max_attempts": 2,
            "retryable_failure_classes": [
                "transport_preflight",
                "worker_busy",
                "hardware_transient",
            ],
        },
    }


def request():
    return make_experiment_request(
        proposal=proposal(),
        model_spec=model_spec(),
        train_config=train_config(),
        acceptance=acceptance(),
        repo_commit=COMMIT,
        benchmark_id="ami-smoke-v1",
        manifest_path="work/speech-asr/ami/ami-smoke-v1/manifest.jsonl",
        manifest_sha256=SHA_A,
    )


class ExperimentContractTests(unittest.TestCase):
    def test_reviewed_documents_validate(self):
        self.assertEqual(validate_experiment_proposal(proposal())["version"], 1)
        self.assertEqual(validate_experiment_model_spec(model_spec())["model_id"], "cnn_ctc_v2")
        self.assertEqual(validate_train_config(train_config())["device"], "cuda")
        self.assertEqual(validate_acceptance_policy(acceptance())["version"], 1)

    def test_request_identity_is_deterministic(self):
        first = request()
        second = request()
        self.assertEqual(first, second)
        self.assertRegex(first["experiment_id"], r"^exp-[0-9a-f]{16}$")
        self.assertEqual(validate_request_identity(first)["experiment_id"], first["experiment_id"])
        self.assertEqual(validate_experiment_request(first)["state"], "APPROVED")

    def test_training_change_creates_new_experiment(self):
        first = request()
        changed = train_config()
        changed["epochs"] = 2
        second = make_experiment_request(
            proposal=proposal(),
            model_spec=model_spec(),
            train_config=changed,
            acceptance=acceptance(),
            repo_commit=COMMIT,
            benchmark_id="ami-smoke-v1",
            manifest_path="work/speech-asr/ami/ami-smoke-v1/manifest.jsonl",
            manifest_sha256=SHA_A,
        )
        self.assertNotEqual(first["experiment_id"], second["experiment_id"])
        self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])

    def test_identity_tampering_is_rejected(self):
        document = request()
        document["benchmark"]["id"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity_sha256 mismatch"):
            validate_request_identity(document)


class ExperimentAttemptTests(unittest.TestCase):
    def test_successful_state_machine(self):
        req = request()
        attempt = make_attempt(req, index=1, at_utc="2026-10-05T10:00:00Z")
        self.assertEqual(attempt["state"], "APPROVED")
        self.assertEqual(attempt["request_sha256"], request_sha256(req))

        attempt = transition_attempt(
            attempt, "EXECUTE", at_utc="2026-10-05T10:01:00Z"
        )
        attempt = set_artifact(
            attempt,
            "checkpoint",
            path="work/checkpoint.pt",
            sha256=SHA_A,
        )
        attempt = set_stage_result(
            attempt,
            "training",
            path="attempts/attempt-0001/results/training.json",
            sha256=SHA_B,
        )
        attempt = transition_attempt(
            attempt, "EVALUATE", at_utc="2026-10-05T10:02:00Z"
        )
        attempt = set_artifact(
            attempt,
            "openvino_xml",
            path="work/model.xml",
            sha256=SHA_A,
        )
        attempt = set_artifact(
            attempt,
            "openvino_bin",
            path="work/model.bin",
            sha256=SHA_B,
        )
        attempt = transition_attempt(
            attempt,
            "AWAIT_REVIEW",
            outcome="completed",
            at_utc="2026-10-05T10:03:00Z",
        )
        self.assertEqual(validate_experiment_attempt(attempt)["outcome"], "completed")
        summary = make_summary(attempt)
        self.assertEqual(validate_experiment_summary(summary)["review_state"], "AWAIT_REVIEW")
        self.assertEqual(summary["artifacts"]["checkpoint"]["sha256"], SHA_A)

    def test_completed_attempt_must_pass_evaluate(self):
        attempt = make_attempt(request(), index=1, at_utc="2026-10-05T10:00:00Z")
        attempt = transition_attempt(
            attempt, "EXECUTE", at_utc="2026-10-05T10:01:00Z"
        )
        with self.assertRaisesRegex(ValueError, "must pass through EVALUATE"):
            transition_attempt(
                attempt,
                "AWAIT_REVIEW",
                outcome="completed",
                at_utc="2026-10-05T10:02:00Z",
            )

    def test_failed_attempt_requires_failure_class(self):
        attempt = make_attempt(request(), index=1, at_utc="2026-10-05T10:00:00Z")
        attempt = transition_attempt(
            attempt, "EXECUTE", at_utc="2026-10-05T10:01:00Z"
        )
        with self.assertRaisesRegex(ValueError, "requires failure_class"):
            transition_attempt(
                attempt,
                "AWAIT_REVIEW",
                outcome="failed",
                at_utc="2026-10-05T10:02:00Z",
            )

    def test_artifact_reference_is_immutable(self):
        attempt = make_attempt(request(), index=1, at_utc="2026-10-05T10:00:00Z")
        attempt = transition_attempt(
            attempt, "EXECUTE", at_utc="2026-10-05T10:01:00Z"
        )
        attempt = set_artifact(
            attempt, "onnx", path="work/model.onnx", sha256=SHA_A
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            set_artifact(
                attempt, "onnx", path="work/model.onnx", sha256=SHA_B
            )


class DeploymentAndHistoryTests(unittest.TestCase):
    def test_deployment_manifest_binds_request_attempt_and_artifacts(self):
        req = request()
        attempt = make_attempt(req, index=1, at_utc="2026-10-05T10:00:00Z")
        attempt = transition_attempt(
            attempt, "EXECUTE", at_utc="2026-10-05T10:01:00Z"
        )
        attempt = set_artifact(
            attempt, "openvino_xml", path="work/model.xml", sha256=SHA_A
        )
        attempt = set_artifact(
            attempt, "openvino_bin", path="work/model.bin", sha256=SHA_B
        )
        attempt = transition_attempt(
            attempt, "EVALUATE", at_utc="2026-10-05T10:02:00Z"
        )
        manifest = make_deployment_manifest(
            request=req,
            attempt=attempt,
            worker_commit=COMMIT,
            model_id=model_spec()["model_id"],
            model_spec_sha256=req["documents"]["model_spec"]["sha256"],
            evaluator_id="cnn-ctc-myriad-v1",
            result_path="attempt/results/hardware.json",
            artifacts={
                "openvino_xml": attempt["artifacts"]["openvino_xml"],
                "openvino_bin": attempt["artifacts"]["openvino_bin"],
            },
        )
        self.assertEqual(validate_deployment_manifest(manifest)["runtime"]["target"], "arm64")

    def test_history_is_sorted(self):
        entries = [
            {
                "experiment_id": "exp-ffffffffffffffff",
                "parent_experiment_id": None,
                "request_sha256": SHA_A,
                "path": "exp-ffffffffffffffff",
                "latest_attempt_id": None,
                "latest_outcome": None,
            },
            {
                "experiment_id": "exp-0000000000000000",
                "parent_experiment_id": None,
                "request_sha256": SHA_B,
                "path": "exp-0000000000000000",
                "latest_attempt_id": "attempt-0001",
                "latest_outcome": "completed",
            },
        ]
        history = history_document(entries)
        self.assertEqual(
            [value["experiment_id"] for value in history["experiments"]],
            ["exp-0000000000000000", "exp-ffffffffffffffff"],
        )


if __name__ == "__main__":
    unittest.main()
