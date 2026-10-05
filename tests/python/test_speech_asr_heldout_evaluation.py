import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
SPLIT = (
    SPEECH
    / "datasets"
    / "ami"
    / "splits"
    / "eval-full-corpus-asr-sc-v1.json"
)
QUALIFIER = SPEECH / "tools" / "qualify_heldout_evaluation_manifest.py"
SPEC = SPEECH / "models" / "cnn_ctc_v3" / "model_spec.json"
VOCAB = SPEECH / "models" / "cnn_ctc_v3" / "vocab.json"

EXPECTED_MEETINGS = {
    f"{prefix}{suffix}"
    for prefix in ("EN2002", "ES2004", "IS1009", "TS3003")
    for suffix in ("a", "b", "c", "d")
}


def load_qualifier():
    spec = importlib.util.spec_from_file_location(
        "qualify_heldout_evaluation",
        QUALIFIER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load held-out qualification tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(
    sample_id: str,
    meeting: str,
    speaker: str = "A",
    end_sample: int = 16000,
) -> dict:
    return {
        "schema": "speech-asr/sample",
        "version": 1,
        "id": sample_id,
        "audio": {
            "path": f"audio/{sample_id}.f32",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_type": "float32",
            "encoding": "f32le",
            "start_sample": 0,
            "end_sample": end_sample,
        },
        "transcript": {
            "text": "hello world",
            "normalization": "text-v1",
        },
        "metadata": {
            "corpus": "AMI",
            "meeting": meeting,
            "speaker": speaker,
        },
    }


def write_manifest(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class HeldoutSplitContractTests(unittest.TestCase):
    def test_split_is_official_full_corpus_asr_sc_boundary(self):
        value = json.loads(SPLIT.read_text(encoding="utf-8"))
        self.assertEqual(value["id"], "ami-eval-full-corpus-asr-sc-v1")
        self.assertEqual(value["partition"]["name"], "Full-corpus-ASR")
        self.assertEqual(
            value["partition"]["role"],
            "SC-unseen-evaluation",
        )
        self.assertFalse(
            value["partition"]["checkpoint_selection_allowed"]
        )
        self.assertFalse(value["partition"]["training_allowed"])
        meetings = {source["meeting"] for source in value["sources"]}
        self.assertEqual(meetings, EXPECTED_MEETINGS)
        self.assertEqual(len(value["sources"]), 16)
        for source in value["sources"]:
            self.assertEqual(source["audio"]["stream"], "Mix-Headset")
            self.assertEqual(len(source["audio"]["sha256"]), 64)
            self.assertEqual(
                [entry["speaker"] for entry in source["selections"]],
                ["A", "B", "C", "D"],
            )
            self.assertTrue(
                all(
                    entry.get("all_segments") is True
                    for entry in source["selections"]
                )
            )


class HeldoutQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_qualifier()

    def official_records(self) -> list[dict]:
        return [
            sample(f"eval-{meeting}", meeting)
            for meeting in sorted(EXPECTED_MEETINGS)
        ]

    def test_qualifier_emits_test_only_disjoint_manifest(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = pathlib.Path(temp_name)
            source = temp / "source" / "manifest.jsonl"
            train = temp / "train" / "manifest.jsonl"
            selection = temp / "selection" / "manifest.jsonl"
            write_manifest(source, self.official_records())
            write_manifest(
                train,
                [sample("train-1", "ES2005a")],
            )
            write_manifest(
                selection,
                [sample("selection-1", "ES2002a", "D")],
            )
            output = temp / "heldout"
            result = self.tool.qualify(
                source_manifest=source,
                train_manifest=train,
                selection_manifest=selection,
                output_dir=output,
                spec_path=SPEC,
                vocab_path=VOCAB,
            )
            self.assertEqual(result["status"], "structurally_valid")
            self.assertEqual(result["role"], "test_only")
            self.assertFalse(
                result["partition"]["checkpoint_selection_allowed"]
            )
            self.assertFalse(result["partition"]["training_allowed"])
            self.assertEqual(
                result["overlap"],
                {
                    "training_records": 0,
                    "selection_records": 0,
                    "training_meetings": 0,
                    "selection_meetings": 0,
                },
            )
            self.assertEqual(result["test"]["stats"]["records"], 16)
            self.assertTrue((output / "manifest.jsonl").is_file())
            self.assertTrue((output / "qualification.json").is_file())
            verified = self.tool.verify_existing(
                source_manifest=source,
                train_manifest=train,
                selection_manifest=selection,
                output_dir=output,
                spec_path=SPEC,
                vocab_path=VOCAB,
            )
            self.assertEqual(
                verified["test"]["manifest_sha256"],
                result["test"]["manifest_sha256"],
            )

    def test_qualifier_rejects_training_meeting_overlap(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = pathlib.Path(temp_name)
            source = temp / "source" / "manifest.jsonl"
            train = temp / "train" / "manifest.jsonl"
            selection = temp / "selection" / "manifest.jsonl"
            write_manifest(source, self.official_records())
            write_manifest(
                train,
                [sample("train-overlap", "ES2004a")],
            )
            write_manifest(
                selection,
                [sample("selection-1", "ES2002a", "D")],
            )
            with self.assertRaisesRegex(
                ValueError,
                "held-out boundary overlap detected",
            ):
                self.tool.qualify(
                    source_manifest=source,
                    train_manifest=train,
                    selection_manifest=selection,
                    output_dir=temp / "heldout",
                    spec_path=SPEC,
                    vocab_path=VOCAB,
                )


class FrozenHeldoutControllerSourceTests(unittest.TestCase):
    def test_controller_is_bound_to_frozen_accepted_source(self):
        source = (
            SPEECH / "agent" / "run_frozen_heldout_evaluation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'SOURCE_EXPERIMENT_ID = "exp-87538823d2bf1562"',
            source,
        )
        self.assertIn('SOURCE_ATTEMPT_ID = "attempt-0001"', source)
        self.assertIn('acceptance.get("status") != "accepted"', source)
        for artifact in (
            '"checkpoint"',
            '"onnx"',
            '"openvino_xml"',
            '"openvino_bin"',
        ):
            self.assertIn(artifact, source)

    def test_controller_cannot_train_or_reselect_checkpoint(self):
        source = (
            SPEECH / "agent" / "run_frozen_heldout_evaluation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("training_command(", source)
        self.assertNotIn("train-cnn", source)
        self.assertIn('"checkpoint_selection_allowed": False', source)
        self.assertIn('"training_allowed": False', source)
        self.assertIn('"repeat_for_model_selection": False', source)
        self.assertIn("held-out evaluation is already sealed", source)

    def test_controller_pins_edge_and_uses_hash_bound_ir(self):
        source = (
            SPEECH / "agent" / "run_frozen_heldout_evaluation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("edge-speech-bootstrap.sh", source)
        self.assertIn("validate_deployment_manifest", source)
        self.assertIn('dict(artifacts["openvino_xml"])', source)
        self.assertIn('dict(artifacts["openvino_bin"])', source)
        self.assertIn("frame_argmax_agreement", source)
        self.assertIn("qualify-speech-heldout-eval.sh", source)
        self.assertIn('"--verify-only"', source)

    def test_reference_failure_reports_agreement_diagnostics(self):
        source = (
            SPEECH / "evaluation" / "evaluate_frozen_cnn_ctc_v3_reference.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"frame_argmax_mismatches"', source)
        self.assertIn('"max_mismatched_reference_top2_margin"', source)
        self.assertIn('"mean_mismatched_reference_top2_margin"', source)
        self.assertIn('"mismatch_sample_examples"', source)
        self.assertIn('"decoded_hypothesis_mismatches"', source)
        self.assertIn('"decoded_hypothesis_mismatch_examples"', source)
        self.assertIn('"wer_delta"', source)
        self.assertIn('"cer_delta"', source)
        self.assertIn("json.dumps(agreement, sort_keys=True)", source)

    def test_reference_retry_can_reuse_validated_cpu_result(self):
        controller = (
            SPEECH / "agent" / "run_frozen_heldout_evaluation.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--reuse-reference"', controller)
        self.assertIn("validate_cached_reference(", controller)
        self.assertIn('"cached reference checkpoint hash mismatch"', controller)
        self.assertIn('"cached reference ONNX hash mismatch"', controller)
        self.assertIn('"cached reference manifest hash mismatch"', controller)
        self.assertIn('default="cpu"', controller)
        self.assertIn(
            '"held-out reference: reusing validated "',
            controller,
        )

    def test_reference_progress_is_streamed(self):
        controller = (
            SPEECH / "agent" / "run_frozen_heldout_evaluation.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            SPEECH / "evaluation" / "evaluate_frozen_cnn_ctc_v3_reference.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def run_streaming(", controller)
        self.assertIn("reference_proc = run_streaming(", controller)
        self.assertIn('"--progress-every"', controller)
        self.assertIn('"--quiet-result"', controller)
        self.assertIn("[heldout-reference]", evaluator)
        self.assertIn("flush=True", evaluator)

    def test_reference_runtime_dependencies_are_declared(self):
        requirements = (
            ROOT / "requirements" / "training.txt"
        ).read_text(encoding="utf-8")
        prepare_env = (
            ROOT / "scripts" / "prepare-python-env.sh"
        ).read_text(encoding="utf-8")
        evaluator = (
            SPEECH / "evaluation" / "evaluate_frozen_cnn_ctc_v3_reference.py"
        ).read_text(encoding="utf-8")

        self.assertIn("import onnxruntime as ort", evaluator)
        self.assertIn("onnxruntime", requirements)
        self.assertIn("import numpy, torch, onnx, onnxruntime", prepare_env)

    def test_edge_provisioning_keeps_dataset_out_of_band(self):
        source = (
            ROOT / "scripts" / "provision-speech-heldout-eval-edge.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("BatchMode=yes", source)
        self.assertNotIn("StrictHostKeyChecking=no", source)
        self.assertIn("edge-speech-bootstrap.sh", source)
        self.assertIn("rsync -a --delete --checksum", source)
        self.assertIn("heldout-eval-v1", source)
        self.assertIn("--verify-only", source)


if __name__ == "__main__":
    unittest.main()
