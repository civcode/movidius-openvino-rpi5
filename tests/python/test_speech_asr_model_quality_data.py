import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
TOOL = SPEECH / "tools" / "qualify_model_quality_manifests.py"
SPEC = SPEECH / "models" / "cnn_ctc_v3" / "model_spec.json"
VOCAB = SPEECH / "models" / "cnn_ctc_v3" / "vocab.json"


def load_tool():
    spec = importlib.util.spec_from_file_location("qualify_model_quality", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load model-quality qualification tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(sample_id: str, speaker: str, end_sample: int = 16000) -> dict:
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
            "meeting": "TEST",
            "speaker": speaker,
        },
    }


class ModelQualityQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_partition_is_speaker_disjoint_and_filters_ineligible_records(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = pathlib.Path(temp_name)
            source = temp / "manifest.jsonl"
            records = [
                sample("a1", "A"),
                sample("b1", "B"),
                sample("c1", "C"),
                sample("d1", "D"),
                sample("d-too-long", "D", 90000),
            ]
            source.write_text(
                "".join(
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for value in records
                ),
                encoding="utf-8",
            )
            output = temp / "quality"
            result = self.tool.qualify(
                source_manifest=source,
                spec_path=SPEC,
                vocab_path=VOCAB,
                output_dir=output,
                train_speakers=("A", "B", "C"),
                validation_speakers=("D",),
            )
            self.assertEqual(result["status"], "structurally_valid")
            self.assertEqual(result["partition"]["speaker_overlap"], 0)
            self.assertEqual(result["partition"]["record_overlap"], 0)
            self.assertEqual(result["train"]["stats"]["records"], 3)
            self.assertEqual(result["validation"]["stats"]["records"], 1)
            self.assertEqual(
                result["validation"]["excluded"]["too_long"],
                1,
            )
            self.assertTrue((output / "train.manifest.jsonl").is_file())
            self.assertTrue((output / "validation.manifest.jsonl").is_file())
            self.assertTrue((output / "qualification.json").is_file())
            train_record = json.loads(
                (output / "train.manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )
            self.assertEqual(
                train_record["audio"]["path"],
                "../audio/a1.f32",
            )

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = pathlib.Path(temp_name)
            source = temp / "manifest.jsonl"
            source.write_text(
                json.dumps(sample("a1", "A")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "speaker sets overlap"):
                self.tool.qualify(
                    source_manifest=source,
                    spec_path=SPEC,
                    vocab_path=VOCAB,
                    output_dir=temp / "quality",
                    train_speakers=("A",),
                    validation_speakers=("A",),
                )


if __name__ == "__main__":
    unittest.main()
