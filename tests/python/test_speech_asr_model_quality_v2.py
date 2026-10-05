import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
sys.path.insert(0, str(SPEECH / "python"))

from speech_asr.ami import validate_split_spec

SPLIT = SPEECH / "datasets" / "ami" / "splits" / "train-es2005-v1.json"


class ExpandedAmiContractTests(unittest.TestCase):
    def test_es2005_split_is_frozen_and_official_training_only(self):
        value = validate_split_spec(json.loads(SPLIT.read_text(encoding="utf-8")))
        self.assertEqual(
            [source["meeting"] for source in value["sources"]],
            ["ES2005a", "ES2005b", "ES2005c", "ES2005d"],
        )
        self.assertEqual(value["annotations"]["version"], "1.6.2")
        self.assertEqual(value["expected"]["records"], 1865)
        self.assertEqual(
            value["expected"]["manifest_sha256"],
            "4e4ba1b8df2e3041b9dbbf654b761777abd256c4a272f8103a24575e49114c6b",
        )
        for source in value["sources"]:
            self.assertEqual(source["audio"]["stream"], "Mix-Headset")
            self.assertEqual(
                [item["speaker"] for item in source["selections"]],
                ["A", "B", "C", "D"],
            )

    def test_qualifier_enforces_meeting_disjointness_and_eligibility(self):
        source = (
            SPEECH / "tools" / "qualify_expanded_model_quality_manifests.py"
        ).read_text(encoding="utf-8")
        self.assertIn("manifest_record_eligibility", source)
        self.assertIn("train/validation meeting overlap", source)
        self.assertIn('"meeting_overlap": 0', source)
        self.assertIn("relocate(", source)

    def test_initializer_binds_expanded_train_and_frozen_validation(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_expanded_data.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'TRAIN_MANIFEST_SHA256 = "abb4ed8fcf054497e1d81b63acfc8aa42ad0a49e542184581299be8496fefbb6"',
            source,
        )
        self.assertIn(
            'VALIDATION_MANIFEST_SHA256 = "07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923"',
            source,
        )
        self.assertIn('"model-quality-v1"', source)
        self.assertIn('"checkpoint_selection": "validation_cer"', source)
        self.assertNotIn('"ctc_objective": {', source)
        self.assertNotIn('"augmentation": {', source)

    def test_commands_are_executable(self):
        for name in (
            "qualify-speech-model-data-v2.sh",
            "init-cnn-ctc-v3-expanded-data.sh",
        ):
            path = ROOT / "scripts" / name
            self.assertTrue(path.is_file())
            self.assertNotEqual(path.stat().st_mode & 0o111, 0)
