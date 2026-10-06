import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
POLICY = (
    SPEECH
    / "datasets"
    / "ami"
    / "splits"
    / "model-quality-v4-edinburgh-v1.json"
)
FREEZER = (
    SPEECH
    / "datasets"
    / "ami"
    / "freeze_model_quality_v4_sources.py"
)


def load_freezer():
    spec = importlib.util.spec_from_file_location(
        "freeze_model_quality_v4_sources",
        FREEZER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load model-quality-v4 source freezer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelQualityV4PolicyTests(unittest.TestCase):
    def test_policy_uses_fresh_official_development_boundary(self):
        value = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(
            value["id"],
            "ami-model-quality-v4-edinburgh-full-corpus-asr-v1",
        )
        self.assertEqual(value["authority"]["partition_name"], "Full-corpus-ASR")
        self.assertEqual(set(value["validation_groups"]), {"ES2011"})
        self.assertEqual(
            set(value["train_groups"]),
            {
                "ES2003",
                "ES2005",
                "ES2006",
                "ES2007",
                "ES2008",
                "ES2009",
                "ES2010",
                "ES2012",
                "ES2013",
                "ES2014",
                "ES2015",
                "ES2016",
            },
        )
        self.assertEqual(
            set(value["excluded_prior_selection_groups"]),
            {"ES2002"},
        )
        self.assertEqual(
            set(value["sealed_test_groups"]),
            {"EN2002", "ES2004", "IS1009", "TS3003"},
        )
        self.assertTrue(value["policy"]["training_allowed_on_train"])
        self.assertFalse(value["policy"]["training_allowed_on_validation"])
        self.assertTrue(
            value["policy"]["checkpoint_selection_allowed_on_validation"]
        )
        self.assertFalse(
            value["policy"]["heldout_metrics_allowed_for_model_selection"]
        )

    def test_source_freezer_enforces_reviewed_policy(self):
        module = load_freezer()
        value = json.loads(POLICY.read_text(encoding="utf-8"))
        module.validate_policy(value)
        self.assertEqual(
            module.EXPECTED_TRAIN_GROUPS,
            set(value["train_groups"]),
        )
        self.assertEqual(module.EXPECTED_VALIDATION_GROUPS, {"ES2011"})
        self.assertEqual(
            module.EXPECTED_SEALED_TEST_GROUPS,
            {"EN2002", "ES2004", "IS1009", "TS3003"},
        )


class ModelQualityV4ImplementationTests(unittest.TestCase):
    def test_qualification_guards_fresh_boundary(self):
        source = (
            SPEECH / "tools" / "qualify_model_quality_v4.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTED_VALIDATION_GROUPS = {\"ES2011\"}", source)
        self.assertIn("FORBIDDEN_GROUPS", source)
        self.assertIn("training population did not expand beyond v3", source)
        self.assertIn("training duration did not expand beyond v3", source)
        self.assertIn('"heldout_metrics_allowed_for_model_selection": False', source)
        self.assertIn("source_lock_sha256", source)
        self.assertIn("EXPECTED_VALIDATION_INVALID_INTERVAL", source)
        self.assertIn('"meeting": "ES2011c"', source)
        self.assertIn('"source_start_sample": 1249952', source)
        self.assertIn('"source_end_sample": 1248016', source)
        self.assertIn("if train_exclusions:", source)
        self.assertIn("len(validation_exclusions) != 1", source)

    def test_baseline_changes_data_only(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_model_quality_v4.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PARENT = "exp-87538823d2bf1562"', source)
        self.assertIn('"model_id": "cnn_ctc_v3"', source)
        self.assertIn('"frontend": {"kind": "logmel-v1"}', source)
        self.assertIn('"checkpoint_selection": "validation_cer"', source)
        self.assertIn('"max_wer": None', source)
        self.assertIn('"max_cer": None', source)
        self.assertNotIn('"augmentation": {', source)
        self.assertNotIn('"ctc_objective": {', source)
        self.assertIn("sealed held-out metrics must remain forbidden", source)
        self.assertIn("training source exclusions are not allowed", source)
        self.assertIn("exactly one reviewed validation exclusion", source)
        self.assertIn('"meeting": "ES2011c"', source)
        self.assertIn(
            'TRAIN_MANIFEST_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"',
            source,
        )
        self.assertIn(
            'VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"',
            source,
        )
        self.assertIn("TRAIN_RECORDS = 15738", source)
        self.assertIn("TRAIN_AUDIO_SECONDS = 25846.327", source)
        self.assertIn("VALIDATION_RECORDS = 1273", source)
        self.assertIn("VALIDATION_AUDIO_SECONDS = 2279.385", source)

    def test_shell_entrypoints_exist(self):
        for relative in (
            "scripts/prepare-speech-model-data-v4.sh",
            "scripts/qualify-speech-model-data-v4.sh",
            "scripts/provision-speech-model-data-v4-edge.sh",
            "scripts/init-cnn-ctc-v3-model-quality-v4.sh",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
