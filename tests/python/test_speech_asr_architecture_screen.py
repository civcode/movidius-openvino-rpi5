import hashlib
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"
TOOL = SPEECH / "tools" / "prepare_architecture_screen_v1.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("prepare_architecture_screen_v1", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load architecture screen tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchitectureScreenV1Tests(unittest.TestCase):
    def test_hash_bucket_rule_is_exact_and_deterministic(self):
        module = load_tool()
        for sample_id in (
            "ami-ES2003a-A-1",
            "ami-ES2010c-D-77",
            "ami-ES2016d-B-1234",
        ):
            digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
            expected = (
                int.from_bytes(digest[:8], byteorder="big", signed=False) % 4 == 0
            )
            self.assertEqual(module.selected_by_rule(sample_id), expected)
        self.assertEqual(module.MODULUS, 4)
        self.assertEqual(module.REMAINDER, 0)
        self.assertEqual(
            module.RULE_ID,
            "sha256-id-first-u64-be-mod4-eq0-v1",
        )
        self.assertEqual(len(module.EXPECTED_MEETINGS), 48)

    def test_screen_freezes_full_validation_and_short_budget(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_architecture_screen.py"
        ).read_text(encoding="utf-8")
        self.assertIn('SCREEN_EPOCHS = 12', source)
        self.assertIn('"epochs": SCREEN_EPOCHS', source)
        self.assertIn('"model_id": "cnn_ctc_v3"', source)
        self.assertIn('"checkpoint_selection": "validation_cer"', source)
        self.assertIn(
            'VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"',
            source,
        )
        self.assertIn('"max_cer": None', source)
        self.assertNotIn('"augmentation": {', source)
        self.assertNotIn('"ctc_objective": {', source)
        self.assertIn("sealed held-out data forbidden", source)

    def test_entrypoints_exist(self):
        for relative in (
            "scripts/prepare-speech-architecture-screen-v1.sh",
            "scripts/init-cnn-ctc-v3-architecture-screen.sh",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
