import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts" / "patch-model-optimizer.py"


def load_patcher():
    spec = importlib.util.spec_from_file_location("patch_model_optimizer", PATCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load patch-model-optimizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelOptimizerPatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = load_patcher()

    def make_tree(self, root: pathlib.Path, versions_text: str, emitter_text: str = ""):
        versions = root / "mo" / "utils" / "versions_checker.py"
        emitter = root / "mo" / "back" / "ie_ir_ver_2" / "emitter.py"
        versions.parent.mkdir(parents=True)
        emitter.parent.mkdir(parents=True)
        versions.write_text(versions_text, encoding="utf-8")
        emitter.write_text(emitter_text, encoding="utf-8")
        return versions, emitter

    def test_pinned_2020_3_shape_does_not_require_packaging_exec(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            versions, emitter = self.make_tree(
                root,
                "def check_requirements(framework=None):\n    return 0\n",
                "children = node.getchildren()\n",
            )
            changed = self.patcher.patch_tree(root)
            self.assertEqual(
                changed,
                ["mo/back/ie_ir_ver_2/emitter.py"],
            )
            self.assertIn("def check_requirements(", versions.read_text())
            self.assertNotIn(".getchildren()", emitter.read_text())

    def test_legacy_packaging_variant_is_patched_and_idempotent(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            versions, _ = self.make_tree(
                root,
                'def check_requirements(framework=None):\n'
                '    exec("import platform,sys,packaging")\n'
                '    exec("del packaging,platform,sys")\n',
            )
            first = self.patcher.patch_tree(root)
            second = self.patcher.patch_tree(root)
            text = versions.read_text()
            self.assertEqual(first, ["mo/utils/versions_checker.py"])
            self.assertEqual(second, [])
            self.assertIn("import packaging, platform, sys", text)
            self.assertIn("del packaging, platform, sys", text)

    def test_unexpected_versions_checker_layout_fails_loudly(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            self.make_tree(root, "print('not versions checker')\n")
            with self.assertRaises(SystemExit):
                self.patcher.patch_tree(root)


if __name__ == "__main__":
    unittest.main()
