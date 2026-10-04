import importlib.util
import pathlib
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "examples"
    / "speech-asr"
    / "evaluation"
    / "benchmark_rm_cnn4a.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_rm_cnn4a", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark_rm_cnn4a")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RmCnn4aReferenceCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_cpu_reference_uses_amd64_reference_mode(self):
        command = self.module.build_command("cpu", "amd64")
        self.assertEqual(command[-3:], ["--platform", "amd64", "speech-reference"])

    def test_cpu_reference_rejects_arm_targets(self):
        for platform in ("armv7", "arm64"):
            with self.subTest(platform=platform):
                with self.assertRaisesRegex(ValueError, "requires platform amd64"):
                    self.module.build_command("cpu", platform)

    def test_model_identity_uses_ir_inspector(self):
        contract = {
            "xml_sha256": "1" * 64,
            "graph_sha256": "2" * 64,
            "bin_sha256": "3" * 64,
        }
        xml = pathlib.Path("/tmp/model.xml")
        binary = pathlib.Path("/tmp/model.bin")
        with mock.patch.object(self.module, "inspect_ir", return_value=contract) as inspect:
            identity = self.module.model_identity(xml, binary)
        inspect.assert_called_once_with(xml, binary)
        self.assertEqual(identity["id"], "rm_cnn4a-fp16")
        self.assertEqual(identity["family"], "rm_cnn4a")
        self.assertEqual(identity["xml_sha256"], "1" * 64)
        self.assertEqual(identity["graph_sha256"], "2" * 64)
        self.assertEqual(identity["bin_sha256"], "3" * 64)

    def test_git_head_has_no_model_artifact_dependency(self):
        with mock.patch.object(self.module.subprocess, "run") as run:
            run.return_value.stdout = "abcdef123456\n"
            self.assertEqual(self.module.git_head(), "abcdef123456")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["git", "rev-parse", "HEAD"])
        self.assertTrue(kwargs["check"])

    def test_output_tail_keeps_last_nonempty_lines(self):
        text = "one\n\n two \nthree\nfour\n"
        self.assertEqual(self.module.output_tail(text, lines=2), "three\nfour")

    def test_myriad_uses_device_regression_mode(self):
        for platform in ("armv7", "arm64", "amd64"):
            with self.subTest(platform=platform):
                command = self.module.build_command("myriad", platform)
                self.assertEqual(command[-1], "speech-regress")


if __name__ == "__main__":
    unittest.main()
