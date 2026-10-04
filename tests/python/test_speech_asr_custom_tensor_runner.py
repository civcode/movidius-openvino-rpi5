import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "smoke-test" / "main.cpp"


class CustomTensorRunnerSourceTests(unittest.TestCase):
    def test_custom_runner_has_f32_tensor_io(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('arg == "--tensor"', source)
        self.assertIn('arg == "--output"', source)
        self.assertIn("floatToHalf(values[i])", source)
        self.assertIn("writeF32(outputPath, decoded)", source)

    def test_input_layout_is_rank_derived(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("TensorDesc::getLayoutByDims", source)
        self.assertNotIn("item.second->setLayout(Layout::NCHW)", source)

    def test_custom_work_mount_is_writable(self):
        run = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn('DOCKER_ARGS+=(-v "${ROOT}/work:/work")', run)


if __name__ == "__main__":
    unittest.main()
