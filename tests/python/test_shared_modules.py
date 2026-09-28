import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import ov203
from ov203 import camera, display, process, runtime

class SharedModuleTests(unittest.TestCase):
    def test_package_exports(self):
        self.assertTrue(callable(ov203.resolve_servers))
        self.assertTrue(callable(ov203.spawn_servers))
        self.assertTrue(hasattr(camera, "LatestFrame"))
        self.assertTrue(hasattr(display, "WindowWatcher"))
        self.assertTrue(hasattr(process, "ProcCpu"))

    def test_source_repo_discovery(self):
        self.assertEqual(runtime.source_repo_root(__file__), ROOT)
        self.assertEqual(runtime.models_root(__file__), ROOT / "vendor" / "models")

    def test_fake_camera_unbounded(self):
        class Image:
            shape = (8, 9, 3)
        fake = camera.FakeCamera(Image(), fps=0)
        ok, image = fake.read()
        self.assertTrue(ok)
        self.assertIs(image, fake.image)
        fake.release()
        self.assertFalse(fake.isOpened())

if __name__ == "__main__":
    unittest.main()
