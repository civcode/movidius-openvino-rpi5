import subprocess
import sys
import unittest

sys.path.insert(0, "python")

from ov203.process import server_exit_meaning, stop_server


class ProcessHelpersTest(unittest.TestCase):
    def test_stop_server_prefers_eof(self):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdin.buffer.read(); raise SystemExit(0)",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        stop_server(proc, graceful_timeout=1, term_timeout=1)
        self.assertEqual(proc.returncode, 0)

    def test_exit_meaning(self):
        self.assertIn("model", server_exit_meaning(3))
        self.assertEqual(server_exit_meaning(99), "unknown")


if __name__ == "__main__":
    unittest.main()
