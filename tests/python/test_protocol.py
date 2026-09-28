import os
import threading
import time
import unittest

import sys
sys.path.insert(0, "python")

from ov203.protocol import LinePipe, write_all


class ProtocolHelpersTest(unittest.TestCase):
    def test_line_then_binary_then_line(self):
        rfd, wfd = os.pipe()
        try:
            def writer():
                os.write(wfd, b"MASK 2 2\n")
                time.sleep(0.01)
                write_all(wfd, b"\x01\x00\x02\x00\x03\x00\x04\x00END\n")
                os.close(wfd)

            thread = threading.Thread(target=writer)
            thread.start()
            reader = LinePipe(rfd)
            self.assertEqual(reader.readline(timeout=1), "MASK 2 2")
            self.assertEqual(
                reader.read_exact(8, timeout=1),
                b"\x01\x00\x02\x00\x03\x00\x04\x00",
            )
            self.assertEqual(reader.readline(timeout=1), "END")
            thread.join(timeout=1)
        finally:
            try:
                os.close(rfd)
            except OSError:
                pass
            try:
                os.close(wfd)
            except OSError:
                pass

    def test_readline_timeout(self):
        rfd, wfd = os.pipe()
        try:
            reader = LinePipe(rfd)
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                reader.readline(timeout=0.05)
            self.assertLess(time.monotonic() - start, 0.5)
        finally:
            os.close(rfd)
            os.close(wfd)


if __name__ == "__main__":
    unittest.main()
