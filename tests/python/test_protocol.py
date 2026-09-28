import os
import threading
import time
import unittest
import pathlib
import tempfile
import textwrap

import numpy as np

import sys
sys.path.insert(0, "python")

from ov203.protocol import LinePipe, MobileNetPipeClient, SegPipeClient, SsdPipeClient, write_all


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

class ProtocolClientTest(unittest.TestCase):
    def make_server(self, body):
        tmp = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmp.name) / "server.py"
        path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
        path.chmod(0o755)
        self.addCleanup(tmp.cleanup)
        return str(path)

    def test_mobilenet_pipe_client(self):
        server = self.make_server(r"""
            import struct, sys
            raw = sys.stdin.buffer.read(4)
            assert raw == b"ABCD"
            sys.stdout.buffer.write(struct.pack("<2f", 1.25, -2.5))
            sys.stdout.buffer.flush()
        """)
        client = MobileNetPipeClient(server, "host", "fp16", "MYRIAD", 1,
                                     input_bytes=4, output_elements=2)
        try:
            out = client.infer(b"ABCD")
            np.testing.assert_allclose(out, [1.25, -2.5])
        finally:
            client.close()

    def test_ssd_pipe_client(self):
        server = self.make_server(r"""
            import struct, sys
            w, h = struct.unpack("<II", sys.stdin.buffer.read(8))
            assert (w, h) == (1, 1)
            assert len(sys.stdin.buffer.read(3)) == 3
            sys.stdout.write("FRAME 1 1 3.5\nDET fire hydrant 0.75 1 2 3 4\nEND\n")
            sys.stdout.flush()
        """)
        client = SsdPipeClient(server, "host", "MYRIAD", 0.5, 1)
        try:
            result = client.detect(np.zeros((1, 1, 3), dtype=np.uint8))
            self.assertEqual(result[:3], (1, 1, 3.5))
            self.assertEqual(result[3][0], ("fire hydrant", 0.75, 1, 2, 3, 4))
        finally:
            client.close()

    def test_seg_pipe_client(self):
        server = self.make_server(r"""
            import struct, sys
            w, h = struct.unpack("<II", sys.stdin.buffer.read(8))
            assert (w, h) == (1, 1)
            assert len(sys.stdin.buffer.read(3)) == 3
            sys.stdout.buffer.write(
                b"FRAME 1 1 4.0 2.0\nCLASSES 1\nCLASS 15 person 1\nMASK 1 1\n"
                + struct.pack("<H", 15) + b"END\n"
            )
            sys.stdout.buffer.flush()
        """)
        client = SegPipeClient(server, "host", "MYRIAD", 1)
        try:
            classes, mask, total_ms, infer_ms = client.segment(
                np.zeros((1, 1, 3), dtype=np.uint8), 1, 1
            )
            self.assertEqual(classes, [(15, "person", 1)])
            self.assertEqual(mask, b"\x0f\x00")
            self.assertEqual((total_ms, infer_ms), (4.0, 2.0))
        finally:
            client.close()



if __name__ == "__main__":
    unittest.main()
