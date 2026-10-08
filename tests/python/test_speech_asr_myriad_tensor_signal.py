"""Hardware-free SIGINT regression for the persistent MYRIAD tensor client.

A private process group receives Ctrl+C. A mock tensor server must continue
serving the final inference request until the Python client closes it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH_PYTHON = ROOT / "examples/speech-asr/python"


MOCK_SERVER = """import sys

sys.stderr.write(
    "READY protocol=tensor-stream-v2 input_elements=4 "
    "output_elements=4 load_ms=1.0 warmup=1\\n"
)
sys.stderr.flush()

index = 0
while True:
    payload = sys.stdin.buffer.read(16)
    if not payload:
        break
    if len(payload) != 16:
        raise RuntimeError("mock server got a partial tensor")
    index += 1
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    sys.stderr.write(f"TIMING request={index} inference_ms=3.5\\n")
    sys.stderr.flush()
"""

CLIENT = """import os
import pathlib
import signal
import sys

import numpy as np

sys.path.insert(0, sys.argv[3])
from speech_asr.myriad_tensor_client import PersistentTensorServer

received = []
signal.signal(signal.SIGINT, lambda number, _frame: received.append(number))
with PersistentTensorServer(
    command=[sys.executable, "-u", sys.argv[1]],
    cwd=pathlib.Path(sys.argv[2]),
    input_elements=4,
    output_elements=4,
    log_path=pathlib.Path(sys.argv[2]) / "mock-server.log",
    startup_timeout=10,
    request_timeout=10,
) as server:
    if server.warmup != 1:
        raise AssertionError("mock startup handshake failed")
    # The test client has its own process group. Terminal Ctrl+C arrives
    # here, but MUST NOT kill the persistent model-server subprocess.
    os.killpg(os.getpgrp(), signal.SIGINT)
    if received != [signal.SIGINT]:
        raise AssertionError(f"unexpected terminal signal state: {received}")
    values = np.array([1., 2., 3., 4.], dtype=np.float32)
    reply, latency = server.infer(values)
    np.testing.assert_array_equal(reply, values)
    if latency != 3.5:
        raise AssertionError(f"bad timing response: {latency}")
    print("final MYRIAD inference survived Ctrl+C", flush=True)
"""


class MyriadSignalTests(unittest.TestCase):
    def test_ctrl_c_does_not_kill_persistent_tensor_server(self):
        if not hasattr(os, "killpg"):
            self.skipTest("requires POSIX process groups")
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            server_file = folder / "mock_server.py"
            client_file = folder / "client.py"
            server_file.write_text(MOCK_SERVER, encoding="utf-8")
            client_file.write_text(CLIENT, encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable, str(client_file), str(server_file),
                    tmp, str(SPEECH_PYTHON),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"client rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}",
            )
            self.assertIn("final MYRIAD inference survived Ctrl+C", proc.stdout)


if __name__ == "__main__":
    unittest.main()
