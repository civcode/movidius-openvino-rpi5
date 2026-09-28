"""Binary pipe helpers shared by the inference clients."""

import fcntl
import os
import select
import subprocess
import time


class LinePipe:
    """Buffered line/binary reader over a server stdout file descriptor."""

    def __init__(self, fd, proc=None):
        self.fd = fd
        self.proc = proc
        self.buf = bytearray()

    def _fill(self):
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            chunk = b""
        self.buf += chunk
        return chunk

    def readline(self, timeout=None):
        """Return the next line without its newline, or None on clean EOF."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            pos = self.buf.find(b"\n")
            if pos != -1:
                line, self.buf = self.buf[:pos], self.buf[pos + 1:]
                return line.decode("utf-8", "replace")

            if self.proc is not None and self.proc.poll() is not None:
                self._fill()
                pos = self.buf.find(b"\n")
                if pos == -1:
                    if self.buf:
                        line, self.buf = bytes(self.buf), bytearray()
                        return line.decode("utf-8", "replace")
                    raise RuntimeError(
                        "inference server exited with code %s before the response "
                        "was complete (see its diagnostics above)"
                        % self.proc.returncode
                    )
                line, self.buf = self.buf[:pos], self.buf[pos + 1:]
                return line.decode("utf-8", "replace")

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no server response within %.0f s" % timeout)

            if deadline is not None:
                ready, _, _ = select.select(
                    [self.fd], [], [], max(0.001, deadline - time.monotonic())
                )
                if not ready:
                    continue

            if not self._fill():
                if self.proc is not None:
                    if self.proc.poll() is None:
                        try:
                            self.proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    if self.proc.returncode is not None:
                        raise RuntimeError(
                            "inference server exited with code %s before the response "
                            "was complete (see its diagnostics above)"
                            % self.proc.returncode
                        )
                line, self.buf = bytes(self.buf), bytearray()
                return line.decode("utf-8", "replace") if line else None

    def read_exact(self, n, timeout=None):
        """Return exactly *n* bytes, preserving buffered bytes after the payload."""
        out = bytearray(self.buf[:n])
        self.buf = self.buf[n:]
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(out) < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("no server payload within %.0f s" % timeout)
                ready, _, _ = select.select([self.fd], [], [], max(0.001, remaining))
                if not ready:
                    continue
            chunk = os.read(self.fd, n - len(out))
            if not chunk:
                raise RuntimeError("server closed while reading a binary payload")
            out += chunk
        return bytes(out)


def write_all(fd, data):
    """Write every byte of a bytes-like object directly to *fd*."""
    view = memoryview(data)
    if view.itemsize != 1 or view.ndim != 1:
        view = view.cast("B")
    off = 0
    while off < view.nbytes:
        off += os.write(fd, view[off:])


def pipe_capacity(stream, want=0):
    """Return pipe capacity, best-effort growing it to *want* bytes first."""
    try:
        fd = stream.fileno()
        if want:
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, int(want))
        return fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
    except (OSError, AttributeError, ValueError):
        return 0
