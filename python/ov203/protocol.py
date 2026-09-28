"""Binary pipe helpers shared by the inference clients."""

import fcntl
import os
import select
import subprocess
import time
import struct
import sys

import numpy as np

from .process import server_exit_meaning, stop_server


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

class MobileNetPipeClient:
    """Client for the fixed-size MobileNet stdin/stdout protocol."""
    def __init__(self,server_cmd,backend,ir,device,request_timeout,
                 input_bytes=224*224*3*4,output_elements=1000):
        self.request_timeout=request_timeout; self.input_bytes=input_bytes
        self.output_bytes=output_elements*4
        self.proc=subprocess.Popen([server_cmd,backend,ir,device],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=None,start_new_session=True)
        self.pipe_bytes=pipe_capacity(self.proc.stdin,self.input_bytes+4096)
        print("inference backend: %s ir=%s device=%s (server pid %d, request pipe %.0f KiB)" %
              (backend,ir,device,self.proc.pid,self.pipe_bytes/1024.0),file=sys.stderr,flush=True)
    def _stdin_closed_message(self):
        code=self.proc.poll()
        if code is None: return "server closed its stdin pipe but has not exited yet (see its diagnostics above)"
        return "server closed its stdin pipe (exit code %s: %s)"%(code,server_exit_meaning(code))
    def send(self,payload):
        view=payload if isinstance(payload,memoryview) else memoryview(payload).cast("B")
        if view.nbytes!=self.input_bytes:
            raise AssertionError("payload is %d bytes, expected %d"%(view.nbytes,self.input_bytes))
        if not view.c_contiguous: raise AssertionError("payload must be C-contiguous to send")
        try: write_all(self.proc.stdin.fileno(),view)
        except (BrokenPipeError,OSError) as exc: raise RuntimeError(self._stdin_closed_message()) from exc
    def recv(self):
        pipe=LinePipe(self.proc.stdout.fileno(),self.proc)
        try: raw=pipe.read_exact(self.output_bytes,timeout=self.request_timeout)
        except TimeoutError as exc: raise RuntimeError(str(exc)) from exc
        return np.frombuffer(raw,dtype="<f4")
    def infer(self,tensor):
        self.send(tensor); return self.recv()
    def close(self):
        stop_server(self.proc)
        for stream in (self.proc.stdin,self.proc.stdout):
            try: stream.close()
            except Exception: pass

class SsdPipeClient:
    """Client for the SSD framed RGB/text detection protocol."""
    def __init__(self,server_cmd,backend,device,min_conf,request_timeout):
        self.request_timeout=request_timeout
        self.proc=subprocess.Popen([server_cmd,backend,device,str(min_conf)],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=None,start_new_session=True)
        self.pipe=LinePipe(self.proc.stdout.fileno(),self.proc)
        print("inference backend: %s device=%s min-conf=%s (server pid %d)" %
              (backend,device,min_conf,self.proc.pid),file=sys.stderr,flush=True)
    def _readline(self):
        try: return self.pipe.readline(timeout=self.request_timeout)
        except TimeoutError as exc: raise RuntimeError(str(exc)) from exc
    def detect(self,rgb):
        h,w=rgb.shape[:2]
        try:
            write_all(self.proc.stdin.fileno(),struct.pack("<II",w,h))
            write_all(self.proc.stdin.fileno(),np.ascontiguousarray(rgb,dtype=np.uint8))
        except (BrokenPipeError,OSError) as exc:
            code=self.proc.poll()
            if code is None: raise RuntimeError("server closed its stdin pipe but has not exited yet") from exc
            raise RuntimeError("server closed its stdin pipe (exit code %s: %s)"%(code,server_exit_meaning(code))) from exc
        line=self._readline()
        if line is None: raise RuntimeError("server closed the response pipe")
        parts=line.split()
        if parts and parts[0]=="ERROR": raise RuntimeError("server reported: "+line[5:].strip())
        if len(parts)!=4 or parts[0]!="FRAME": raise RuntimeError("unexpected server line: %r"%line)
        try: infer_ms=float(parts[3])
        except ValueError: raise RuntimeError("unexpected server line: %r"%line) from None
        dets=[]
        while True:
            line=self._readline()
            if line is None: raise RuntimeError("server closed the response before END")
            if line=="END": break
            if line.startswith("ERROR"): raise RuntimeError("server reported: "+line[5:].strip())
            if line.split(" ",1)[0]!="DET": raise RuntimeError("unexpected server line: %r"%line)
            fields=line.split("DET",1)[1].rsplit(" ",5)
            if len(fields)!=6: raise RuntimeError("unexpected server line: %r"%line)
            try: det=(fields[0].strip(),float(fields[1]),int(fields[2]),int(fields[3]),int(fields[4]),int(fields[5]))
            except ValueError: raise RuntimeError("unexpected server line: %r"%line) from None
            dets.append(det)
        return w,h,infer_ms,dets
    def close(self):
        stop_server(self.proc)
        for stream in (self.proc.stdin,self.proc.stdout):
            try: stream.close()
            except Exception: pass

class SegPipeClient:
    """Client for the segmentation framed RGB/class-map protocol."""
    def __init__(self,server_cmd,backend,device,request_timeout):
        self.request_timeout=request_timeout
        self.proc=subprocess.Popen([server_cmd,backend,device],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=None,start_new_session=True)
        self.pipe=LinePipe(self.proc.stdout.fileno(),self.proc)
        print("inference backend: %s device=%s (server pid %d)" %
              (backend,device,self.proc.pid),file=sys.stderr,flush=True)
    def _readline(self):
        try: return self.pipe.readline(timeout=self.request_timeout)
        except TimeoutError as exc: raise RuntimeError(str(exc)) from exc
    def segment(self,rgb,w,h):
        try:
            write_all(self.proc.stdin.fileno(),struct.pack("<II",w,h))
            write_all(self.proc.stdin.fileno(),np.ascontiguousarray(rgb,dtype=np.uint8))
        except (BrokenPipeError,OSError) as exc:
            code=self.proc.poll()
            if code is None: raise RuntimeError("server closed its stdin pipe but has not exited yet") from exc
            raise RuntimeError("server closed its stdin pipe (exit code %s: %s)"%(code,server_exit_meaning(code))) from exc
        classes=[]; mask_bytes=None; total_ms=infer_ms=0.0
        while True:
            line=self._readline()
            if line is None: raise RuntimeError("server closed stdout mid-frame")
            if line.startswith("ERROR"): raise RuntimeError("server reported: "+line[5:].strip())
            tok=line.split()
            if not tok: continue
            if tok[0]=="FRAME":
                if len(tok)!=5: raise RuntimeError("unexpected server line: %r"%line)
                try: total_ms=float(tok[3]); infer_ms=float(tok[4])
                except ValueError: raise RuntimeError("unexpected server line: %r"%line) from None
            elif tok[0]=="CLASSES":
                for _ in range(int(tok[1])):
                    row=self._readline()
                    if row is None: raise RuntimeError("server closed mid-CLASS")
                    fields=row.split()
                    if len(fields)<4 or fields[0]!="CLASS": raise RuntimeError("unexpected server line: %r"%row)
                    classes.append((int(fields[1])," ".join(fields[2:-1]),int(fields[-1])))
            elif tok[0]=="MASK":
                if len(tok)!=3: raise RuntimeError("unexpected server line: %r"%line)
                fw,fh=int(tok[1]),int(tok[2])
                if fw<=0 or fh<=0: raise RuntimeError("bad MASK dimensions: %r"%line)
                try: mask_bytes=self.pipe.read_exact(fw*fh*2,timeout=self.request_timeout)
                except TimeoutError as exc: raise RuntimeError(str(exc)) from exc
            elif tok[0]=="END":
                return classes,mask_bytes,total_ms,infer_ms
    def close(self):
        stop_server(self.proc)
        for stream in (self.proc.stdin,self.proc.stdout):
            try: stream.close()
            except Exception: pass
