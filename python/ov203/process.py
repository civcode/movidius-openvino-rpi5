"""Inference server process lifecycle helpers."""

import os
import signal
import subprocess
import sys
import time


def server_exit_meaning(code):
    return {
        1: "runtime failure (see its diagnostics)",
        2: "device not available or bad command line",
        3: "model or IO failure",
        4: "bad command line",
    }.get(code, "unknown")


def _docker_container_for(proc):
    """Return the matching named docker-run container for *proc*, if any."""
    try:
        with open("/proc/%d/cmdline" % proc.pid, "rb") as handle:
            cmd = handle.read().decode()
    except OSError:
        return None
    if "docker" not in cmd or "run" not in cmd:
        return None
    try:
        out = subprocess.run(
            [
                "docker", "ps", "-a", "--filter", "name=ov203-",
                "--format", "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        name = line.strip()
        if name.endswith("-%d" % proc.pid):
            return name
    return None


def stop_server(proc, graceful_timeout=5, term_timeout=5):
    """Stop an inference server, preferring protocol EOF over signals."""
    if proc.poll() is not None:
        return

    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except (OSError, ValueError):
        pass

    try:
        proc.wait(timeout=graceful_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    container = _docker_container_for(proc)
    if container:
        try:
            subprocess.run(
                ["docker", "stop", "-t", "0", container],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()

    try:
        proc.wait(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=term_timeout)
        except subprocess.TimeoutExpired:
            pass

class ProcCpu:
    """Average CPU cores busy across a set of processes."""
    def __init__(self,pids=(),include_children=False):
        self.hz=os.sysconf("SC_CLK_TCK") or 100; self.logical=os.cpu_count() or 1
        self.include_children=include_children; self._pids=[]; self._cached=None; self._cached_at=0.0
        self.add(pids); self.reset()
    def add(self,pids):
        for pid in pids:
            if pid is not None and int(pid) not in self._pids: self._pids.append(int(pid))
        return self
    def _live_pids(self):
        if not self.include_children: return self._pids
        parent_of={}
        for pid in os.listdir("/proc"):
            if not pid.isdigit(): continue
            try:
                with open("/proc/%s/stat"%pid,"rb") as fh:
                    fields=fh.read().decode("ascii","replace").rsplit(")",1)[1].split()
                parent_of[int(pid)]=int(fields[1])
            except (OSError,IndexError,ValueError): pass
        wanted=set(self._pids); grew=True
        while grew:
            grew=False
            for pid,ppid in parent_of.items():
                if ppid in wanted and pid not in wanted: wanted.add(pid); grew=True
        return sorted(wanted)
    def _total_ticks(self):
        total=0
        for pid in self._live_pids():
            try:
                with open("/proc/%d/stat"%pid,"rb") as fh: fields=fh.read().decode("ascii","replace")
                fields=fields.rsplit(")",1)[1].split(); total+=int(fields[11])+int(fields[12])
            except (OSError,IndexError,ValueError): pass
        return total
    def reset(self):
        self._ticks=self._total_ticks(); self._when=time.monotonic(); self._cached_at=0.0; return self
    def cores(self,min_interval=0.25):
        if not self._pids: return None
        now=time.monotonic()
        if now-self._cached_at<min_interval: return self._cached
        span=now-self._when
        if span<=0: return None
        self._cached=(self._total_ticks()-self._ticks)/self.hz/span; self._cached_at=now
        return self._cached
    def percent(self):
        cores=self.cores(); return None if cores is None else 100.0*cores/self.logical

def wait_alive(proc,timeout=2.5):
    t0=time.monotonic()
    while time.monotonic()-t0<timeout:
        code=proc.poll()
        if code is not None:
            raise RuntimeError("inference server exited during startup (exit code %s: %s)" %
                               (code,server_exit_meaning(code)))
        time.sleep(0.05)

def _parent_cpuset():
    try:
        with open("/proc/%d/status"%os.getppid()) as fh:
            spec=next((line.split(":",1)[1].strip() for line in fh if line.startswith("Cpus_allowed_list:")),None)
        if not spec: return None
        out=set()
        for part in spec.split(","):
            if "-" in part:
                lo,hi=part.split("-"); out.update(range(int(lo),int(hi)+1))
            elif part.isdigit(): out.add(int(part))
        return out or None
    except (OSError,ValueError): return None

def restore_parent_affinity():
    try: cur=os.sched_getaffinity(0)
    except (OSError,AttributeError): return None
    parent=_parent_cpuset()
    if not parent or parent==set(cur): return None
    want=set(cur)|set(parent)
    try: os.sched_setaffinity(0,want)
    except OSError: return None
    return "CPU affinity restored from %d to %d cores"%(len(cur),len(want))

def spawn_servers(factory,count,startup_window=2.5):
    fixed=restore_parent_affinity()
    if fixed: print("note: "+fixed,file=sys.stderr,flush=True)
    servers=[factory() for _ in range(max(1,int(count)))]
    deadline=time.monotonic()+startup_window
    while time.monotonic()<deadline:
        for srv in servers:
            code=srv.proc.poll()
            if code is not None:
                raise RuntimeError("inference server exited during startup (exit code %s: %s)" %
                                   (code,server_exit_meaning(code)))
        time.sleep(0.05)
    return servers
