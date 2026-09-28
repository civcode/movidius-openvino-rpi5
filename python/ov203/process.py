"""Inference server process lifecycle helpers."""

import os
import signal
import subprocess


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
