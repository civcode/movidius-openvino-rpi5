#!/usr/bin/env python3
"""Stage and run the QuartzNet multi-resident-network capacity test on MA2450."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.orchestration import rsync_push_command, ssh_command  # noqa: E402

DEFAULT_IR = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "myriad"
    / "openvino"
    / "fp16"
)
DEFAULT_TIMES = "512,1024,1536,2048"


def run_capture(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stdout}"
        )
    return proc


def run_stream(argv: list[str], label: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    rc = proc.wait()
    value = subprocess.CompletedProcess(argv, rc, "".join(lines), None)
    if rc != 0:
        raise RuntimeError(
            f"{label} failed ({rc}):\n" + "\n".join(value.stdout.splitlines()[-80:])
        )
    return value


def remote_last_line(worker: str, argv: list[str]) -> str:
    proc = run_capture(ssh_command(worker, argv))
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"remote command produced no output: {argv!r}")
    return lines[-1]


def require_main_clean() -> str:
    branch = run_capture(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "main":
        raise ValueError(f"controller checkout must be main, got {branch!r}")
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout.strip()
    if dirty:
        raise ValueError("controller tracked worktree must be clean")
    return run_capture(["git", "rev-parse", "HEAD"]).stdout.strip()


def parse_times(value: str) -> list[int]:
    try:
        times = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--times must be comma-separated integers") from exc
    if not times or any(item <= 0 or item % 16 for item in times):
        raise ValueError("--times must contain positive multiples of 16")
    if len(set(times)) != len(times):
        raise ValueError("--times must not contain duplicates")
    if any(item >= 3104 for item in times):
        raise ValueError(
            "--times must remain below the proven QuartzNet MYRIAD catastrophic "
            "boundary at T=3104"
        )
    return times


def ensure_runtime(
    *,
    worker: str,
    remote_repo: str,
    refresh_runtime: bool,
) -> None:
    check_cmd = [
        f"{remote_repo}/scripts/run-myriad-tensor.sh",
        "--platform",
        "arm64",
        "--backend",
        "host",
        "check-resident",
    ]
    check = run_capture(ssh_command(worker, check_cmd), check=False)
    if check.returncode == 0:
        print(check.stdout, end="" if check.stdout.endswith("\n") else "\n")
        return
    if not refresh_runtime:
        raise RuntimeError(
            "edge host runtime lacks resident-network test support; "
            "rerun with --refresh-runtime.\n" + check.stdout
        )

    run_stream(
        ssh_command(worker, [f"{remote_repo}/build.sh", "--platform", "arm64"]),
        "edge arm64 runtime rebuild",
    )
    run_stream(
        ssh_command(
            worker,
            [f"{remote_repo}/scripts/pull-runtime.sh", "--platform", "arm64"],
        ),
        "edge host-runtime extraction",
    )
    check = run_capture(ssh_command(worker, check_cmd))
    print(check.stdout, end="" if check.stdout.endswith("\n") else "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-repo")
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR)
    parser.add_argument("--times", default=DEFAULT_TIMES)
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            ROOT
            / "work"
            / "speech-asr"
            / "quartznet15x5-reference"
            / "myriad-residency"
            / "resident-networks.log"
        ),
    )
    args = parser.parse_args()

    try:
        times = parse_times(args.times)
        head = require_main_clean()

        xml = args.ir_dir / "quartznet15x5_nvidia_ref.xml"
        binary = args.ir_dir / "quartznet15x5_nvidia_ref.bin"
        for path in (xml, binary):
            if not path.is_file():
                raise ValueError(
                    f"required carrier IR missing: {path}; run "
                    "./scripts/prepare-quartznet15x5-reference-myriad.sh first"
                )

        home = remote_last_line(
            args.worker,
            ["sh", "-c", 'printf "%s\\n" "$HOME"'],
        )
        remote_repo = args.edge_repo or f"{home}/workspace/movidius-openvino-rpi5"

        run_stream(
            ssh_command(
                args.worker,
                ["git", "-C", remote_repo, "pull", "--ff-only"],
            ),
            "edge git pull",
        )
        remote_head = remote_last_line(
            args.worker,
            ["git", "-C", remote_repo, "rev-parse", "HEAD"],
        )
        if remote_head != head:
            raise ValueError(
                f"controller/edge revision mismatch: {head} != {remote_head}"
            )

        ensure_runtime(
            worker=args.worker,
            remote_repo=remote_repo,
            refresh_runtime=args.refresh_runtime,
        )

        remote_rel = (
            "speech-asr/quartznet15x5-reference/"
            f"myriad-residency/{head[:8]}"
        )
        remote_dir = f"{remote_repo}/work/{remote_rel}"
        run_capture(ssh_command(args.worker, ["mkdir", "-p", remote_dir]))
        run_stream(
            rsync_push_command(
                worker=args.worker,
                sources=[xml, binary],
                remote_dir=remote_dir,
            ),
            "resident-model staging",
        )

        remote_model = f"/work/{remote_rel}/{xml.name}"
        remote_weights = f"/work/{remote_rel}/{binary.name}"
        command = [
            f"{remote_repo}/scripts/run-myriad-tensor.sh",
            "--platform",
            "arm64",
            "--backend",
            "host",
            "custom",
            "--model",
            remote_model,
            "--weights",
            remote_weights,
            "--resident-times",
            ",".join(str(item) for item in times),
        ]
        result = run_stream(
            ssh_command(args.worker, command),
            "resident-network test",
        )

        result_line = next(
            (
                line
                for line in reversed(result.stdout.splitlines())
                if line.startswith("RESIDENT_RESULT ")
            ),
            None,
        )
        if result_line is None:
            raise ValueError("resident-network test did not emit RESIDENT_RESULT")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.stdout, encoding="utf-8")
        print(f"evidence: {args.output}")
        print(result_line)

        if "status=RECHECK_FAIL" in result_line or "status=FAIL" in result_line:
            return 3
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
