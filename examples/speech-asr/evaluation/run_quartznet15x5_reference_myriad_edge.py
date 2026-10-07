#!/usr/bin/env python3
"""Prepare, stage, run and validate the qualified QuartzNet LibriSpeech MYRIAD reproof."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.orchestration import rsync_pull_command, rsync_push_command, ssh_command  # noqa: E402

EXPECTED_SAMPLES = 2703
QUALIFIED_WER = 0.037939781625675524
MAX_WER = 0.05
MAX_ABS_WER_DELTA = 0.005
MAX_FRAME_TOTAL_VARIATION = 0.002


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run_stream(argv: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    return subprocess.CompletedProcess(argv, returncode, "".join(lines), None)


def remote_text(worker: str, argv: list[str]) -> str:
    proc = run_capture(ssh_command(worker, argv))
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"remote command returned no output: {argv!r}")
    return lines[-1]


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_main_and_clean() -> str:
    branch = run_capture(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "main":
        raise ValueError(f"controller checkout must remain on main, got {branch!r}")
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no]
    ).stdout.strip()
    if dirty:
        raise ValueError("controller tracked worktree must be clean")
    return run_capture(["git", "rev-parse", "HEAD"]).stdout.strip()


def run_local_reference_preparation(device: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    run_stream([str(ROOT / "scripts" / "prepare-cnn-ctc-v18-pretrained.sh")])
    run_stream([str(ROOT / "scripts" / "prepare-librispeech-dev-clean.sh")])
    qual = run_stream(
        [
            str(ROOT / "scripts" / "qualify-quartznet15x5-reference-numpy.sh"),
            "--device",
            device,
        ]
    )
    if qual.returncode != 0:
        raise RuntimeError("NumPy source-reference qualification failed")
    prepared = run_stream(
        [str(ROOT / "scripts" / "prepare-quartznet15x5-reference-myriad.sh")]
    )
    if prepared.returncode != 0:
        raise RuntimeError("QuartzNet MYRIAD artifact preparation failed")

    dataset = ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean"
    myriad = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad"
    ir = myriad / "openvino" / "fp16"
    dynamic = myriad / "dynamic" / "quartznet15x5_nvidia_ref.onnx"
    for path in (
        dataset / "manifest.jsonl",
        dataset / "dataset.json",
        ir / "quartznet15x5_nvidia_ref.xml",
        ir / "quartznet15x5_nvidia_ref.bin",
        ir / "artifacts.json",
        dynamic,
    ):
        if not path.is_file():
            raise ValueError(f"prepared artifact missing: {path}")
    return dataset, ir, dynamic


def validate_local_dataset(dataset: pathlib.Path) -> tuple[pathlib.Path, str]:
    manifest = dataset / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"dev-clean manifest count changed: {len(records)}")
    metadata = load_json(dataset / "dataset.json")
    if metadata.get("id") != "librispeech-dev-clean":
        raise ValueError("dev-clean dataset identity changed")
    if metadata.get("archive_md5") != "42e2234ba48799c1f50f24a7926300a1":
        raise ValueError("dev-clean archive MD5 changed")
    return manifest, sha256_path(manifest)


def ensure_remote_runtime(
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
        "check-reshape",
    ]
    check = run_capture(ssh_command(worker, check_cmd), check=False)
    if check.returncode == 0:
        print(check.stdout, end="" if check.stdout.endswith("\n") else "\n")
        return
    if not refresh_runtime:
        raise RuntimeError(
            "edge host runtime lacks the new reshape-capable hello_myriad. "
            "Rerun with --refresh-runtime.\n"
            + check.stdout
        )

    print("[quartznet-edge] rebuilding arm64 runtime image for reshape support", flush=True)
    build = run_stream(
        ssh_command(
            worker,
            [f"{remote_repo}/build.sh", "--platform", "arm64"],
        )
    )
    if build.returncode != 0:
        raise RuntimeError("edge arm64 runtime rebuild failed")
    pull = run_stream(
        ssh_command(
            worker,
            [
                f"{remote_repo}/scripts/pull-runtime.sh",
                "--platform",
                "arm64",
            ],
        )
    )
    if pull.returncode != 0:
        raise RuntimeError("edge host-runtime extraction failed")
    check = run_capture(ssh_command(worker, check_cmd))
    print(check.stdout, end="" if check.stdout.endswith("\n") else "\n")


def validate_result(
    *,
    result: dict[str, Any],
    local_head: str,
    manifest_sha: str,
    xml_sha: str,
    bin_sha: str,
    onnx_sha: str,
    full: bool,
) -> None:
    expected_status = "pass" if full else "diagnostic"
    if result.get("status") != expected_status:
        raise ValueError(
            f"edge result status {result.get('status')!r} != {expected_status!r}"
        )
    if result.get("model_id") != "quartznet15x5_nvidia_ref":
        raise ValueError("edge result model identity mismatch")
    if result.get("training_performed") is not False:
        raise ValueError("edge result unexpectedly performed training")
    if result.get("runtime", {}).get("repo_commit") != local_head:
        raise ValueError("edge result repository commit mismatch")
    if result.get("manifest", {}).get("sha256") != manifest_sha:
        raise ValueError("edge result manifest hash mismatch")
    artifacts = result.get("artifacts", {})
    if artifacts.get("xml_sha256") != xml_sha:
        raise ValueError("edge result XML hash mismatch")
    if artifacts.get("bin_sha256") != bin_sha:
        raise ValueError("edge result BIN hash mismatch")
    if artifacts.get("dynamic_onnx_sha256") != onnx_sha:
        raise ValueError("edge result ONNX hash mismatch")
    myriad = result.get("myriad", {})
    if myriad.get("runtime_backend") != "host" or myriad.get("runtime_target") != "arm64":
        raise ValueError("edge result runtime backend/target mismatch")
    if myriad.get("execution_mode") != "exact-time-runtime-reshape-v1":
        raise ValueError("edge result execution mode changed")

    parity = result.get("semantic_parity", {})
    comparison = parity.get("comparison", {})
    if comparison.get("frame_argmax_agreement") != 1.0:
        raise ValueError("edge ONNX/MYRIAD frame argmax parity failed")
    if float(comparison.get("max_frame_total_variation", 1.0)) > MAX_FRAME_TOTAL_VARIATION:
        raise ValueError("edge ONNX/MYRIAD probability parity failed")

    if full:
        if result.get("samples") != EXPECTED_SAMPLES or result.get("full_dev_clean") is not True:
            raise ValueError("edge full dev-clean sample boundary changed")
        quality = result.get("quality", {})
        wer = float(quality.get("wer", 1.0))
        delta = abs(wer - QUALIFIED_WER)
        if quality.get("pass") is not True or wer > MAX_WER or delta > MAX_ABS_WER_DELTA:
            raise ValueError(
                f"edge clean-speech quality gate failed: wer={wer}, delta={delta}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-repo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpuset", default="0-3")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()

    try:
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        if not all(ch.isdigit() or ch in ",-" for ch in args.cpuset):
            raise ValueError("invalid --cpuset")

        local_head = require_main_and_clean()
        dataset, ir, dynamic = run_local_reference_preparation(args.device)
        manifest, manifest_sha = validate_local_dataset(dataset)
        xml = ir / "quartznet15x5_nvidia_ref.xml"
        binary = ir / "quartznet15x5_nvidia_ref.bin"
        artifacts_json = ir / "artifacts.json"
        xml_sha = sha256_path(xml)
        bin_sha = sha256_path(binary)
        onnx_sha = sha256_path(dynamic)

        home = remote_text(
            args.worker,
            ["sh", "-c", 'printf "%s\\n" "$HOME"'],
        )
        remote_repo = args.edge_repo or f"{home}/workspace/movidius-openvino-rpi5"
        remote_branch = remote_text(
            args.worker,
            ["git", "-C", remote_repo, "branch", "--show-current"],
        )
        if remote_branch != "main":
            raise ValueError(
                f"edge human checkout must remain on main, got {remote_branch!r}"
            )
        pull = run_capture(
            ssh_command(
                args.worker,
                ["git", "-C", remote_repo, "pull", "--ff-only"],
            )
        )
        print(pull.stdout, end="" if pull.stdout.endswith("\n") else "\n")
        remote_head = remote_text(
            args.worker,
            ["git", "-C", remote_repo, "rev-parse", "HEAD"],
        )
        if remote_head != local_head:
            raise ValueError(
                f"controller/edge revision mismatch: {local_head} != {remote_head}"
            )

        ensure_remote_runtime(
            worker=args.worker,
            remote_repo=remote_repo,
            refresh_runtime=args.refresh_runtime,
        )

        run_id = f"quartznet15x5-ref-{local_head[:8]}-{manifest_sha[:8]}-{xml_sha[:8]}"
        remote_base = f"{remote_repo}/work/speech-asr/reference-myriad-eval/{run_id}"
        remote_dataset_parent = f"{remote_base}/dataset/librispeech"
        remote_ir = f"{remote_base}/input/openvino/fp16"
        remote_dynamic = f"{remote_base}/input/dynamic"
        remote_evidence = f"{remote_base}/evidence"
        run_capture(
            ssh_command(
                args.worker,
                [
                    "mkdir",
                    "-p",
                    remote_dataset_parent,
                    remote_ir,
                    remote_dynamic,
                    remote_evidence,
                ],
            )
        )

        print("[quartznet-edge] staging LibriSpeech dev-clean", flush=True)
        run_stream(
            rsync_push_command(
                worker=args.worker,
                sources=[dataset],
                remote_dir=remote_dataset_parent,
            )
        )
        run_stream(
            rsync_push_command(
                worker=args.worker,
                sources=[xml, binary, artifacts_json],
                remote_dir=remote_ir,
            )
        )
        run_stream(
            rsync_push_command(
                worker=args.worker,
                sources=[dynamic],
                remote_dir=remote_dynamic,
            )
        )

        remote_manifest = f"{remote_dataset_parent}/dev-clean/manifest.jsonl"
        remote_manifest_sha = remote_text(
            args.worker,
            ["sha256sum", remote_manifest],
        ).split()[0]
        if remote_manifest_sha != manifest_sha:
            raise ValueError(
                f"edge staged manifest SHA mismatch: {remote_manifest_sha}"
            )

        remote_onnx = f"{remote_dynamic}/{dynamic.name}"
        preflight = run_capture(
            ssh_command(
                args.worker,
                [
                    f"{remote_repo}/scripts/evaluate-quartznet15x5-reference-myriad.sh",
                    "--manifest",
                    remote_manifest,
                    "--ir-dir",
                    remote_ir,
                    "--dynamic-onnx",
                    remote_onnx,
                    "--preflight-only",
                ],
            )
        )
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") else "\n")

        command = [
            f"{remote_repo}/scripts/evaluate-quartznet15x5-reference-myriad.sh",
            "--manifest",
            remote_manifest,
            "--ir-dir",
            remote_ir,
            "--dynamic-onnx",
            remote_onnx,
            "--work-dir",
            f"{remote_evidence}/sessions",
            "--output",
            f"{remote_evidence}/result.json",
            "--log",
            f"{remote_evidence}/evaluator.log",
            "--cpuset",
            args.cpuset,
            "--fresh",
        ]
        if args.max_samples is not None:
            command += ["--max-samples", str(args.max_samples)]

        proc = run_stream(ssh_command(args.worker, command))
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else ROOT
            / "work"
            / "speech-asr"
            / "deployed-evaluations"
            / "quartznet15x5-reference-libri-dev-clean"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pull_evidence = run_capture(
            rsync_pull_command(
                worker=args.worker,
                remote_dir=remote_evidence,
                local_dir=output_dir,
            ),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"edge QuartzNet MYRIAD evaluation failed ({proc.returncode}); "
                f"evidence pull rc={pull_evidence.returncode}\n"
                + "\n".join(proc.stdout.splitlines()[-60:])
            )
        if pull_evidence.returncode != 0:
            raise RuntimeError(
                "edge evaluation completed but evidence pull failed:\n"
                + pull_evidence.stdout
            )

        result_path = output_dir / "result.json"
        result = load_json(result_path)
        validate_result(
            result=result,
            local_head=local_head,
            manifest_sha=manifest_sha,
            xml_sha=xml_sha,
            bin_sha=bin_sha,
            onnx_sha=onnx_sha,
            full=args.max_samples is None,
        )
        quality = result["quality"]
        myriad = result["myriad"]
        summary = {
            "status": result["status"],
            "repo_commit": local_head,
            "samples": result["samples"],
            "result": str(result_path),
            "wer": quality["wer"],
            "cer": quality["cer"],
            "qualified_pytorch_wer": quality["qualified_pytorch_wer"],
            "absolute_wer_delta": quality["absolute_wer_delta"],
            "semantic_frame_argmax_agreement": result["semantic_parity"]["comparison"][
                "frame_argmax_agreement"
            ],
            "semantic_max_frame_total_variation": result["semantic_parity"]["comparison"][
                "max_frame_total_variation"
            ],
            "shape_sessions": myriad["shape_sessions"],
            "inference_latency_p50_ms": myriad["inference_latency_p50_ms"],
            "inference_latency_p95_ms": myriad["inference_latency_p95_ms"],
            "inference_realtime_factor": myriad["inference_realtime_factor"],
        }
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
