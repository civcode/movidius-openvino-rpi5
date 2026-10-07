#!/usr/bin/env python3
"""Run the fixed-T=512 QuartzNet streaming qualification on Pi 5 + MA2450."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]

from run_quartznet15x5_reference_myriad_edge import (  # noqa: E402
    ensure_remote_runtime,
    load_json,
    remote_text,
    require_main_and_clean,
    run_capture,
    run_local_reference_preparation,
    run_stream,
    run_stream_checked,
    sha256_path,
    validate_local_dataset,
)

sys.path.insert(0, str(SPEECH_ROOT / "python"))
from speech_asr.orchestration import rsync_pull_command, rsync_push_command, ssh_command  # noqa: E402
from speech_asr.quartznet_fixed512 import FIXED_TENSOR_FRAMES  # noqa: E402

EXPECTED_SAMPLES = 2703


def validate_result(
    *,
    result: dict,
    local_head: str,
    manifest_sha: str,
    xml_sha: str,
    bin_sha: str,
    onnx_sha: str,
    full: bool,
    hop_output_frames: int,
) -> None:
    if result.get("schema") != "speech-asr/quartznet-reference-fixed512-evaluation":
        raise ValueError("fixed512 result schema changed")
    if result.get("model_id") != "quartznet15x5_nvidia_ref":
        raise ValueError("fixed512 result model identity changed")
    if result.get("training_performed") is not False:
        raise ValueError("fixed512 evaluation unexpectedly performed training")
    if result.get("runtime", {}).get("repo_commit") != local_head:
        raise ValueError("fixed512 result repository commit mismatch")
    if result.get("manifest", {}).get("sha256") != manifest_sha:
        raise ValueError("fixed512 result manifest hash mismatch")

    policy = result.get("policy", {})
    if policy.get("tensor_feature_frames") != FIXED_TENSOR_FRAMES:
        raise ValueError("fixed512 result tensor shape changed")
    if policy.get("hop_output_frames") != hop_output_frames:
        raise ValueError("fixed512 result hop changed")
    if policy.get("decode") != "single_global_ctc_collapse_after_logit_stitch":
        raise ValueError("fixed512 result stitch/decode policy changed")

    execution = result.get("execution", {})
    if execution.get("engine") != "myriad":
        raise ValueError("fixed512 physical result did not use MYRIAD")
    if execution.get("runtime_backend") != "host":
        raise ValueError("fixed512 physical result did not use host runtime")
    if execution.get("runtime_target") != "arm64":
        raise ValueError("fixed512 physical result did not use arm64")
    if execution.get("network_loads") != 1:
        raise ValueError("fixed512 physical result loaded more than one network")
    if execution.get("warmup_inferences") != 1:
        raise ValueError("fixed512 physical warmup policy changed")

    artifacts = result.get("artifacts", {})
    if artifacts.get("xml_sha256") != xml_sha:
        raise ValueError("fixed512 result XML hash mismatch")
    if artifacts.get("bin_sha256") != bin_sha:
        raise ValueError("fixed512 result BIN hash mismatch")
    if artifacts.get("dynamic_onnx_sha256") != onnx_sha:
        raise ValueError("fixed512 result ONNX hash mismatch")

    parity = result.get("semantic_parity", {})
    hard_gate = parity.get("hard_gate", {})
    if hard_gate.get("pass") is not True:
        raise ValueError("fixed512 ONNX/MYRIAD first-window semantic probe failed")
    if float(
        parity.get("valid_comparison", {}).get("frame_argmax_agreement", 0.0)
    ) < 0.99:
        raise ValueError("fixed512 ONNX/MYRIAD first-window argmax agreement < 0.99")

    if full:
        if result.get("samples") != EXPECTED_SAMPLES:
            raise ValueError("fixed512 full dev-clean sample count changed")
        if result.get("full_dev_clean") is not True:
            raise ValueError("fixed512 full-dev-clean marker missing")
        if result.get("status") not in {"pass", "fail"}:
            raise ValueError("fixed512 full result has invalid status")
    elif result.get("status") != "diagnostic":
        raise ValueError("fixed512 partial run must be diagnostic")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-repo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpuset", default="0-15")
    parser.add_argument("--hop-output-frames", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()

    try:
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        if not 1 <= args.hop_output_frames <= 256:
            raise ValueError("--hop-output-frames must be in 1..256")
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

        run_id = (
            f"quartznet15x5-fixed512-{local_head[:8]}-"
            f"{manifest_sha[:8]}-{xml_sha[:8]}-hop{args.hop_output_frames}"
        )
        remote_base = f"{remote_repo}/work/speech-asr/fixed512-eval/{run_id}"
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

        print("[fixed512-edge] staging LibriSpeech dev-clean", flush=True)
        run_stream_checked(
            rsync_push_command(
                worker=args.worker,
                sources=[dataset],
                remote_dir=remote_dataset_parent,
            ),
            "LibriSpeech staging",
        )
        run_stream_checked(
            rsync_push_command(
                worker=args.worker,
                sources=[xml, binary, artifacts_json],
                remote_dir=remote_ir,
            ),
            "OpenVINO artifact staging",
        )
        run_stream_checked(
            rsync_push_command(
                worker=args.worker,
                sources=[dynamic],
                remote_dir=remote_dynamic,
            ),
            "dynamic ONNX staging",
        )

        remote_manifest = f"{remote_dataset_parent}/dev-clean/manifest.jsonl"
        remote_onnx = f"{remote_dynamic}/{dynamic.name}"
        preflight = run_capture(
            ssh_command(
                args.worker,
                [
                    f"{remote_repo}/scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh",
                    "--manifest",
                    remote_manifest,
                    "--ir-dir",
                    remote_ir,
                    "--dynamic-onnx",
                    remote_onnx,
                    "--hop-output-frames",
                    str(args.hop_output_frames),
                    "--preflight-only",
                ],
            )
        )
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") else "\n")

        command = [
            f"{remote_repo}/scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh",
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
            "--hop-output-frames",
            str(args.hop_output_frames),
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
            / "quartznet15x5-reference-fixed512-libri-dev-clean"
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

        if pull_evidence.returncode != 0:
            raise RuntimeError(
                "fixed512 evidence pull failed:\n" + pull_evidence.stdout
            )

        result_path = output_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"fixed512 evaluation produced no result.json; edge rc={proc.returncode}"
            )
        result = load_json(result_path)
        validate_result(
            result=result,
            local_head=local_head,
            manifest_sha=manifest_sha,
            xml_sha=xml_sha,
            bin_sha=bin_sha,
            onnx_sha=onnx_sha,
            full=args.max_samples is None,
            hop_output_frames=args.hop_output_frames,
        )

        quality = result["quality"]
        execution = result["execution"]
        summary = {
            "status": result["status"],
            "edge_exit_status": proc.returncode,
            "repo_commit": local_head,
            "samples": result["samples"],
            "windows": execution["windows"],
            "network_loads": execution["network_loads"],
            "window_policy": result["policy"],
            "wer": quality["wer"],
            "cer": quality["cer"],
            "qualified_whole_utterance_pytorch_wer": quality[
                "qualified_whole_utterance_pytorch_wer"
            ],
            "absolute_wer_delta": quality["absolute_wer_delta"],
            "semantic_valid_frame_argmax_agreement": result["semantic_parity"][
                "valid_comparison"
            ]["frame_argmax_agreement"],
            "inference_latency_p50_ms": execution["inference_latency_p50_ms"],
            "inference_latency_p95_ms": execution["inference_latency_p95_ms"],
            "inference_realtime_factor": execution["inference_realtime_factor"],
            "result": str(result_path),
        }
        print(json.dumps(summary, sort_keys=True, indent=2))

        if proc.returncode not in {0, 3}:
            raise RuntimeError(
                f"edge fixed512 evaluation failed unexpectedly ({proc.returncode})"
            )
        return 0 if result["status"] != "fail" else 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
