#!/usr/bin/env python3
"""Diagnose QuartzNet MYRIAD drift using an exact static IR with no runtime reshape."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import conv1d_output_length  # noqa: E402
from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402
from speech_asr.orchestration import (  # noqa: E402
    rsync_pull_command,
    rsync_push_command,
    ssh_command,
)
from speech_asr.quartznet_reference_frontend import (  # noqa: E402
    nemo_reference_features_numpy,
    reference_feature_lengths,
)

DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean" / "manifest.jsonl"
)
DEFAULT_SPEC = (
    SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
)
DEFAULT_PRETRAINED = (
    ROOT
    / "work"
    / "speech-asr"
    / "pretrained"
    / "quartznet15x5-en-base-v2"
    / "QuartzNet15x5-En-Base.nemo"
)
DEFAULT_DYNAMIC_ONNX = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "myriad"
    / "dynamic"
    / "quartznet15x5_nvidia_ref.onnx"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "work"
    / "speech-asr"
    / "quartznet15x5-reference"
    / "static-shape-diagnostic"
)


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


def run_stream(argv: list[str], label: str) -> None:
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
    if rc != 0:
        raise RuntimeError(
            f"{label} failed ({rc}):\n" + "\n".join(lines[-60:])
        )


def git_head() -> str:
    return run_capture(["git", "rev-parse", "HEAD"]).stdout.strip()


def require_main_clean() -> str:
    branch = run_capture(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "main":
        raise ValueError(f"diagnostic checkout must be main, got {branch!r}")
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout.strip()
    if dirty:
        raise ValueError("diagnostic checkout has tracked local modifications")
    return git_head()


def select_record(
    manifest: pathlib.Path,
    *,
    sample_id: str | None,
    tensor_frames: int | None,
    spec: dict,
) -> dict:
    if sample_id is not None and tensor_frames is not None:
        raise ValueError("--sample-id and --tensor-frames are mutually exclusive")

    first: dict | None = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("manifest record must be an object")
        if first is None:
            first = value
        if sample_id is not None and str(value.get("id")) == sample_id:
            return value
        if tensor_frames is not None:
            _, observed_tensor_frames = reference_feature_lengths(
                int(value["sample_count"]),
                spec,
            )
            if observed_tensor_frames == tensor_frames:
                return value

    if sample_id is not None:
        raise ValueError(f"sample id not found in manifest: {sample_id}")
    if tensor_frames is not None:
        raise ValueError(
            f"no manifest record has tensor_feature_frames={tensor_frames}"
        )
    if first is None:
        raise ValueError("manifest is empty")
    return first


def output_frames(feature_frames: int, spec: dict) -> int:
    layer = spec["network"]["prologue"]
    return conv1d_output_length(
        int(feature_frames),
        kernel=int(layer["kernel"]),
        stride=int(layer["stride"]),
        padding=int(layer["padding"]),
        dilation=int(layer["dilation"]),
    )


def remote_last_line(worker: str, argv: list[str]) -> str:
    proc = run_capture(ssh_command(worker, argv))
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"remote command produced no output: {argv!r}")
    return lines[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-repo")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--pretrained", type=pathlib.Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--dynamic-onnx", type=pathlib.Path, default=DEFAULT_DYNAMIC_ONNX)
    parser.add_argument(
        "--sample-id",
        help="LibriSpeech utterance id to diagnose; defaults to the first manifest record",
    )
    parser.add_argument(
        "--tensor-frames",
        type=int,
        help="diagnose the first manifest record with this exact padded feature length",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        head = require_main_clean()
        for path in (args.manifest, args.spec, args.pretrained, args.dynamic_onnx):
            if not path.is_file():
                raise ValueError(f"required input missing: {path}")

        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if args.tensor_frames is not None and args.tensor_frames <= 0:
            raise ValueError("--tensor-frames must be positive")
        record = select_record(
            args.manifest,
            sample_id=args.sample_id,
            tensor_frames=args.tensor_frames,
            spec=spec,
        )
        sample_count = int(record["sample_count"])
        valid_frames, tensor_frames = reference_feature_lengths(sample_count, spec)
        valid_out = output_frames(valid_frames, spec)
        tensor_out = output_frames(tensor_frames, spec)

        audio_path = args.manifest.parent / str(record["audio_path"])
        samples, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != 16000 or np.asarray(samples).ndim != 1:
            raise ValueError("first LibriSpeech record is not 16 kHz mono")
        if len(samples) != sample_count:
            raise ValueError("first LibriSpeech sample count changed")
        features, observed_valid = nemo_reference_features_numpy(samples, spec)
        if observed_valid != valid_frames or int(features.shape[2]) != tensor_frames:
            raise ValueError("first-sample frontend geometry changed")
        features = np.ascontiguousarray(features, dtype=np.float32)

        sample_tag = str(record["id"]).replace("/", "_")
        out = args.output_dir.resolve() / sample_tag
        export = out / f"export-t{tensor_frames}"
        ir = out / f"openvino-t{tensor_frames}"
        stage = out / f"edge-t{tensor_frames}"
        for path in (export, ir, stage):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

        print(
            json.dumps(
                {
                    "event": "static_shape_diagnostic",
                    "sample_id": record["id"],
                    "sample_count": sample_count,
                    "valid_feature_frames": valid_frames,
                    "tensor_feature_frames": tensor_frames,
                    "valid_output_frames": valid_out,
                    "tensor_output_frames": tensor_out,
                    "runtime_reshape_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        run_stream(
            [
                str(ROOT / "scripts" / "python-training.sh"),
                str(SPEECH_ROOT / "training" / "export_quartznet15x5_reference.py"),
                "--archive",
                str(args.pretrained),
                "--output-dir",
                str(export),
                "--fixed-time-frames",
                str(tensor_frames),
            ],
            "exact-shape ONNX export",
        )
        static_onnx = export / "quartznet15x5_nvidia_ref.onnx"
        run_stream(
            [
                str(ROOT / "scripts" / "run-mo.sh"),
                "--framework",
                "onnx",
                "--",
                "--input_model",
                str(static_onnx),
                "--output_dir",
                str(ir),
                "--model_name",
                "quartznet15x5_nvidia_ref",
                "--data_type",
                "FP16",
            ],
            "exact-shape OpenVINO conversion",
        )
        xml = ir / "quartznet15x5_nvidia_ref.xml"
        binary = ir / "quartznet15x5_nvidia_ref.bin"
        for path in (static_onnx, xml, binary):
            if not path.is_file():
                raise ValueError(f"diagnostic artifact missing: {path}")

        feature_path = stage / "features.f32"
        reference_path = stage / "dynamic-reference.f32"
        static_onnx_path = stage / "static-onnx.f32"
        myriad_path = stage / "myriad.f32"
        features.tofile(feature_path)

        dynamic_session = ort.InferenceSession(
            str(args.dynamic_onnx),
            providers=["CPUExecutionProvider"],
        )
        static_session = ort.InferenceSession(
            str(static_onnx),
            providers=["CPUExecutionProvider"],
        )
        dynamic_logits = dynamic_session.run(
            ["logits"],
            {"features": features},
        )[0].astype(np.float32)
        static_logits = static_session.run(
            ["logits"],
            {"features": features},
        )[0].astype(np.float32)
        if tuple(dynamic_logits.shape) != (1, tensor_out, 29):
            raise ValueError(f"unexpected dynamic ONNX output: {dynamic_logits.shape}")
        if static_logits.shape != dynamic_logits.shape:
            raise ValueError(
                f"static/dynamic ONNX shape mismatch: "
                f"{static_logits.shape} != {dynamic_logits.shape}"
            )
        dynamic_logits.tofile(reference_path)
        static_logits.tofile(static_onnx_path)

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

        remote_rel = (
            f"speech-asr/quartznet15x5-reference/static-shape-diagnostic/"
            f"{head[:8]}-{sample_tag}-t{tensor_frames}"
        )
        remote_dir = f"{remote_repo}/work/{remote_rel}"
        run_capture(
            ssh_command(args.worker, ["mkdir", "-p", remote_dir])
        )
        run_stream(
            rsync_push_command(
                worker=args.worker,
                sources=[xml, binary, feature_path],
                remote_dir=remote_dir,
            ),
            "static diagnostic staging",
        )

        remote_model = f"/work/{remote_rel}/{xml.name}"
        remote_weights = f"/work/{remote_rel}/{binary.name}"
        remote_features = f"/work/{remote_rel}/{feature_path.name}"
        remote_output = f"/work/{remote_rel}/myriad.f32"
        run_stream(
            ssh_command(
                args.worker,
                [
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
                    "--tensor",
                    remote_features,
                    "--output",
                    remote_output,
                ],
            ),
            "exact-shape MYRIAD inference",
        )
        run_stream(
            rsync_pull_command(
                worker=args.worker,
                remote_dir=remote_dir,
                local_dir=stage,
            ),
            "static diagnostic evidence pull",
        )
        pulled = stage / pathlib.Path(remote_dir).name / "myriad.f32"
        if not pulled.is_file():
            # rsync may copy contents directly depending on destination state.
            pulled = stage / "myriad.f32"
        if not pulled.is_file():
            raise ValueError("MYRIAD diagnostic output was not collected")
        myriad_logits = np.fromfile(pulled, dtype=np.float32)
        if myriad_logits.size != tensor_out * 29:
            raise ValueError(
                f"MYRIAD output elements {myriad_logits.size} != {tensor_out * 29}"
            )
        myriad_logits = myriad_logits.reshape(1, tensor_out, 29)
        myriad_logits.tofile(myriad_path)

        comparisons = {
            "dynamic_vs_static_onnx_full": compare_arrays(
                dynamic_logits,
                static_logits,
            ),
            "dynamic_vs_static_onnx_valid": compare_arrays(
                dynamic_logits[:, :valid_out, :],
                static_logits[:, :valid_out, :],
            ),
            "dynamic_onnx_vs_myriad_full": compare_arrays(
                dynamic_logits,
                myriad_logits,
            ),
            "dynamic_onnx_vs_myriad_valid": compare_arrays(
                dynamic_logits[:, :valid_out, :],
                myriad_logits[:, :valid_out, :],
            ),
            "static_onnx_vs_myriad_valid": compare_arrays(
                static_logits[:, :valid_out, :],
                myriad_logits[:, :valid_out, :],
            ),
        }

        result = {
            "schema": "speech-asr/quartznet-reference-myriad-static-diagnostic",
            "version": 1,
            "repo_commit": head,
            "sample_id": record["id"],
            "sample_count": sample_count,
            "valid_feature_frames": valid_frames,
            "tensor_feature_frames": tensor_frames,
            "valid_output_frames": valid_out,
            "tensor_output_frames": tensor_out,
            "runtime_reshape_used": False,
            "precision": "FP16",
            "comparisons": comparisons,
            "artifacts": {
                "dynamic_onnx_sha256": sha256_path(args.dynamic_onnx),
                "static_onnx_sha256": sha256_path(static_onnx),
                "xml_sha256": sha256_path(xml),
                "bin_sha256": sha256_path(binary),
                "features_sha256": sha256_path(feature_path),
                "myriad_output_sha256": sha256_path(myriad_path),
            },
        }
        result_path = out / "result.json"
        result_path.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
