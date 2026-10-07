#!/usr/bin/env python3
"""Evaluate fixed-T=512 overlapping-window QuartzNet on LibriSpeech dev-clean."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from evaluate_quartznet15x5_reference_myriad import (  # noqa: E402
    DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION,
    PersistentMyriadServer,
    container_work,
    git_head,
    load_spec_vocab,
    read_manifest,
    require_runtime,
    sha256_path,
)
from speech_asr.cnn_ctc import (  # noqa: E402
    aggregate_decoder_diagnostics,
    greedy_decode_logits_diagnostics,
)
from speech_asr.cnn_ctc_compare import compare_arrays  # noqa: E402
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)
from speech_asr.quartznet_fixed512 import (  # noqa: E402
    DEFAULT_HOP_OUTPUT_FRAMES,
    FIXED_TENSOR_FRAMES,
    FULL_OUTPUT_FRAMES,
    Fixed512LogitStitcher,
    fixed512_features,
    fixed512_policy_dict,
    plan_fixed512_windows,
    quartznet_output_frames,
    validate_fixed512_geometry,
)
from speech_asr.quartznet_reference_frontend import (  # noqa: E402
    reference_feature_lengths,
)

QUALIFIED_WER = 0.037939781625675524
QUALIFIED_CER = 0.012506494000177396
MAX_STREAMING_WER = 0.05
MIN_SEMANTIC_ARGMAX_AGREEMENT = 0.99
DEFAULT_SPEC = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "work" / "speech-asr" / "librispeech" / "dev-clean" / "manifest.jsonl"
DEFAULT_IR = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad" / "openvino" / "fp16"
DEFAULT_ONNX = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "myriad" / "dynamic" / "quartznet15x5_nvidia_ref.onnx"
DEFAULT_OUTPUT = ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "fixed512-evaluation" / "result.json"


def sum_rate(counts) -> float:
    errors = sum(item.errors for item in counts)
    refs = sum(item.reference_units for item in counts)
    return errors / max(1, refs)


def validate_artifacts(
    *,
    ir_dir: pathlib.Path,
    dynamic_onnx: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    xml = ir_dir / "quartznet15x5_nvidia_ref.xml"
    binary = ir_dir / "quartznet15x5_nvidia_ref.bin"
    artifacts_path = ir_dir / "artifacts.json"
    for path in (xml, binary, artifacts_path, dynamic_onnx):
        if not path.is_file():
            raise ValueError(f"required file missing: {path}")

    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    if artifacts.get("model_id") != "quartznet15x5_nvidia_ref":
        raise ValueError("MYRIAD artifact model identity changed")
    if artifacts.get("carrier_time_frames") != FIXED_TENSOR_FRAMES:
        raise ValueError("MYRIAD carrier is not fixed at T=512")
    hashes = artifacts.get("artifacts", {})
    if hashes.get("xml_sha256") != sha256_path(xml):
        raise ValueError("MYRIAD XML hash mismatch")
    if hashes.get("bin_sha256") != sha256_path(binary):
        raise ValueError("MYRIAD BIN hash mismatch")
    if hashes.get("dynamic_onnx_sha256") != sha256_path(dynamic_onnx):
        raise ValueError("dynamic ONNX hash mismatch")
    return xml, binary, artifacts_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("onnx", "myriad"), required=True)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR)
    parser.add_argument("--dynamic-onnx", type=pathlib.Path, default=DEFAULT_ONNX)
    parser.add_argument("--platform", choices=("arm64", "armv7", "amd64"), default="arm64")
    parser.add_argument(
        "--runtime-backend",
        choices=("auto", "host", "docker"),
        default="host",
    )
    parser.add_argument(
        "--hop-output-frames",
        type=int,
        default=DEFAULT_HOP_OUTPUT_FRAMES,
        help="window hop on QuartzNet output lattice; 128 is ~50%% overlap",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "quartznet15x5-reference" / "fixed512-evaluation",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    server: PersistentMyriadServer | None = None
    try:
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        if args.progress_interval < 0:
            raise ValueError("--progress-interval must be >= 0")

        spec, vocab = load_spec_vocab(args.spec, args.vocab)
        validate_fixed512_geometry(spec)
        policy = fixed512_policy_dict(args.hop_output_frames)
        records = read_manifest(args.manifest)
        full = args.max_samples is None
        if args.max_samples is not None:
            records = records[: args.max_samples]

        if not args.dynamic_onnx.is_file():
            raise ValueError(f"dynamic ONNX missing: {args.dynamic_onnx}")
        ort_session = ort.InferenceSession(
            str(args.dynamic_onnx),
            providers=["CPUExecutionProvider"],
        )

        xml = binary = artifacts_path = None
        resolved_backend = None
        if args.engine == "myriad":
            xml, binary, artifacts_path = validate_artifacts(
                ir_dir=args.ir_dir,
                dynamic_onnx=args.dynamic_onnx,
            )
            resolved_backend = require_runtime(args.platform, args.runtime_backend)
            command = [
                str(ROOT / "scripts" / "run-myriad-tensor.sh"),
                "--platform",
                args.platform,
                "--backend",
                resolved_backend,
                "custom-server",
                "--model",
                container_work(xml),
                "--weights",
                container_work(binary),
                "--warmup",
                "1",
            ]
            args.work_dir.mkdir(parents=True, exist_ok=True)
            server = PersistentMyriadServer(
                command=command,
                input_elements=64 * FIXED_TENSOR_FRAMES,
                output_elements=FULL_OUTPUT_FRAMES * 29,
                log_path=args.work_dir / "fixed512-myriad.log",
            )
            if server.warmup != 1:
                raise RuntimeError("MYRIAD fixed512 warmup count changed")
            print(
                "[fixed512] MYRIAD graph ready "
                f"load_ms={server.load_ms:.3f} tensor_frames={FIXED_TENSOR_FRAMES}",
                flush=True,
            )

        word_counts = []
        char_counts = []
        decoder_samples = []
        frontend_latencies: list[float] = []
        inference_latencies: list[float] = []
        per_sample: list[dict] = []
        windows_per_sample: list[int] = []
        total_audio_samples = 0
        total_frontend_seconds = 0.0
        total_inference_seconds = 0.0
        request_index = 0
        parity = None
        started = time.monotonic()

        for sample_index, record in enumerate(records, start=1):
            audio_path = args.manifest.parent / str(record["audio_path"])
            samples, sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=False,
            )
            samples = np.asarray(samples, dtype=np.float32)
            if sample_rate != 16000 or samples.ndim != 1:
                raise ValueError(f"{record['id']}: expected 16 kHz mono audio")
            if samples.size != int(record["sample_count"]):
                raise ValueError(f"{record['id']}: sample count changed")

            valid_features, _ = reference_feature_lengths(samples.size, spec)
            global_output_frames = quartznet_output_frames(valid_features, spec)
            windows = plan_fixed512_windows(
                samples.size,
                spec,
                hop_output_frames=args.hop_output_frames,
            )
            stitcher = Fixed512LogitStitcher(global_output_frames, 29)
            sample_frontend_ms = 0.0
            sample_inference_ms = 0.0

            for window in windows:
                segment = samples[window.start_sample : window.end_sample]

                frontend_start = time.perf_counter()
                features, observed_valid_features, observed_valid_outputs = (
                    fixed512_features(segment, spec)
                )
                frontend_ms = (time.perf_counter() - frontend_start) * 1000.0
                frontend_latencies.append(frontend_ms)
                sample_frontend_ms += frontend_ms
                total_frontend_seconds += frontend_ms / 1000.0
                if observed_valid_features != window.valid_feature_frames:
                    raise ValueError(
                        f"{record['id']}: window {window.index} valid features changed"
                    )
                if observed_valid_outputs != window.valid_output_frames:
                    raise ValueError(
                        f"{record['id']}: window {window.index} valid outputs changed"
                    )

                if args.engine == "myriad":
                    assert server is not None
                    request_index += 1
                    logits_flat, inference_ms = server.infer(
                        features,
                        request_index=request_index,
                    )
                    logits = logits_flat.reshape(1, FULL_OUTPUT_FRAMES, 29)
                    if parity is None:
                        reference_logits = ort_session.run(
                            ["logits"],
                            {"features": features},
                        )[0].astype(np.float32)
                        valid = window.valid_output_frames
                        valid_comparison = compare_arrays(
                            reference_logits[:, :valid, :],
                            logits[:, :valid, :],
                        )
                        full_comparison = compare_arrays(reference_logits, logits)
                        argmax_agreement = float(
                            valid_comparison["frame_argmax_agreement"]
                        )
                        semantic_probe_pass = (
                            argmax_agreement >= MIN_SEMANTIC_ARGMAX_AGREEMENT
                        )
                        if not semantic_probe_pass:
                            raise RuntimeError(
                                "fixed512 ONNX/MYRIAD semantic probe failed: "
                                f"argmax={argmax_agreement} "
                                f"< {MIN_SEMANTIC_ARGMAX_AGREEMENT}"
                            )
                        parity = {
                            "sample_id": record["id"],
                            "window_index": window.index,
                            "valid_output_frames": valid,
                            "hard_gate": {
                                "minimum_valid_frame_argmax_agreement": (
                                    MIN_SEMANTIC_ARGMAX_AGREEMENT
                                ),
                                "observed_valid_frame_argmax_agreement": (
                                    argmax_agreement
                                ),
                                "pass": semantic_probe_pass,
                            },
                            "valid_comparison": valid_comparison,
                            "full_tensor_comparison": full_comparison,
                            "probability_drift_diagnostic": {
                                "reference_max_frame_total_variation": (
                                    DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION
                                ),
                                "within_reference_tolerance": (
                                    float(
                                        valid_comparison[
                                            "max_frame_total_variation"
                                        ]
                                    )
                                    <= DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION
                                ),
                                "gating": False,
                            },
                        }
                        print(
                            "[fixed512] semantic probe PASS "
                            f"sample={record['id']} "
                            f"argmax={argmax_agreement:.9f} "
                            f"max_tv={valid_comparison['max_frame_total_variation']:.9f}",
                            flush=True,
                        )
                else:
                    infer_start = time.perf_counter()
                    logits = ort_session.run(
                        ["logits"],
                        {"features": features},
                    )[0].astype(np.float32)
                    inference_ms = (time.perf_counter() - infer_start) * 1000.0

                if logits.shape != (1, FULL_OUTPUT_FRAMES, 29):
                    raise ValueError(
                        f"{record['id']}: fixed512 logits shape changed: {logits.shape}"
                    )
                if not np.isfinite(logits).all():
                    raise RuntimeError(
                        f"{record['id']}: non-finite {args.engine} logits"
                    )

                inference_latencies.append(inference_ms)
                sample_inference_ms += inference_ms
                total_inference_seconds += inference_ms / 1000.0
                stitcher.add(window, logits[0])

            stitched_logits, owners = stitcher.finish()
            decoder = greedy_decode_logits_diagnostics(stitched_logits, vocab)
            hypothesis = decoder["hypothesis"]
            reference = str(record["text"])
            words = word_error_counts(reference, hypothesis)
            chars = character_error_counts(reference, hypothesis)
            word_counts.append(words)
            char_counts.append(chars)
            windows_per_sample.append(len(windows))
            total_audio_samples += int(samples.size)
            decoder_samples.append(
                {
                    "reference_words": len(reference.split()),
                    "reference_characters_no_spaces": len(reference.replace(" ", "")),
                    "decoder": {
                        key: value
                        for key, value in decoder.items()
                        if key != "hypothesis"
                    },
                    "hypothesis": hypothesis,
                }
            )
            per_sample.append(
                {
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "valid_feature_frames": valid_features,
                    "global_output_frames": global_output_frames,
                    "windows": len(windows),
                    "frontend_ms_total": sample_frontend_ms,
                    "inference_ms_total": sample_inference_ms,
                    "owner_window_min": int(owners.min()),
                    "owner_window_max": int(owners.max()),
                }
            )

            if args.progress_interval and (
                sample_index == 1
                or sample_index == len(records)
                or sample_index % args.progress_interval == 0
            ):
                elapsed = time.monotonic() - started
                print(
                    "[fixed512] "
                    f"processed={sample_index}/{len(records)} "
                    f"windows={sum(windows_per_sample)} "
                    f"wer={sum_rate(word_counts):.6f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        if args.engine == "myriad" and parity is None:
            raise RuntimeError("fixed512 semantic parity probe did not run")

        wer = sum_rate(word_counts)
        cer = sum_rate(char_counts)
        quality_pass = wer <= MAX_STREAMING_WER
        status = "diagnostic" if not full else ("pass" if quality_pass else "fail")
        audio_seconds = total_audio_samples / 16000.0
        infer_summary = latency_summary_ms(inference_latencies)
        frontend_summary = latency_summary_ms(frontend_latencies)
        hypotheses_path = args.output.with_name("hypotheses.jsonl")
        hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
        hypotheses_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in per_sample),
            encoding="utf-8",
        )

        execution = {
            "engine": args.engine,
            "windows": sum(windows_per_sample),
            "windows_per_utterance_mean": (
                sum(windows_per_sample) / len(windows_per_sample)
            ),
            "inference_latency_p50_ms": infer_summary["p50_ms"],
            "inference_latency_p95_ms": infer_summary["p95_ms"],
            "inference_latency_mean_ms": (
                sum(inference_latencies) / len(inference_latencies)
            ),
            "inference_realtime_factor": total_inference_seconds / audio_seconds,
            "frontend_latency_p50_ms": frontend_summary["p50_ms"],
            "frontend_latency_p95_ms": frontend_summary["p95_ms"],
            "frontend_latency_mean_ms": (
                sum(frontend_latencies) / len(frontend_latencies)
            ),
            "frontend_realtime_factor": total_frontend_seconds / audio_seconds,
        }
        if args.engine == "myriad":
            assert server is not None
            execution.update(
                {
                    "runtime_backend": resolved_backend,
                    "runtime_target": args.platform,
                    "openvino_version": "2020.3.2",
                    "precision": "FP16",
                    "network_loads": 1,
                    "model_load_ms": server.load_ms,
                    "warmup_inferences": server.warmup,
                    "measurement_scope": (
                        "one persistent fixed-T=512 MYRIAD network; one unmeasured "
                        "warmup; inference RTF includes overlapping-window compute; "
                        "model load excluded"
                    ),
                }
            )
        else:
            execution.update(
                {
                    "onnxruntime_provider": "CPUExecutionProvider",
                    "measurement_scope": (
                        "fixed-T=512 window policy executed with dynamic ONNX on CPU"
                    ),
                }
            )

        result = {
            "schema": "speech-asr/quartznet-reference-fixed512-evaluation",
            "version": 1,
            "status": status,
            "model_id": "quartznet15x5_nvidia_ref",
            "training_performed": False,
            "dataset_id": "librispeech-dev-clean",
            "full_dev_clean": full,
            "samples": len(records),
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256_path(args.manifest),
            },
            "policy": policy,
            "quality": {
                "wer": wer,
                "cer": cer,
                "qualified_whole_utterance_pytorch_wer": QUALIFIED_WER,
                "qualified_whole_utterance_pytorch_cer": QUALIFIED_CER,
                "absolute_wer_delta": abs(wer - QUALIFIED_WER),
                "max_streaming_wer": MAX_STREAMING_WER,
                "pass": quality_pass if full else None,
            },
            "decoder": aggregate_decoder_diagnostics(decoder_samples),
            "semantic_parity": parity,
            "execution": execution,
            "artifacts": {
                "dynamic_onnx_sha256": sha256_path(args.dynamic_onnx),
                **(
                    {
                        "xml_sha256": sha256_path(xml),
                        "bin_sha256": sha256_path(binary),
                        "artifacts_json_sha256": sha256_path(artifacts_path),
                    }
                    if args.engine == "myriad"
                    else {}
                ),
            },
            "runtime": {
                "repo_commit": git_head(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "onnxruntime_version": ort.__version__,
                "soundfile_version": sf.__version__,
            },
            "hypotheses": str(hypotheses_path),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if status != "fail" else 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
