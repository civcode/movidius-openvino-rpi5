#!/usr/bin/env python3
"""Evaluate cnn_ctc_v3 FP16 IR on MYRIAD against a normalized AMI manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.audio import read_f32le  # noqa: E402
from speech_asr.cnn_ctc import (  # noqa: E402
    acoustic_output_length,
    aggregate_decoder_diagnostics,
    canonical_sha256,
    greedy_decode_logits_diagnostics,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.cnn_ctc_frontend import logmel_features  # noqa: E402
from speech_asr.contracts import validate_experiment_result, validate_speech_sample  # noqa: E402
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "ami-smoke-v1" / "manifest.jsonl"
DEFAULT_IR = ROOT / "work" / "speech-asr" / "cnn_ctc_v3" / "openvino" / "fp16"

INFER_RE = re.compile(r"inference\s+1\s*:\s*([0-9.]+)\s*ms")
LOAD_RE = re.compile(r"load\+compile\s*:\s*([0-9.]+)\s*ms")


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def records(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield validate_speech_sample(json.loads(line))


def resolve_audio(manifest: pathlib.Path, record: dict) -> pathlib.Path:
    path = pathlib.Path(record["audio"]["path"])
    return path if path.is_absolute() else manifest.parent / path


def container_work(path: pathlib.Path) -> str:
    try:
        relative = path.resolve().relative_to((ROOT / "work").resolve())
    except ValueError as exc:
        raise ValueError(f"path must be below repository work/: {path}") from exc
    return "/work/" + relative.as_posix()


def sum_rate(counts) -> float:
    errors = sum(value.errors for value in counts)
    refs = sum(value.reference_units for value in counts)
    return errors / max(1, refs)


def parse_myriad_timing(text: str) -> tuple[float, float] | None:
    if "RESULT: PASS" not in text:
        return None
    infer_match = INFER_RE.search(text)
    load_match = LOAD_RE.search(text)
    if infer_match is None or load_match is None:
        return None
    return float(infer_match.group(1)), float(load_match.group(1))


def write_sample_cache(
    *,
    cache_path: pathlib.Path,
    record_id: str,
    platform: str,
    feature_path: pathlib.Path,
    logits_path: pathlib.Path,
    log_path: pathlib.Path,
    feature_sha256: str,
    xml_sha256: str,
    bin_sha256: str,
    output_elements: int,
    infer_ms: float,
    load_ms: float,
) -> None:
    value = {
        "schema": "speech-asr/myriad-sample-cache",
        "version": 1,
        "sample_id": record_id,
        "platform": platform,
        "feature_sha256": feature_sha256,
        "xml_sha256": xml_sha256,
        "bin_sha256": bin_sha256,
        "output_elements": output_elements,
        "logits_sha256": sha256_path(logits_path),
        "log_sha256": sha256_path(log_path),
        "inference_ms": infer_ms,
        "load_ms": load_ms,
    }
    cache_path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_sample_cache(
    *,
    cache_path: pathlib.Path,
    record_id: str,
    platform: str,
    feature_path: pathlib.Path,
    logits_path: pathlib.Path,
    log_path: pathlib.Path,
    feature_sha256: str,
    xml_sha256: str,
    bin_sha256: str,
    output_elements: int,
) -> tuple[float, float] | None:
    required = (feature_path, logits_path, log_path)
    if any(not path.is_file() for path in required):
        return None
    if feature_path.stat().st_size == 0:
        return None
    if sha256_path(feature_path) != feature_sha256:
        return None
    if logits_path.stat().st_size != output_elements * np.dtype(np.float32).itemsize:
        return None

    log_text = log_path.read_text(encoding="utf-8")
    timing = parse_myriad_timing(log_text)
    if timing is None:
        return None
    infer_ms, load_ms = timing

    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        expected = {
            "schema": "speech-asr/myriad-sample-cache",
            "version": 1,
            "sample_id": record_id,
            "platform": platform,
            "feature_sha256": feature_sha256,
            "xml_sha256": xml_sha256,
            "bin_sha256": bin_sha256,
            "output_elements": output_elements,
            "logits_sha256": sha256_path(logits_path),
            "log_sha256": sha256_path(log_path),
            "inference_ms": infer_ms,
            "load_ms": load_ms,
        }
        if cache != expected:
            return None
    else:
        # Adopt successful outputs from runs made before resumable caches existed.
        # The held-out work directory is manifest-hash scoped, and current
        # feature/model hashes are recorded when the legacy result is adopted.
        write_sample_cache(
            cache_path=cache_path,
            record_id=record_id,
            platform=platform,
            feature_path=feature_path,
            logits_path=logits_path,
            log_path=log_path,
            feature_sha256=feature_sha256,
            xml_sha256=xml_sha256,
            bin_sha256=bin_sha256,
            output_elements=output_elements,
            infer_ms=infer_ms,
            load_ms=load_ms,
        )
    return infer_ms, load_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-id", default="ami-smoke-v1")
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=DEFAULT_IR)
    parser.add_argument("--experiment-id", default="cnn_ctc_v3-ami-smoke-myriad")
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v3" / "evaluation",
    )
    parser.add_argument("--platform", required=True, choices=("armv7", "arm64", "amd64"))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v3" / "ami-smoke-result.json",
    )
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        xml = args.ir_dir / "cnn_ctc_v3.xml"
        binary = args.ir_dir / "cnn_ctc_v3.bin"
        for path in (args.manifest, xml, binary):
            if not path.is_file():
                raise ValueError(f"required file is missing: {path}")

        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)
        word_counts = []
        char_counts = []
        latencies = []
        load_times = []
        per_sample = []
        skipped = {"too_long": [], "target_too_long": []}
        manifest_samples = 0
        total_audio_samples = 0
        total_infer_seconds = 0.0
        output_shape = tuple(int(value) for value in spec["output_contract"]["shape"])
        output_elements = int(np.prod(output_shape))
        xml_sha = sha256_path(xml)
        bin_sha = sha256_path(binary)
        reused_samples = 0
        manifest_total = sum(
            1
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        print(
            f"[myriad-eval] manifest_records={manifest_total} resumable=true",
            flush=True,
        )

        for record in records(args.manifest):
            manifest_samples += 1
            decision = manifest_record_eligibility(record, spec, vocab)
            if not decision["eligible"]:
                reason = decision["reason"]
                if reason not in skipped:
                    raise ValueError(
                        f"{record['id']}: unknown cnn_ctc_v3 eligibility reason {reason!r}"
                    )
                skipped[reason].append(record["id"])
                continue

            audio_path = resolve_audio(args.manifest, record)
            audio = read_f32le(audio_path)
            if audio.sample_count != decision["sample_count"]:
                raise ValueError(
                    f"{record['id']}: manifest sample count {decision['sample_count']} "
                    f"does not match audio sample count {audio.sample_count}"
                )
            features, valid_frames = logmel_features(audio.samples, spec)
            if valid_frames != decision["valid_feature_frames"]:
                raise ValueError(
                    f"{record['id']}: frontend frame count {valid_frames} does not match "
                    f"eligibility frame count {decision['valid_feature_frames']}"
                )
            sample_dir = work / record["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            feature_path = sample_dir / "features.f32"
            logits_path = sample_dir / "logits.f32"
            log_path = sample_dir / "myriad.log"
            cache_path = sample_dir / "sample-cache.json"
            feature_values = features.astype(np.float32, copy=False)
            feature_sha = hashlib.sha256(
                feature_values.tobytes(order="C")
            ).hexdigest()

            cached_timing = load_sample_cache(
                cache_path=cache_path,
                record_id=str(record["id"]),
                platform=args.platform,
                feature_path=feature_path,
                logits_path=logits_path,
                log_path=log_path,
                feature_sha256=feature_sha,
                xml_sha256=xml_sha,
                bin_sha256=bin_sha,
                output_elements=output_elements,
            )
            if cached_timing is not None:
                infer_ms, load_ms = cached_timing
                reused_samples += 1
            else:
                feature_values.tofile(feature_path)
                command = [
                    str(ROOT / "run.sh"),
                    "--platform",
                    args.platform,
                    "custom",
                    "--model",
                    container_work(xml),
                    "--weights",
                    container_work(binary),
                    "--tensor",
                    container_work(feature_path),
                    "--output",
                    container_work(logits_path),
                    "--iterations",
                    "1",
                ]
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_path.write_text(proc.stdout, encoding="utf-8")
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{record['id']}: MYRIAD inference failed ({proc.returncode})\n"
                        + "\n".join(proc.stdout.splitlines()[-20:])
                    )
                timing = parse_myriad_timing(proc.stdout)
                if timing is None:
                    raise RuntimeError(
                        f"{record['id']}: could not parse MYRIAD timing"
                    )
                infer_ms, load_ms = timing
                write_sample_cache(
                    cache_path=cache_path,
                    record_id=str(record["id"]),
                    platform=args.platform,
                    feature_path=feature_path,
                    logits_path=logits_path,
                    log_path=log_path,
                    feature_sha256=feature_sha,
                    xml_sha256=xml_sha,
                    bin_sha256=bin_sha,
                    output_elements=output_elements,
                    infer_ms=infer_ms,
                    load_ms=load_ms,
                )
            latencies.append(infer_ms)
            load_times.append(load_ms)
            total_infer_seconds += infer_ms / 1000.0
            total_audio_samples += audio.sample_count

            logits_flat = np.fromfile(logits_path, dtype=np.float32)
            if logits_flat.size != output_elements:
                raise RuntimeError(
                    f"{record['id']}: output elements {logits_flat.size} != "
                    f"{output_elements}"
                )
            logits = logits_flat.reshape(output_shape)
            output_frames = acoustic_output_length(valid_frames, spec)
            if output_frames != decision["valid_output_frames"]:
                raise ValueError(
                    f"{record['id']}: output frame count {output_frames} does not match "
                    f"eligibility output count {decision['valid_output_frames']}"
                )
            decoder = greedy_decode_logits_diagnostics(
                logits[0, :output_frames, :],
                vocab,
            )
            hypothesis = decoder["hypothesis"]
            reference = record["transcript"]["text"]
            words = word_error_counts(reference, hypothesis)
            chars = character_error_counts(reference, hypothesis)
            word_counts.append(words)
            char_counts.append(chars)
            per_sample.append(
                {
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "audio_samples": audio.sample_count,
                    "valid_feature_frames": valid_frames,
                    "valid_output_frames": output_frames,
                    "load_ms": load_ms,
                    "inference_ms": infer_ms,
                    "wer": words.rate,
                    "cer": chars.rate,
                    "reference_words": len(reference.split()),
                    "reference_characters_no_spaces": len(reference.replace(" ", "")),
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "decoder": {
                        key: value
                        for key, value in decoder.items()
                        if key != "hypothesis"
                    },
                }
            )
            if manifest_samples == 1 or manifest_samples % 100 == 0:
                print(
                    "[myriad-eval] "
                    f"processed={manifest_samples}/{manifest_total} "
                    f"evaluated={len(per_sample)} reused={reused_samples}",
                    flush=True,
                )

        print(
            "[myriad-eval] "
            f"processed={manifest_samples}/{manifest_total} "
            f"evaluated={len(per_sample)} reused={reused_samples} "
            "hardware_pass_complete=true",
            flush=True,
        )

        if not per_sample:
            raise ValueError(
                "no eligible cnn_ctc_v3 samples in manifest; "
                f"skipped={skipped}"
            )

        latency = latency_summary_ms(latencies)
        decoder_summary = aggregate_decoder_diagnostics(per_sample)
        manifest_sha = sha256_path(args.manifest)
        spec_sha = canonical_sha256(spec)
        result = {
            "schema": "speech-asr/experiment-result",
            "version": 1,
            "experiment_id": args.experiment_id,
            "status": "completed",
            "benchmark": {
                "id": args.benchmark_id,
                "contract_version": 1,
                "manifest_sha256": manifest_sha,
                "text_normalization_version": "text-v1",
            },
            "model": {
                "id": spec["id"],
                "family": spec["family"],
                "spec_sha256": spec_sha,
                "artifact_sha256": {
                    "cnn_ctc_v3.xml": xml_sha,
                    "cnn_ctc_v3.bin": bin_sha,
                },
            },
            "runtime": {
                "repo_commit": git_head(),
                "backend": "MYRIAD",
                "openvino_version": "2020.3.2",
                "target": args.platform,
            },
            "metrics": {
                "wer": sum_rate(word_counts),
                "cer": sum_rate(char_counts),
                "realtime_factor": total_infer_seconds / (total_audio_samples / 16000.0),
                "inference_latency_p50_ms": latency["p50_ms"],
                "inference_latency_p95_ms": latency["p95_ms"],
                "failures": 0,
                "manifest_samples": manifest_samples,
                "evaluated_samples": len(per_sample),
                "skipped": skipped,
                "measurement_scope": "MYRIAD inference only; frontend/decoder excluded",
                "model_load_ms": {
                    "min": min(load_times),
                    "max": max(load_times),
                    "mean": sum(load_times) / len(load_times),
                },
                "decoder": decoder_summary,
                "per_sample": per_sample,
            },
            "provenance": {
                "dataset_manifest_sha256": manifest_sha,
                "model_spec_sha256": spec_sha,
                "vocab_sha256": canonical_sha256(vocab),
                "frontend_kind": spec["frontend"]["kind"],
            },
        }
        validate_experiment_result(result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"status: completed\n"
        f"samples: {len(per_sample)} evaluated / {manifest_samples} manifest\n"
        f"skipped: too_long={len(skipped['too_long'])}, "
        f"target_too_long={len(skipped['target_too_long'])}\n"
        f"WER: {result['metrics']['wer']:.6f}\n"
        f"CER: {result['metrics']['cer']:.6f}\n"
        f"inference-only RTF: {result['metrics']['realtime_factor']:.6f}\n"
        f"latency p50/p95 ms: {latency['p50_ms']:.3f}/{latency['p95_ms']:.3f}\n"
        f"result: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
