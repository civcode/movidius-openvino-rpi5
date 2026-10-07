#!/usr/bin/env python3
"""Attribute AMI quality changes across the QuartzNet adaptation boundary.

This is a CPU/CUDA PyTorch reference analysis. It deliberately performs no
training, OpenVINO conversion, Docker execution, SSH, or MYRIAD inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
import time
from typing import Callable

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
TRAINING = SPEECH_ROOT / "training"
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(TRAINING))

from cnn_ctc_v19 import CnnCtcV19  # noqa: E402
from pretrained_cnn_ctc_v18 import load_pretrained_quartznet  # noqa: E402
from quartznet15x5_reference import (  # noqa: E402
    load_reference_model,
    load_reference_spec,
    load_reference_vocab,
    nemo_reference_features,
    reference_output_length,
)
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
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    word_error_counts,
)

EXPECTED_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
EXPECTED_SAMPLES = 1273
SOURCE_LIBRISPEECH_DEV_CLEAN_WER = 0.037939781625675524
EXPECTED_EPOCH0_WER = 0.7393651479162168
EXPECTED_EPOCH0_CER = 0.5776651500316765
EXPECTED_FINETUNED_WER = 0.5983588857698121
EXPECTED_FINETUNED_CER = 0.46201693255773774
FROZEN_METRIC_TOLERANCE = 1e-12

DEFAULT_MANIFEST = (
    ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4" / "validation.manifest.jsonl"
)
DEFAULT_ARCHIVE = (
    ROOT
    / "work"
    / "speech-asr"
    / "pretrained"
    / "quartznet15x5-en-base-v2"
    / "QuartzNet15x5-En-Base.nemo"
)
DEFAULT_SOURCE_SPEC = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "model_spec.json"
DEFAULT_SOURCE_VOCAB = SPEECH_ROOT / "models" / "quartznet15x5_nvidia_ref" / "vocab.json"
DEFAULT_V19_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
DEFAULT_V19_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"
DEFAULT_EXPERIMENT = (
    ROOT / "work" / "speech-asr" / "experiments" / "exp-f4adb44ab833e896"
)
DEFAULT_ATTEMPT = "attempt-0002"
DEFAULT_OUTPUT_DIR = (
    ROOT / "work" / "speech-asr" / "analysis" / "quartznet15x5-ami-attribution-v1"
)


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device: {request}")


def read_records(manifest: pathlib.Path) -> list[dict]:
    actual_sha = sha256_path(manifest)
    if actual_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"AMI validation manifest SHA-256 changed: "
            f"{actual_sha} != {EXPECTED_MANIFEST_SHA256}"
        )
    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(validate_speech_sample(json.loads(line)))
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(
            f"AMI validation sample count changed: {len(records)} != {EXPECTED_SAMPLES}"
        )
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("AMI validation manifest contains duplicate sample ids")
    return records


def resolve_audio(manifest: pathlib.Path, record: dict) -> pathlib.Path:
    path = pathlib.Path(record["audio"]["path"])
    return path if path.is_absolute() else manifest.parent / path


def sum_rate(counts) -> float:
    errors = sum(value.errors for value in counts)
    refs = sum(value.reference_units for value in counts)
    return errors / max(1, refs)


def validate_fixed_manifest(records: list[dict], spec: dict, vocab: dict) -> None:
    rejected = []
    for record in records:
        decision = manifest_record_eligibility(record, spec, vocab)
        if not decision["eligible"]:
            rejected.append((record["id"], decision["reason"]))
    if rejected:
        preview = ", ".join(f"{sample}:{reason}" for sample, reason in rejected[:10])
        raise ValueError(
            f"frozen AMI validation contains {len(rejected)} records outside "
            f"the v19 fixed-shape contract: {preview}"
        )


def discover_v19_checkpoint(
    *,
    explicit: pathlib.Path | None,
    experiment: pathlib.Path,
    attempt: str,
    spec: dict,
    vocab: dict,
) -> tuple[pathlib.Path, dict]:
    if explicit is not None:
        candidates = [explicit]
    else:
        if not experiment.is_dir():
            raise ValueError(
                f"v19 experiment directory missing: {experiment}; "
                "pass --v19-checkpoint explicitly"
            )
        candidates = sorted(
            path
            for path in experiment.rglob("checkpoint.pt")
            if attempt in path.parts or attempt in str(path)
        )
        if not candidates:
            raise ValueError(
                f"no {attempt} checkpoint.pt found below {experiment}; "
                "pass --v19-checkpoint explicitly"
            )

    valid: list[tuple[pathlib.Path, dict, str]] = []
    errors = []
    for path in candidates:
        if not path.is_file():
            errors.append(f"{path}: missing")
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu")
            if checkpoint.get("format") != "speech-asr/cnn-ctc-checkpoint-v1":
                raise ValueError("unexpected checkpoint format")
            if checkpoint.get("model_id") != "cnn_ctc_v19":
                raise ValueError("checkpoint is not cnn_ctc_v19")
            if checkpoint.get("spec_sha256") != canonical_sha256(spec):
                raise ValueError("checkpoint spec hash does not match current v19 spec")
            if checkpoint.get("vocab_sha256") != canonical_sha256(vocab):
                raise ValueError("checkpoint vocab hash does not match current v19 vocab")
            if int(checkpoint.get("best_epoch", -1)) != 5:
                raise ValueError(
                    f"selected v19 checkpoint best_epoch is "
                    f"{checkpoint.get('best_epoch')!r}, expected 5"
                )
            valid.append((path, checkpoint, sha256_path(path)))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not valid:
        raise ValueError("no valid selected v19 checkpoint found: " + "; ".join(errors))

    hashes = {digest for _, _, digest in valid}
    if len(hashes) != 1:
        choices = ", ".join(f"{path} sha256={digest}" for path, _, digest in valid)
        raise ValueError(
            "multiple distinct valid v19 checkpoints found; pass --v19-checkpoint: "
            + choices
        )
    path, checkpoint, _ = valid[0]
    return path, checkpoint


def evaluate_stage(
    *,
    stage_id: str,
    change: str,
    model: torch.nn.Module,
    vocab: dict,
    records: list[dict],
    manifest: pathlib.Path,
    feature_fn: Callable[[tuple[float, ...]], tuple[torch.Tensor, int, int]],
    device: torch.device,
    output_dir: pathlib.Path,
    progress_interval: int,
) -> dict:
    model.to(device)
    model.eval()
    word_counts = []
    char_counts = []
    decoder_samples = []
    sample_rows = []
    started = time.monotonic()

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            audio_path = resolve_audio(manifest, record)
            audio = read_f32le(audio_path)
            features, valid_feature_frames, valid_output_frames = feature_fn(audio.samples)
            logits = model(features.to(device))[0]
            if valid_output_frames < 1 or valid_output_frames > int(logits.shape[0]):
                raise ValueError(
                    f"{record['id']}: output length {valid_output_frames} "
                    f"is invalid for {tuple(logits.shape)}"
                )
            decoder = greedy_decode_logits_diagnostics(
                logits[:valid_output_frames].detach().cpu().numpy(),
                vocab,
            )
            reference = str(record["transcript"]["text"])
            hypothesis = decoder["hypothesis"]
            words = word_error_counts(reference, hypothesis)
            chars = character_error_counts(reference, hypothesis)
            word_counts.append(words)
            char_counts.append(chars)
            decoder_samples.append(
                {
                    "reference_words": len(reference.split()),
                    "reference_characters_no_spaces": len(reference.replace(" ", "")),
                    "decoder": {
                        key: value for key, value in decoder.items() if key != "hypothesis"
                    },
                    "hypothesis": hypothesis,
                }
            )
            sample_rows.append(
                {
                    "id": record["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "word_edits": words.to_dict(),
                    "character_edits": chars.to_dict(),
                    "valid_feature_frames": valid_feature_frames,
                    "valid_output_frames": valid_output_frames,
                }
            )

            if progress_interval and (
                index == 1
                or index == len(records)
                or index % progress_interval == 0
            ):
                elapsed = time.monotonic() - started
                print(
                    f"[ami-attribution:{stage_id}] processed={index}/{len(records)} "
                    f"wer={sum_rate(word_counts):.6f} "
                    f"rate={index / max(elapsed, 1e-9):.2f}/s",
                    file=sys.stderr,
                    flush=True,
                )

    hypotheses_path = output_dir / f"{stage_id}.hypotheses.jsonl"
    hypotheses_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    result = {
        "stage": stage_id,
        "isolated_change": change,
        "samples": len(records),
        "wer": sum_rate(word_counts),
        "cer": sum_rate(char_counts),
        "decoder": aggregate_decoder_diagnostics(decoder_samples),
        "wall_seconds": time.monotonic() - started,
        "hypotheses": str(hypotheses_path),
        "training_performed": False,
    }
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def delta(previous: dict, current: dict, label: str) -> dict:
    return {
        "change": label,
        "from": previous["stage"],
        "to": current["stage"],
        "wer_delta": current["wer"] - previous["wer"],
        "cer_delta": current["cer"] - previous["cer"],
        "relative_wer_change": (
            (current["wer"] - previous["wer"]) / previous["wer"]
            if previous["wer"]
            else None
        ),
    }


def metric_match(observed: float, expected: float) -> bool:
    return abs(float(observed) - float(expected)) <= FROZEN_METRIC_TOLERANCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-spec", type=pathlib.Path, default=DEFAULT_SOURCE_SPEC)
    parser.add_argument("--source-vocab", type=pathlib.Path, default=DEFAULT_SOURCE_VOCAB)
    parser.add_argument("--v19-spec", type=pathlib.Path, default=DEFAULT_V19_SPEC)
    parser.add_argument("--v19-vocab", type=pathlib.Path, default=DEFAULT_V19_VOCAB)
    parser.add_argument("--v19-experiment", type=pathlib.Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--v19-attempt", default=DEFAULT_ATTEMPT)
    parser.add_argument("--v19-checkpoint", type=pathlib.Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        records = read_records(args.manifest)
        full = args.max_samples is None
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("--max-samples must be positive")
            records = records[: args.max_samples]

        source_spec = load_reference_spec(args.source_spec)
        source_vocab = load_reference_vocab(args.source_vocab)
        v19_spec = load_spec(args.v19_spec)
        v19_vocab = load_vocab(args.v19_vocab)
        validate_fixed_manifest(records, v19_spec, v19_vocab)
        device = choose_device(args.device)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path, checkpoint = discover_v19_checkpoint(
            explicit=args.v19_checkpoint,
            experiment=args.v19_experiment,
            attempt=args.v19_attempt,
            spec=v19_spec,
            vocab=v19_vocab,
        )

        source_model, source_import = load_reference_model(
            args.archive,
            spec=source_spec,
            vocab=source_vocab,
        )

        def source_frontend(samples):
            features, valid = nemo_reference_features(
                samples,
                source_spec,
                device=device,
            )
            return features, valid, reference_output_length(valid, source_spec)

        def fixed_frontend(samples):
            values, valid = logmel_features(samples, v19_spec)
            return (
                torch.from_numpy(np.asarray(values, dtype=np.float32)),
                valid,
                acoustic_output_length(valid, v19_spec),
            )

        stages = []
        stages.append(
            evaluate_stage(
                stage_id="s0-source-head-source-frontend",
                change="AMI domain only",
                model=source_model,
                vocab=source_vocab,
                records=records,
                manifest=args.manifest,
                feature_fn=source_frontend,
                device=device,
                output_dir=args.output_dir,
                progress_interval=args.progress_interval,
            )
        )
        stages.append(
            evaluate_stage(
                stage_id="s1-source-head-fixed-frontend",
                change="introduce AMI fixed-shape logmel-v3 frontend",
                model=source_model,
                vocab=source_vocab,
                records=records,
                manifest=args.manifest,
                feature_fn=fixed_frontend,
                device=device,
                output_dir=args.output_dir,
                progress_interval=args.progress_interval,
            )
        )

        seed = int(v19_spec["training"]["seed"])
        torch.manual_seed(seed)
        epoch0_model = CnnCtcV19(v19_spec, len(v19_vocab["tokens"]))
        epoch0_import = load_pretrained_quartznet(
            epoch0_model,
            args.archive,
            list(v19_vocab["tokens"]),
        )
        if epoch0_import["target_only_symbols_random_init"] != list("0123456789"):
            raise ValueError(
                "v19 epoch-zero target-only rows are not the frozen digits 0-9"
            )
        stages.append(
            evaluate_stage(
                stage_id="s2-ami-head-fixed-frontend-epoch0",
                change="expand source CTC head from 29 to 39 AMI classes",
                model=epoch0_model,
                vocab=v19_vocab,
                records=records,
                manifest=args.manifest,
                feature_fn=fixed_frontend,
                device=device,
                output_dir=args.output_dir,
                progress_interval=args.progress_interval,
            )
        )

        finetuned_model = CnnCtcV19(v19_spec, len(v19_vocab["tokens"]))
        finetuned_model.load_state_dict(checkpoint["state_dict"], strict=True)
        stages.append(
            evaluate_stage(
                stage_id="s3-v19-finetuned",
                change="apply selected conservative AMI fine-tuning",
                model=finetuned_model,
                vocab=v19_vocab,
                records=records,
                manifest=args.manifest,
                feature_fn=fixed_frontend,
                device=device,
                output_dir=args.output_dir,
                progress_interval=args.progress_interval,
            )
        )

        transitions = [
            delta(stages[0], stages[1], "fixed_frontend"),
            delta(stages[1], stages[2], "39_class_head"),
            delta(stages[2], stages[3], "v19_fine_tuning"),
        ]

        frozen_checks = {
            "epoch0": {
                "expected_wer": EXPECTED_EPOCH0_WER,
                "observed_wer": stages[2]["wer"],
                "wer_match": metric_match(stages[2]["wer"], EXPECTED_EPOCH0_WER),
                "expected_cer": EXPECTED_EPOCH0_CER,
                "observed_cer": stages[2]["cer"],
                "cer_match": metric_match(stages[2]["cer"], EXPECTED_EPOCH0_CER),
            },
            "finetuned": {
                "expected_wer": EXPECTED_FINETUNED_WER,
                "observed_wer": stages[3]["wer"],
                "wer_match": metric_match(stages[3]["wer"], EXPECTED_FINETUNED_WER),
                "expected_cer": EXPECTED_FINETUNED_CER,
                "observed_cer": stages[3]["cer"],
                "cer_match": metric_match(stages[3]["cer"], EXPECTED_FINETUNED_CER),
            },
        }
        frozen_match = all(
            item["wer_match"] and item["cer_match"]
            for item in frozen_checks.values()
        )

        result = {
            "schema": "speech-asr/quartznet-ami-adaptation-attribution",
            "version": 1,
            "status": (
                "pass"
                if full and frozen_match
                else ("fail" if full else "diagnostic")
            ),
            "analysis_kind": "zero-new-training staged CPU/CUDA reference attribution",
            "full_validation": full,
            "samples": len(records),
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256_path(args.manifest),
                "expected_samples": EXPECTED_SAMPLES,
            },
            "source": {
                "archive": str(args.archive),
                "sha384": source_import["source"]["sha384"],
                "qualified_librispeech_dev_clean_wer": SOURCE_LIBRISPEECH_DEV_CLEAN_WER,
                "source_classes": len(source_vocab["tokens"]),
            },
            "v19": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_path(checkpoint_path),
                "best_epoch": checkpoint["best_epoch"],
                "classes": len(v19_vocab["tokens"]),
            },
            "stages": stages,
            "transitions": transitions,
            "frozen_evidence_checks": frozen_checks,
            "frozen_evidence_match": frozen_match,
            "runtime": {
                "device": str(device),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
            "scope": (
                "PyTorch reference attribution only; no new training, "
                "OpenVINO conversion, Docker, SSH, or MYRIAD execution"
            ),
        }
        output = args.output_dir / "result.json"
        output.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["status"] != "fail" else 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
