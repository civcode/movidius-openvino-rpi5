#!/usr/bin/env python3
"""Tune CPU CTC beam/LM decoding on frozen cnn_ctc_v19 MYRIAD logits."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import (  # noqa: E402
    greedy_decode_logits,
    load_spec,
    load_vocab,
    manifest_record_eligibility,
)
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.ctc_beam import (  # noqa: E402
    CharacterNgramLM,
    build_decoder_artifact,
    prefix_beam_decode,
)
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    latency_summary_ms,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"
DEFAULT_TRAIN = ROOT / "work" / "speech-asr" / "ami" / "model-quality-v4-architecture-screen-v1" / "train.manifest.jsonl"

_WORKER_VOCAB = None
_WORKER_LM = None
_WORKER_CONFIG = None
_WORKER_OUTPUT_SHAPE = None


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_records(path: pathlib.Path) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(validate_speech_sample(json.loads(line)))
    return result


def latest_attempt(experiment: pathlib.Path) -> pathlib.Path:
    attempts = sorted((experiment / "attempts").glob("attempt-*"))
    completed = [
        path
        for path in attempts
        if (path / "results" / "hardware.json").is_file()
        and (path / "edge" / "evaluation").is_dir()
    ]
    if not completed:
        raise ValueError("no attempt contains hardware.json plus pulled evaluation logits")
    return completed[-1]


def parse_csv_numbers(raw: str, cast):
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("parameter grid cannot be empty")
    return values


def _init_worker(vocab, lm, config, output_shape):
    global _WORKER_VOCAB, _WORKER_LM, _WORKER_CONFIG, _WORKER_OUTPUT_SHAPE
    _WORKER_VOCAB = vocab
    _WORKER_LM = lm
    _WORKER_CONFIG = config
    _WORKER_OUTPUT_SHAPE = tuple(output_shape)


def _decode_one(task):
    record_id, reference, logits_path, output_frames = task
    started = time.perf_counter()
    values = np.fromfile(logits_path, dtype=np.float32)
    expected = int(np.prod(_WORKER_OUTPUT_SHAPE))
    if values.size != expected:
        raise ValueError(f"{record_id}: logits elements {values.size} != {expected}")
    logits = values.reshape(_WORKER_OUTPUT_SHAPE)[0, :output_frames, :]
    kind = _WORKER_CONFIG["kind"]
    if kind == "greedy":
        hypothesis = greedy_decode_logits(logits, _WORKER_VOCAB)
    elif kind == "prefix-beam":
        hypothesis = prefix_beam_decode(
            logits,
            _WORKER_VOCAB,
            beam_width=_WORKER_CONFIG["beam_width"],
            token_top_k=_WORKER_CONFIG["token_top_k"],
            lm=_WORKER_LM,
            lm_weight=_WORKER_CONFIG["lm_weight"],
            word_bonus=_WORKER_CONFIG["word_bonus"],
        )["hypothesis"]
    else:
        raise ValueError(f"unsupported decoder kind: {kind}")
    decode_ms = (time.perf_counter() - started) * 1000.0
    words = word_error_counts(reference, hypothesis)
    chars = character_error_counts(reference, hypothesis)
    return {
        "id": record_id,
        "reference": reference,
        "hypothesis": hypothesis,
        "decode_ms": decode_ms,
        "word_errors": words.errors,
        "word_reference_units": words.reference_units,
        "char_errors": chars.errors,
        "char_reference_units": chars.reference_units,
    }


def evaluate_config(*, tasks, vocab, lm, config, output_shape, jobs, progress_interval):
    started = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_init_worker,
        initargs=(vocab, lm, config, output_shape),
    ) as pool:
        for index, result in enumerate(pool.map(_decode_one, tasks, chunksize=8), start=1):
            results.append(result)
            if progress_interval and (
                index == 1 or index % progress_interval == 0 or index == len(tasks)
            ):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                eta = (len(tasks) - index) / max(rate, 1e-9)
                print(
                    "[decoder-sweep] "
                    f"kind={config['kind']} beam={config.get('beam_width', 1)} "
                    f"lm={config.get('lm_weight', 0.0):.3f} "
                    f"bonus={config.get('word_bonus', 0.0):.3f} "
                    f"processed={index}/{len(tasks)} "
                    f"rate={rate:.2f}/s eta={eta:.1f}s",
                    flush=True,
                )

    word_errors = sum(item["word_errors"] for item in results)
    word_refs = sum(item["word_reference_units"] for item in results)
    char_errors = sum(item["char_errors"] for item in results)
    char_refs = sum(item["char_reference_units"] for item in results)
    decode_ms = [item["decode_ms"] for item in results]
    timing = latency_summary_ms(decode_ms)
    return {
        "config": config,
        "samples": len(results),
        "wer": word_errors / max(1, word_refs),
        "cer": char_errors / max(1, char_refs),
        "decode_latency_p50_ms": timing["p50_ms"],
        "decode_latency_p95_ms": timing["p95_ms"],
        "decode_latency_mean_ms": statistics.fmean(decode_ms),
        "wall_seconds": time.perf_counter() - started,
        "per_sample": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=pathlib.Path, required=True)
    parser.add_argument("--attempt")
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--train-manifest", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--beam-widths", default="8,16,32")
    parser.add_argument("--lm-weights", default="0.15,0.30,0.45,0.60")
    parser.add_argument("--word-bonuses", default="-0.20,0.0,0.20")
    parser.add_argument("--token-top-k", type=int, default=12)
    parser.add_argument("--lm-order", type=int, default=5)
    parser.add_argument("--lm-smoothing", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        if args.jobs < 1:
            raise ValueError("jobs must be >= 1")
        if args.token_top_k < 1:
            raise ValueError("token_top_k must be >= 1")
        experiment = args.experiment.resolve()
        attempt = (
            experiment / "attempts" / args.attempt
            if args.attempt
            else latest_attempt(experiment)
        )
        hardware_path = attempt / "results" / "hardware.json"
        cache_dir = attempt / "edge" / "evaluation"
        if not hardware_path.is_file() or not cache_dir.is_dir():
            raise ValueError("attempt lacks completed physical hardware evidence/logit cache")

        request = load_json(experiment / "request" / "experiment.json")
        benchmark_path = pathlib.Path(request["benchmark"]["manifest_path"])
        validation_manifest = (
            benchmark_path if benchmark_path.is_absolute() else ROOT / benchmark_path
        )
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        output_shape = tuple(int(v) for v in spec["output_contract"]["shape"])
        expected_elements = int(np.prod(output_shape))

        validation = manifest_records(validation_manifest)
        if args.max_samples is not None:
            validation = validation[: args.max_samples]
        tasks = []
        for record in validation:
            decision = manifest_record_eligibility(record, spec, vocab)
            if not decision["eligible"]:
                continue
            logits_path = cache_dir / record["id"] / "logits.f32"
            if not logits_path.is_file():
                raise ValueError(f"missing pulled MYRIAD logits cache: {logits_path}")
            if logits_path.stat().st_size != expected_elements * 4:
                raise ValueError(f"invalid logits cache size: {logits_path}")
            tasks.append(
                (
                    str(record["id"]),
                    str(record["transcript"]["text"]),
                    str(logits_path),
                    int(decision["valid_output_frames"]),
                )
            )
        if not tasks:
            raise ValueError("no decoder-eligible cached samples found")

        train = manifest_records(args.train_manifest)
        lm_alphabet = tuple(vocab["tokens"][1:])
        lm = CharacterNgramLM.train(
            (item["transcript"]["text"] for item in train),
            alphabet=lm_alphabet,
            order=args.lm_order,
            smoothing=args.lm_smoothing,
        )

        greedy_config = {
            "kind": "greedy",
            "beam_width": 1,
            "token_top_k": 1,
            "lm_weight": 0.0,
            "word_bonus": 0.0,
        }
        greedy = evaluate_config(
            tasks=tasks,
            vocab=vocab,
            lm=None,
            config=greedy_config,
            output_shape=output_shape,
            jobs=args.jobs,
            progress_interval=args.progress_interval,
        )

        hardware = load_json(hardware_path)
        if args.max_samples is None:
            recorded = hardware["metrics"]
            if abs(greedy["wer"] - float(recorded["wer"])) > 1e-12:
                raise ValueError(
                    f"cached greedy WER {greedy['wer']} != physical {recorded['wer']}"
                )
            if abs(greedy["cer"] - float(recorded["cer"])) > 1e-12:
                raise ValueError(
                    f"cached greedy CER {greedy['cer']} != physical {recorded['cer']}"
                )

        beam_widths = parse_csv_numbers(args.beam_widths, int)
        lm_weights = parse_csv_numbers(args.lm_weights, float)
        word_bonuses = parse_csv_numbers(args.word_bonuses, float)
        beam_results = []
        for width in beam_widths:
            config = {
                "kind": "prefix-beam",
                "beam_width": width,
                "token_top_k": args.token_top_k,
                "lm_weight": 0.0,
                "word_bonus": 0.0,
            }
            beam_results.append(
                evaluate_config(
                    tasks=tasks,
                    vocab=vocab,
                    lm=None,
                    config=config,
                    output_shape=output_shape,
                    jobs=args.jobs,
                    progress_interval=args.progress_interval,
                )
            )

        best_beam = min(
            beam_results,
            key=lambda item: (
                item["wer"],
                item["cer"],
                item["decode_latency_p95_ms"],
                item["config"]["beam_width"],
            ),
        )
        lm_results = []
        for weight in lm_weights:
            for bonus in word_bonuses:
                config = {
                    "kind": "prefix-beam",
                    "beam_width": best_beam["config"]["beam_width"],
                    "token_top_k": args.token_top_k,
                    "lm_weight": weight,
                    "word_bonus": bonus,
                }
                lm_results.append(
                    evaluate_config(
                        tasks=tasks,
                        vocab=vocab,
                        lm=lm,
                        config=config,
                        output_shape=output_shape,
                        jobs=args.jobs,
                        progress_interval=args.progress_interval,
                    )
                )

        candidates = [greedy, *beam_results, *lm_results]
        selected = min(
            candidates,
            key=lambda item: (
                item["wer"],
                item["cer"],
                item["decode_latency_p95_ms"],
            ),
        )
        output = {
            "schema": "speech-asr/ctc-decoder-sweep",
            "version": 1,
            "acoustic_model": "cnn_ctc_v19",
            "source_experiment_id": request["experiment_id"],
            "source_attempt_id": attempt.name,
            "source_hardware_sha256": sha256_path(hardware_path),
            "validation_manifest": {
                "path": str(validation_manifest),
                "sha256": sha256_path(validation_manifest),
                "samples": len(tasks),
            },
            "lm_training_manifest": {
                "path": str(args.train_manifest),
                "sha256": sha256_path(args.train_manifest),
                "samples": len(train),
            },
            "lm": lm.summary(),
            "greedy_baseline": {k: v for k, v in greedy.items() if k != "per_sample"},
            "beam_only": [{k: v for k, v in item.items() if k != "per_sample"} for item in beam_results],
            "beam_plus_lm": [{k: v for k, v in item.items() if k != "per_sample"} for item in lm_results],
            "selected": {k: v for k, v in selected.items() if k != "per_sample"},
            "delta_vs_greedy": {
                "wer": selected["wer"] - greedy["wer"],
                "cer": selected["cer"] - greedy["cer"],
                "wer_ratio": selected["wer"] / greedy["wer"] if greedy["wer"] else None,
                "cer_ratio": selected["cer"] / greedy["cer"] if greedy["cer"] else None,
            },
        }
        output_path = args.output or (attempt / "results" / "decoder-v1.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        decoder_artifact = build_decoder_artifact(
            acoustic_model="cnn_ctc_v19",
            vocab=vocab,
            config=selected["config"],
            lm=lm,
            provenance={
                "sweep_result_sha256": sha256_path(output_path),
                "source_experiment_id": request["experiment_id"],
                "source_attempt_id": attempt.name,
                "source_hardware_sha256": sha256_path(hardware_path),
                "validation_manifest_sha256": sha256_path(validation_manifest),
                "lm_training_manifest_sha256": sha256_path(args.train_manifest),
                "selected_wer": selected["wer"],
                "selected_cer": selected["cer"],
                "decoder_latency_p50_ms_on_oberon": selected[
                    "decode_latency_p50_ms"
                ],
                "decoder_latency_p95_ms_on_oberon": selected[
                    "decode_latency_p95_ms"
                ],
            },
        )
        artifact_path = output_path.with_name("decoder-artifact-v1.json")
        artifact_path.write_text(
            json.dumps(
                decoder_artifact,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        print(json.dumps(output, sort_keys=True, indent=2))
        print(f"decoder sweep: {output_path}", flush=True)
        print(
            f"decoder artifact: {artifact_path} "
            f"sha256={sha256_path(artifact_path)}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
