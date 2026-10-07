#!/usr/bin/env python3
"""Freeze the selected cnn_ctc_v19 beam/LM decoder into a runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import load_vocab  # noqa: E402
from speech_asr.contracts import validate_speech_sample  # noqa: E402
from speech_asr.ctc_beam import (  # noqa: E402
    CharacterNgramLM,
    build_decoder_artifact,
)

DEFAULT_TRAIN = (
    ROOT
    / "work"
    / "speech-asr"
    / "ami"
    / "model-quality-v4-architecture-screen-v1"
    / "train.manifest.jsonl"
)
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"

EXPECTED = {
    "acoustic_model": "cnn_ctc_v19",
    "source_hardware_sha256": (
        "5c2edb2da169474dc448ffa84b0b949286faa64081f88db5c77452a49500491b"
    ),
    "validation_manifest_sha256": (
        "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
    ),
    "lm_training_manifest_sha256": (
        "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"
    ),
    "greedy_wer": 0.5985748218527316,
    "greedy_cer": 0.46230490122674656,
    "selected_wer": 0.5497732671129346,
    "selected_cer": 0.4634567759027818,
    "decoder": {
        "kind": "prefix-beam",
        "beam_width": 8,
        "token_top_k": 12,
        "lm_weight": 0.3,
        "word_bonus": -0.2,
    },
}


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def manifest_texts(path: pathlib.Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = validate_speech_sample(json.loads(line))
                texts.append(str(record["transcript"]["text"]))
    if not texts:
        raise ValueError("LM training manifest is empty")
    return texts


def require_close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ValueError(f"{label} must be numeric")
    if abs(float(actual) - expected) > 1e-12:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def validate_sweep(sweep: dict[str, Any], train_manifest: pathlib.Path) -> None:
    if sweep.get("schema") != "speech-asr/ctc-decoder-sweep":
        raise ValueError("decoder sweep schema mismatch")
    if sweep.get("version") != 1:
        raise ValueError("decoder sweep version mismatch")
    if sweep.get("acoustic_model") != EXPECTED["acoustic_model"]:
        raise ValueError("decoder sweep acoustic model changed")
    if sweep.get("source_hardware_sha256") != EXPECTED["source_hardware_sha256"]:
        raise ValueError("decoder sweep physical-v19 source changed")

    validation = sweep.get("validation_manifest")
    lm_training = sweep.get("lm_training_manifest")
    if not isinstance(validation, dict) or (
        validation.get("sha256") != EXPECTED["validation_manifest_sha256"]
        or validation.get("samples") != 1273
    ):
        raise ValueError("decoder sweep validation benchmark changed")
    if not isinstance(lm_training, dict) or (
        lm_training.get("sha256") != EXPECTED["lm_training_manifest_sha256"]
        or lm_training.get("samples") != 3904
    ):
        raise ValueError("decoder sweep LM training corpus changed")
    if sha256_path(train_manifest) != EXPECTED["lm_training_manifest_sha256"]:
        raise ValueError("local LM training manifest hash mismatch")

    greedy = sweep.get("greedy_baseline")
    selected = sweep.get("selected")
    if not isinstance(greedy, dict) or not isinstance(selected, dict):
        raise ValueError("decoder sweep baseline/selection is missing")
    require_close(greedy.get("wer"), EXPECTED["greedy_wer"], "greedy WER")
    require_close(greedy.get("cer"), EXPECTED["greedy_cer"], "greedy CER")
    require_close(selected.get("wer"), EXPECTED["selected_wer"], "selected WER")
    require_close(selected.get("cer"), EXPECTED["selected_cer"], "selected CER")
    if selected.get("config") != EXPECTED["decoder"]:
        raise ValueError(
            f"selected decoder changed: {selected.get('config')!r} "
            f"!= {EXPECTED['decoder']!r}"
        )

    lm = sweep.get("lm")
    if not isinstance(lm, dict) or (
        lm.get("kind") != "character-ngram-additive-v1"
        or lm.get("order") != 5
        or float(lm.get("smoothing", -1)) != 0.1
        or lm.get("training_utterances") != 3904
        or lm.get("training_characters") != 63507
    ):
        raise ValueError("decoder sweep LM contract changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-result", type=pathlib.Path, required=True)
    parser.add_argument("--train-manifest", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        sweep_result = args.sweep_result.resolve()
        train_manifest = args.train_manifest.resolve()
        vocab = load_vocab(args.vocab)
        sweep = load_json(sweep_result)
        validate_sweep(sweep, train_manifest)

        lm = CharacterNgramLM.train(
            manifest_texts(train_manifest),
            alphabet=tuple(vocab["tokens"][1:]),
            order=5,
            smoothing=0.1,
        )
        artifact = build_decoder_artifact(
            acoustic_model="cnn_ctc_v19",
            vocab=vocab,
            config=sweep["selected"]["config"],
            lm=lm,
            provenance={
                "sweep_result_sha256": sha256_path(sweep_result),
                "source_experiment_id": sweep["source_experiment_id"],
                "source_attempt_id": sweep["source_attempt_id"],
                "source_hardware_sha256": sweep["source_hardware_sha256"],
                "validation_manifest_sha256": sweep["validation_manifest"]["sha256"],
                "lm_training_manifest_sha256": sweep["lm_training_manifest"]["sha256"],
                "selected_wer": sweep["selected"]["wer"],
                "selected_cer": sweep["selected"]["cer"],
                "decoder_latency_p50_ms_on_oberon": (
                    sweep["selected"]["decode_latency_p50_ms"]
                ),
                "decoder_latency_p95_ms_on_oberon": (
                    sweep["selected"]["decode_latency_p95_ms"]
                ),
            },
        )
        output = args.output
        if output is None:
            output = sweep_result.with_name("decoder-artifact-v1.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "frozen",
                    "path": str(output),
                    "sha256": sha256_path(output),
                    "decoder": artifact["decoder"],
                    "lm": lm.summary(),
                    "provenance": artifact["provenance"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
