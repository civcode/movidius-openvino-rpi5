# Deterministic evaluation

Evaluation is authoritative code, not an LLM judgment.

The implementation lives in `python/speech_asr/evaluation.py` and provides:

- WER plus substitution/deletion/insertion counts;
- CER plus edit counts;
- real-time factor;
- p50/p95 latency summaries;
- sample-index timing error in milliseconds.

All transcript scoring applies `text-v1` internally. Callers must not maintain a
separate normalization implementation.

## WER/CER conventions

- WER and CER are edit errors divided by the number of reference units.
- Both are non-negative and may exceed `1.0` when insertions exceed the
  reference length.
- CER removes spaces after `text-v1` normalization.
- An empty normalized reference uses denominator `max(1, reference_units)`.
  This deliberately produces finite machine-readable output instead of
  NaN/Infinity.
- Detailed S/D/I counts are retained alongside the aggregate rates.
- Equal-cost Levenshtein paths use a fixed substitution, deletion, insertion
  tie order so detailed counts are deterministic.

## CLI

```bash
python3 examples/speech-asr/evaluation/score_transcripts.py \
    --reference "Hello world" \
    --hypothesis "hello word" \
    --pretty
```

## Authority boundary

An AI agent may orchestrate the evaluator, explain failures and summarize
results. It must not edit metric outputs to make a candidate appear better.

Hardware results must record the benchmark version, model artifact hashes,
repository revision and relevant runtime/firmware provenance.
