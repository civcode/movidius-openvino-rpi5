# Deterministic evaluation

Evaluation is authoritative code, not an LLM judgment.

## Transcript metrics

`python/speech_asr/evaluation.py` provides WER/CER, real-time factor, latency
percentiles and sample-index timing error. All transcript scoring applies
`text-v1` internally.

WER and CER may exceed 1.0 when insertions exceed reference length. CER removes
spaces after normalization. Detailed S/D/I counts are retained.

## rm_cnn4a acoustic regression

`rm_cnn4a` is not an end-to-end transcript fixture. Its vendor package
provides feature and score ARKs, so the authoritative first hardware test is
score regression plus device timing:

```bash
python3 examples/speech-asr/evaluation/benchmark_rm_cnn4a.py --platform arm64
```

The command executes `./run.sh speech-regress`, stores the raw OpenVINO sample
log, parses each utterance's frame count/inference time/error statistics, and
writes `work/speech-asr/rm_cnn4a/result.json`.

The result uses `contracts/acoustic-regression-result-v1.schema.json`.
Per-frame latency is reported directly; no RTF is invented because the vendor
feature ARK is not the project's canonical raw-audio benchmark.

## Authority boundary

An AI agent may orchestrate evaluation, diagnose failures and summarize results.
It must not edit metric outputs to make a candidate appear better.
