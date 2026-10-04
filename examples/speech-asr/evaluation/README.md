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
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend myriad --platform arm64
```

The command executes `./run.sh speech-regress`, stores the raw OpenVINO sample
log, parses each utterance's frame count/inference time/error statistics, and
writes `work/speech-asr/rm_cnn4a/result.json`.

The result uses `contracts/acoustic-regression-result-v1.schema.json`.
Before the file is written it is also checked by the stdlib semantic validator,
so a parser/wiring bug cannot silently emit an invalid benchmark result.
Relative `--output` and `--log` paths are resolved from the repository root.

Per-frame latency is reported directly; no RTF is invented because the vendor
feature ARK is not the project's canonical raw-audio benchmark.

## Authority boundary

An AI agent may orchestrate evaluation, diagnose failures and summarize results.
It must not edit metric outputs to make a candidate appear better.


### CPU reference

The authoritative Phase 4 CPU result is checked in at
`models/rm_cnn4a/cpu-reference-v1.json`. Re-running

```bash
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend cpu
```

should use the same XML/BIN and vendor feature/reference-score ARKs. The frozen
result records 10 utterances, 3401 frames, zero failures and the exact numerical
error/latency metrics used as the Phase 5 comparison reference.


### Frozen amd64 MYRIAD result

The successful physical-device result is checked in at
`models/rm_cnn4a/myriad-amd64-v1.json`, with its evidence-only comparison at
`models/rm_cnn4a/cpu-vs-myriad-amd64-v1.json`. Both use the same XML/BIN and
vendor feature/reference-score hashes as the frozen CPU result.

The comparison intentionally applies no acceptance threshold; it records exact
deltas/ratios so later Pi 5 and custom-model results can be judged from explicit
policy rather than retrofitted thresholds.


### Reproducible CPU-to-MYRIAD comparison

Use the checked-in CPU reference as the default baseline and provide any
completed MYRIAD result as the candidate:

```bash
./scripts/python.sh examples/speech-asr/evaluation/compare_rm_cnn4a.py \
    --candidate work/speech-asr/rm_cnn4a/result-myriad-arm64.json \
    --output work/speech-asr/rm_cnn4a/comparison-myriad-arm64.json
```

The command first validates both acoustic-regression result documents. It then
requires identical BIN, vendor feature/reference-score fixtures and OpenVINO
version. Raw XML must also match unless both results contain the same
`graph_sha256` executable-graph fingerprint. The fingerprint excludes only
Model Optimizer's top-level non-executable `meta_data` section; graph
layers/edges/ports/blobs remain part of the identity. The command then reports
count agreement plus exact metric deltas and ratios and deliberately applies no
MYRIAD pass/fail thresholds.


### Pi 5 evidence

The first completed arm64 MYRIAD run is frozen at
`models/rm_cnn4a/myriad-arm64-pi5-v1.json`. It completed all 3401 frames with
zero failures and produced the same vendor-reference error metrics as the
frozen amd64 MYRIAD run. Its raw XML hash differs from the earlier amd64
preparation, so cross-host artifact identity remains explicitly pending until
the new executable-graph fingerprint is regenerated and compared on both
prepared IRs.
