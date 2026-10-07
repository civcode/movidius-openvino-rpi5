# ADR: freeze cnn_ctc_v19 acoustics and add CPU CTC beam/LM decoding

Status: selected decoder frozen; Raspberry Pi integrated deployment reproof complete
Date: 2026-10-07
Acoustic reference: `exp-f4adb44ab833e896/attempt-0002`

## Acoustic reference

cnn_ctc_v19 is frozen as the current acoustic-quality reference.

The selected epoch-5 checkpoint reached validation CER
`0.46201693255773774` and validation WER `0.5983588857698121`.

The complete physical MYRIAD evaluation processed all 1,273 validation
utterances in one persistent server session with zero restarts and zero
failures. Physical metrics were:

- CER `0.46230490122674656`;
- WER `0.5985748218527316`;
- inference-only RTF `0.21930080610559416`;
- p50 inference latency `392.428065 ms`;
- p95 inference latency `395.7003068 ms`.

The PyTorch/ONNX/OpenVINO/MYRIAD path therefore preserves acoustic quality
closely enough that the next isolated system variable is decoding.

## Decision

Do not create another acoustic-model version for the decoder experiment.

Add a CPU-side decoder sweep over the frozen physical v19 logits already
pulled into:

`attempts/<attempt>/edge/evaluation/<sample-id>/logits.f32`.

The sweep must:

1. re-run greedy decoding over the cached physical logits and require exact
   WER/CER agreement with the recorded hardware result before trusting any
   decoder comparison;
2. evaluate CTC prefix beam search with no language model over beam widths
   8, 16 and 32;
3. choose the best beam-only width by WER, then CER, then decoder p95 latency;
4. train a compact character 5-gram language model only from the frozen
   architecture-screen training manifest;
5. evaluate LM weights 0.15, 0.30, 0.45 and 0.60 with word-boundary bonuses
   -0.20, 0.0 and 0.20 at the selected beam width;
6. choose the overall decoder by WER, then CER, then decoder p95 latency;
7. record CPU decoder p50/p95/mean latency and wall time for every candidate.

No held-out corpus is used in this decoder-development sweep.

## Sweep result

The completed 1,273-sample sweep reproduced the physical greedy baseline
exactly at WER `0.5985748218527316` and CER `0.46230490122674656`.

Beam search alone did not materially improve WER. Beam width 8 was the best
beam-only point at WER `0.5974951414381343`, CER
`0.457582215055002`, and Oberon decoder p95 `28.727534600329808 ms`.

The selected decoder is:

- prefix beam width: `8`;
- acoustic token top-k per frame: `12`;
- character LM: additive-smoothed suffix-backoff 5-gram;
- LM weight: `0.30`;
- word-boundary bonus: `-0.20`;
- validation WER: `0.5497732671129346`;
- validation CER: `0.4634567759027818`;
- Oberon decoder p50/p95: `13.642333 / 48.499758 ms`.

Relative to greedy, WER improves by about 8.15%. CER is effectively flat but
slightly worse by `0.001151874676035225`, so selection is explicitly a
word-error optimization.

The selected profile is frozen into a versioned decoder artifact containing the
trained LM counts plus sweep/model/dataset provenance. The v19 model spec and
OpenVINO IR remain unchanged.

## Raspberry Pi CPU latency reproof

The frozen artifact was re-benchmarked on the Raspberry Pi 5 over the same
1,273 cached physical-logit utterances with CPython 3.11.17, NumPy 1.26.4,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
CPU affinity restricted to cores 0-3.

The Pi result reproduced the selected decoder quality exactly:

- WER `0.5497732671129346`;
- CER `0.4634567759027818`;
- decoder p50 `39.573781999934 ms`;
- decoder p95 `142.60984940001433 ms`;
- decoder mean `55.34163030793814 ms`;
- benchmark wall time `71.22903373599956 s`.

This closes the CPU-only Raspberry Pi latency reproof. The higher Pi latency
relative to Oberon is now deployment evidence rather than an estimate.

## Integrated Raspberry Pi deployment reproof

The final paired deployment measurement completed on 2026-10-07 at repository
commit `9ec080227130df59d66dd65da99e91ee52e6da73`. The OpenVINO 2020.3.2
ARM64 runtime was exported from the pinned Docker image and executed directly
on the Raspberry Pi host; the measurement is therefore pinned to
`launcher_backend=host`, not Docker or automatic fallback.

The evaluator processed all 1,273 ES2011 validation utterances in one persistent
MYRIAD tensor-stream-v2 session with zero restarts. The first-sample persistent
parity gate passed with valid-frame argmax agreement `1.0` and maximum absolute
error `0.0`.

Quality exactly matched the frozen selected decoder:

- WER `0.5497732671129346`;
- CER `0.4634567759027818`;
- frozen-quality match `true`.

Measured compute latency was:

- MYRIAD inference p50 `392.424125 ms`;
- MYRIAD inference p95 `392.5454856 ms`;
- decoder p50 `44.00145399995381 ms`;
- decoder p95 `150.2441755998006 ms`;
- decoder mean `60.10071448467321 ms`;
- paired acoustic-plus-decoder p50 `436.4124369999538 ms`;
- paired acoustic-plus-decoder p95 `542.6246731998006 ms`;
- paired acoustic-plus-decoder mean `452.5247842749317 ms`;
- paired acoustic-plus-decoder realtime factor
  `0.25272784122997616`.

The paired values are the authoritative deployment percentiles. Their scope is
per-utterance MYRIAD `Infer()` plus CPU decoder compute only; frontend work and
IPC are excluded.

Do not derive a deployed p95 by adding acoustic p95 and decoder p95. Percentiles
must be computed from paired per-utterance measurements. The integrated reproof
above closes that measurement requirement.

## CPU execution policy

The sweep parallelizes utterances with 16 Python worker processes. The shell
entry point defaults OpenMP, MKL and OpenBLAS to one thread per worker. On
Oberon's 16 physical cores the intended launch policy is therefore:

`taskset -c 0-15 ... --jobs 16`

rather than 16 BLAS threads inside each of 16 workers.

## Deployment split

The VPU graph remains unchanged and still ends at CTC logits. Prefix beam and
language-model scoring remain CPU-only. Decoder quality can therefore improve
without increasing MYRIAD graph size or changing the validated v19 acoustic
latency.
