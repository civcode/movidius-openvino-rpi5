# ADR: freeze cnn_ctc_v19 acoustics and add CPU CTC beam/LM decoding

Status: implemented; decoder sweep execution pending
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
