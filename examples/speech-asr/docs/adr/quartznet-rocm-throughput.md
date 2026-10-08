# QuartzNet ROCm throughput benchmark (experimental)

Status: benchmark-only; **not yet measured on Oberon**.

The frozen single-utterance QuartzNet reference evaluator is retained as the
accuracy baseline. Oberon's 200-utterance, batch-one ROCm diagnostic processed
32.37 utterances/s at 15.12 ms median acoustic-model forward latency.
The full 2,703-sample progress log reported roughly 28–30 utterances/s.
The goal for the throughput path is **greater than 80 utterances/s**, subject
to physical measurement rather than an assumed result.

## Design

The new `benchmark_quartznet15x5_rocm.py` uses the exact source model and
individually normalized NeMo reference frontend. It groups utterances by their
**identical padded frontend length**, then forms GPU batches (default size 8).
This does not append additional temporal padding to any example. The order of
GPU execution may differ from manifest order; the emitted hypotheses always
follow manifest order. No weights or CTC semantics are changed.

The path also caches the immutable Hann window and Slaney mel filterbank,
uses vectorized NumPy argmax with first-index tie behavior, and checks the
first 20 utterances against the reference Python decoder. The accelerator
forward pass is synchronized **once per batch**, rather than once per
utterance. CPU worker threads may have narrower affinity masks; the main
thread must retain its requested mask, and all workers must remain within the
allowed CPU set.

Stage timings include audio reads, frontend preprocessing and GPU feature
concatenation, acoustic model forward time, logit transfer, and decoding plus
WER/CER scoring. The diagnostic reports **utterances/s** and RTF excluding
model loading, plus mean actual batch size and batch latency percentiles.
GPU synchronizations at stage boundaries affect the measured throughput;
these timings should be considered conservative diagnostics.

## Running on Oberon

From the repository root after pulling `main`:

```bash
SPEECH_TORCH_BACKEND=rocm SPEECH_ROCM_CPUSET=0-15 SPEECH_ROCM_THREADS=16 \
  ./scripts/python-training.sh \
  examples/speech-asr/evaluation/benchmark_quartznet15x5_rocm.py \
  --device rocm --frontend torch --batch-size 8 --max-samples 1000 \
  --verify-rocm-affinity --progress-interval 100
```

Use a unique `--output` and `--hypotheses` for each run, then compare
`runtime.utterances_per_second`, `batch_policy.mean_batch_size`,
`runtime.stages`, and WER/CER against the frozen reference.

Because batching is grouped by exact feature length, shorter subsets may
have many partially filled batches. Judge the 80 utterance/s target using the
full corpus after confirming WER/CER and transcript parity on a smaller subset.
Do not infer 80/s from theoretical GEMM throughput.

## Validation gates

1. Check the dependency-free scheduler unit tests locally.
2. Run a batch-one version of this throughput path and compare transcript
   files to the reference evaluator for the same subset.
3. Run batch sizes 4, 8, and 16; examine actual batch fill, stage timing,
   peak memory, and recognition accuracy. Do not assume increasing batch size
   always improves throughput.
4. Only if accuracy holds, measure all 2,703 utterances and accept full-corpus
   WER <= 0.05, with stable ROCm CPU affinity.
5. Report whether physical end-to-end throughput exceeds 80 utterances/s.
