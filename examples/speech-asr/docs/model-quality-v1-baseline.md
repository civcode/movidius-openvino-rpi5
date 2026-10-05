# Model-quality v1 baseline review

Date: 2026-10-05
Status: reviewed and approved for interim baseline execution

## Qualification input

The post-Phase-11 data qualification was generated from the frozen
`ami-benchmark-v1` manifest:

- source records: 277;
- source manifest SHA-256:
  `9f4444c6c0e54cf25d92a69723727bb46a73632e34ec33402c71beca1b64cd3e`;
- source meeting: ES2002a.

The selected model contract is `cnn_ctc_v3` with:

- model-spec canonical SHA-256:
  `beb28c4666687d634aa4a954b0c5cdabd82c373b7d028d6ffa96c9f56aaf275f`;
- vocabulary canonical SHA-256:
  `79f4dc2b628f5f61b3d5361ca67fadff044a91569c24e0af3916786fd8ddce4f`.

## Reviewed partition

Training:

- speakers A/B/C;
- 125 eligible records;
- 219.998 seconds of audio;
- 420 normalized words;
- 1,610 normalized characters excluding spaces;
- 36 records excluded for fixed-audio overflow;
- 1 record excluded because CTC target length exceeded available output frames;
- manifest SHA-256:
  `89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a`.

Validation:

- speaker D;
- 95 eligible records;
- 122.52 seconds of audio;
- 254 normalized words;
- 992 normalized characters excluding spaces;
- 19 records excluded for fixed-audio overflow;
- 1 record excluded because CTC target length exceeded available output frames;
- manifest SHA-256:
  `07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923`.

There is zero record overlap and zero speaker overlap.

## Interpretation

This partition is a materially stronger model-selection population than the
Phase 11 smoke set and is approved for the next controlled baseline.

It is not a final generalization benchmark because both sides come from the
same meeting, ES2002a. Speaker holdout tests cross-speaker transfer inside one
meeting/session context; it does not test cross-meeting robustness.

## Baseline decision

The first experiment on this data must keep `cnn_ctc_v3` architecture and its
training policy unchanged. The purpose is to establish a new WER/CER reference
without conflating data quality with another architecture change.

Frozen training policy:

- CUDA model execution;
- deterministic CPU CTC loss;
- seed 1337;
- 32 epochs;
- batch size 1;
- Adam learning rate 3e-4;
- cosine decay to 3e-5;
- gradient clipping at norm 5;
- best-validation-loss checkpoint selection.

With 125 training records and batch size 1, the declared schedule corresponds
to 125 optimizer steps per epoch and 4,000 optimizer steps total. It also
processes 219.998 seconds of eligible training audio per epoch, or about
7,039.936 seconds across 32 epochs.

## Acceptance semantics

This experiment establishes an accuracy baseline; it is not compared against a
smoke-derived CER/WER ceiling.

Required gates remain:

- training;
- ONNX export;
- OpenVINO 2020.3 conversion;
- physical MYRIAD execution;
- accuracy evaluation.

Hardware/compatibility constraints remain:

- ONNX frame argmax agreement >= 1.0;
- MYRIAD p95 latency <= 25 ms.

WER, CER and corpus-level inference-only RTF are measured and reviewed but have
no numeric acceptance threshold for this baseline. RTF depends on the
benchmark's clip-duration distribution for this fixed-shape graph, whereas
per-inference latency is directly comparable across manifests.

## Edge provisioning

The validation manifest and referenced benchmark audio must be provisioned
out-of-band on edge before experiment execution. Ordinary experiment execution
does not sync datasets.

Use:

```bash
./scripts/provision-speech-model-data-edge.sh
```

The provisioning command pins the dedicated edge worker worktree to the current
commit, rsyncs the frozen benchmark dataset once, verifies the prepared corpus,
regenerates the derived train/validation manifests on edge, and verifies their
reviewed SHA-256 identities.

## Baseline execution

After provisioning:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v3-quality-baseline.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"

./scripts/run-speech-experiment.sh \
  --experiment "$EXP" \
  --worker edge

./scripts/review-speech-experiment.sh \
  --experiment "$EXP"
```

A later architecture generation may be proposed only after this baseline result
is reviewed.


## Executed baseline result

The reviewed baseline executed as
`exp-62a94f36aefddb58/attempt-0001`.

Execution was healthy:

- CUDA model/logits placement was verified on NVIDIA GeForce RTX 4070 Ti SUPER;
- 4,000 optimizer steps completed;
- 7,039.936 seconds of training audio were processed;
- OpenVINO IR validation passed;
- PyTorch/ONNX frame argmax agreement was 1.0;
- initialized MYRIAD numerical compatibility passed;
- MYRIAD p50/p95 latency was 16.60 / 16.65 ms;
- zero hardware failures occurred.

The original immutable request was rejected only because it applied
`max_realtime_factor = 0.01`. The measured RTF was 0.012877. That rejection
is a benchmark-policy defect rather than a hardware regression: p95 latency was
effectively unchanged from the smoke parent, while the validation population
contains shorter clips. Future model-quality experiments leave RTF ungated and
retain the fixed-shape latency gate.

Accuracy/decoder evidence:

- WER: 1.0;
- CER: 0.980847;
- blank-frame fraction: 0.959331;
- empty hypotheses: 84 / 95 (0.884211);
- emitted characters excluding spaces: 25;
- reference characters excluding spaces: 992;
- emitted/reference character ratio: 0.025202.

Training evidence showed strong overfit:

- best validation loss: 4.798953 at epoch 11;
- final epoch train loss: 0.475595;
- final epoch validation loss: 10.958259.

The baseline therefore establishes a real held-out-speaker failure mode: severe
CTC blank collapse. It does not justify another inference-graph architecture
change yet, because checkpoint selection still uses CTC loss even though prior
evidence showed that loss and greedy CER do not rank candidates consistently.

## Next diagnostic

The immediate child experiment keeps the exact graph, manifests and optimizer
trajectory but records held-out WER/CER and decoder-collapse metrics at every
epoch and selects the exported checkpoint by minimum validation CER.

Use:

```bash
./scripts/init-cnn-ctc-v3-cer-selection.sh
```

If a materially better epoch exists, subsequent work should retain
metric-aligned checkpointing. If every epoch remains blank-collapsed, the next
intervention should be training regularization/augmentation rather than a
hardware-graph redesign.
