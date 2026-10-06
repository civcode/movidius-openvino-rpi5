# ADR: sealed held-out AMI Full-corpus-ASR evaluation

Status: completed and sealed
Date: 2026-10-05
Physical evaluation completed: 2026-10-06
Frozen source checkpoint: `exp-87538823d2bf1562/attempt-0001`

## Context

The scaled-data `cnn_ctc_v3` result improved CER and under-emission metrics on
the ES2002a selection manifest after repeated model and data decisions had
already used that meeting. ES2002a is therefore model-selection evidence, not
an unbiased final test.

The next boundary must evaluate the frozen checkpoint without retraining,
checkpoint selection, threshold tuning or another pass over ES2002a.

## Partition decision

Use the AMI **Full-corpus-ASR** unseen scenario-component evaluation partition.
For speech recognition this is the authoritative scenario evaluation boundary
used here.

The frozen meetings are:

- EN2002a, EN2002b, EN2002c, EN2002d;
- ES2004a, ES2004b, ES2004c, ES2004d;
- IS1009a, IS1009b, IS1009c, IS1009d;
- TS3003a, TS3003b, TS3003c, TS3003d.

The split spec pins:

- manual annotations v1.6.2;
- official AMI Mix-Headset source URLs;
- SHA-256 for every source WAV;
- all A/B/C/D annotated segments;
- `training_allowed=false`;
- `checkpoint_selection_allowed=false`.

The repository intentionally does not use ES2014 or TS3007 here; those belong
to AMI's broader scenario-only evaluation framing rather than the selected
Full-corpus-ASR boundary.

## Qualification

Preparation creates the canonical source corpus under:

```text
work/speech-asr/ami/ami-eval-full-corpus-asr-sc-v1/
```

Qualification applies the frozen `cnn_ctc_v3` fixed-shape/CTC eligibility
contract and writes:

```text
work/speech-asr/ami/heldout-eval-v1/
├── manifest.jsonl
└── qualification.json
```

The qualifier refuses the boundary unless:

- the prepared source contains exactly the 16 meetings above;
- held-out record overlap with model-quality-v3 training is zero;
- held-out meeting overlap with model-quality-v3 training is zero;
- held-out record overlap with the ES2002a selection manifest is zero;
- held-out meeting overlap with the ES2002a selection manifest is zero.

Once written, both the qualified manifest and qualification document are
immutable. Later provisioning uses `--verify-only`; it never regenerates the
reviewed test boundary.

The split file pins source identities. The first real preparation determines
the deterministic prepared-manifest/logical-tree identity under the checked-in
preparation code. The qualified manifest SHA-256 then becomes the sealed test
identity recorded by the evaluation result.

## Frozen artifact evaluation

`scripts/run-speech-heldout-eval.sh` is not a training experiment.

It is hard-bound to:

- experiment `exp-87538823d2bf1562`;
- attempt `attempt-0001`;
- model `cnn_ctc_v3`;
- that attempt's recorded checkpoint, ONNX, OpenVINO XML and BIN hashes.

Before MA2450 execution it:

1. verifies the source attempt completed and was accepted;
2. verifies every recorded source artifact against the Phase 9 artifact index;
3. evaluates the frozen checkpoint in PyTorch on the held-out manifest;
4. evaluates the frozen ONNX artifact on the same samples;
5. requires exact valid-frame PyTorch/ONNX argmax agreement;
6. pins the edge automation worktree to the current evaluation commit;
7. deploys only the source attempt's already-recorded XML/BIN;
8. runs the normal locked MA2450 evaluator against the held-out manifest;
9. verifies returned benchmark and artifact hashes;
10. seals one result under `work/speech-asr/heldout-evaluations/`.

A completed result refuses a second execution for model-selection purposes.

### Persistent MA2450 execution

The held-out hardware path uses one long-running container and one compiled
OpenVINO `ExecutableNetwork` per server session. The evaluator sends each
fixed-shape float32 feature tensor over a binary stdin protocol and receives
one float32 logits tensor over stdout. The network is loaded once, one
unmeasured warmup inference is performed, and then every held-out utterance is
measured with the same live `InferRequest`.

The authoritative latency samples are the server-side `Infer()` durations.
Frontend calculation, pipe transfer, decoding, scoring, container startup and
model load/compile time are excluded from p50/p95 and realtime-factor
aggregation. Model-load time is recorded separately per persistent session.

Hardware execution is resumable after transport/device failures. A reusable
sample result is accepted only when its cache entry is bound to:

- execution mode `persistent-tensor-stream-v2`;
- sample feature SHA-256;
- exact frozen XML and BIN SHA-256;
- evaluator source SHA-256;
- output tensor element count and logits SHA-256;
- recorded persistent-server session identity and inference latency.

Earlier one-container-per-utterance cache entries are deliberately incompatible
with this contract and are not reused. This prevents cold-start measurements
from being mixed into the steady-state held-out latency distribution.

A retry may create a new persistent server session after a hardware transient;
each new session again performs one unmeasured warmup. The final result records
the number of server sessions and their model-load timings. No model weights,
decoder behavior, scoring rule, held-out sample, or acceptance threshold changes
when execution resumes.

The runtime image must contain the tensor-stream-capable `hello_myriad`. The
evaluator checks that capability before touching the held-out hardware path and
fails with an explicit rebuild instruction if the image predates the protocol.

Before the full hardware pass proceeds, the evaluator runs the first uncached
eligible feature tensor once through the trusted single-shot MYRIAD path and
once through the persistent server. It requires exact valid-frame argmax
agreement and a maximum absolute logit error no greater than 0.01. This parity
gate prevents binary framing, launcher stdout pollution, or persistent-runtime
I/O defects from reaching corpus scoring or sealing.

### Corrective rerun after the v1 transport defect

The first physical persistent evaluation was sealed under evaluation commit
`f5c059d2028d215777b28c1fc971b2f482185a05` before a launcher defect was
identified. `custom-server` had not set `OV_QUIET=1`, so the container
entrypoint wrote its text banner to stdout ahead of the binary logits stream.
That shifted every fixed-size client read and corrupted the hardware logits.
The defect signature was zero blank argmax frames on hardware while the frozen
reference had nonzero blank occupancy. A one-sample diagnostic subsequently
showed 0% argmax agreement and a 200-byte excess in the stream output.

The corrected `tensor-stream-v2` launcher suppresses the banner and a
one-sample physical comparison produced byte-for-byte identical logits between
the single-shot and persistent paths.

A corrective held-out execution is permitted only with
`--correct-invalid-v1-transport-result`. The controller accepts that flag only
when the existing seal matches the exact frozen source, benchmark manifest,
known defective evaluation commit, v1 execution mode, and zero-blank corruption
signature. It archives and hash-verifies both the old sealed result and old
hardware result, writes an explicit invalidation record, and allows at most one
replacement seal. The replacement remains test-only, cannot be used for model
selection, must use the same frozen checkpoint/manifest/reference, and must pass
the v2 single-shot/persistent parity gate before sealing.

The result records both the original training commit and the evaluation-code
commit so model weights and evaluation implementation remain independently
auditable.

## Final sealed outcome

The corrective physical evaluation completed on 2026-10-06 at evaluation
commit `db0e1465f0e3641891a45fe10a86d99a57dd5a82` and sealed the replacement
result for manifest
`b304a6e064b6e014bc7b1029b21202dbefbc7be2813a5b7b7dfd7847df887315`.

The frozen CPU/ONNX reference result is:

- WER: `1.0544630762763323`;
- CER: `0.8215929990002099`;
- blank-frame fraction: `0.6846211673964842`;
- empty-hypothesis fraction: `0.1932932616260677`;
- emitted/reference character ratio: `0.5857165780021477`;
- PyTorch/ONNX valid-frame argmax agreement: `1.0`.

The corrected MA2450/OpenVINO 2020.3.2 result is:

- WER: `1.0540426923256574`;
- CER: `0.8217904884160114`;
- blank-frame fraction: `0.6846721342807511`;
- empty-hypothesis fraction: `0.19297690604239165`;
- emitted/reference character ratio: `0.585642519471222`;
- inference-only RTF: `0.011201942869795709`;
- steady-state latency p50/p95: `16.7213195 / 16.7796900 ms`;
- evaluated samples: `6322 / 6322`;
- persistent server sessions: `1`;
- hardware failures: `0`.

The cross-runtime deltas are small: WER `-0.0004203839506748963`,
CER `+0.000197489415801555`, blank-frame fraction
`+0.000050966884266978596`, and empty-hypothesis fraction
`-0.00031635558367604233`. The physical first-sample parity gate also
produced byte-identical single-shot and persistent logits
(`valid_argmax_agreement=1.0`, `max_abs_error=0.0`).

These results establish that the MA2450/OpenVINO execution path is not the
material source of the ASR quality gap. The poor held-out recognition quality
is therefore model/data/decoder generalization evidence for the frozen
checkpoint, not a VPU transport or numerical-parity failure.

This held-out boundary is now consumed. It must not be rerun or used to choose
future model architectures, objectives, decoder variants, thresholds,
hyperparameters, or training-data decisions. Further ASR development must use
training/validation evidence only; this sealed result remains final
promotion/generalization evidence for the frozen checkpoint.

## Commands

Prepare and qualify on oberon:

```bash
./scripts/prepare-speech-heldout-eval.sh
cat work/speech-asr/ami/heldout-eval-v1/qualification.json
```

After reviewing that qualification, provision the frozen corpus once to edge:

```bash
./scripts/provision-speech-heldout-eval-edge.sh
```

Then execute the sealed evaluation:

```bash
./scripts/run-speech-heldout-eval.sh
```

Do not use the resulting held-out metrics to select among already-tried
variants. They are promotion/generalization evidence for the frozen scaled-data
checkpoint.
