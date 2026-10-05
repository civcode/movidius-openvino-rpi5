# ADR: sealed held-out AMI Full-corpus-ASR evaluation

Status: implemented, awaiting physical evaluation
Date: 2026-10-05
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

- execution mode `persistent-tensor-stream-v1`;
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

The result records both the original training commit and the evaluation-code
commit so model weights and evaluation implementation remain independently
auditable.

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
