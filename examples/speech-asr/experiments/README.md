# Experiment records

Phase 9 makes every model-development attempt a self-contained, immutable
experiment request plus one or more execution attempts.

The controller-side records live below:

```text
work/speech-asr/experiments/
├── history.json
└── exp-<16 hex>/
    ├── request/
    │   ├── experiment.json
    │   ├── proposal.json
    │   ├── model-spec.json
    │   ├── train-config.json
    │   └── acceptance.json
    └── attempts/
        └── attempt-0001/
            ├── attempt.json
            ├── deployment-manifest.json
            ├── results/
            │   ├── training.json
            │   ├── compatibility.json
            │   ├── hardware.json
            │   ├── accuracy.json
            │   └── result.json
            └── logs/
```

Generated checkpoints, ONNX models and OpenVINO XML/BIN artifacts remain under
`work/` and are not copied into Git. Each attempt records their repository-
relative storage path and SHA-256.

## Identity

The experiment ID is content-derived:

```text
exp-<first 16 hex characters of immutable request identity SHA-256>
```

The immutable request identity covers:

- parent experiment ID;
- exact 40-character repository commit;
- benchmark ID, manifest path and manifest SHA-256;
- canonical SHA-256 of proposal, model spec, training config and acceptance
  policy.

Changing any of those inputs creates a new experiment ID. A transient retry
does **not** create a new experiment: it creates `attempt-0002`,
`attempt-0003`, and so on according to the frozen retry policy.

## Lifecycle

An attempt follows:

```text
APPROVED
   |
   v
EXECUTE
   |
   v
EVALUATE
   |
   v
AWAIT_REVIEW
```

A failed or blocked attempt may transition directly to `AWAIT_REVIEW` from
an earlier state, but a successful `completed` attempt must pass through
`EVALUATE`.

Once an artifact or stage-result reference is recorded, the name/path/hash
binding is immutable for that attempt.

## Manager

Use the standard-library Phase 9 manager through the repository Python helper:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py init \
  --proposal proposal.json \
  --model-spec model-spec.json \
  --train-config train-config.json \
  --acceptance acceptance.json \
  --benchmark-id ami-smoke-v1 \
  --manifest work/speech-asr/ami/ami-smoke-v1/manifest.jsonl
```

The command validates the four reviewed request documents, verifies declared
training/validation manifest hashes, records the current Git commit, computes
the immutable experiment ID and updates `history.json`.

Start the first execution attempt:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py start-attempt \
  --experiment work/speech-asr/experiments/exp-...
```

Advance into execution:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py transition \
  --experiment work/speech-asr/experiments/exp-... \
  --attempt attempt-0001 \
  --state EXECUTE
```

As deterministic tools produce evidence, record it instead of parsing logs
later:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py record-stage \
  --experiment work/speech-asr/experiments/exp-... \
  --attempt attempt-0001 \
  --stage training \
  --source work/.../training-result.json

./scripts/python.sh examples/speech-asr/tools/manage_experiment.py record-artifact \
  --experiment work/speech-asr/experiments/exp-... \
  --attempt attempt-0001 \
  --name openvino_xml \
  --source work/.../model.xml
```

After local execution completes, transition to `EVALUATE`, record
`openvino_xml` and `openvino_bin`, then generate the future edge-worker
handoff:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py deployment-manifest \
  --experiment work/speech-asr/experiments/exp-... \
  --attempt attempt-0001 \
  --worker-commit "$(git rev-parse HEAD)" \
  --evaluator cnn-ctc-myriad-v1 \
  --result-path attempts/attempt-0001/results/hardware.json
```

By default that manifest includes only `openvino_xml` and `openvino_bin`.
Additional recorded artifacts must be explicitly requested with repeated
`--artifact NAME`.

When evaluation is complete:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py transition \
  --experiment work/speech-asr/experiments/exp-... \
  --attempt attempt-0001 \
  --state AWAIT_REVIEW \
  --outcome completed
```

This writes the compact `results/result.json` handoff. Failed or blocked
attempts require an explicit failure class. A retry is allowed only when the
frozen acceptance policy lists that failure class as retryable and the maximum
attempt count has not been reached.

Validate an experiment bundle at any point with:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py validate \
  --experiment work/speech-asr/experiments/exp-...
```

## Authority boundary

The manager executes lifecycle mechanics only. It does not design a new
architecture, change benchmark/scoring rules, loosen acceptance policy or
silently change the reviewed request.

The remote oberon -> edge execution implementation belongs to Phase 10 and
consumes the deployment manifest defined here. See
[`../docs/remote-experiment-orchestration.md`](../docs/remote-experiment-orchestration.md).


## Phase 9 acceptance

Accepted on 2026-10-05 from oberon. The repository static gate passed and the
speech-ASR suite completed 143 tests successfully with 3 expected skips for
NumPy-only frontend tests in the tools environment. The Phase 9 experiment
identity, transition, deployment, lineage and immutability tests all passed.


## Phase 10 execution

An approved experiment is executed from the controller host with:

```bash
./scripts/run-speech-experiment.sh \
  --experiment work/speech-asr/experiments/exp-... \
  --worker edge
```

The controller creates a new attempt according to the frozen retry policy. It
records controller/edge logs, attempt-local training artifacts, compatibility
evidence, the deployment manifest, collected hardware/accuracy results,
acceptance-policy evaluation and the final compact REVIEW handoff.

The baseline acceptance request for the already-qualified model can be created
without manually authoring JSON:

```bash
./scripts/init-cnn-ctc-v1-experiment.sh
```

That initializer does not propose a new architecture; it declares the frozen
`cnn_ctc_v1` executor specifically to qualify Phase 10 orchestration.


## Phase 10 acceptance

Accepted on 2026-10-05 with experiment `exp-f915ec624a63caf6`,
attempt `attempt-0001`. Oberon executed the approved baseline request through
the dedicated edge worktree and physical MA2450 worker, collected the returned
evidence, applied the frozen acceptance policy and produced:

- `status: completed`;
- `acceptance: accepted`;
- `review_state: AWAIT_REVIEW`.

This is the first accepted end-to-end Phase 9/10 experiment bundle and is the
reference handoff shape for later architecture experiments.


## Phase 11 generation-1 review

The first `cnn_ctc_v2` experiment completed on 2026-10-05 as
`exp-1c682f4eda475a01/attempt-0001`. Execution completed normally and the
attempt reached `AWAIT_REVIEW`, but its frozen acceptance policy returned
`rejected`.

Use the review helper to inspect the authoritative reason instead of inferring
from controller exit status:

```bash
./scripts/review-speech-experiment.sh \
  --experiment work/speech-asr/experiments/exp-1c682f4eda475a01
```

The review output includes threshold failures and candidate-vs-parent metric
deltas. Future controller runs also print acceptance reasons and threshold
checks directly in their terminal JSON.


## Phase 11 generation 2

Generation-1 review identified optimization quality rather than hardware
compatibility as the active bottleneck. The reviewed generation-2 initializer is:

```bash
./scripts/init-cnn-ctc-v3-experiment.sh
```

It creates a child of `exp-1c682f4eda475a01` with the frozen
normalization-free `cnn_ctc_v3` model declaration and 32-epoch/batch-1
training request. The experiment keeps the v1 CER ceiling and tightens the
hardware acceptance limits to RTF <= 0.01 and p95 <= 25 ms.

After execution, use:

```bash
./scripts/review-speech-experiment.sh --experiment work/speech-asr/experiments/exp-...
```

The review output now includes optimizer-step count, selected best epoch and
best validation loss for generation-2 training.


## Phase 11 final review

Generation 2 completed as `exp-a6f83c0451532122/attempt-0001` and reached
`AWAIT_REVIEW`. Execution was healthy: CUDA placement was verified, OpenVINO
and MYRIAD compatibility passed, p95 latency was 16.59 ms and RTF was
0.004579. The frozen policy rejected only CER because the candidate scored
1.0 against a 0.913043 ceiling.

The 32-epoch/64-step v3 run reduced best validation loss to 2.66385 but did not
improve greedy transcript accuracy. Together with generation 1, this satisfies
the Phase 11 two-generation REVIEW exit criterion and demonstrates that the
two-eligible-record smoke population is not suitable for further architecture
ranking.

Do not create another model lineage from the smoke result. First run:

```bash
./scripts/qualify-speech-model-data.sh
```

and review the resulting
`work/speech-asr/ami/model-quality-v1/qualification.json`. A later baseline
experiment must bind the accepted train/validation manifest hashes explicitly.


## Model-quality v1 baseline

The reviewed post-Phase-11 qualification is accepted for an interim
speaker-held-out baseline:

- train: 125 eligible A/B/C records, 219.998 s,
  SHA-256 `89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a`;
- validation: 95 eligible D records, 122.52 s,
  SHA-256 `07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923`.

The first experiment keeps `cnn_ctc_v3` and its 32-epoch/batch-1 training
policy unchanged. It is initialized with:

```bash
./scripts/init-cnn-ctc-v3-quality-baseline.sh
```

The validation manifest is also the hardware benchmark manifest. Provision it
out-of-band before the experiment:

```bash
./scripts/provision-speech-model-data-edge.sh
```

The baseline has no numeric WER/CER acceptance threshold; its purpose is to
establish those metrics on the reviewed validation population. Review tooling
does not compute accuracy/RTF deltas against the smoke parent because benchmark
manifest identities differ.


## CER-aligned checkpoint diagnostic

`exp-3c7727ca3f37ba2c/attempt-0001` completed and was accepted. Relative to
the same-manifest validation-loss-selected parent, minimum-CER checkpoint
selection changed the exported epoch from 11 to 21 and improved CER from
0.980847 to 0.866935.

Decoder collapse improved substantially:

- blank frames: 95.93% -> 61.33%;
- empty hypotheses: 84/95 -> 19/95;
- emitted/reference character ratio: 2.52% -> 47.98%.

The inference graph and MA2450 latency were unchanged. WER remained poor at
1.1614, so the next child retains CER-aligned selection and tests deterministic
training-only SpecAugment:

```bash
./scripts/init-cnn-ctc-v3-specaugment.sh
```


## SpecAugment sibling and v4 frontend generation

`exp-2de4643d351299ec/attempt-0001` completed successfully but is rejected as
a model-quality direction. Relative to its accepted CER-selected parent, it
regressed CER and decoder emission while leaving hardware cost unchanged.

The next lineage therefore branches again from
`exp-3c7727ca3f37ba2c`, not from the SpecAugment sibling.

`cnn_ctc_v4` keeps the v3 graph and changes only the frontend contract from
`logmel-v1` to `logmel-v2`, where padding is excluded from CMVN and padded
normalized frames are zero.

Initialize with:

```bash
./scripts/init-cnn-ctc-v4-valid-cmvn.sh
```


## v4 result and CTC-objective child

`exp-92ee80e9dfdfcc79/attempt-0001` completed successfully, but valid-frame
CMVN regressed CER and blank/empty-hypothesis diagnostics relative to
`exp-3c7727ca3f37ba2c`. v4 is therefore not the new parent.

The next child again branches from `exp-3c7727ca3f37ba2c` and keeps
`cnn_ctc_v3` / `logmel-v1` unchanged. Its only training-identity delta is
`blank-logit-penalty-v1` with `blank_logit_penalty=0.25`.

Initialize with:

```bash
./scripts/init-cnn-ctc-v3-blank-penalty.sh
```


## Sealed held-out evaluation is not a training experiment

The accepted scaled-data checkpoint
`exp-87538823d2bf1562/attempt-0001` is evaluated on the official held-out AMI
ASR boundary without creating a new trainable experiment.

The evaluation controller is hard-bound to that source experiment and attempt.
It verifies the source attempt is completed/accepted and verifies the recorded
SHA-256 identities of:

- checkpoint;
- ONNX;
- OpenVINO XML;
- OpenVINO BIN.

It then scores the frozen checkpoint/ONNX locally, requires exact valid-frame
argmax agreement, and deploys the already-recorded XML/BIN to the normal locked
MA2450 worker. No training command or checkpoint-selection path is reachable
from the held-out controller.

The completed result lives separately under:

```text
work/speech-asr/heldout-evaluations/
  exp-87538823d2bf1562/
    <heldout-manifest-sha-prefix>/
      result.json
```

A completed result is sealed. Read it with:

```bash
./scripts/review-speech-heldout-eval.sh
```

Do not rerun the held-out corpus to choose among previously explored model or
training variants.

## Model-quality-v4 ES2011 baseline and width ablation

The fresh Full-corpus-ASR development baseline completed as
`exp-63fdb8d218673527/attempt-0001` with the unchanged `cnn_ctc_v3` graph.
On the frozen ES2011 validation manifest it measured CER `0.758855`, WER
`1.042755`, 69.07% blank frames, 15.79% empty hypotheses and 58.96%
emitted/reference characters. MA2450 p95 remained `16.781 ms`.

Because the parent scaled-data experiment used a different validation manifest,
its CER/WER are not compared directly. This experiment is the new
model-quality-v4 reference.

The next child is `cnn_ctc_v7`, a width-only capacity ablation:

- second stem/residual width: 96 -> 112;
- parameters: 1,346,343 -> 1,818,183;
- fixed-input MACs: 174,804,992 -> 235,177,984;
- frontend, kernels, receptive field, CTC objective, decoder, data and training
  schedule unchanged.

Initialize with:

```bash
./scripts/init-cnn-ctc-v7-wide.sh
```

Its immutable CER acceptance ceiling is the v3 ES2011 baseline
`0.7588550365720209`; physical p95 remains capped at 25 ms.

