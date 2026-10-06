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

## v7 rejection and architecture-screen-v1

The 112-channel `cnn_ctc_v7` experiment
`exp-ed00410b3da5d26c/attempt-0001` completed but was rejected on the exact
same ES2011 validation manifest as its v3 parent:

- v3 CER: `0.7588550365720209`;
- v7 CER: `0.7670333467718712`;
- v7 WER: `1.0462103217447636`;
- v7 MA2450 p95: `19.7972674 ms`.

Widening the full-context v3 graph therefore increased cost without improving
quality.

Subsequent architecture ideas first use `architecture-screen-v1`: an
approximately 25% deterministic hash-bucket subset of the frozen v4 training
manifest, 12 epochs, and the full ES2011 validation manifest. The first screen
experiment is an unchanged v3 control initialized with:

```bash
./scripts/prepare-speech-architecture-screen-v1.sh
./scripts/init-cnn-ctc-v3-architecture-screen.sh
```

Later screen candidates must use the exact derived screen manifest and budget
and should parent the recorded v3 screen control so review metrics are directly
comparable.

## v8 screen rejection and v9

`cnn_ctc_v8` completed as `exp-aa5f9f4edd71919b/attempt-0001`. Runtime
compatibility was clean, but the screen rejected the architecture:

- CER `0.8442089500662328` vs v3-screen `0.7984219316938317`;
- emitted/reference characters `0.2442550250532742`;
- empty-hypothesis fraction `0.3307148468185389`;
- MA2450 p95 `26.6580554 ms`;
- final PyTorch/ONNX argmax agreement `1.0`.

The 256-frame output increased hardware cost and worsened under-emission. Do
not give v8 extra epochs under the architecture screen.

The next candidate is `cnn_ctc_v9`, still parented to the v3 screen control.
It returns to 128 CTC frames and 96 channels while replacing five large
residual kernels with eight kernel-7 blocks at dilations
`[1,2,3,4,4,3,2,1]`.

Initialize with:

```bash
./scripts/init-cnn-ctc-v9-architecture-screen.sh
```

## v9 efficiency Pareto and v10

`cnn_ctc_v9` completed as `exp-df5bbe0ea05c2238/attempt-0001`:

- CER `0.8032598053331798` vs v3-screen `0.7984219316938317`;
- WER `1.005182465990067`;
- emitted/reference characters `0.39399873293785637`;
- empty-hypothesis fraction `0.2545168892380204`;
- MA2450 p95 `12.0927212 ms`;
- final PyTorch/ONNX argmax agreement `1.0`.

It is rejected for promotion but retained as an efficiency Pareto point.

The next candidate, `cnn_ctc_v10`, keeps v9's eight kernel-7 blocks and
dilations `[1,2,3,4,4,3,2,1]` and changes only width from 96 to 112
channels. It remains parented to the frozen v3 screen control so its CER gate
is directly comparable.

Initialize with:

```bash
./scripts/init-cnn-ctc-v10-architecture-screen.sh
```

## v10 screen pass and next width step

`cnn_ctc_v10` completed as `exp-884807b8e192aa9c/attempt-0001`:

- CER `0.7888613718827392` vs v3-screen `0.7984219316938317`;
- WER `1.0308788598574823`;
- emitted/reference characters `0.4818867707193457`;
- empty-hypothesis fraction `0.24509033778476041`;
- MA2450 p95 `13.6566998 ms`;
- final PyTorch/ONNX argmax agreement `1.0`;
- selected checkpoint epoch 12.

The candidate passes the frozen screen acceptance gate but misses the stronger
promotion CER `0.7744692737430167`. The 96 -> 112 width step improved CER
while retaining ample latency headroom, so the next screen keeps all v10
temporal geometry and tests one final width increase to 128 channels.

## v11 result and capacity-regime change

`cnn_ctc_v11` completed as `exp-4fb0f93165d0f76d/attempt-0001`:

- CER `0.7865576225306686`;
- WER `1.0205139278773483`;
- emitted/reference characters `0.4790646777630594`;
- empty-hypothesis fraction `0.22152395915161036`;
- MA2450 p95 `14.9809824 ms`;
- selected checkpoint epoch 12.

It passes the frozen v3-screen CER gate but misses the promotion target.
The 128-channel result closes the small width sweep. The next model moves
directly to a large capacity regime, with latency recorded but no longer
rejected against the inherited 25 ms architecture-screen threshold.


## v12 large-capacity probe

The active model is `cnn_ctc_v12`. It intentionally leaves the small-model
regime at 19,627,687 parameters and 2,515,673,088 fixed-input MACs, using two
stride-2 stems and sixteen 448-channel kernel-5 residual blocks.

The experiment reuses the exact architecture-screen training and ES2011
validation manifests. It keeps the 12-epoch validation-CER-selected ranking
budget, but removes numeric CER/WER/RTF/p95 rejection thresholds. OpenVINO
conversion, numerical compatibility and physical MYRIAD execution are the
capacity feasibility gates.

Initialize with:

```bash
./scripts/init-cnn-ctc-v12-capacity-probe.sh
```

See
`../docs/adr/model-quality-v4-cnn-ctc-v12-capacity-probe.md`.


## v12 capacity result

`cnn_ctc_v12` completed as `exp-4e16fd348004571b/attempt-0001`.

Hardware feasibility was established:

- 19,627,687 parameters;
- 2,515,673,088 fixed-input MACs;
- MA2450 p95 `103.8896558 ms`;
- RTF `0.058019043467865204`;
- valid FP16 IR;
- final PyTorch/ONNX frame-argmax agreement `1.0`.

The training attempt collapsed to the CTC blank solution. CER/WER were both
`1.0`, blank fraction was `1.0`, and all 1,273 hypotheses were empty.
The `3e-3` OneCycle peak was reached too quickly: epoch 2 recorded a raw
pre-clip gradient norm of `36,104,904` and mean train loss `90.963`.

The next experiment retains the exact v12 inference graph and changes only the
learning-rate trajectory to `3e-4 -> 1.2e-3 -> 3e-5`, with a 30% OneCycle
ramp.


## v13 stabilized LR retry

The active follow-up is `cnn_ctc_v13`, parented to completed v12 experiment
`exp-4e16fd348004571b`. The inference graph is byte-for-byte equivalent in
declared geometry to v12; only the optimizer schedule changes.

Initialize with:

```bash
./scripts/init-cnn-ctc-v13-stabilized-lr.sh
```

The schedule is `3e-4 -> 1.2e-3 -> 3e-5` with a 30% OneCycle ramp.


## v13-v15 result and v16 QuartzNet architecture reset

`cnn_ctc_v13` completed as `exp-968150d44c4479ac/attempt-0001`; its
OneCycle trajectory returned to near-total blank emission.

`cnn_ctc_v14` completed as `exp-1e2478b84317ab02/attempt-0001` using the
stable `3e-4 -> 3e-5` cosine schedule. It selected epoch 8 at validation CER
`0.7898404653573691`, with deployed MA2450 p95 `103.8884392 ms`.

`cnn_ctc_v15` completed as `exp-5d1209f5120388f8/attempt-0001` at deployed
CER `0.8428267004549905`, WER `0.9941697257611747`, blank-frame fraction
`0.8668262967005881`, and MA2450 p95 `103.8926016 ms`. BatchNorm therefore
failed the v15 decision rule: it regressed against v14 and remained behind the
much smaller v11 accuracy/latency point. Further local tuning of the v12-v15
large residual topology is closed.

The active follow-up is `cnn_ctc_v16`, parented to completed v15 evidence but
replacing its acoustic graph with the published QuartzNet-15x5 topology:

- C1 256 channels, kernel 33, stride 2;
- B1-B5 kernels `33/39/51/63/75`, each block type repeated 3 times and each
  block containing 5 time-channel-separable modules;
- channels `256/256/512/512/512`;
- projected residuals with BatchNorm/ReLU/dropout;
- C2 512 channels, kernel 87, dilation 2;
- C3 1024 channels, kernel 1;
- project to the frozen 39-token character CTC vocabulary;
- 18,934,631 trainable parameters and `[1,256,39]` output.

The VPU boundary ends at CTC logits. Greedy decoding remains the architecture
screen decoder; CPU prefix-beam search and language-model scoring are deferred
to the next isolated system experiment if v16 improves acoustic quality.

Initialize with:

```bash
./scripts/init-cnn-ctc-v16-quartznet.sh
```

See
`../docs/adr/model-quality-v4-cnn-ctc-v16-quartznet.md`.
