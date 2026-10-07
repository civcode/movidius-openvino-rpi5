# Speech ASR on Movidius

This directory is the development area for automatic speech recognition on the
Movidius/OpenVINO 2020.3.2 platform provided by this repository.

The first development model is **rm_cnn4a**. It is a conversion/device
qualification fixture: its published Intel package provides Kaldi feature and
reference-score ARKs, so it exercises FP16 conversion, tensor transport,
MYRIAD inference, numerical comparison, timing and result/provenance handling.
It is **not** the raw-audio AMI model and is not the long-term architecture.

The long-term workflow is designed for custom acoustic models trained with a
normal PyTorch/CUDA toolchain, exported through ONNX, converted to OpenVINO
2020.3 FP16 IR, and evaluated on the Raspberry Pi/Movidius target.

## Project boundary

The repository root owns the generic platform:

- OpenVINO 2020.3.2 build and runtime
- MYRIAD plugin and Movidius firmware
- USB/device access
- generic model loading and deployment support
- platform build/container infrastructure

This subproject owns the speech-specific work:

- AMI corpus preparation and manifests
- audio, benchmark and model contracts
- speech feature extraction and chunking
- model-specific adapters
- decoding
- WER/CER/timing evaluation
- experiment records
- training/export workflow
- agent-driven model development

Speech-specific assumptions must not leak into the generic Movidius runtime.

## Development lifecycle

The lifecycle has a hard boundary between architecture design and experiment
execution:

```text
DESIGN (frontier model)
  |
  v
APPROVED model/experiment spec
  |
  v
EXECUTE (local model + deterministic tools)
  |
  v
EVALUATE (deterministic scoring + local orchestration)
  |
  v
AWAIT_REVIEW
  |
  v
REVIEW / next DESIGN (frontier model)
```

The execution side may train, export, convert, test, benchmark, retry declared
operations and collect diagnostics. It may not silently redesign the network or
change the benchmark contract.

See [docs/agent-workflow.md](docs/agent-workflow.md).

The implementation sequence and phase exit criteria are defined in
[docs/implementation-roadmap.md](docs/implementation-roadmap.md).

## Data and measurement baseline

AMI is the canonical development corpus for end-to-end transcript evaluation.
Corpus-specific input is normalized to the internal speech sample format before
it reaches inference code.

Core invariants:

- 16 kHz audio
- mono
- float32 f32le normalized sample representation
- integer sample indices as the canonical timing coordinate
- versioned dataset splits/manifests
- versioned text normalization and scoring rules
- pinned OpenVINO/MYRIAD runtime for hardware comparisons

Feature geometry, chunking and future model architecture are explicit,
versioned experiment parameters rather than hidden constants.

See [docs/parameters.md](docs/parameters.md).

## Model paths

Initial device qualification:

```text
Intel rm_cnn4a Kaldi feature ARK
  -> OpenVINO 2020.3 FP16 IR
  -> MYRIAD
  -> acoustic score ARK
  -> deterministic comparison to Intel reference score ARK
```

AMI/custom-model path:

```text
AMI/WAV
  -> canonical 16 kHz mono audio
  -> model-declared frontend
  -> custom acoustic model
  -> decoder
  -> normalized hypothesis
  -> deterministic WER/CER + timing evaluation
```

Future custom-model loop:

```text
architecture spec
  -> PyTorch/CUDA training
  -> checkpoint
  -> ONNX
  -> OpenVINO 2020.3 FP16 IR
  -> Pi + Movidius evaluation
  -> results
  -> architecture review
  -> next architecture
```

The benchmark, dataset and hardware measurement contracts remain stable while
model architectures evolve.

## Python environment

Host-side speech tooling follows the repository-wide uv policy. Dependency-free speech utilities run through `./scripts/python.sh`; Python packages must not be installed into system Python.

## Tests

The speech-specific fixture tests are deliberately independent of network and
MYRIAD hardware:

```bash
./scripts/test-speech-asr.sh
```

Physical-device qualification, the repeated Phase 6 worker, and Phase 7
recorded replay remain explicit operations:

```bash
./scripts/prepare-rm-cnn4a.sh
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend myriad --platform arm64

./scripts/benchmark-speech.sh \
    --model rm_cnn4a --backend myriad --platform arm64

./scripts/python.sh examples/speech-asr/tools/make_streaming_updates.py \
    work/speech-asr/ami/ami-smoke-v1/manifest.jsonl \
    --output work/speech-asr/streaming/updates.json
# The helper prints the corresponding ./scripts/replay-speech.sh command.
```

## Directory layout

```text
speech-asr/
├── README.md
├── agent/                 execution/design authority notes
├── contracts/             versioned machine-readable contracts
├── datasets/
│   └── ami/               AMI preparation and manifest rules
├── docs/
│   ├── architecture.md
│   ├── parameters.md
│   ├── agent-workflow.md
│   ├── implementation-roadmap.md
│   └── adr/
├── evaluation/            deterministic scoring/measurement
├── experiments/           experiment artifact conventions
├── models/
│   └── rm_cnn4a/          initial development fixture
├── runtime/               Pi/MYRIAD speech runtime boundary
└── training/              future PyTorch/CUDA training/export path
```

No corpus audio, checkpoints or generated model binaries are intended to be
committed here unless a later decision explicitly says otherwise.


## Trainable custom-model skeleton

Milestone B is complete with `cnn_ctc_v1`, the first custom PyTorch/CTC
skeleton. The model keeps log-mel extraction on the host so the deployment
graph remains small and fixed-shape for OpenVINO 2020.3/MYRIAD. The accepted
2026-10-05 lifecycle includes deterministic CUDA training, ONNX export,
OpenVINO 2020.3 FP16 conversion, physical Pi 5/arm64 + MA2450 execution, and a
contract-valid AMI smoke result.

Always run compatibility before training:

```bash
./scripts/probe-cnn-ctc-v1.sh --platform amd64 --no-device
./scripts/train-cnn-ctc-v1.sh --device cuda
```

On the Pi, run the physical compatibility probe and trained-model evaluator.

Phase 9 is complete and provides the explicit immutable experiment lifecycle
used for further model development. Reviewed proposal/model/training/acceptance
documents are managed with:

```bash
./scripts/python.sh examples/speech-asr/tools/manage_experiment.py --help
```

Phase 10 is complete and implements the accepted oberon -> edge SSH/rsync execution path
using those experiment and deployment-manifest contracts. For the frozen
`cnn_ctc_v1` baseline:

```bash
./scripts/init-cnn-ctc-v1-experiment.sh
./scripts/run-speech-experiment.sh \
  --experiment work/speech-asr/experiments/exp-... \
  --worker edge
```

The executor rejects unknown architecture declarations rather than silently
changing or ignoring them. Runtime-image rebuilds and benchmark/acceptance
changes remain outside ordinary experiment execution.


The Phase 10 physical acceptance run on 2026-10-05 used
`exp-f915ec624a63caf6/attempt-0001` and completed with
`acceptance: accepted` at `AWAIT_REVIEW`. The project is ready to move into
Phase 11 architecture optimization while keeping this execution path fixed.


## Phase 11 generation 1

The first architecture optimization candidate is `cnn_ctc_v2`: a residual
large-kernel temporal CTC encoder inspired by the low-rate computation of
FastConformer/Zipformer and the residual temporal-convolution approach of
QuartzNet/Citrinet, constrained to the legacy OpenVINO/MYRIAD deployment
envelope.

It preserves `[1,64,512] -> [1,128,39]` and the existing frontend/decoder, but
moves to two stride-2 stems plus five 96-channel residual blocks with kernels
`11,19,27,35,43`. The fixed estimate is 1,347,463 parameters and
174,804,992 MACs.

Run it through the formal lifecycle with:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v2-experiment.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
```

The controller physically probes the initialized v2 graph on MA2450 before full
CUDA training. Phase 11 remains open until generation 1 is reviewed and a
second architecture generation is selected and evaluated from that evidence.


## Phase 11 generation 2

Generation-1 review showed that `cnn_ctc_v2` was hardware-safe but
optimization-starved: only CER failed, while OpenVINO/MYRIAD compatibility and
realtime performance retained large margin.

`cnn_ctc_v3` keeps v2's 4x reduction, 96-channel residual stack, kernel
schedule and 174,804,992 MAC envelope, but removes BatchNorm and small-initializes
each residual block's final 1x1 projection with 1%-scaled Kaiming weights. Training is increased to 32 epochs
with batch size 1, Adam at 3e-4, cosine decay to 3e-5, gradient clipping and
best-validation-loss checkpoint selection.

Run generation 2 with:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v3-experiment.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The acceptance policy retains the v1 CER ceiling while tightening the already
comfortable v2 hardware limits to RTF <= 0.01 and p95 <= 25 ms.


## Phase 11 completion and next data boundary

Phase 11 completed after two physical architecture generations. `cnn_ctc_v2`
and `cnn_ctc_v3` both remained comfortably realtime on MA2450, but the
two-eligible-record smoke population could not rank model changes reliably:
v3 substantially reduced CTC loss while greedy CER regressed to 1.0.

The final review is in
`docs/phase11-final-review.md`.

Before another architecture generation, derive and review the larger
speaker-disjoint model-quality manifests:

```bash
./scripts/python.sh examples/speech-asr/datasets/ami/prepare_ami.py \
  --subset benchmark
./scripts/qualify-speech-model-data.sh
```

Future custom-model evaluation results include per-sample CTC collapse and edit
diagnostics so blank collapse or pathological emission can be distinguished
from ordinary substitution/deletion errors.


## Model-quality v1 baseline

The post-Phase-11 data qualification is reviewed and accepted for the next
controlled baseline. It uses the frozen ES2002a benchmark preparation but
partitions by speaker:

- train A/B/C: 125 eligible records, 219.998 seconds;
- validation D: 95 eligible records, 122.52 seconds;
- zero record and speaker overlap.

This is intentionally an interim speaker-holdout benchmark, not cross-meeting
generalization evidence.

Provision the reviewed dataset once on edge, then run the frozen v3 baseline:

```bash
./scripts/provision-speech-model-data-edge.sh

EXP="$(
  ./scripts/init-cnn-ctc-v3-quality-baseline.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The experiment keeps `cnn_ctc_v3` unchanged and removes smoke-derived WER/CER
acceptance ceilings. It establishes the larger-data accuracy reference while
retaining the proven OpenVINO/MYRIAD realtime gates.


## cnn_ctc_v4 valid-frame frontend

After the deterministic SpecAugment diagnostic regressed decoder emission,
model development moved to a structural frontend correction rather than
stronger regularization.

`cnn_ctc_v4` keeps the v3 Conv1D/ReLU inference graph and MA2450 workload but
changes host feature normalization from padding-inclusive `logmel-v1` to
valid-frame `logmel-v2`. Padding no longer contributes to CMVN and padded
normalized frames are zero.

Run the reviewed child with:

```bash
./scripts/init-cnn-ctc-v4-valid-cmvn.sh
```


## Held-out promotion boundary

The frozen scaled-data `cnn_ctc_v3` checkpoint
`exp-87538823d2bf1562/attempt-0001` has now completed its one-time sealed AMI
Full-corpus-ASR unseen scenario-component evaluation. The corrected
`tensor-stream-v2` MA2450 path tracked the CPU/ONNX reference closely, so the
remaining recognition gap is model/data generalization evidence rather than a
VPU transport problem.

That held-out boundary is consumed. Do not rerun it or use its metrics for
future model, objective, decoder, threshold, hyperparameter, or data selection.

## Model-quality v4 fresh development boundary

The fresh Full-corpus-ASR development boundary is now frozen:

- training: 15,738 eligible utterances / 25,846.327 s;
- validation: 1,273 eligible ES2011 utterances / 2,279.385 s;
- zero train/validation meeting or record overlap;
- ES2002 retired from new model selection;
- sealed EN2002/ES2004/IS1009/TS3003 held-out meetings excluded.

The unchanged `cnn_ctc_v3` baseline completed as
`exp-63fdb8d218673527/attempt-0001`:

- CER `0.7588550365720209`;
- WER `1.0427553444180522`;
- blank frames `69.07%`;
- empty hypotheses `15.79%`;
- emitted/reference characters `58.96%`;
- MA2450 p95 `16.7812226 ms`;
- 32-epoch training wall time about 56.4 minutes.

That establishes the direct-comparison ES2011 baseline. The next controlled
candidate is `cnn_ctc_v7`: the same v3 topology widened from 96 to 112
channels in the second stem and five residual blocks. Everything else remains
fixed.

Run:

```bash
./scripts/provision-speech-model-data-v4-edge.sh

EXP="$(
  ./scripts/init-cnn-ctc-v7-wide.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The controller performs an initialized ONNX/OpenVINO/physical-MYRIAD
compatibility probe before training. The v7 decision gate requires ES2011 CER
no worse than the v3 baseline and MA2450 p95 no greater than 25 ms.

See `docs/adr/model-quality-v4-fresh-development.md` and
`docs/adr/model-quality-v4-cnn-ctc-v7-wide.md`.

The sealed held-out benchmark remains consumed and must not be used for this
architecture decision.

## Fast architecture screen

Full model-quality-v4 architecture runs are now preceded by
`architecture-screen-v1`. It deterministically selects about 25% of the
frozen qualified training manifest by sample-ID hash, requires representation
from all 48 training meetings, trains for 12 epochs, and keeps the full frozen
ES2011 validation set.

Prepare and verify it with:

```bash
./scripts/prepare-speech-architecture-screen-v1.sh
./scripts/prepare-speech-architecture-screen-v1.sh --verify-only
```

Establish the proxy reference with unchanged `cnn_ctc_v3`:

```bash
EXP="$(
  ./scripts/init-cnn-ctc-v3-architecture-screen.sh |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])'
)"
./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The v3 proxy completed as `exp-00e6b1e434d187d8/attempt-0001` with
CER `0.7984219316938317` in about 6.75 minutes of training.

The first candidate, `cnn_ctc_v8`, was rejected: CER regressed to
`0.8442089500662328`, emitted/reference characters fell to `0.244255`,
empty hypotheses rose to `0.330715`, and MA2450 p95 rose to `26.6580554
ms`. The 256-frame CTC direction is retired.

`cnn_ctc_v9` finished at CER `0.8032598053331798` and MA2450 p95
`12.0927212 ms`. It narrowly misses the v3-screen CER gate but is retained as
an efficiency Pareto point because emission diagnostics are healthy and latency
is much lower.

`cnn_ctc_v10` completed as `exp-884807b8e192aa9c/attempt-0001`. It
passed the screen gate with CER `0.7888613718827392`, emitted/reference
characters `0.4818867707193457`, empty hypotheses `0.24509033778476041`,
and MA2450 p95 `13.6566998 ms`. That is about a 1.20% relative CER
improvement over the v3 screen control, but it does not reach the 3% relative
promotion threshold `0.7744692737430167`.

The final small-width candidate, `cnn_ctc_v11`, completed as
`exp-4fb0f93165d0f76d/attempt-0001` with CER
`0.7865576225306686` and MA2450 p95 `14.9809824 ms`. It passes the
frozen v3-screen CER gate, but the 112 -> 128 width step only slightly
improves v10. The small width sweep is closed.

## v12 large-capacity result

`cnn_ctc_v12` completed as `exp-4e16fd348004571b/attempt-0001`.
The capacity result is positive: the 19,627,687-parameter,
2,515,673,088-MAC graph converted and ran physically on MA2450 at
`103.8896558 ms` p95 and RTF `0.058019043467865204`.

The quality result is not usable because the aggressive `3e-3`, 10%-warmup
OneCycle schedule drove training into permanent CTC blank collapse. CER and
WER were both `1.0`, all 56,798 validation argmax frames were blank, and all
1,273 validation hypotheses were empty. Epoch 2 reached a raw pre-clip
gradient norm of about 36.1 million and mean train loss `90.963`.

The architecture is retained. The next experiment keeps the same 20M model and
changes only optimization: OneCycle `3e-4 -> 1.2e-3 -> 3e-5`, with the peak
moved to 30% of optimizer steps. This is still a 4x higher peak LR than the
historical small-model schedule, but avoids reaching the maximum during the
second epoch.



## v13 stabilized large-capacity training retry

The active candidate is `cnn_ctc_v13`. It keeps the exact v12 inference
graph and changes only the OneCycle trajectory:

- start LR: `3e-4`;
- peak LR: `1.2e-3`;
- peak position: 30% of optimizer steps;
- final LR: `3e-5`;
- Adam, gradient clipping 5.0, 12 epochs and validation-CER checkpoint
  selection remain unchanged.

Run:

```bash
INIT_JSON="$(./scripts/init-cnn-ctc-v13-stabilized-lr.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

The model-quality-v4 data are already provisioned if the v12 run was executed
on the same edge worker.


## v13-v16 result and v17 QuartzNet reference training

`cnn_ctc_v14` remains the strongest large-model acoustic result at CER
`0.7898404653573691`, while the much smaller `cnn_ctc_v11` remains the
accuracy/latency Pareto point at CER `0.7865576225306686` and MA2450 p95
`14.9809824 ms`. `cnn_ctc_v15` BatchNorm regressed to physical CER
`0.8428267004549905`.

`cnn_ctc_v16` reset the acoustic graph to QuartzNet-15x5. The important
result is mixed but clear:

- the fixed `[1,64,512] -> [1,256,39]` QuartzNet graph exported to ONNX and
  converted to OpenVINO 2020.3 successfully;
- the initialized physical MYRIAD gate achieved frame-argmax agreement
  `1.0` with max absolute error `7.482245564460754e-06`;
- training with the inherited Adam/dropout-0.2 recipe was unstable;
- best validation CER was only `0.9297356447618499` at epoch 5;
- epoch 12 returned to CER `1.0`, validation loss `43.97023439682033`
  and near-total blank emission;
- full-corpus edge evaluation then lost its persistent MYRIAD server after
  806 successful requests. That is an execution-session failure, not evidence
  that the graph is unsupported.

The evaluator now restarts a failed persistent MYRIAD session and retries the
current uncached sample, retaining the existing per-sample cache.

The active candidate is `cnn_ctc_v17`. It keeps the QuartzNet inference
topology and corrects the training recipe instead of inventing another graph:
dropout `0.0`, NovoGrad at peak LR `0.01`, betas `0.8/0.5`, weight
decay `0.001`, a 12% step warmup, then cosine decay to `1e-5`. The exact
12-epoch screen data/budget, batch size 1, standard CTC and validation-CER
checkpoint selection remain frozen.

Run:

```bash
INIT_JSON="$(./scripts/init-cnn-ctc-v17-quartznet-reference-training.sh)"
EXP="$(printf '%s\n' "$INIT_JSON" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

./scripts/run-speech-experiment.sh --experiment "$EXP" --worker edge
./scripts/review-speech-experiment.sh --experiment "$EXP"
```

See `docs/adr/model-quality-v4-cnn-ctc-v17-quartznet-reference-training.md`.

## v19 frozen acoustic and decoder result

`cnn_ctc_v19` is the current frozen acoustic reference. Conservative
pretrained fine-tuning selected epoch 5 at validation CER
`0.46201693255773774` / WER `0.5983588857698121`; the full 1,273-utterance
Pi 5 + MA2450 evaluation reproduced that closely at CER
`0.46230490122674656` / WER `0.5985748218527316`, with inference-only p50
`392.428065 ms`, p95 `395.7003068 ms`, RTF `0.21930080610559416`, zero
failures and zero persistent-server restarts.

Decoder development is frozen separately from the acoustic graph. The selected
CPU prefix-beam decoder uses beam width 8, token top-k 12, a character 5-gram
LM, LM weight 0.30 and word bonus -0.20. It improves WER to
`0.5497732671129346` while CER is effectively flat at
`0.4634567759027818`.

The Raspberry Pi CPU-only decoder reproof over the same cached physical logits
measured p50 `39.573781999934 ms`, p95 `142.60984940001433 ms`, mean
`55.34163030793814 ms`, and 71.229 seconds wall time for all 1,273 samples.

The final integrated Raspberry Pi deployment reproof is also complete. It used
the OpenVINO 2020.3.2 ARM64 runtime exported from the pinned Docker image and
executed directly on the host, with all 1,273 utterances in one persistent
MYRIAD session and zero restarts. Frozen decoder quality matched exactly at WER
`0.5497732671129346` / CER `0.4634567759027818`.

Integrated compute metrics were MYRIAD p50/p95
`392.424125 / 392.5454856 ms`, decoder p50/p95/mean
`44.00145399995381 / 150.2441755998006 / 60.10071448467321 ms`, and paired
acoustic-plus-decoder p50/p95/mean
`436.4124369999538 / 542.6246731998006 / 452.5247842749317 ms`. The paired
acoustic-plus-decoder realtime factor was `0.25272784122997616`.

Those paired values are authoritative; do not estimate combined p95 by adding
independent acoustic and decoder percentiles. The deployment timing scope is
MYRIAD `Infer()` plus CPU decoder compute only, excluding frontend and IPC.

See
`docs/adr/model-quality-v4-cnn-ctc-v19-conservative-transfer.md` and
`docs/adr/model-quality-v4-cnn-ctc-v19-decoder-v1.md`.

## Pretrained QuartzNet source-domain qualification

The next controlled step is no longer another AMI training or decoder
experiment. The pinned NVIDIA Multidataset QuartzNet15x5Base-En checkpoint is
now qualified first against its known source-domain benchmark, LibriSpeech
`dev-clean`.

This path is intentionally separate from v19. It restores the original
29-class CTC projection and source token order (blank index 28), uses
historical NeMo-style feature framing/normalization, performs zero training,
and compares full-corpus greedy WER with NVIDIA's published `3.79%`
dev-clean result. The initial reproduction gate is WER <= `5%`.

Run:

```bash
./scripts/test-speech-asr-quartznet-reference.sh
./scripts/qualify-quartznet15x5-reference.sh --device cuda
```

The qualification is complete. The full 2,703-utterance dev-clean run reached
WER `0.037939781625675524` (3.793978%) versus NVIDIA's published `0.0379`
(3.79%), with CER `0.012506494000177396`. All 29 source decoder symbols were
loaded, no target-only rows existed, and no training was performed.

The same untouched reconstruction also passed PyTorch/ONNX parity with frame
argmax agreement `1.0`, zero mismatches in 256 tested frames, and maximum
absolute logit error `6.0677528381347656e-05`.

This source-domain reproduction is now qualified. The original QuartzNet
architecture/frontend/import path should not be changed to explain the AMI
error rate; the next REVIEW must focus on the AMI-specific adaptation boundary.
The qualification performed **no OpenVINO conversion or MYRIAD execution**.

LibriSpeech `test-clean` remains available as an independent confirmation
against NVIDIA's published 3.85% result, but dev-clean already establishes the
source-pipeline reproduction.

See
`docs/adr/quartznet15x5-librispeech-reference-qualification.md`.

## AMI adaptation attribution

With the source QuartzNet pipeline qualified, the next measurement isolates the
AMI-specific changes without any new training or hardware execution.

The same frozen 1,273-record ES2011 validation set is evaluated through four
ordered stages:

1. original 29-class source head + original NeMo frontend;
2. original 29-class source head + AMI fixed `logmel-v3` frontend;
3. seeded epoch-zero 39-class AMI head + fixed frontend;
4. existing selected best-epoch-5 v19 checkpoint + fixed frontend.

Only adjacent deltas are interpreted causally: frontend, then head expansion,
then the already-completed fine-tuning. Stage 0 itself measures the untouched
source recognizer's AMI domain gap.

The full run self-checks stage 2 against the frozen epoch-zero WER/CER
`0.7393651479162168 / 0.5776651500316765` and stage 3 against the selected
v19 validation WER/CER
`0.5983588857698121 / 0.46201693255773774`.

Run:

```bash
./ci/verify-static.sh
./scripts/test-speech-asr-quartznet-ami-attribution.sh
./scripts/evaluate-quartznet15x5-ami-attribution.sh --device cuda
```

This path performs no new optimizer updates, OpenVINO conversion, Docker
execution, SSH, or MYRIAD inference.

The full attribution is complete:

- source head + source frontend on AMI: WER `0.7806089397538328`, CER
  `0.6027184242354432`;
- source head + fixed AMI frontend: WER `0.7393651479162168`, CER
  `0.5776651500316765`;
- 39-class AMI head at epoch zero: identical WER/CER
  `0.7393651479162168 / 0.5776651500316765`;
- selected v19 fine-tuning: WER `0.5983588857698121`, CER
  `0.46201693255773774`.

The fixed frontend improves, rather than harms, the source model on AMI. The
39-class head is neutral at epoch zero. Existing fine-tuning is also beneficial
but does not close the large source-to-AMI domain gap. The project therefore
returns to REVIEW with domain/data adaptation as the next problem to solve.

See
`docs/adr/quartznet15x5-ami-adaptation-attribution.md`.

## Clean QuartzNet Pi/MYRIAD qualification

LibriSpeech clean speech is now the primary deployment-quality reference. The
already-qualified 29-class NVIDIA QuartzNet source model is carried unchanged
through ONNX -> OpenVINO 2020.3 FP16 -> Pi 5 -> MA2450/MYRIAD.

The source frontend remains variable-length. A fixed 512-frame ONNX is used
only as a Model Optimizer carrier; the generic `hello_myriad` runner reshapes
the OpenVINO network to each exact source padded time length before
`LoadNetwork`. No utterance is truncated or expanded into a larger inference
bucket.

Before hardware execution, the Pi-safe NumPy implementation of the historical
NeMo frontend must reproduce the full 2,703-utterance clean reference within
`0.001` absolute WER of the qualified PyTorch WER
`0.037939781625675524`.

Physical acceptance requires:

- first-sample ONNX/MYRIAD valid decoded-frame argmax agreement `1.0`;
- maximum valid decoded-frame total-variation distance <= `0.002`;
- full dev-clean WER <= `0.05`;
- absolute WER drift from the qualified PyTorch result <= `0.005`;
- all 2,703 utterances;
- zero training.

First run:

```bash
./scripts/run-quartznet15x5-reference-myriad-edge.sh \
  --device cuda \
  --refresh-runtime
```

`--refresh-runtime` is needed when the Pi's extracted host runtime predates
the new `hello_myriad --reshape-time` support.

See
`docs/adr/quartznet15x5-librispeech-myriad-qualification.md`.

