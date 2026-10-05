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
