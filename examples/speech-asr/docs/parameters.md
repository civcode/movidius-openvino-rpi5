# Parameter space

The project divides configuration into **fixed invariants**, **tunable
parameters**, and **derived values**. A value must belong to exactly one of
those classes for a given experiment family.

This prevents contradictory configuration and makes experiments reproducible.

## Fixed invariants

These are outside normal architecture optimization.

| Area | Parameter | Contract |
|---|---|---|
| audio | sample rate | 16 kHz |
| audio | channels | mono |
| audio | internal representation | float32 |
| timing | canonical coordinate | integer sample index |
| dataset | corpus/split manifests | versioned and immutable per benchmark version |
| scoring | text normalization | versioned and immutable per benchmark version |
| scoring | WER/CER implementation | deterministic |
| deployment | hardware target | recorded for every hardware result |
| deployment | OpenVINO/MYRIAD runtime | pinned by this repository |
| deployment | MYRIAD precision | FP16 baseline |
| measurement | warmup/iteration/statistics rules | versioned benchmark contract |

Source WAV bit depth is not a model parameter. Input is converted to the
canonical internal audio representation before processing.

## Frontend tunables

For custom models these are part of the search space and are recorded in the
model/experiment profile:

| Parameter | Initial search region |
|---|---|
| analysis window | about 20-40 ms |
| frame hop | about 5-20 ms |
| FFT size | 256 / 512 / 1024 candidates |
| mel bins | about 32-128 |
| low/high frequency cutoff | explicit per profile |
| feature normalization | none / global / utterance, as declared |
| feature type | log-mel is the initial custom-model family |

The `rm_cnn4a` bring-up model is an exception: its compatible frontend is a
model-specific frozen profile once identified. We do not alter its feature
geometry merely to match the future custom-model search space.

## Structural model tunables

These require a new model/export and normally a new training run:

- block count
- channel widths
- convolution kernel sizes
- dilation
- temporal stride/downsampling
- normalization type
- activation
- residual topology
- receptive field
- input feature count
- fixed input frame count where required by MYRIAD
- output vocabulary/target representation for a model family

The agent may propose these only during DESIGN.

## Runtime/streaming tunables

These normally do not require changing trained weights, although some may be
constrained by a model's fixed input shape:

- inference chunk duration
- chunk overlap
- left context
- right/lookahead context
- VAD threshold
- minimum speech duration
- end-of-speech timeout
- decoder beam width
- language-model weight
- word insertion penalty
- token/word stabilization policy

## Derived values

Derived values are computed, never configured independently.

At 16 kHz:

```text
samples_per_ms = 16
window_samples = window_ms * 16
hop_samples    = hop_ms * 16
chunk_samples  = chunk_ms * 16
```

Other derived values include:

- frames per second
- frames per chunk
- overlap frames
- tensor dimensions
- model parameter count
- effective receptive field
- algorithmic minimum latency

A configuration must not contain both an authoritative value in milliseconds
and another authoritative value in samples for the same quantity.

## Optimization order

Do not begin with a full Cartesian search. Optimize in stages:

1. establish a model-compatible frontend baseline;
2. optimize custom-model frontend geometry;
3. optimize model efficiency/capacity;
4. optimize streaming/chunking;
5. optimize decoder/language-model settings.

Each stage keeps the other layers fixed enough that gains remain attributable.

## Metrics

No single metric defines "best". The project records at least:

- WER
- CER
- model conversion/load success
- real-time factor (RTF)
- inference latency p50/p95
- end-to-end/word stabilization latency when available
- model size
- device failures/timeouts
- numerical divergence against a reference backend when measured

Architecture review should reason over a Pareto frontier rather than optimize
WER while ignoring real-time operation.
