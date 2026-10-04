# rm_cnn4a development fixture

`rm_cnn4a_smbr` is the initial known acoustic network used to develop the
OpenVINO/MYRIAD inference and measurement path.

## Important boundary

This public Intel package is **not a raw-audio end-to-end ASR model**. Its
development fixture is:

```text
Kaldi feature ARK -> rm_cnn4a acoustic network -> acoustic score ARK
```

The package ships `feat1_10.ark` and `score1_10.ark`, which make it useful
for deterministic CPU/MYRIAD numerical regression. It does not ship the full
raw-audio feature-transform/decoder bundle required to turn arbitrary AMI WAV
clips into text.

Full Kaldi WER decoding requires additional resources such as `HCLG.fst`,
`final.mdl`, and `words.txt`. Those are outside the first
hardware-qualification path.

Therefore:

- **AMI remains the canonical end-to-end dataset** for future custom models;
- **rm_cnn4a is the device/conversion regression fixture**;
- no guessed MFCC/log-mel frontend is attached to rm_cnn4a;
- vendor score error, not AMI WER, is the first acceptance metric for this model.

## Source and conversion

The source descriptor is `source-v1.json`. It pins the Intel
`models_contrib/speech/2021.2` release and the exact expected filenames.

Prepare it with:

```bash
./scripts/prepare-rm-cnn4a.sh
```

Intel's storage index does not publish SHA-256 values, so first acquisition
creates `vendor/models/rm_cnn4a_smbr/source/SOURCE-LOCK.sha256`; every
subsequent run verifies that content lock. The generated model spec records all
source and IR hashes.

Conversion uses the repository's pinned OpenVINO 2020.3.2 Model Optimizer with
the Kaldi frontend, removes the output softmax, and emits FP16 IR. The script
writes `ir-contract.json` by inspecting the actual generated IR instead of
hard-coding tensor names or dimensions.

## Historical MYRIAD constraint

Intel's published NCS2 test used the FP16 IR and batch size 1. That report also
showed substantial score divergence versus other targets, so numerical
comparison against `score1_10.ark` is a required metric.

## Non-goals

The eventual custom model does not inherit rm_cnn4a's frontend, acoustic
targets, decoder, topology, or training framework. The custom path remains
PyTorch/CUDA -> ONNX -> OpenVINO -> MYRIAD.


## Runtime regression

After rebuilding the runtime image with this checkpoint:

```bash
./build.sh --platform arm64
./run.sh --platform arm64 speech-regress
```

The Docker build includes the pinned OpenVINO 2020.3 `speech_sample` only; it
does not enable the full sample suite. The wrapper forces `MYRIAD`, FP16 IR and
batch size 1, and compares generated acoustic scores with the vendor
`score1_10.ark` reference.


## CPU reference regression

Phase 4 now has a hardware-independent OpenVINO reference path on amd64. It
uses the exact FP16 IR plus `feat1_10.ark` and `score1_10.ark` that the
MYRIAD path consumes:

```bash
./build.sh --platform amd64
./scripts/python.sh examples/speech-asr/evaluation/benchmark_rm_cnn4a.py \
    --backend cpu
```

The result is written to
`work/speech-asr/rm_cnn4a/result-cpu-reference.json`. The same evaluator can
later run the device path with `--backend myriad --platform arm64`.

Prepared bundles can be checked without network, Docker, or hardware:

```bash
./scripts/prepare-rm-cnn4a.sh --verify-only
```

That verification re-hashes the source lock and source artifacts, re-inspects
the generated IR, and verifies that `model-spec.json`, `ir-contract.json`,
the XML and the BIN all agree. The generated model spec also records the source
lock and license hashes so a completed reference run can be frozen from actual
evidence.


## Frozen Phase 4 CPU reference

The first completed amd64 CPU reference run is frozen in
`cpu-reference-v1.json`. It uses OpenVINO 2020.3.2, the FP16 IR identified by
XML SHA-256 `53a22c26746eeedf053864385606d64b20ce6c4020732a2a55f8ce2609152f27`
and BIN SHA-256
`a27a0f9ebf4a235e6789d746940ba07941c236b5930efdafdf8ffd771be79195`.

Observed reference results:

- 10 utterances / 3401 frames;
- 0 failures;
- weighted mean inference: 2.021289091443693 ms/frame;
- utterance p50: 2.019845 ms/frame;
- utterance p95: 2.0324755000000003 ms/frame;
- maximum absolute score error: 0.00880814;
- mean average error: 0.00137728;
- mean RMS error: 0.001708947;
- model load time: 150.739 ms.

These are evidence from the frozen run, not acceptance thresholds for MYRIAD.
Phase 5 should compare the device result against this exact artifact/fixture
identity and report the measured deltas.


## Runtime platform versus Movidius device

The `--platform` argument selects the architecture of the host/runtime Docker
image, **not** the architecture of the Movidius stick. For example, a stick
plugged into an x86_64 workstation should use `--platform amd64`; the same
stick plugged into a Raspberry Pi 5 running 64-bit userspace should use
`--platform arm64`.

The requested runtime image must already exist locally. If it does not,
`run.sh` now stops before Docker attempts a registry pull and prints the exact
`./build.sh --platform ...` command required.


## Frozen Phase 5 amd64 MYRIAD result

The first completed physical MYRIAD regression is frozen in
`myriad-amd64-v1.json`. It uses the same OpenVINO 2020.3.2 FP16 IR and vendor
feature/reference-score ARKs as `cpu-reference-v1.json`.

Observed device results on the amd64 host:

- 10 utterances / 3401 frames;
- 0 failures;
- weighted mean inference: 21.67994733901794 ms/frame;
- utterance p50: 21.6799 ms/frame;
- utterance p95: 21.792255 ms/frame;
- maximum absolute score error: 0.0915527;
- mean average error: 0.006222882000000001;
- mean RMS error: 0.007926627000000002;
- model load time: 1825.88 ms.

`cpu-vs-myriad-amd64-v1.json` records the deterministic evidence-only
comparison. On this host the MYRIAD weighted per-frame inference time was
10.725802375717159x the CPU reference, while mean average error and mean RMS
error were 4.51824029972119x and 4.638310608813499x the CPU reference,
respectively. These ratios are measurements, not pass/fail thresholds.

This validates the physical MYRIAD execution path on an amd64 host. Raspberry
Pi 5 / arm64 host validation remains a separate deployment gate.


## Frozen Raspberry Pi 5 / arm64 MYRIAD result

The first completed Pi 5 physical-device regression is frozen in
`myriad-arm64-pi5-v1.json`.

Observed Pi result:

- 10 utterances / 3401 frames;
- 0 failures;
- weighted mean inference: 21.31601402528668 ms/frame;
- utterance p50: 21.31425 ms/frame;
- utterance p95: 21.323745 ms/frame;
- maximum absolute score error: 0.0915527;
- mean average error: 0.006222882000000001;
- mean RMS error: 0.007926627000000002;
- model load time: 1972.68 ms.

The numerical error metrics are the same as the frozen amd64 MYRIAD result.
Steady-state MYRIAD inference on the Pi was about 1.68% faster by weighted
per-frame latency (21.3160 vs 21.6799 ms/frame), while model loading was about
8.04% slower.

The prepared BIN and both vendor ARK hashes match the frozen amd64 result.
The raw XML SHA-256 does not: the Pi-prepared XML is
`5fbc7dc2327a22d75bcd4b27d98f73689fdaf047fc16ee57be3ea507670d4d35`,
while the earlier amd64-prepared XML is
`53a22c26746eeedf053864385606d64b20ce6c4020732a2a55f8ce2609152f27`.
Cross-host diagnostics confirmed that both prepared IRs contain the same 31
layers, 30 edges, BIN payload, network attributes and per-layer executable
section hashes. The only difference is nondeterministic numeric suffixes on two
Model Optimizer-generated Const names matching `Cast_<number>_const`; the
suffix assignments are swapped between hosts while the Const output/blob
sections and consumer ports are identical. The canonical graph fingerprint
therefore normalizes only that generated suffix and disambiguates each such
Const by its consumer edge. This closes the Phase 5 IR-identity gate.


## Frozen Phase 6 Pi benchmark worker result

`phase6-pi5-benchmark-v1.json` freezes the first accepted repeated benchmark
worker run on Raspberry Pi 5 / arm64. The benchmark-v1 policy executed one
excluded warmup followed by five measured independent full-fixture runs.

Observed repeatability:

- 5/5 measured runs completed;
- 0 inference failures;
- weighted inference mean: 21.313979870626287 ms/frame;
- weighted inference min/max: 21.312113701852397 / 21.316284092913847;
- weighted inference relative range: 0.019566458665927117%;
- weighted inference population CV: 0.007851929831317018%;
- model-load mean: 1996.562 ms;
- model-load relative range: 1.1199251513351402%;
- max/average/RMS vendor score-error metrics had exactly zero cross-run
  variation.

The observed result was generated at repository commit `1f0ce936`. Its
`canonical_graph_sha256` value was produced before fixing an escaping bug in
the autogenerated `Cast_<number>_const` matcher. That fingerprint value is
therefore historical provenance rather than the canonical cross-host identity.
The raw XML/BIN and fixture hashes, run logs, timing and numerical measurements
remain valid and unchanged.
