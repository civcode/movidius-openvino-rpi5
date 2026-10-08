# ADR: fixed-T=512 QuartzNet streaming on MA2450

Status: accepted; full 2,703-utterance physical qualification passed
Date: 2026-10-07

## Context

Whole-utterance qualification preserved the source model's variable time axis
and therefore created 152 exact MYRIAD shape sessions on LibriSpeech dev-clean.
That methodology exposed a catastrophic MA2450 execution boundary between
`T=3088` and `T=3104`, but it is not a production requirement.

The physical residency experiment also proved that at least four QuartzNet
graphs can coexist on one MA2450. Multiple resident shapes are therefore
available for future applications, but they are unnecessary if one fixed
streaming shape preserves ASR quality.

## Decision

Qualify a production-oriented policy using exactly one `[1,64,512]` QuartzNet
graph loaded once and retained for the full evaluation.

The source frontend remains unchanged. Each inference window contains 81,760
audio samples (5.11 s), which produces 511 valid frontend frames plus one zero
pad frame for an exact 512-frame tensor.

Default window starts advance by 128 QuartzNet output frames, or 40,960 audio
samples (2.56 s). Consecutive full windows therefore overlap by 40,800 samples
(2.55 s).

Window starts are aligned to the network's output lattice. Overlapping logits
are stitched before decoding: each global CTC frame is owned by the window
where that frame is closest to the fixed 256-frame window center. Ties retain
the earlier window. The stitched global logits are decoded with one ordinary
greedy CTC collapse, so duplicate symbols spanning a window boundary remain
subject to normal CTC semantics.

The last partial waveform window is passed through the unchanged source
frontend and its feature tensor is zero-padded to `T=512`. Raw audio is not
zero-extended before per-feature normalization.

## Evidence required

For the physical MA2450 run:

- one persistent fixed-`T=512` MYRIAD network;
- one unmeasured warmup;
- first-window ONNX/MYRIAD valid-frame argmax agreement at least `0.99` as a non-catastrophic hardware smoke gate;
- all 2,703 LibriSpeech dev-clean utterances;
- full-corpus WER/CER after logit stitching, which remains the authoritative semantic acceptance gate;
- WER <= `0.05` as the initial fixed-window acceptance ceiling;
- inference RTF includes all overlapping-window inference work;
- model load time is reported separately.

The evaluator also supports `--engine onnx` so the same fixed-window policy can
be run without MYRIAD. This isolates chunking/stitching accuracy from VPU
numerical execution if the physical result regresses.

## 200-utterance physical diagnostic

A 200-utterance LibriSpeech dev-clean diagnostic was run on the Pi 5 + MA2450
using the fixed policy above.

Controller-side preparation on Oberon was constrained independently to CPUs
`0-15` with `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, and
`OPENBLAS_NUM_THREADS=16`. The Pi-side evaluator used its own four-core policy:
CPUs `0-3` with the three thread counts set to `4`.

Measured diagnostic result:

- samples: `200`
- fixed-window inferences: `443`
- network loads: `1`
- model load: `1970.619 ms`
- first-window ONNX/MYRIAD valid-frame argmax agreement: `0.99609375`
- first-window max frame total variation: `0.06952664` (diagnostic only)
- WER: `0.046166529266281946`
- CER: `0.014760597743214395`
- absolute WER delta from the qualified whole-utterance PyTorch source:
  `0.008226747640606422`
- MYRIAD inference p50 / p95: `393.939 / 395.967 ms`
- inference-only RTF including overlap compute: `0.12868995426179905`
- result status: `diagnostic`

The same 200-sample WER/CER was reproduced before and after separating the
Oberon and Pi CPU-affinity policies, so the host CPU-limit correction did not
change recognition semantics.

## Full 2,703-utterance physical qualification

The authoritative LibriSpeech dev-clean qualification completed successfully on
Pi 5 + MA2450 from repository commit
`98ac27e45e69a883de8b7f9ae0a0fe8977e058cf`. Later ROCm/CUDA selector work
does not alter this frozen edge result.

Measured result:

- samples: `2703 / 2703`
- fixed-window inferences: `6429`
- windows per utterance, mean: `2.378468368479467`
- network loads: `1`
- unmeasured warmups: `1`
- model load: `2009.517596 ms`
- first-window ONNX/MYRIAD valid-frame argmax agreement: `0.99609375`
- first-window max frame total variation: `0.0695266401494683`
  (diagnostic only)
- WER: `0.03922649902577111`
- CER: `0.012895078075833026`
- frozen whole-utterance PyTorch reference WER:
  `0.037939781625675524`
- absolute WER delta from whole-utterance reference:
  `0.0012867174000955883`
- fixed-window WER acceptance ceiling: `0.05`
- WER gate: **PASS**
- MYRIAD inference p50: `394.101793 ms`
- MYRIAD inference p95: `396.0079184 ms`
- MYRIAD inference mean: `394.1237425708507 ms`
- overlap-inclusive inference-only RTF: `0.13063547982851126`
- frontend p50 / p95: `4.861740 / 6.517406 ms`
- frontend RTF: `0.0015521516708959379`
- result status: **PASS**

The fixed-window policy therefore increases absolute WER by only about
`0.1287` percentage points relative to the frozen whole-utterance PyTorch
reference while eliminating runtime shape switching and all exposure to the
large-temporal-shape MYRIAD failure regime.

### Deployment consequence

The accepted production architecture is one permanently loaded
`[1,64,512]` QuartzNet network, approximately 5.11-second inference windows,
2.56-second hop, center-owned overlap stitching, and one global greedy CTC
collapse.

Multiple resident networks remain a proven and potentially useful MA2450
capability, but they are not required for this ASR deployment.


## Live runtime

The accepted policy is implemented as an online state machine in
`speech_asr.quartznet_fixed512_live` and exposed by
`scripts/quartznet-fixed512-live.sh`.

The runtime keeps exactly one `T=512` MYRIAD network loaded through the
existing persistent tensor-stream protocol. Incoming audio is retained only
until it is no longer needed by a future overlapping window. CTC frames are
committed only when the next overlapping window can no longer replace them;
uncommitted frames may still be shown as tentative command-line text. Final
decoding therefore preserves the center-owned overlap policy used by the full
2,703-utterance qualification.

The Pi wrapper is pinned independently to CPUs `0-3` with four OMP, MKL, and
OpenBLAS threads.

WAV input:

```bash
./scripts/quartznet-fixed512-live.sh --wav recording.wav --show-timing
```

Add `--wav-realtime` to replay the file at wall-clock speed rather than as
fast as inference permits.

ALSA microphone input:

```bash
./scripts/quartznet-fixed512-live.sh --microphone --show-timing
```

Use `--alsa-device NAME` to select a non-default ALSA source and
`--list-microphones` to print ALSA device names. Microphone capture uses
`arecord` at 16 kHz, mono, signed 16-bit PCM. Ctrl+C stops capture, processes
the remaining partial window, and prints the final transcript.

The live CLI looks for the fixed512 XML/BIN/manifest in the usual
`work/speech-asr/quartznet15x5-reference/myriad/openvino/fp16/` build output.
On the Pi, the existing qualification may instead have staged those files
under `work/speech-asr/fixed512-eval/<run-id>/input/openvino/fp16/`.
If the usual build output is absent, the app automatically reuses the most
recent *complete, shape-validated* staged model. An explicit `--ir-dir PATH`
always takes precedence. If neither exists, prepare the model on Oberon with
`./scripts/prepare-quartznet15x5-reference-myriad.sh` and copy the three
compiled artifacts (`.xml`, `.bin`, `artifacts.json`) to the Pi before running
the WAV/microphone app. The live CLI itself does not compile models.

Quick start on the **Pi 5 with the MA2450 attached**:

```bash
./scripts/quartznet-fixed512-live.sh --list-microphones
./scripts/quartznet-fixed512-live.sh --wav recording.wav --show-timing
./scripts/quartznet-fixed512-live.sh --microphone --alsa-device default --show-timing
```

Microphone enumeration and `--help` do not acquire the MYRIAD device lock.
Actual inference retains the existing single-MYRIAD, four-core Pi policy; it
does not use the Oberon ROCm training launcher. WAV loading automatically
converts supported integer-PCM WAV formats to 16 kHz mono. A missing,
inaccessible, or busy ALSA device reports an explicit capture error rather
than silently returning an empty transcript. PCM framing is preserved even
if the capture pipe splits a 16-bit sample between reads.

Hardware-free I/O tests:

```bash
./scripts/python-apps.sh -m unittest discover -s tests/python \
  -p 'test_speech_asr_quartznet_live_io.py' -v
```

The accepted fixed512 inference and stitching implementation remains
unchanged.

The terminal emits only the new **committed** portion of the transcript,
without repeatedly repainting the growing full line. This prevents older
text from being duplicated when it wraps across terminal rows, and makes
stdout usable as a continuous stream for downstream consumers. Tentative
overlap-window previews are not printed because they can change; finalization
prints any remaining decoded text and ends with a newline. A long recording
can thus produce multiple screen rows but every confirmed character is sent
only once. `--show-timing` still prints diagnostics on stderr.

For microphone listening, the live terminal can *display* a space after
a long CTC blank interval (600 ms by default). The original CTC transcript,
logit stitching, MYRIAD model, and qualified WAV-mode behavior are unchanged.
This is a heuristic for speech pauses, not an ASR language model; it cannot
correct all word boundaries. Tune or disable it on the Pi:

```bash
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" --pause-space-ms 400
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" --pause-space-ms 0
```

The pause separator can be any **non-empty printable string**, not just the
default space. Use `--pause-delimiter TEXT` to insert a literal marker whenever
the microphone detects a speech pause long enough for `--pause-space-ms`.
Shell-quote markers containing punctuation, spaces, or special characters:

```bash
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --pause-delimiter '[ref_delimiter]'

./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --pause-delimiter ' to stop
silent-room microphone noise being decoded as stray vowels such as
`e o e o`. It measures audio energy in 20 ms blocks, requires several active
blocks to open, and allows 200 ms of surrounding audio for soft consonants.
Below-threshold regions are forced to the CTC blank token *after* MYRIAD
inference. Source WAV evaluation, the trained model, and the fixed512 overlap
policy are unchanged. This is a simple noise gate, not a trained speech VAD.

For the Marantz MPM-2000U on edge:

```bash
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --hop-output-frames 64
```

Use `--squelch-dbfs -35` for stronger suppression if background noise still
produces characters, or `--squelch-dbfs -50` to keep quieter speech. A
higher (less negative) threshold suppresses more audio; if set too high,
soft speech may be lost. `--no-squelch` disables the gate entirely.
`--show-timing` remains optional. The gate cannot distinguish a loud noise
from speech, so the Pi microphone run is the decisive real-world test.

The first model inference still needs the first **5.11 seconds** of audio;
the qualified MA2450 inference time is approximately **394 ms** per window.
The live decoder now commits all frames guaranteed to retain their
center-owned logit even after the next overlapping window arrives. This
reduces the delay of confirmed text while preserving final offline parity:

- 64-frame hop: updates every 1.28 seconds; first complete window safely
  commits 160 output frames (3.20 seconds of audio), rather than just 64.
- 128-frame hop (qualified configuration): updates every 2.56 seconds;
  first complete window safely commits 192 output frames (3.84 seconds),
  rather than just 128.

Use `--show-preview` to see a **short, tentative** suffix of the current
window as `[preview] ...` on stderr. Preview text may be revised by later
windows and is not intended for downstream parsing. The normal stdout
stream remains append-only committed text, with no repeated full sentences.

```bash
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --hop-output-frames 64 --show-preview
```

Neither the earlier confirmation boundary nor the preview option reduces the
initial 5.11-second window requirement. Achieving first speech output sooner
would need a separate policy for short/padded windows and accuracy checks.

## Commands

Local ONNX policy diagnostic:

```bash
./scripts/evaluate-quartznet15x5-reference-fixed512.sh \
  --engine onnx \
  --max-samples 100
```

Physical Pi 5 + MA2450 run from the controller host:

```bash
./scripts/run-quartznet15x5-reference-fixed512-edge.sh
```

Use `--max-samples N` for a non-authoritative diagnostic. The full 2,703
dev-clean qualification above is the frozen acceptance result.


./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --pause-delimiter '#'
```

For example, two phrases separated by a detected pause display as
`first phrase[ref_delimiter]second phrase`. The chosen marker survives
text normalization verbatim. Normal model-predicted spaces *within* speech
are not replaced; the marker represents only detected pauses. The original
CTC transcript used for model parity remains unchanged. The
`--pause-space-ms 0` option disables generated delimiters irrespective
of `--pause-delimiter`.

Microphone-only squelch is enabled by default at **-40 dBFS** to stop
silent-room microphone noise being decoded as stray vowels such as
`e o e o`. It measures audio energy in 20 ms blocks, requires several active
blocks to open, and allows 200 ms of surrounding audio for soft consonants.
Below-threshold regions are forced to the CTC blank token *after* MYRIAD
inference. Source WAV evaluation, the trained model, and the fixed512 overlap
policy are unchanged. This is a simple noise gate, not a trained speech VAD.

For the Marantz MPM-2000U on edge:

```bash
./scripts/quartznet-fixed512-live.sh --microphone \
  --alsa-device "plughw:CARD=MICROPHONE,DEV=0" \
  --hop-output-frames 64
```

Use `--squelch-dbfs -35` for stronger suppression if background noise still
produces characters, or `--squelch-dbfs -50` to keep quieter speech. A
higher (less negative) threshold suppresses more audio; if set too high,
soft speech may be lost. `--no-squelch` disables the gate entirely.
`--show-timing` remains optional. The gate cannot distinguish a loud noise
from speech, so the Pi microphone run is the decisive real-world test.

With the accepted 128-output-frame hop, the first partial text requires the
initial 5.11-second inference window; subsequent updates occur every 2.56
seconds of incoming audio. The qualified MA2450 inference time is about 394 ms
per fixed window.

## Commands

Local ONNX policy diagnostic:

```bash
./scripts/evaluate-quartznet15x5-reference-fixed512.sh \
  --engine onnx \
  --max-samples 100
```

Physical Pi 5 + MA2450 run from the controller host:

```bash
./scripts/run-quartznet15x5-reference-fixed512-edge.sh
```

Use `--max-samples N` for a non-authoritative diagnostic. The full 2,703
dev-clean qualification above is the frozen acceptance result.
