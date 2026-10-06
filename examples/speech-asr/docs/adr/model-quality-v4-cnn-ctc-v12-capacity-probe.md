# ADR: cnn_ctc_v12 large-capacity MYRIAD probe

Status: completed; hardware-feasible; training collapsed
Date: 2026-10-06
Experiment: `exp-4e16fd348004571b/attempt-0001`
Comparison parent: `exp-00e6b1e434d187d8/attempt-0001`

## Result

`cnn_ctc_v12` successfully answered the hardware-capacity question.

- parameters: `19,627,687`;
- fixed-input MACs: `2,515,673,088`;
- MA2450 p50: `103.822604 ms`;
- MA2450 p95: `103.8896558 ms`;
- inference-only RTF: `0.058019043467865204`;
- OpenVINO IR: valid;
- final PyTorch/ONNX frame-argmax agreement: `1.0`;
- pretraining MYRIAD numerical gate: accepted;
- training duration: `2037.7116871379985 s`.

The graph therefore converts and executes physically on the original MA2450.
The 20M-parameter / 2.5B-MAC regime is not beyond the device. Its measured p95
is close to the deliberately exploratory ~100 ms capacity target and remains
substantially faster than realtime by the measured RTF.

Recognition quality from this particular training attempt is unusable:

- CER: `1.0`;
- WER: `1.0`;
- blank-frame fraction: `1.0`;
- emitted/reference characters: `0.0`;
- empty hypotheses: `1273/1273`.

## Training diagnosis

The `3e-3` OneCycle peak was too aggressive for this large normalization-free
CTC network, especially with the peak reached after only 10% of optimizer
steps.

Epoch 1 started at `3e-4` but ended at about `2.82e-3`. Its maximum raw
pre-clip gradient norm reached `402,556.7` and validation was already all
blank. Epoch 2 operated around the `3e-3` peak, reached a maximum raw
pre-clip gradient norm of `36,104,904`, and its mean train loss exploded to
`90.963`.

The optimizer later returned to ordinary finite losses as the learning rate
fell, but the decoder never recovered from the blank attractor. Every
validation epoch remained CER `1.0`, blank fraction `1.0`, and emitted
characters `0`.

This is therefore an optimization-policy failure, not evidence that the
20M-parameter architecture lacks useful capacity.

## Decision

Keep the exact v12 inference architecture as the active capacity regime.
Do not shrink back toward the v9-v11 models and do not enlarge the network
again yet.

The next candidate changes only the training schedule:

- same 19,627,687-parameter architecture;
- same frontend, decoder, manifests and 12-epoch screen budget;
- Adam;
- OneCycle start `3e-4`;
- peak `1.2e-3` (4x the old small-model peak, rather than 10x);
- peak position 30% of optimizer steps instead of 10%;
- final LR `3e-5`;
- gradient clipping remains 5.0;
- checkpoint selection remains validation CER.

This slower ramp gives the large model several epochs to form useful acoustic
representations before the highest learning rate is reached, while still
testing a materially higher LR than the historical `3e-4` ceiling.

The sealed held-out benchmark remains unavailable for selection.
