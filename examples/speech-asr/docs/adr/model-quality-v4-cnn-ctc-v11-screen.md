# ADR: cnn_ctc_v11 128-channel deep-dilated architecture screen

Status: completed; screen accepted; promotion threshold not met
Date: 2026-10-06
Experiment: `exp-4fb0f93165d0f76d/attempt-0001`
Parent: `exp-00e6b1e434d187d8/attempt-0001`

## Result

`cnn_ctc_v11` completed successfully on the frozen architecture screen.

- physical CER: `0.7865576225306686`;
- validation CER at the selected epoch: `0.7865000287968669`;
- physical WER: `1.0205139278773483`;
- emitted/reference characters: `0.4790646777630594`;
- empty-hypothesis fraction: `0.22152395915161036`;
- blank-frame fraction: `0.757843586041762`;
- MA2450 p50: `14.921604 ms`;
- MA2450 p95: `14.9809824 ms`;
- inference-only RTF: `0.008344295021244756`;
- final PyTorch/ONNX frame-argmax agreement: `1.0`;
- pretraining MYRIAD numerical gate: accepted;
- training duration: `694.0145067159974 s`;
- selected checkpoint: epoch 12;
- parameters: `1,117,287`;
- fixed-input MACs: `145,342,464`.

The candidate improves the frozen v3-screen CER from
`0.7984219316938317` to `0.7865576225306686`, about 1.49% relative, and
passes the screen acceptance gate. It does not reach the stronger promotion
CER `0.7744692737430167`.

Relative to v10, the 112 -> 128 width step improves CER only from
`0.7888613718827392` to `0.7865576225306686`, while both candidates remain
far from useful transcription quality. The width sweep therefore has enough
evidence to stop: further small width increments are not justified.

## Interpretation

The important result is not the small CER gain. The physical MA2450 continues
to execute the graph with very large throughput margin. The project had been
treating the inherited 25 ms p95 screen gate as though it were a deployment
limit, but that threshold was an internal conservative policy derived after
earlier models already ran comfortably below 20 ms. It is not a speech-frame
deadline or a demonstrated hardware ceiling.

The current fixed input contains 512 log-mel frames with a 10 ms hop, about
5.14 seconds of audio. A model taking substantially more than 25 ms to process
that chunk can still be many times faster than realtime.

## Decision

v11 closes the small width-only architecture-screen sequence.

The next experiment will deliberately move into a much larger capacity regime
rather than testing 144/160-channel variants. It will:

- keep the frozen data and decoder so acoustic-model quality remains comparable;
- retain ordinary Conv1D/ReLU/residual operators already proven on OpenVINO
  2020.3/MYRIAD;
- increase both width and depth by roughly an order of magnitude in total model
  capacity;
- measure latency without using 25 ms as a rejection gate;
- treat OpenVINO conversion and physical MYRIAD execution as the hardware
  feasibility test;
- keep sealed held-out evidence unavailable for model selection.

If the large graph fails to convert or execute, that failure becomes useful
capacity-boundary evidence. If it executes, recognition quality determines
whether the larger regime is worth refining.
