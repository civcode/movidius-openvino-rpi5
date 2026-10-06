# cnn_ctc_v12

Large-capacity acoustic-model probe for the frozen model-quality-v4
architecture-screen data boundary.

The model intentionally leaves the small-width regime:

- input: `[1,64,512]`;
- two stride-2 Conv1D stems: 128 then 448 channels;
- 16 normalization-free residual temporal blocks;
- residual kernel: 5;
- dilations: `[1,2,3,4,4,3,2,1]` repeated twice;
- output: `[1,128,39]`;
- parameters: 19,627,687;
- fixed-input MACs: 2,515,673,088;
- FP16 weight bytes: 39,255,374;
- receptive field estimate: 653 feature frames.

This is a capacity-boundary experiment, not a latency-optimized deployment
candidate. The 25 ms p95 policy used by the small architecture screen is not
an acceptance threshold here. OpenVINO conversion and physical MYRIAD
execution determine whether the graph is hardware-feasible; measured latency
is evidence for later downsizing if quality improves.

The frontend, character vocabulary and greedy CTC decoder remain unchanged so
the acoustic-quality result is directly comparable to the existing screen
lineage.
