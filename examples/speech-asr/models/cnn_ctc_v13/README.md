# cnn_ctc_v13

Training-only retry of the physically proven cnn_ctc_v12 large-capacity graph.

Inference architecture is unchanged from v12: 19,627,687 parameters,
2,515,673,088 fixed-input MACs, two stride-2 stems (128 -> 448), sixteen
448-channel kernel-5 residual blocks, and 128 CTC output frames.

v12 established MA2450 feasibility at 103.8896558 ms p95 but its 3e-3
OneCycle peak caused blank collapse. v13 changes only optimization:

- Adam;
- initial LR: `3e-4`;
- peak LR: `1.2e-3`;
- peak position: 30% of optimizer steps;
- final LR: `3e-5`;
- gradient clipping: 5.0;
- screen checkpoint selection: validation CER.

The peak remains 4x above the historical small-model learning rate.
