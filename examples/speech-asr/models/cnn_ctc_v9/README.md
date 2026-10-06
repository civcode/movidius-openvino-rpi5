# cnn_ctc_v9

`cnn_ctc_v9` is the second architecture-screen-v1 redesign.

v8 showed that doubling the CTC time axis to 256 frames worsened under-emission
and pushed MA2450 p95 above the 25 ms screen ceiling. v9 therefore restores the
proven v3 tensor rate and width while changing temporal modeling depth.

Frozen geometry:

- input: `[1,64,512]`;
- stems: 64 then 96 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 96 channels;
- eight residual blocks, all kernel 7;
- dilations: `[1,2,3,4,4,3,2,1]`;
- effective receptive field: 493 feature frames;
- parameters: 646,503;
- fixed-input MACs: 85,151,744.

The first experiment uses the exact architecture-screen-v1 data and 12-epoch
budget. Frontend, vocabulary, standard CTC objective, optimizer, checkpoint
selection and greedy decoder remain unchanged.
