# cnn_ctc_v10

`cnn_ctc_v10` is the third architecture-screen-v1 redesign.

v9 restored healthy 128-frame emission behavior and reduced MA2450 p95 to about
12.09 ms, but missed the v3-screen CER by only ~0.61% relative. v10 therefore
keeps v9's temporal geometry and spends some of that latency headroom on width.

Frozen geometry:

- input: `[1,64,512]`;
- stems: 64 then 112 channels, both stride 2;
- output: `[1,128,39]`;
- residual width: 112 channels;
- eight residual blocks, all kernel 7;
- dilations: `[1,2,3,4,4,3,2,1]`;
- effective receptive field: 493 feature frames;
- parameters: 865,511;
- fixed-input MACs: 113,149,952.

The first experiment uses the exact architecture-screen-v1 data and 12-epoch
budget. Frontend, vocabulary, standard CTC objective, optimizer, checkpoint
selection and greedy decoder remain unchanged.
