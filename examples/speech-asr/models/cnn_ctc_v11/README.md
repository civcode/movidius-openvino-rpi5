# cnn_ctc_v11

`cnn_ctc_v11` is the fourth architecture-screen-v1 redesign.

v10 proved that the v9 deep-dilated temporal geometry benefits from additional
capacity: widening 96 -> 112 channels improved physical CER from
`0.8032598053331798` to `0.7888613718827392` while MA2450 p95 remained
`13.6566998 ms`. v10 passed the v3-screen CER gate but did not reach the
stronger promotion threshold `0.7744692737430167`.

v11 keeps the complete v10 temporal geometry and changes only the second stem
and residual width from 112 to 128 channels.

Frozen graph contract:

- input: `[1,64,512]`;
- stems: 64 then 128 channels, both stride 2;
- eight 128-channel kernel-7 residual blocks;
- dilations: `[1,2,3,4,4,3,2,1]`;
- output: `[1,128,39]`;
- receptive field: 493 feature frames;
- parameters: 1,117,287;
- fixed-input MACs: 145,342,464.

The frontend, vocabulary, standard CTC objective, greedy decoder, optimizer,
architecture-screen-v1 manifest, full ES2011 validation set and 12-epoch screen
budget remain unchanged.
