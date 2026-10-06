# cnn_ctc_v15

BatchNorm-conditioning control for the physically proven 20M large-capacity graph.

Relative to cnn_ctc_v14, v15 keeps the complete temporal geometry and training
schedule unchanged:

- input: `[1,64,512]`;
- stems: 128 then 448 channels, both stride 2;
- 16 448-channel kernel-5 residual blocks;
- dilations: `[1,2,3,4,4,3,2,1]` repeated twice;
- output: `[1,128,39]`;
- fixed-input convolution MACs: 2,515,673,088;
- Adam cosine schedule from `3e-4` to `3e-5`;
- gradient clipping: 5.0;
- architecture-screen checkpoint selection: validation CER.

The single controlled change is BatchNorm after both stem convolutions and after
the temporal and projection convolutions in every residual branch. Convolutions
covered by BatchNorm omit their bias terms. Because BatchNorm would cancel the
old 0.01 projection-weight scale during training, the residual near-identity
scale is applied as a 0.01 gamma on each block's final BatchNorm instead. The
final CTC projection remains bias-bearing.

The deterministic resource estimate is 19,642,599 trainable parameters. The
receptive field and fixed output geometry remain unchanged.

v14 proved that the large graph can train without blank collapse under the
stable cosine schedule, but its best validation CER was 0.7898404653573691 at
epoch 8 and validation loss diverged while training loss kept falling. v15 tests
whether normalization improves optimization conditioning/generalization without
changing width, depth, receptive field, data, decoder, or learning-rate policy.

BatchNorm has already executed through this project's OpenVINO 2020.3/MYRIAD
path in cnn_ctc_v2. The normal pretraining compatibility gate remains mandatory
before full training.
