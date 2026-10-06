# cnn_ctc_v17

`cnn_ctc_v17` keeps the v16 QuartzNet-15x5 inference topology and changes
only the training recipe needed to evaluate that topology fairly.

v16 established two useful facts: the graph converts through ONNX/OpenVINO
2020.3 and executes numerically on MA2450, but the inherited Adam +
dropout-0.2 training setup is unstable. v17 therefore aligns the screen with
the QuartzNet/NeMo training family:

- same 64-bin `[1,64,512]` frontend;
- same QuartzNet-15x5 convolution/channel/kernel schedule;
- same 18,934,631 trainable parameters and `[1,256,39]` output;
- dropout `0.0`;
- NovoGrad, peak LR `0.01`, betas `0.8/0.5`, eps `1e-8`;
- weight decay `0.001`;
- 12% linear warmup followed by cosine decay to `1e-5`;
- gradient clipping `5.0`;
- architecture-screen batch size 1 and 12 epochs;
- checkpoint selection by validation CER.

The deployed VPU graph remains QuartzNet through CTC logits. Greedy CTC remains
the acoustic screen decoder. Prefix beam + LM stays deferred until the acoustic
model produces competitive logits.

The long-corpus evaluator can restart a failed persistent MYRIAD session and
retry the current uncached sample, preventing a transient server death from
discarding hundreds of already cached inferences.
