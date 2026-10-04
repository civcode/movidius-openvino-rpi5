# ADR-0001: Keep speech ASR in the toolkit repository

## Status

Accepted.

## Decision

The speech-to-text development project initially lives in the existing
`movidius-openvino-rpi5` repository under:

```text
examples/speech-asr/
```

It is a self-contained subproject with explicit interfaces to the existing
OpenVINO 2020.3.2/MYRIAD runtime.

## Rationale

The speech project directly depends on the repository's:

- OpenVINO 2020.3.2 build
- MYRIAD plugin
- Movidius firmware
- Raspberry Pi runtime
- container/build environment
- model conversion and device-access infrastructure

Keeping the work together avoids duplicated platform plumbing and prevents
version drift during early development.

At the same time, speech-specific dataset, training, evaluation and agent
infrastructure is kept below `examples/speech-asr/` so it can be extracted
later.

## Boundary

The existing repository owns generic Movidius/OpenVINO platform behavior.

`examples/speech-asr/` owns speech-specific:

- corpus preparation;
- audio/feature/model contracts;
- model adapters;
- decoding;
- WER/CER/timing evaluation;
- experiment definitions;
- training/export workflow;
- agent-driven development workflow.

Speech-specific assumptions must not be added to generic runtime components
unless they are generalized first.

## Future extraction criteria

A separate repository becomes appropriate when one or more are true:

1. training/architecture development is substantially independent of this
   Movidius toolkit;
2. the same training/evaluation system targets multiple deployment platforms;
3. experiment/training infrastructure dominates the base toolkit;
4. speech develops an independent release lifecycle;
5. other projects need to consume the speech infrastructure independently.

## Initial model

`rm_cnn4a` is the initial known development model used to bring up the speech
inference, measurement and agent workflow.

It is a development fixture, not the long-term architecture.

## Consequences

- platform and speech code evolve together during bring-up;
- experiments can pin one repository revision for host/runtime behavior;
- the subproject must maintain clean interfaces to make later extraction
  straightforward.
