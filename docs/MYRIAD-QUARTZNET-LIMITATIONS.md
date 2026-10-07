# Intel Movidius Compute Stick limitations for QuartzNet ASR

Measured behavior of the MA2450/MYRIAD path on Raspberry Pi 5 using OpenVINO 2020.3.2.

**Evidence-based technical note - 7 October 2026**

## 1. Scope and tested configuration

This document describes limitations observed in the specific tested deployment path. It should not be read as a universal specification for every Movidius device, model, firmware revision, or OpenVINO release.

| Component | Tested configuration |
|---|---|
| Accelerator | Intel Movidius MA2450 / MYRIAD Compute Stick |
| Host | Raspberry Pi 5, native ARM64 OpenVINO runtime for physical MYRIAD evaluation |
| Runtime | OpenVINO 2020.3.2, FP16 IR, MYRIAD plugin |
| Model | NVIDIA QuartzNet15x5Base-En source recognizer, original 29-class CTC head |
| Frontend | 16 kHz NeMo-compatible filterbank frontend; feature time padded to multiples of 16 |
| Primary corpus | LibriSpeech dev-clean, 2,703 utterances |
| Reference quality | Qualified source WER 3.79398%; NumPy frontend reproduction WER 3.79030% |

## 2. Confirmed limitations

### 2.1 Catastrophic temporal-shape boundary

The strongest confirmed limitation is a discrete failure in the MYRIAD execution path for large QuartzNet temporal tensors. The boundary was isolated by holding one speech sample fixed, with a natural tensor length of `T=592`, and changing only the zero-extended static tensor length.

| Static input `T` | Valid output frames | Argmax agreement | Mismatches | Max frame TV | Max raw-logit error |
|---:|---:|---:|---:|---:|---:|
| 3056 | 293 | 1.000000 | 0 | 0.300783 | 10.963 |
| 3072 | 293 | 0.996587 | 1 | 0.777316 | 12.619 |
| 3088 | 293 | 0.996587 | 1 | 0.855861 | 12.994 |
| 3104 | 293 | 0.000000 | 293 | 1.000000 | 630.521 |
| 3120 | 293 | 0.000000 | 293 | 1.000000 | 630.521 |
| 3152 | 293 | 0.000000 | 293 | 1.000000 | 630.521 |
| 3264 | 293 | 0.000000 | 293 | 1.000000 | 630.521 |

For this exact QuartzNet/OpenVINO 2020.3.2/MA2450 combination, `T=3088` is the last tested non-catastrophic shape and `T=3104` is the first tested catastrophic shape. Because the frontend hop is 10 ms, these tensor lengths correspond to roughly 30.9-31.0 seconds of padded feature time.

This is an observed stack/model boundary, not a published universal MA2450 hardware specification.

### 2.2 The failure is in the MYRIAD path, not the model or IR

At `T=3264`, an exact static ONNX model was generated and converted directly to an OpenVINO FP16 IR. Runtime reshape was not used. Dynamic ONNX and static ONNX were bit-identical, eliminating ONNX shape specialization as the cause.

| Execution path at `T=3264` | Valid-frame agreement | Mismatches | Max abs error | Max frame TV | Observed output range |
|---|---:|---:|---:|---:|---|
| Dynamic ONNX vs static ONNX | 1.000000 | 0 | 0.0 | 0.0 | Reference-equivalent |
| ONNX vs OpenVINO CPU | 0.999384 | 1 / 1624 | 0.394 | 0.01578 | about -7.97 to +41.07 |
| ONNX vs MYRIAD | 0.000000 | 1624 / 1624 | 632.720 | 1.0 | about -0.44 to +626.5 |

The CPU result is essentially correct while the same IR is completely wrong on MYRIAD. This exonerates the source model, frontend, ONNX export, Model Optimizer, static FP16 IR, and runtime reshape from the catastrophic failure.

The remaining fault domain is the MYRIAD compiler/runtime/device execution path at large temporal dimensions.

### 2.3 Significant numerical drift can exist even below the cliff

The MYRIAD output can show substantial probability-space drift while still preserving greedy CTC decisions. At ordinary shapes the first physical sample achieved 100% valid-frame argmax agreement even though maximum frame total variation was about `0.059`.

Near the large-shape boundary, maximum frame total variation can exceed `0.8` while only one of 293 valid argmax frames changes.

For this deployment, raw-logit or probability closeness is therefore not a reliable acceptance criterion by itself. Argmax parity and full-corpus WER/CER are more meaningful semantic checks. Rising drift near the boundary is still a warning sign and argues for operating with substantial margin.

### 2.4 Variable-length exact-shape execution has high load overhead

LibriSpeech dev-clean produced 152 distinct padded feature lengths. The correctness-first qualification loaded one exact MYRIAD graph per shape. This is scientifically conservative, but it is a poor production strategy on this runtime.

| Metric | Measured value |
|---|---:|
| Distinct shape sessions | 152 |
| Mean MYRIAD model load/compile | 1,963 ms per shape |
| Minimum / maximum load | 1,931 ms / 2,026 ms |
| Total shape-load time | 298.4 s |
| Steady-state inference median | 611.9 ms |
| Steady-state inference p95 | 4,023.1 ms |
| Steady-state inference mean | 1,317.0 ms |
| Inference-only realtime factor | 0.1835, about 5.45x faster than real time overall |
| Pi NumPy frontend median / p95 | 7.80 ms / 17.69 ms |

Repeated `LoadNetwork()`/compile operations dominate wall-clock time when many exact shapes are used. The Pi CPU remains lightly loaded because it spends much of the time waiting on the accelerator or graph-load path; low host CPU use does not imply unused MYRIAD compute capacity.

### 2.5 Long-shape latency is irregular after corruption begins

The long-shape tail shows non-physical timing behavior alongside numerical corruption. For example, a failing `T=3264` static graph completed in roughly 515 ms while producing completely wrong output, whereas a `T=2944` corpus sample took about 55 seconds.

A catastrophically fast result at a failing shape must not be interpreted as performance improvement. It is evidence that the intended computation is no longer being executed correctly.

## 3. Impact on measured ASR quality

The full 2,703-utterance physical run initially appeared to miss the clean-speech quality gate, with WER 5.1763% versus a 5% ceiling. Length analysis showed that this aggregate result was dominated by a tiny set of execution failures rather than broad recognition degradation.

| Subset | Samples / words | WER | Interpretation |
|---|---:|---:|---|
| All dev-clean | 2703 / 54,402 | 5.176% | Aggregate includes catastrophic device executions |
| Nine catastrophic long-shape utterances | 9 / 767 | 100.0% | 767 errors on 767 reference words |
| All remaining utterances | 2694 / 53,635 | about 3.820% | Returns close to qualified clean-reference quality |
| Long speech `>2048` but below catastrophic cases | 46 / 2,893 | about 3.77% | Long clean speech itself remains healthy below the failure region |
| Qualified PyTorch source reference | 2703 / 54,402 | 3.794% | Reference implementation |

Therefore, the observed long-utterance quality collapse is primarily a hardware/runtime execution artifact. It should not be used as evidence that QuartzNet intrinsically fails on long clean speech.

## 4. Practical operating envelope

- Do not submit QuartzNet tensors at or above `T=3104` to this MA2450/OpenVINO 2020.3.2 path. The tested behavior is catastrophically incorrect.
- Do not treat `T=3088` as a recommended production maximum. It is merely the last tested non-catastrophic point and already shows substantial numerical drift.
- Use a materially smaller operating ceiling. `T<=2048`, roughly 20.5 seconds of padded feature time, is a conservative upper bound supported by corpus-wide quality evidence, not an optimal streaming window.
- For interactive ASR, shorter fixed windows or streaming chunks are preferable because latency scales strongly with temporal length.
- Reject, split, or route oversized inputs before MYRIAD inference. A runtime guard should fail closed rather than silently returning corrupted logits.

## 5. Deployment implications and recommended design

### 5.1 Prefer a small set of validated static shapes or streaming windows

The qualification used 152 exact shapes only to preserve source semantics. A production deployment should avoid this. Candidate static shapes such as `512`, `1024`, `1536`, and `2048` would dramatically reduce load churn while staying well below the catastrophic boundary.

Padding an utterance into a larger bucket is **not yet proven semantically neutral** for QuartzNet. Temporal convolutions and boundary effects can change valid logits. Any bucket policy must be qualified against exact-shape hypotheses and WER before adoption. Fixed-window streaming likewise requires its own accuracy validation.

### 5.2 Multiple executable networks can coexist on one MA2450

Physical testing confirmed that the OpenVINO 2020.3 MYRIAD plugin can keep at
least four QuartzNet executable networks resident simultaneously on the tested
MA2450. The test loaded reshaped static graphs at `T=512`, `1024`, `1536`,
and `2048` without releasing earlier `ExecutableNetwork` or `InferRequest`
objects.

After every additional `LoadNetwork()`, all previously loaded graphs were run
again. Their deterministic output fingerprints remained unchanged:

| Resident graph | Load time | Steady-state inference during rechecks | Recheck result |
|---|---:|---:|---|
| `T=512` | 1945 ms | about 392 ms | stable after all later loads |
| `T=1024` | 5470 ms | about 1415 ms | stable after `T=1536` and `T=2048` loads |
| `T=1536` | 5478 ms | about 3704 ms | stable after `T=2048` load |
| `T=2048` | 5495 ms | about 6125 ms | stable |

The final result was `RESIDENT_RESULT status=PASS loaded=4 requested=4`.

This proves that multiple resident compiled networks are supported in practice
by this OpenVINO 2020.3/MA2450 stack. It also shows that loading a later network
did not evict or corrupt the earlier networks in this four-graph test.

The maximum resident-network count remains unknown, as does the device-memory
cost of each compiled graph. Do not assume nominal stick memory translates
directly into available graph capacity. Compiled-network memory includes
transformed weights, activation/work buffers, DMA/runtime state, and
plugin/firmware overhead.

Multiple resident networks can remove repeated graph-load latency for
applications that require several models or shapes. They do not create multiple
independent accelerators and therefore do not imply multiplied inference
throughput.

### 5.3 Long audio should be segmented or routed elsewhere

For speech longer than the validated MYRIAD window, the safe options are to segment/stream before inference or route the workload to another backend.

The exact chunking strategy must preserve enough acoustic context to avoid introducing a new WER regression.

## 6. Broader platform limitations exposed by this work

- **Legacy software dependency:** the tested MA2450 path is tied to OpenVINO 2020.3.2 in this project. That limits access to newer compiler/runtime improvements and makes topology-specific VPU issues harder to remediate.
- **Weak dynamic-shape ergonomics:** the runtime can reshape and compile different temporal lengths, but each distinct shape incurs a costly `LoadNetwork()`/compile step.
- **Latency variability:** large temporal convolutions are a poor fit for predictable interactive latency on this VPU. Corpus p95 was over 4 seconds even though overall inference remained faster than real time.
- **Silent semantic failure risk:** the failing `T>=3104` graphs returned successful inference calls and plausible tensor dimensions. The runtime did not signal an error; semantic guards are therefore required.
- **Numerical parity is topology- and shape-dependent:** acceptable behavior at `T=592` does not imply acceptable behavior at `T=3000+`. Qualification must cover intended production shapes.

## 7. What has not been established

- The exact underlying implementation defect, for example internal tiling, addressing, buffering, or another compiler primitive limit, has not been identified.
- The `T=3088/3104` transition has only been established for this QuartzNet topology and tested software/firmware path; it must not be generalized to unrelated neural networks.
- At least four simultaneously resident QuartzNet networks are proven; the maximum resident-network count has not been measured.
- A production bucket-padding policy has not yet been shown to preserve exact-shape WER.
- A fixed-window or streaming QuartzNet deployment has not yet been qualified for boundary effects and word-error rate.
- The tested amd64 OpenVINO CPU result is an attribution control, not a Raspberry Pi CPU performance comparison.

## 8. Recommended safeguards for this repository

1. Add a hard runtime shape guard that rejects `T>=3104` for the MA2450 QuartzNet path.
2. Use a lower production policy ceiling, for example `T<=2048`, until a validated streaming/bucketing design is frozen.
3. Keep semantic probes, including argmax parity and WER, not only successful `Infer()` return codes, in hardware qualification.
4. Four resident QuartzNet graphs are proven stable; measure the actual capacity ceiling only if a future application needs more than four resident models.
5. Record per-shape load time and inference time separately so graph compilation is never confused with steady-state latency.
6. Retain CPU/ONNX controls for future MYRIAD regressions; they are effective at distinguishing graph-conversion problems from VPU execution problems.

## 9. Bottom line

The MA2450 remains usable for this clean-speech QuartzNet workload when operated inside a validated temporal envelope, but it is not safe as a transparent variable-length accelerator.

The decisive limitation is a silent, catastrophic execution failure beginning between `T=3088` and `T=3104` for this topology on OpenVINO 2020.3.2. Below that cliff, quality is generally strong, but numerical drift and latency increase with shape.

Production should therefore use bounded, validated static or streaming shapes, explicit oversize rejection or segmentation, and semantic hardware guards.

## Appendix A. Key measured evidence

| Evidence item | Measured result |
|---|---|
| Source clean reference | PyTorch WER `0.0379398`; CER `0.0125065` |
| Pi-compatible frontend requalification | WER `0.0379030`; absolute WER delta `0.0000368` |
| First ordinary MYRIAD semantic probe | `T=592`; valid-frame argmax agreement `1.0`; max TV about `0.0594` |
| Exact-static `T=3264` ONNX control | Dynamic vs static ONNX: bit-identical |
| Exact-static `T=3264` CPU control | 1623/1624 valid argmax frames match; max TV about `0.01578` |
| Exact-static `T=3264` MYRIAD | 0/1624 valid argmax frames match; max TV `1.0`; max raw error about `632.72` |
| Same-sample boundary | `T=3088` non-catastrophic; `T=3104` catastrophic; all tested larger shapes remain catastrophic |
| Full physical corpus | 2703/2703; overall WER `0.0517628` before removing known catastrophic execution cases |
| MYRIAD inference | p50 `611.9 ms`; p95 `4023.1 ms`; mean `1317.0 ms`; RTF `0.1835` |
| Shape load overhead | 152 sessions; mean about `1963 ms`; total about `298.4 s` |
| Multi-resident capacity | `T=512,1024,1536,2048` loaded simultaneously; all four rechecked with stable output fingerprints |

## Related repository evidence

The implementation and qualification details are maintained alongside the speech-ASR example, including:

- `examples/speech-asr/docs/adr/quartznet15x5-librispeech-myriad-qualification.md`
- `examples/speech-asr/evaluation/analyze_quartznet15x5_reference_myriad_lengths.py`
- `examples/speech-asr/evaluation/diagnose_quartznet15x5_reference_myriad_static.py`
