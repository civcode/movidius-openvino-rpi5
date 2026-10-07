#!/usr/bin/env bash
# Hardware-free regression checks for the multi-platform refactor.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# Host Python dependency policy: uv-managed environments only.
if grep -R -nE 'python([0-9.]+)?[[:space:]]+-m[[:space:]]+venv|/bin/python[[:space:]]+-m[[:space:]]+pip[[:space:]]+install|/bin/pip[[:space:]]+install' \
    scripts examples --include='*.sh' --exclude='verify-static.sh'; then
    fail 'host shell script bypasses the uv-managed Python environment policy'
fi
grep -q 'uv venv' scripts/lib/python-env.sh || fail 'uv venv helper missing'
grep -q 'uv pip install' scripts/lib/python-env.sh || fail 'uv pip helper missing'

# Shared headers live below include/ov203 and source code must include them
# through the configured include root, never as repository-relative paths.
if grep -R -nE '#include[[:space:]]+[<"]half\.hpp[>"]|#include[[:space:]]+["<]include/ov203/' \
    examples mobilenet-test smoke-test tests/cpp --include='*.cpp' --include='*.hpp'; then
    fail 'invalid shared-header include path; use <ov203/...>'
fi

# All shell entry points must parse.
while IFS= read -r f; do
    bash -n "$f" || fail "bash syntax: $f"
done < <(find . -path './vendor' -prune -o -path './logs' -prune -o -type f -name '*.sh' -print | sort)

echo 'platform helper:'
armv7="$(./build.sh --platform armv7 --print-platform)"
arm64="$(./build.sh --platform arm64 --print-platform)"
amd64="$(./build.sh --platform amd64 --print-platform)"
grep -q '^target=armv7$' <<<"$armv7"
grep -q '^docker_platform=linux/arm/v7$' <<<"$armv7"
grep -q '^use_cmake_toolchain=1$' <<<"$armv7"
grep -q '^expected_elf_class=ELF32$' <<<"$armv7"
grep -q '^expected_elf_machine_id=40$' <<<"$armv7"
grep -q '^host_runtime_kind=armhf-sysroot$' <<<"$armv7"
grep -q '^target=arm64$' <<<"$arm64"
grep -q '^docker_platform=linux/arm64$' <<<"$arm64"
grep -q '^use_cmake_toolchain=0$' <<<"$arm64"
grep -q '^expected_elf_class=ELF64$' <<<"$arm64"
grep -q '^expected_elf_machine_id=183$' <<<"$arm64"
grep -q '^host_runtime_kind=native$' <<<"$arm64"
grep -q '^target=amd64$' <<<"$amd64"
grep -q '^docker_platform=linux/amd64$' <<<"$amd64"
grep -q '^use_cmake_toolchain=0$' <<<"$amd64"
grep -q '^expected_elf_class=ELF64$' <<<"$amd64"
grep -q '^expected_elf_machine_id=62$' <<<"$amd64"
grep -q '^host_runtime_kind=native$' <<<"$amd64"
printf '%s\n%s\n%s\n' "$armv7" "$arm64" "$amd64"

# Common build/runtime files must not contain an unconditional ARMv7 runtime path.
for f in Dockerfile build.sh run.sh container-entry.sh scripts/prepare-deps.sh scripts/prepare-mobilenet.sh; do
    if grep -nE 'lib/armv7l|ld-linux-armhf\.so\.3|--platform[ =]+linux/arm/v7' "$f"; then
        fail "unconditional ARMv7 literal remains in common file: $f"
    fi
done

grep -q 'USE_CMAKE_TOOLCHAIN' Dockerfile || fail 'Dockerfile lacks conditional toolchain selection'
grep -q 'ov203-build-${TARGET}' Dockerfile || fail 'Docker build cache is not target-qualified'
grep -q 'CONFIGURE_REVISION' build.sh || fail 'build.sh lacks CMake cache revision'
grep -q '/work/build/.configure_stamp' Dockerfile || fail 'Dockerfile lacks CMake build-cache stamp'
grep -q 'CMakeError.log' Dockerfile || fail 'Dockerfile does not surface CMake configure failures'
grep -q 'BUILD_IMAGE="${IMAGE}-${BUILD_TARGET}"' build.sh || fail 'stage-only Docker build can overwrite runtime image'
grep -q 'libmyriadPlugin.so' Dockerfile || fail 'MYRIAD plugin validation missing'
grep -q 'usb-ma2450.mvcmd' Dockerfile || fail 'MA2450 firmware validation missing'
grep -q 'readelf -h "${IE_PLUGIN}"' Dockerfile || fail 'MYRIAD plugin ELF architecture validation missing'
grep -q 'EXPECTED_ELF_MACHINE_ID' Dockerfile || fail 'runtime manifest lacks ELF machine metadata'

# OV linking is centralised in cmake/ov203-link.cmake.  The shared module
# must discover lib/<arch> dynamically, every CMakeLists must use it, and
# no CMake file may pin or re-implement an architecture lib dir.
grep -q 'lib/\*' cmake/ov203-link.cmake || fail 'cmake/ov203-link.cmake does not discover architecture lib dirs'
if grep -nE 'lib/(armv7l|aarch64|arm64|intel64|x86_64)' cmake/ov203-link.cmake; then
    fail 'cmake/ov203-link.cmake hard-codes an architecture lib dir'
fi
while IFS= read -r f; do
    grep -q 'ov203-link' "$f" || fail "$f does not include the shared OV link module"
    if grep -nE 'lib/(armv7l|aarch64|arm64|intel64|x86_64)' "$f"; then fail "$f hard-codes an architecture lib dir"; fi
done < <(find . -path './vendor' -prune -o -path './work' -prune -o -type f -name CMakeLists.txt -print | sort)

# ARM64 must be treated as a native target, not as ARMv7 with a renamed image.
grep -q 'arm64)' scripts/platform.sh || fail 'platform helper lacks arm64 target'
grep -q 'DOCKER_PLATFORM="linux/arm64"' scripts/platform.sh || fail 'arm64 Docker platform missing'
grep -q 'EXPECTED_ELF_MACHINE_ID=183' scripts/platform.sh || fail 'arm64 ELF machine ID missing'
grep -q 'runtime_target_can_run_host' scripts/host-run.sh || fail 'host-run bypasses shared host-target acceptance'
grep -q 'arm64:aarch64' scripts/lib/runtime.sh || fail 'runtime helper lacks native aarch64 host acceptance'
grep -q 'arm64:arm64' scripts/lib/runtime.sh || fail 'runtime helper lacks native arm64 host acceptance'
grep -q 'arm64:arm64' scripts/verify.sh || fail 'runtime verifier lacks arm64 image validation'

# Phase 6 speech benchmark worker invariants.
grep -q 'benchmark_worker.py' scripts/benchmark-speech.sh || fail 'speech benchmark wrapper missing worker'
grep -q 'EXIT_EXECUTION = 3' examples/speech-asr/evaluation/benchmark_worker.py || fail 'benchmark worker exit codes missing'
grep -q 'measured_iterations.*5' examples/speech-asr/contracts/benchmark-v1.yaml || fail 'benchmark-v1 measured iteration policy changed'
grep -q 'expected frozen value 5' examples/speech-asr/python/speech_asr/contracts.py || fail 'benchmark-v1 iteration policy is not enforced'
test -f examples/speech-asr/contracts/acoustic-benchmark-result-v1.schema.json || fail 'Phase 6 result schema missing'

# Phase 7 recorded streaming invariants.
test -f examples/speech-asr/python/speech_asr/streaming.py || fail 'Phase 7 streaming core missing'
test -f examples/speech-asr/python/speech_asr/quartznet_fixed512.py || fail 'QuartzNet fixed512 streaming core missing'
grep -q 'FIXED_TENSOR_FRAMES = 512' examples/speech-asr/python/speech_asr/quartznet_fixed512.py || fail 'QuartzNet fixed512 tensor geometry changed'
grep -q 'single_global_ctc_collapse_after_logit_stitch' examples/speech-asr/python/speech_asr/quartznet_fixed512.py || fail 'QuartzNet fixed512 stitch/decode policy changed'
test -f examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_fixed512.py || fail 'QuartzNet fixed512 evaluator missing'
grep -q 'PersistentMyriadServer' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_fixed512.py || fail 'QuartzNet fixed512 evaluator lacks persistent MYRIAD session'
grep -q 'network_loads.*1' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_fixed512.py || fail 'QuartzNet fixed512 evaluator must use one network load'
grep -q 'MIN_SEMANTIC_ARGMAX_AGREEMENT = 0.99' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_fixed512.py || fail 'QuartzNet fixed512 semantic smoke threshold changed'
grep -q 'SPEECH_DECODER_CPUSET:-0-15' scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh || fail 'QuartzNet fixed512 CPU affinity default changed'
grep -q 'OMP_NUM_THREADS=16' scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh || fail 'QuartzNet fixed512 OMP thread count changed'
grep -q 'MKL_NUM_THREADS=16' scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh || fail 'QuartzNet fixed512 MKL thread count changed'
grep -q 'OPENBLAS_NUM_THREADS=16' scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh || fail 'QuartzNet fixed512 OpenBLAS thread count changed'
grep -q 'LOCAL_THREADS = 16' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 controller local thread count changed'
grep -q '"taskset"' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 controller local CPU affinity missing'
grep -q 'OMP_NUM_THREADS=' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 controller local OMP limit missing'
grep -q 'MKL_NUM_THREADS=' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 controller local MKL limit missing'
grep -q 'OPENBLAS_NUM_THREADS=' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 controller local OpenBLAS limit missing'
test -x scripts/evaluate-quartznet15x5-reference-fixed512.sh || fail 'QuartzNet fixed512 local wrapper missing'
test -x scripts/evaluate-quartznet15x5-reference-fixed512-myriad.sh || fail 'QuartzNet fixed512 MYRIAD wrapper missing'
test -f examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 edge controller missing'
grep -q 'network_loads") != 1' examples/speech-asr/evaluation/run_quartznet15x5_reference_fixed512_edge.py || fail 'QuartzNet fixed512 edge controller does not enforce one network load'
test -x scripts/run-quartznet15x5-reference-fixed512-edge.sh || fail 'QuartzNet fixed512 edge wrapper missing'
test -f examples/speech-asr/docs/adr/quartznet15x5-fixed512-streaming.md || fail 'QuartzNet fixed512 ADR missing'
test -f examples/speech-asr/contracts/streaming-v1.json || fail 'Phase 7 streaming contract missing'
test -f examples/speech-asr/contracts/streaming-replay-result-v1.schema.json || fail 'Phase 7 replay result schema missing'
grep -q 'midpoint_partition' examples/speech-asr/contracts/streaming-v1.json || fail 'streaming overlap ownership changed'
grep -q 'ScriptedCumulativeDecoder' examples/speech-asr/python/speech_asr/streaming.py || fail 'streaming decoder fixture missing'
grep -q 'replay_streaming.py' scripts/replay-speech.sh || fail 'streaming replay wrapper missing'

# Phase 8 trainable deployment skeleton invariants.
test -f examples/speech-asr/models/cnn_ctc_v1/model_spec.json || fail 'cnn_ctc_v1 model spec missing'
test -f requirements/training.txt || fail 'training dependency set missing'
grep -q 'work/venv-training' scripts/prepare-python-env.sh || fail 'uv training environment missing'
grep -q '"onnx_opset": 11' examples/speech-asr/models/cnn_ctc_v1/model_spec.json || fail 'cnn_ctc_v1 ONNX opset changed'
grep -q '"fixed_shapes": true' examples/speech-asr/models/cnn_ctc_v1/model_spec.json || fail 'cnn_ctc_v1 export must remain fixed-shape'
grep -q 'TensorDesc::getLayoutByDims' smoke-test/main.cpp || fail 'generic MYRIAD runner still assumes vision layout'
grep -q 'arg == "--tensor"' smoke-test/main.cpp || fail 'generic MYRIAD runner lacks tensor input'
grep -q 'arg == "--output"' smoke-test/main.cpp || fail 'generic MYRIAD runner lacks tensor output'
grep -q 'arg == "--reshape-time"' smoke-test/main.cpp || fail 'generic MYRIAD runner lacks time-axis reshape'
grep -q 'network.reshape(shapes)' smoke-test/main.cpp || fail 'generic MYRIAD runner does not apply network reshape'
grep -q 'check-reshape' scripts/run-myriad-tensor.sh || fail 'MYRIAD tensor launcher lacks reshape capability check'
grep -q 'check-resident' scripts/run-myriad-tensor.sh || fail 'MYRIAD tensor launcher lacks resident-network capability check'
grep -q 'arg == "--resident-times"' smoke-test/main.cpp || fail 'generic MYRIAD runner lacks resident-network test mode'
grep -q 'RESIDENT_RECHECK' smoke-test/main.cpp || fail 'resident-network test does not recheck earlier graphs'
test -x scripts/test-quartznet15x5-reference-myriad-residency.sh || fail 'QuartzNet MYRIAD residency test wrapper missing'
test -x scripts/probe-cnn-ctc-v1.sh || fail 'cnn_ctc_v1 compatibility probe missing'
test -x scripts/train-cnn-ctc-v1.sh || fail 'cnn_ctc_v1 training pipeline missing'
test -x scripts/evaluate-cnn-ctc-v1.sh || fail 'cnn_ctc_v1 evaluator wrapper missing'

# Phase 9 immutable experiment lifecycle invariants.
for f in \
    examples/speech-asr/contracts/experiment-proposal-v1.schema.json \
    examples/speech-asr/contracts/experiment-model-spec-v1.schema.json \
    examples/speech-asr/contracts/train-config-v1.schema.json \
    examples/speech-asr/contracts/acceptance-policy-v1.schema.json \
    examples/speech-asr/contracts/experiment-lifecycle-v1.schema.json \
    examples/speech-asr/contracts/deployment-manifest-v1.schema.json; do
    test -f "$f" || fail "Phase 9 contract missing: $f"
done
test -f examples/speech-asr/python/speech_asr/experiment.py || fail 'Phase 9 experiment core missing'
test -f examples/speech-asr/tools/manage_experiment.py || fail 'Phase 9 experiment manager missing'
grep -q 'experiment_id_from_identity' examples/speech-asr/python/speech_asr/experiment.py || fail 'experiment identity derivation missing'
grep -q 'stage result is immutable once recorded' examples/speech-asr/python/speech_asr/experiment.py || fail 'experiment stage immutability missing'
grep -q 'artifact is immutable once recorded' examples/speech-asr/python/speech_asr/experiment.py || fail 'experiment artifact immutability missing'
grep -q 'must match \$.controller_commit' examples/speech-asr/python/speech_asr/contracts.py || fail 'deployment worker revision pin missing'
grep -q 'retryable_failure_classes' examples/speech-asr/tools/manage_experiment.py || fail 'experiment retry policy enforcement missing'

# Phase 10 local execution agent / edge orchestration invariants.
test -f examples/speech-asr/python/speech_asr/orchestration.py || fail 'Phase 10 orchestration core missing'
test -f examples/speech-asr/agent/run_experiment.py || fail 'Phase 10 controller missing'
test -f examples/speech-asr/agent/edge_worker.py || fail 'Phase 10 edge worker missing'
test -f examples/speech-asr/agent/init_cnn_ctc_v1_experiment.py || fail 'Phase 10 baseline initializer missing'
test -f examples/speech-asr/contracts/edge-worker-result-v1.schema.json || fail 'Phase 10 edge worker result contract missing'
test -f scripts/run-speech-experiment.sh || fail 'Phase 10 controller wrapper missing'
test -f scripts/edge-speech-bootstrap.sh || fail 'Phase 10 edge bootstrap missing'
test -f scripts/edge-speech-preflight.sh || fail 'Phase 10 edge preflight missing'
test -f scripts/edge-speech-worker.sh || fail 'Phase 10 edge worker wrapper missing'
for f in \
    scripts/run-speech-experiment.sh \
    scripts/init-cnn-ctc-v1-experiment.sh \
    scripts/edge-speech-bootstrap.sh \
    scripts/edge-speech-preflight.sh \
    scripts/edge-speech-worker.sh; do
    test -x "$f" || fail "Phase 10 shell entry point is not executable: $f"
done
grep -q 'BatchMode=yes' examples/speech-asr/python/speech_asr/orchestration.py || fail 'Phase 10 SSH is not non-interactive'
if grep -R -n 'StrictHostKeyChecking=no' examples/speech-asr/agent examples/speech-asr/python/speech_asr/orchestration.py scripts/edge-speech-*.sh; then
    fail 'Phase 10 disables SSH host-key verification'
fi
grep -q 'worktree add --detach' scripts/edge-speech-bootstrap.sh || fail 'edge automation checkout is not a dedicated worktree'
grep -q 'flock -n' scripts/edge-speech-worker.sh || fail 'MYRIAD worker lock is not non-blocking'
grep -q 'worker_busy' scripts/edge-speech-worker.sh || fail 'MYRIAD busy state is not explicit'
grep -q 'python_env_resolve_uv' scripts/lib/python-env.sh || fail 'host uv resolver missing'
grep -q '\.local/bin/uv' scripts/lib/python-env.sh || fail 'host uv resolver lacks ~/.local/bin fallback'
grep -q 'python_env_require_uv' scripts/edge-speech-preflight.sh || fail 'edge preflight does not validate uv availability'
grep -q -- '--work-dir' scripts/train-cnn-ctc-v1.sh || fail 'training outputs are not experiment-isolatable'
grep -q 'AWAIT_REVIEW' examples/speech-asr/agent/run_experiment.py || fail 'Phase 10 controller does not hand off to review'
if grep -q 'build.sh --platform' examples/speech-asr/agent/run_experiment.py; then
    fail 'ordinary experiment execution must not rebuild the edge runtime'
fi

# Phase 11 generation-1 model research invariants.
for f in \
    examples/speech-asr/models/cnn_ctc_v2/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v2/vocab.json \
    examples/speech-asr/training/cnn_ctc_v2.py \
    examples/speech-asr/training/train_cnn_ctc_v2.py \
    examples/speech-asr/training/export_cnn_ctc_v2.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v2_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v2.py \
    examples/speech-asr/agent/init_cnn_ctc_v2_experiment.py; do
    test -f "$f" || fail "Phase 11 cnn_ctc_v2 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v2-experiment.sh \
    scripts/train-cnn-ctc-v2.sh \
    scripts/probe-cnn-ctc-v2.sh \
    scripts/prepare-cnn-ctc-v2.sh \
    scripts/evaluate-cnn-ctc-v2.sh \
    scripts/edge-speech-model-probe.sh; do
    test -x "$f" || fail "Phase 11 shell entry point is not executable: $f"
done
grep -q 'residual-temporal-v1' examples/speech-asr/models/cnn_ctc_v2/model_spec.json || fail 'cnn_ctc_v2 residual architecture missing'
grep -q '174804992' examples/speech-asr/models/cnn_ctc_v2/model_spec.json || fail 'cnn_ctc_v2 MAC estimate not frozen'
grep -q 'physical_compatibility_probe' examples/speech-asr/agent/run_experiment.py || fail 'Phase 11 lacks pretraining physical MYRIAD probe'
grep -q 'acceptance_reasons' examples/speech-asr/agent/run_experiment.py || fail 'controller does not surface review rejection reasons'
test -f examples/speech-asr/agent/review_experiment.py || fail 'experiment review tool missing'
test -x scripts/review-speech-experiment.sh || fail 'experiment review wrapper is not executable'
grep -q 'cnn_ctc_v2' examples/speech-asr/agent/edge_worker.py || fail 'edge worker does not register cnn_ctc_v2'
if grep -q 'groups=' examples/speech-asr/training/cnn_ctc_v2.py; then
    fail 'cnn_ctc_v2 generation 1 must not use grouped/depthwise convolution'
fi
if grep -Eq 'MultiheadAttention|LayerNorm' examples/speech-asr/training/cnn_ctc_v2.py; then
    fail 'cnn_ctc_v2 generation 1 must not use attention/layer normalization'
fi

# Phase 11 generation-2 optimization-dynamics invariants.
for f in \
    examples/speech-asr/models/cnn_ctc_v3/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v3/vocab.json \
    examples/speech-asr/training/cnn_ctc_v3.py \
    examples/speech-asr/training/train_cnn_ctc_v3.py \
    examples/speech-asr/training/export_cnn_ctc_v3.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v3_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py \
    examples/speech-asr/agent/init_cnn_ctc_v3_experiment.py; do
    test -f "$f" || fail "Phase 11 cnn_ctc_v3 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v3-experiment.sh \
    scripts/train-cnn-ctc-v3.sh \
    scripts/probe-cnn-ctc-v3.sh \
    scripts/prepare-cnn-ctc-v3.sh \
    scripts/evaluate-cnn-ctc-v3.sh; do
    test -x "$f" || fail "Phase 11 generation-2 shell entry point is not executable: $f"
done
grep -q '"normalization": "none"' examples/speech-asr/models/cnn_ctc_v3/model_spec.json || fail 'cnn_ctc_v3 must remain normalization-free'
grep -q '"residual_projection_init": "kaiming_scaled_0.01"' examples/speech-asr/models/cnn_ctc_v3/model_spec.json || fail 'cnn_ctc_v3 near-identity residual policy missing'
grep -q '"epochs": 32' examples/speech-asr/models/cnn_ctc_v3/model_spec.json || fail 'cnn_ctc_v3 optimization budget changed'
grep -q '"batch_size": 1' examples/speech-asr/models/cnn_ctc_v3/model_spec.json || fail 'cnn_ctc_v3 batch size changed'
grep -q 'clip_grad_norm_' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 gradient clipping missing'
grep -q 'CosineAnnealingLR' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 cosine schedule missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 training progress output missing'
grep -q '"event": "validation_progress"' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 validation progress output missing'
grep -q '"event": "epoch_complete"' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 epoch progress output missing'
grep -q '"eta_seconds"' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 progress ETA missing'
grep -q 'stream_output=True' examples/speech-asr/agent/run_experiment.py || fail 'experiment controller buffers training progress'
grep -q 'subprocess.Popen(' examples/speech-asr/agent/run_experiment.py || fail 'experiment controller lacks streaming subprocess path'
grep -q 'best_validation_loss' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 best-checkpoint selection missing'
grep -q 'optimizer_steps' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 optimizer-step evidence missing'
grep -q '"event": "training_device"' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 does not report training device'
grep -q 'cuda_peak_memory_allocated_bytes' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'cnn_ctc_v3 does not report CUDA memory evidence'
grep -q 'validate_training_device_evidence' examples/speech-asr/agent/run_experiment.py || fail 'controller does not enforce CUDA training evidence'
grep -q 'training_device_evidence' examples/speech-asr/agent/run_experiment.py || fail 'controller does not surface CUDA training evidence'
grep -q 'prepare-cnn-ctc-v3.sh' scripts/train-cnn-ctc-v3.sh || fail 'cnn_ctc_v3 full pipeline uses wrong conversion wrapper'
if grep -q 'prepare-cnn-ctc-v2.sh' scripts/train-cnn-ctc-v3.sh; then
    fail 'cnn_ctc_v3 full pipeline must not invoke v2 conversion'
fi
grep -q 'cnn_ctc_v3.xml' scripts/train-cnn-ctc-v3.sh || fail 'cnn_ctc_v3 full pipeline does not assert v3 XML output'
grep -q 'PRETRAINING_MYRIAD_MAX_ABS_ERROR = 0.01' examples/speech-asr/python/speech_asr/orchestration.py || fail 'pretraining MYRIAD numerical tolerance changed'
grep -q 'PRETRAINING_MYRIAD_PRETRAINED_MAX_FRAME_TOTAL_VARIATION = 0.002' examples/speech-asr/python/speech_asr/orchestration.py || fail 'pretrained MYRIAD probability tolerance changed'
grep -q 'pretrained-semantic-parity-v1' examples/speech-asr/python/speech_asr/orchestration.py || fail 'pretrained MYRIAD semantic gate missing'
grep -q 'pretraining_myriad_compatibility' examples/speech-asr/agent/run_experiment.py || fail 'controller lacks numerical pretraining MYRIAD gate'
grep -q 'model_id=model_id' examples/speech-asr/agent/run_experiment.py || fail 'controller does not select model-aware MYRIAD gate'
if grep -q 'agreement < 0.99' examples/speech-asr/agent/run_experiment.py; then
    fail 'initialized MYRIAD probe must not use brittle argmax threshold'
fi
grep -q 'prepare-cnn-ctc-v3.sh' scripts/probe-cnn-ctc-v3.sh || fail 'cnn_ctc_v3 probe uses wrong conversion wrapper'
if grep -q 'prepare-cnn-ctc-v2.sh' scripts/probe-cnn-ctc-v3.sh; then
    fail 'cnn_ctc_v3 probe must not invoke v2 conversion'
fi
grep -q 'cnn_ctc_v3.xml' scripts/probe-cnn-ctc-v3.sh || fail 'cnn_ctc_v3 probe does not assert v3 XML output'
grep -q 'cnn_ctc_v3' examples/speech-asr/agent/edge_worker.py || fail 'edge worker does not register cnn_ctc_v3'
if grep -Eq 'BatchNorm|groups=|MultiheadAttention|LayerNorm' examples/speech-asr/training/cnn_ctc_v3.py; then
    fail 'cnn_ctc_v3 inference graph violated normalization/operator policy'
fi

# Post-Phase-11 model-quality data boundary.
test -f examples/speech-asr/docs/phase11-final-review.md || fail 'Phase 11 final review missing'
test -f examples/speech-asr/docs/model-quality-v1-baseline.md || fail 'model-quality baseline review missing'
test -f examples/speech-asr/tools/qualify_model_quality_manifests.py || fail 'model-quality qualification tool missing'
test -x scripts/qualify-speech-model-data.sh || fail 'model-quality qualification wrapper is not executable'
grep -q 'default=("A", "B", "C")' examples/speech-asr/tools/qualify_model_quality_manifests.py || fail 'model-quality train speaker partition changed'
grep -q 'default=("D",)' examples/speech-asr/tools/qualify_model_quality_manifests.py || fail 'model-quality validation speaker partition changed'
grep -q 'speaker_overlap' examples/speech-asr/tools/qualify_model_quality_manifests.py || fail 'model-quality speaker overlap evidence missing'
grep -q 'greedy_decode_logits_diagnostics' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'v3 evaluator lacks decoder diagnostics'
grep -q 'blank_frame_fraction' examples/speech-asr/python/speech_asr/cnn_ctc.py || fail 'CTC blank-collapse diagnostics missing'
grep -q 'character_edits' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'v3 evaluator lacks per-sample character edits'
grep -q 'aggregate_decoder_diagnostics' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'v3 evaluator lacks decoder aggregate'
test -f examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'model-quality baseline initializer missing'
test -x scripts/init-cnn-ctc-v3-quality-baseline.sh || fail 'model-quality baseline wrapper is not executable'
test -x scripts/provision-speech-model-data-edge.sh || fail 'model-quality edge provisioning wrapper is not executable'
grep -q '89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a' examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'reviewed train manifest hash changed'
grep -q '07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923' examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'reviewed validation manifest hash changed'
grep -q '"max_wer": None' examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'quality baseline must not inherit smoke WER ceiling'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'quality baseline must not inherit smoke CER ceiling'
grep -q '"max_realtime_factor": None' examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py || fail 'quality baseline must not gate benchmark-dependent RTF'
grep -q 'processed_train_audio_seconds' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 processed-audio budget evidence missing'
grep -q 'validation_cer' examples/speech-asr/contracts/train-config-v1.schema.json || fail 'train config lacks validation-CER checkpoint selection'
grep -q 'evaluate_validation' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer lacks transcript-quality validation'
grep -q 'validation_blank_frame_fraction' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer lacks per-epoch blank-collapse evidence'
grep -q 'best_validation_cer' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer lacks CER checkpoint evidence'
test -f examples/speech-asr/agent/init_cnn_ctc_v3_cer_selection.py || fail 'CER-selection experiment initializer missing'
test -x scripts/init-cnn-ctc-v3-cer-selection.sh || fail 'CER-selection experiment wrapper is not executable'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v3_cer_selection.py || fail 'CER-selection experiment policy changed'
grep -q '"max_realtime_factor": None' examples/speech-asr/agent/init_cnn_ctc_v3_cer_selection.py || fail 'CER-selection experiment must measure rather than gate RTF'
grep -q 'feature_lengths' examples/speech-asr/training/cnn_ctc_v1.py || fail 'training batches do not preserve valid frontend lengths'
grep -q 'specaugment-v1' examples/speech-asr/contracts/train-config-v1.schema.json || fail 'train config lacks SpecAugment policy'
grep -q 'apply_specaugment_v1' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer lacks deterministic SpecAugment'
grep -q 'augmentation_generator.manual_seed' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'SpecAugment RNG is not deterministically seeded'
grep -q 'batch\["feature_lengths"\]' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'SpecAugment does not respect valid frontend lengths'
test -f examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment experiment initializer missing'
test -x scripts/init-cnn-ctc-v3-specaugment.sh || fail 'SpecAugment experiment wrapper is not executable'
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment parent changed'
grep -q '"kind": "specaugment-v1"' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment experiment policy missing'
grep -q '"frequency_masks": 2' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment frequency-mask count changed'
grep -q '"frequency_max_width": 8' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment frequency width changed'
grep -q '"time_masks": 2' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment time-mask count changed'
grep -q '"time_max_width": 20' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment time width changed'
grep -q '"time_max_fraction": 0.1' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment time fraction changed'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py || fail 'SpecAugment must retain CER-aligned checkpointing'
grep -q 'manifest\["benchmark"\]\["id"\]' examples/speech-asr/agent/edge_worker.py || fail 'edge evaluator does not propagate benchmark id'
grep -q 'benchmark id mismatch' examples/speech-asr/agent/edge_worker.py || fail 'edge result does not validate benchmark id'
grep -q 'same_benchmark_manifest' examples/speech-asr/agent/review_experiment.py || fail 'review tool is not benchmark-aware'
grep -q 'BatchMode=yes' scripts/provision-speech-model-data-edge.sh || fail 'edge dataset provisioning is not non-interactive'
if grep -q 'StrictHostKeyChecking=no' scripts/provision-speech-model-data-edge.sh; then
    fail 'edge dataset provisioning disables host-key verification'
fi
grep -q 'rsync -a --delete --checksum' scripts/provision-speech-model-data-edge.sh || fail 'edge dataset provisioning does not use reviewed rsync path'
grep -q 'edge-speech-bootstrap.sh' scripts/provision-speech-model-data-edge.sh || fail 'edge dataset provisioning does not pin worker revision'

# Frontend-correction generation: same v3 graph, padding-independent CMVN.
for f in \
    examples/speech-asr/models/cnn_ctc_v4/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v4/vocab.json \
    examples/speech-asr/training/export_cnn_ctc_v4.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v4_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v4.py \
    examples/speech-asr/agent/init_cnn_ctc_v4_valid_cmvn.py; do
    test -f "$f" || fail "cnn_ctc_v4 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v4-valid-cmvn.sh \
    scripts/train-cnn-ctc-v4.sh \
    scripts/probe-cnn-ctc-v4.sh \
    scripts/prepare-cnn-ctc-v4.sh \
    scripts/evaluate-cnn-ctc-v4.sh; do
    test -x "$f" || fail "cnn_ctc_v4 shell entry point is not executable: $f"
done
grep -q '"kind": "logmel-v2"' examples/speech-asr/models/cnn_ctc_v4/model_spec.json || fail 'cnn_ctc_v4 frontend kind changed'
grep -q '"normalization": "per_mel_bin_mean_valid_zero_pad"' examples/speech-asr/models/cnn_ctc_v4/model_spec.json || fail 'cnn_ctc_v4 valid-frame CMVN policy changed'
grep -q 'logged\[:, :valid_frames\]\.mean' examples/speech-asr/python/speech_asr/cnn_ctc_frontend.py || fail 'logmel-v2 does not normalize over valid frames'
grep -q 'logged\[:, valid_frames:\] = 0.0' examples/speech-asr/python/speech_asr/cnn_ctc_frontend.py || fail 'logmel-v2 does not zero normalized padding'
grep -q 'V4_ARCHITECTURE = dict(V3_ARCHITECTURE)' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v4 graph is no longer frozen to v3'
grep -q '"cnn_ctc_v4": "train-cnn-ctc-v4.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'controller lacks cnn_ctc_v4 training executor'
grep -q '"cnn_ctc_v4": "evaluate-cnn-ctc-v4.sh"' examples/speech-asr/agent/edge_worker.py || fail 'edge lacks cnn_ctc_v4 evaluator'
grep -q 'cnn_ctc_v4' examples/speech-asr/tools/validate_cnn_ctc_ir.py || fail 'IR validator lacks cnn_ctc_v4'
grep -q 'prepare-cnn-ctc-v4.sh' scripts/train-cnn-ctc-v4.sh || fail 'cnn_ctc_v4 training uses wrong conversion wrapper'
grep -q 'prepare-cnn-ctc-v4.sh' scripts/probe-cnn-ctc-v4.sh || fail 'cnn_ctc_v4 probe uses wrong conversion wrapper'
if grep -q 'prepare-cnn-ctc-v3.sh' scripts/train-cnn-ctc-v4.sh scripts/probe-cnn-ctc-v4.sh; then
    fail 'cnn_ctc_v4 must not invoke v3 conversion'
fi
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v4_valid_cmvn.py || fail 'cnn_ctc_v4 parent changed'
grep -q '"frontend": {"kind": "logmel-v2"}' examples/speech-asr/agent/init_cnn_ctc_v4_valid_cmvn.py || fail 'cnn_ctc_v4 experiment frontend changed'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v4_valid_cmvn.py || fail 'cnn_ctc_v4 must retain CER checkpoint selection'

# CTC-objective diagnostic: exact v3 graph/frontend, training-only blank pressure.
test -f examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'v3 blank-penalty initializer missing'
test -x scripts/init-cnn-ctc-v3-blank-penalty.sh || fail 'v3 blank-penalty wrapper is not executable'
grep -q 'blank-logit-penalty-v1' examples/speech-asr/contracts/train-config-v1.schema.json || fail 'train config lacks blank-logit objective contract'
grep -q 'blank_logit_penalty' examples/speech-asr/python/speech_asr/contracts.py || fail 'semantic train-config validator lacks blank-logit objective'
grep -q -- '--ctc-objective-kind' scripts/train-cnn-ctc-v3.sh || fail 'v3 training wrapper lacks CTC objective argument'
grep -q -- '--blank-logit-penalty' scripts/train-cnn-ctc-v3.sh || fail 'v3 training wrapper lacks blank penalty argument'
grep -q 'adjusted_logits\[\.\.\., blank_index\]' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer does not adjust only blank logits'
grep -q '"ctc_objective": ctc_objective' examples/speech-asr/training/train_cnn_ctc_v3.py || fail 'v3 trainer does not record CTC objective provenance'
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'blank-penalty parent changed'
grep -q '"model_id": "cnn_ctc_v3"' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'blank-penalty experiment changed model graph identity'
grep -q '"frontend": {"kind": "logmel-v1"}' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'blank-penalty experiment changed frontend'
grep -q '"blank_logit_penalty": 0.25' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'blank-penalty diagnostic value changed'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py || fail 'blank-penalty experiment must retain CER checkpoint selection'
if grep -q '"augmentation": {' examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py; then
    fail 'blank-penalty experiment must not add augmentation'
fi

# Reviewed objective/encoder redesign: compact v5 plus training-only InterCTC.
for f in \
    examples/speech-asr/models/cnn_ctc_v5/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v5/vocab.json \
    examples/speech-asr/training/cnn_ctc_v5.py \
    examples/speech-asr/training/train_cnn_ctc_v5.py \
    examples/speech-asr/training/export_cnn_ctc_v5.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v5_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v5.py \
    examples/speech-asr/agent/init_cnn_ctc_v5_interctc.py; do
    test -f "$f" || fail "cnn_ctc_v5 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v5-interctc.sh \
    scripts/train-cnn-ctc-v5.sh \
    scripts/probe-cnn-ctc-v5.sh \
    scripts/prepare-cnn-ctc-v5.sh \
    scripts/evaluate-cnn-ctc-v5.sh; do
    test -x "$f" || fail "cnn_ctc_v5 shell entry point is not executable: $f"
done
grep -q 'intermediate-ctc-v1' examples/speech-asr/contracts/train-config-v1.schema.json || fail 'train config lacks InterCTC objective contract'
grep -q 'V5_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'controller lacks frozen v5 architecture'
grep -q '"cnn_ctc_v5": "train-cnn-ctc-v5.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'controller lacks v5 training executor'
grep -q '"cnn_ctc_v5": "evaluate-cnn-ctc-v5.sh"' examples/speech-asr/agent/edge_worker.py || fail 'edge lacks v5 evaluator'
grep -q 'forward_with_intermediate' examples/speech-asr/training/train_cnn_ctc_v5.py || fail 'v5 trainer lacks intermediate supervision'
grep -q 'intermediate_ctc_weight \* intermediate_loss' examples/speech-asr/training/train_cnn_ctc_v5.py || fail 'v5 trainer does not combine intermediate CTC'
grep -q 'return self.projection(encoded)' examples/speech-asr/training/cnn_ctc_v5.py || fail 'v5 export forward changed'
grep -q '"intermediate_ctc_weight": 0.3' examples/speech-asr/agent/init_cnn_ctc_v5_interctc.py || fail 'v5 InterCTC weight changed'
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v5_interctc.py || fail 'v5 parent changed'

# Controlled v3-capacity InterCTC ablation after compact v5 regressed.
for f in \
    examples/speech-asr/models/cnn_ctc_v6/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v6/vocab.json \
    examples/speech-asr/training/cnn_ctc_v6.py \
    examples/speech-asr/training/train_cnn_ctc_v6.py \
    examples/speech-asr/training/export_cnn_ctc_v6.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v6_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v6.py \
    examples/speech-asr/agent/init_cnn_ctc_v6_interctc.py; do
    test -f "$f" || fail "cnn_ctc_v6 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v6-interctc.sh \
    scripts/train-cnn-ctc-v6.sh \
    scripts/probe-cnn-ctc-v6.sh \
    scripts/prepare-cnn-ctc-v6.sh \
    scripts/evaluate-cnn-ctc-v6.sh; do
    test -x "$f" || fail "cnn_ctc_v6 shell entry point is not executable: $f"
done
grep -q 'V6_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'controller lacks frozen v6 architecture'
grep -q '"cnn_ctc_v6": "train-cnn-ctc-v6.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'controller lacks v6 trainer'
grep -q '"cnn_ctc_v6": "evaluate-cnn-ctc-v6.sh"' examples/speech-asr/agent/edge_worker.py || fail 'edge lacks v6 evaluator'
grep -q 'self.projection = nn.Conv1d' examples/speech-asr/training/cnn_ctc_v6.py || fail 'v6 deployed head missing'
grep -q '"intermediate_ctc_weight": 0.3' examples/speech-asr/agent/init_cnn_ctc_v6_interctc.py || fail 'v6 InterCTC weight changed'
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v6_interctc.py || fail 'v6 reference parent changed'

# Expanded meeting-disjoint AMI training boundary.
test -f examples/speech-asr/datasets/ami/splits/train-es2005-v1.json || fail 'expanded AMI split missing'
test -f examples/speech-asr/tools/qualify_expanded_model_quality_manifests.py || fail 'expanded AMI qualifier missing'
test -f examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py || fail 'expanded-data initializer missing'
test -x scripts/qualify-speech-model-data-v2.sh || fail 'expanded qualifier wrapper is not executable'
test -x scripts/init-cnn-ctc-v3-expanded-data.sh || fail 'expanded-data initializer wrapper is not executable'
grep -q '"meeting": "ES2005a"' examples/speech-asr/datasets/ami/splits/train-es2005-v1.json || fail 'expanded AMI split lacks ES2005a'
grep -q '"meeting": "ES2005d"' examples/speech-asr/datasets/ami/splits/train-es2005-v1.json || fail 'expanded AMI split lacks ES2005d'
grep -q '"records": 1865' examples/speech-asr/datasets/ami/splits/train-es2005-v1.json || fail 'expanded AMI source count changed'
grep -q 'meeting_overlap' examples/speech-asr/tools/qualify_expanded_model_quality_manifests.py || fail 'expanded qualifier lacks meeting isolation'
grep -q '1487' examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py || fail 'expanded eligible count changed'
grep -q '2393.989' examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py || fail 'expanded training duration changed'
grep -q 'DEFAULT_PARENT = "exp-3c7727ca3f37ba2c"' examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py || fail 'expanded-data parent changed'
if grep -q '"ctc_objective": {' examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py; then
    fail 'expanded-data experiment must use standard CTC'
fi

# Held-out Full-corpus-ASR SC evaluation boundary.
test -f examples/speech-asr/datasets/ami/splits/eval-full-corpus-asr-sc-v1.json || fail 'held-out AMI split missing'
test -f examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py || fail 'held-out qualifier missing'
test -f examples/speech-asr/evaluation/evaluate_frozen_cnn_ctc_v3_reference.py || fail 'frozen held-out reference evaluator missing'
test -f examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'sealed held-out controller missing'
test -f examples/speech-asr/agent/review_heldout_evaluation.py || fail 'held-out review helper missing'
test -f examples/speech-asr/docs/adr/heldout-full-corpus-asr-sc-v1.md || fail 'held-out evaluation ADR missing'
for f in \
    scripts/prepare-speech-heldout-eval.sh \
    scripts/qualify-speech-heldout-eval.sh \
    scripts/provision-speech-heldout-eval-edge.sh \
    scripts/run-speech-heldout-eval.sh \
    scripts/review-speech-heldout-eval.sh; do
    test -x "$f" || fail "held-out shell entry point is not executable: $f"
done
python3 - <<'PY_HELDOUT'
import json
from pathlib import Path

path = Path("examples/speech-asr/datasets/ami/splits/eval-full-corpus-asr-sc-v1.json")
value = json.loads(path.read_text())
expected = {
    f"{prefix}{suffix}"
    for prefix in ("EN2002", "ES2004", "IS1009", "TS3003")
    for suffix in ("a", "b", "c", "d")
}
actual = {source["meeting"] for source in value["sources"]}
assert actual == expected, (sorted(actual), sorted(expected))
assert len(value["sources"]) == 16
assert value["partition"]["name"] == "Full-corpus-ASR"
assert value["partition"]["role"] == "SC-unseen-evaluation"
assert value["partition"]["training_allowed"] is False
assert value["partition"]["checkpoint_selection_allowed"] is False
for source in value["sources"]:
    assert source["audio"]["stream"] == "Mix-Headset"
    assert len(source["audio"]["sha256"]) == 64
    assert [entry["speaker"] for entry in source["selections"]] == ["A", "B", "C", "D"]
    assert all(entry.get("all_segments") is True for entry in source["selections"])
PY_HELDOUT
grep -q 'OFFICIAL_MEETINGS' examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py || fail 'held-out qualifier lacks exact meeting authority'
grep -q '"role": "test_only"' examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py || fail 'held-out qualifier does not label test-only role'
grep -q '"training_allowed": False' examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py || fail 'held-out qualifier permits training'
grep -q '"checkpoint_selection_allowed": False' examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py || fail 'held-out qualifier permits checkpoint selection'
grep -q 'SOURCE_EXPERIMENT_ID = "exp-87538823d2bf1562"' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out source checkpoint changed'
grep -q 'SOURCE_ATTEMPT_ID = "attempt-0001"' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out source attempt changed'
grep -q 'held-out evaluation is already sealed' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out controller permits resealing/rerun'
grep -q -- '--correct-invalid-v1-transport-result' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out controller lacks explicit v1 transport correction gate'
grep -q 'KNOWN_BAD_V1_EVALUATION_COMMIT = "f5c059d2028d215777b28c1fc971b2f482185a05"' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out transport correction is not pinned to the known defective commit'
grep -q 'persistent-tensor-stream-v1-stdout-banner' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out transport defect identity missing'
grep -q 'maximum_replacement_seals.*1' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out corrective replacement count is not bounded'
grep -q 'persistent_parity_gate' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out controller does not require hardware parity evidence before sealing'
grep -q 'edge-speech-bootstrap.sh' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out controller does not pin edge revision'
grep -q 'frame_argmax_agreement' examples/speech-asr/evaluation/evaluate_frozen_cnn_ctc_v3_reference.py || fail 'held-out reference lacks PyTorch/ONNX agreement gate'
grep -q 'persistent-tensor-stream-v2' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator lacks persistent execution contract'
grep -q 'class PersistentMyriadServer' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator lacks persistent server client'
grep -q 'CACHE_VERSION = 3' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out persistent cache version changed'
grep -q 'if not parity_checked:' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out evaluator may reuse caches before reproving parity'
grep -q 'custom-server)' run.sh || fail 'runtime launcher lacks persistent custom server mode'
grep -q 'OV_QUIET=1' run.sh || fail 'persistent tensor server does not suppress container stdout banner'
grep -q 'def run_single_shot_parity' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator lacks single-shot parity gate'
grep -q 'persistent MYRIAD parity gate failed' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator does not enforce persistent parity'
grep -q 'READY protocol=tensor-stream-v2' smoke-test/main.cpp || fail 'hello_myriad lacks persistent tensor protocol handshake'
grep -q 'TIMING request=' smoke-test/main.cpp || fail 'hello_myriad lacks per-request persistent timing'
if grep -q 'training_command(' examples/speech-asr/agent/run_frozen_heldout_evaluation.py; then
    fail 'held-out controller must not invoke training'
fi
if grep -q 'train-cnn' examples/speech-asr/agent/run_frozen_heldout_evaluation.py; then
    fail 'held-out controller must not invoke a training wrapper'
fi
grep -q 'BatchMode=yes' scripts/provision-speech-heldout-eval-edge.sh || fail 'held-out data provisioning is not non-interactive'
grep -q -- '--verify-only' scripts/provision-speech-heldout-eval-edge.sh || fail 'held-out provisioning must verify rather than regenerate qualification'
if grep -q 'StrictHostKeyChecking=no' scripts/provision-speech-heldout-eval-edge.sh; then
    fail 'held-out data provisioning disables host-key verification'
fi

# Scaled three-team follow-up after accepted expanded-data result.
test -f examples/speech-asr/datasets/ami/splits/train-es2005-es2007-v1.json || fail 'scaled AMI split missing'
test -f examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py || fail 'scaled-data initializer missing'
test -x scripts/qualify-speech-model-data-v3.sh || fail 'scaled qualifier wrapper is not executable'
test -x scripts/init-cnn-ctc-v3-scaled-data.sh || fail 'scaled initializer wrapper is not executable'
grep -q '"records": 5596' examples/speech-asr/datasets/ami/splits/train-es2005-es2007-v1.json || fail 'scaled AMI source count changed'
grep -q 'DEFAULT_PARENT = "exp-aa5380b542b0d784"' examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py || fail 'scaled-data parent changed'
grep -q '4429' examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py || fail 'scaled eligible count changed'
grep -q '7150.12' examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py || fail 'scaled training duration changed'
if grep -q '"ctc_objective": {' examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py; then
    fail 'scaled-data experiment must use standard CTC'
fi

# Fresh model-quality-v4 Full-corpus-ASR development boundary.
for f in \
    examples/speech-asr/datasets/ami/splits/model-quality-v4-edinburgh-v1.json \
    examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py \
    examples/speech-asr/tools/qualify_model_quality_v4.py \
    examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py \
    examples/speech-asr/docs/adr/model-quality-v4-fresh-development.md \
    tests/python/test_speech_asr_model_quality_v4.py; do
    test -f "$f" || fail "model-quality-v4 file missing: $f"
done
for f in \
    scripts/prepare-speech-model-data-v4.sh \
    scripts/qualify-speech-model-data-v4.sh \
    scripts/provision-speech-model-data-v4-edge.sh \
    scripts/init-cnn-ctc-v3-model-quality-v4.sh; do
    test -x "$f" || fail "model-quality-v4 shell entry point is not executable: $f"
done
python3 - <<'PY_MQV4'
import json
from pathlib import Path

path = Path("examples/speech-asr/datasets/ami/splits/model-quality-v4-edinburgh-v1.json")
value = json.loads(path.read_text())
assert value["id"] == "ami-model-quality-v4-edinburgh-full-corpus-asr-v1"
assert value["authority"]["partition_name"] == "Full-corpus-ASR"
assert set(value["train_groups"]) == {
    "ES2003",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2014",
    "ES2015",
    "ES2016",
}
assert set(value["validation_groups"]) == {"ES2011"}
assert set(value["excluded_prior_selection_groups"]) == {"ES2002"}
assert set(value["sealed_test_groups"]) == {"EN2002", "ES2004", "IS1009", "TS3003"}
assert value["policy"]["training_allowed_on_train"] is True
assert value["policy"]["training_allowed_on_validation"] is False
assert value["policy"]["checkpoint_selection_allowed_on_validation"] is True
assert value["policy"]["heldout_metrics_allowed_for_model_selection"] is False
PY_MQV4
grep -q 'fetch_official_for_freeze' examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py || fail 'model-quality-v4 source freeze does not acquire fresh official bytes'
grep -q 'KNOWN_PINNED_TRAIN_SPLIT' examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py || fail 'model-quality-v4 source freeze does not reuse existing pinned source identities'
grep -q 'partially present; refuse to overwrite' examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py || fail 'model-quality-v4 source lock is not immutable'
grep -q 'FORBIDDEN_GROUPS' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 qualifier lacks forbidden held-out/selection meetings'
grep -q 'training population did not expand beyond v3' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 qualifier does not require data expansion'
grep -q 'source_lock_sha256' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 qualification lacks source-lock provenance'
grep -q 'EXPECTED_VALIDATION_INVALID_INTERVAL' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 qualifier lacks reviewed invalid-interval identity'
grep -q '"meeting": "ES2011c"' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 reviewed anomaly meeting changed'
grep -q '"source_start_sample": 1249952' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 reviewed anomaly start changed'
grep -q '"source_end_sample": 1248016' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 reviewed anomaly end changed'
grep -q 'if train_exclusions:' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 permits training source exclusions'
grep -q 'len(validation_exclusions) != 1' examples/speech-asr/tools/qualify_model_quality_v4.py || fail 'model-quality-v4 validation exclusion count is not frozen'
grep -q 'channels not in (1, 2)' examples/speech-asr/python/speech_asr/ami.py || fail 'AMI normalizer does not accept pinned stereo headset mixes'
grep -q 'stereo-average-v1' examples/speech-asr/python/speech_asr/ami.py || fail 'AMI stereo normalization provenance is missing'
grep -q 'test_stereo_mix_is_deterministically_downmixed_to_mono' tests/python/test_speech_asr_ami.py || fail 'AMI stereo normalization regression test missing'
grep -q 'class AmiInvalidSegmentInterval' examples/speech-asr/python/speech_asr/ami.py || fail 'AMI invalid-segment interval classification missing'
grep -q 'non_positive_annotated_segment_interval' examples/speech-asr/python/speech_asr/ami.py || fail 'AMI invalid-segment provenance reason missing'
grep -q 'test_non_positive_segment_interval_is_excluded_with_provenance' tests/python/test_speech_asr_ami.py || fail 'AMI invalid-segment regression test missing'
grep -q 'DEFAULT_PARENT = "exp-87538823d2bf1562"' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline parent changed'
grep -q '"model_id": "cnn_ctc_v3"' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline changed model'
grep -q '"frontend": {"kind": "logmel-v1"}' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline changed frontend'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline changed checkpoint selection'
grep -q '"max_wer": None' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline must establish a new WER baseline'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 baseline must establish a new CER baseline'
grep -q 'TRAIN_MANIFEST_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 training manifest changed'
grep -q 'VALIDATION_MANIFEST_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 validation manifest changed'
grep -q 'TRAIN_RECORDS = 15738' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 training count changed'
grep -q 'TRAIN_AUDIO_SECONDS = 25846.327' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 training duration changed'
grep -q 'VALIDATION_RECORDS = 1273' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 validation count changed'
grep -q 'VALIDATION_AUDIO_SECONDS = 2279.385' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'reviewed model-quality-v4 validation duration changed'
grep -q 'heldout_metrics_allowed_for_model_selection' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py || fail 'model-quality-v4 initializer does not enforce held-out isolation'
if grep -q '"augmentation": {' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py; then
    fail 'model-quality-v4 baseline must not add augmentation'
fi
if grep -q '"ctc_objective": {' examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py; then
    fail 'model-quality-v4 baseline must use standard CTC'
fi
grep -q 'BatchMode=yes' scripts/provision-speech-model-data-v4-edge.sh || fail 'model-quality-v4 provisioning is not non-interactive'
if grep -q 'StrictHostKeyChecking=no' scripts/provision-speech-model-data-v4-edge.sh; then
    fail 'model-quality-v4 provisioning disables host-key verification'
fi
grep -q -- '--verify-only' scripts/provision-speech-model-data-v4-edge.sh || fail 'model-quality-v4 provisioning does not verify the frozen boundary'

# cnn_ctc_v7 width-only capacity ablation on the frozen model-quality-v4 boundary.
for f in \
    examples/speech-asr/models/cnn_ctc_v7/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v7/vocab.json \
    examples/speech-asr/models/cnn_ctc_v7/README.md \
    examples/speech-asr/training/cnn_ctc_v7.py \
    examples/speech-asr/training/train_cnn_ctc_v7.py \
    examples/speech-asr/training/export_cnn_ctc_v7.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v7_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v7.py \
    examples/speech-asr/agent/init_cnn_ctc_v7_wide.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v7-wide.md \
    tests/python/test_speech_asr_cnn_ctc_v7.py; do
    test -f "$f" || fail "cnn_ctc_v7 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v7-wide.sh \
    scripts/train-cnn-ctc-v7.sh \
    scripts/probe-cnn-ctc-v7.sh \
    scripts/prepare-cnn-ctc-v7.sh \
    scripts/evaluate-cnn-ctc-v7.sh; do
    test -x "$f" || fail "cnn_ctc_v7 shell entry point is not executable: $f"
done
python3 - <<'PY_V7'
import hashlib
import json
from pathlib import Path
spec=json.loads(Path("examples/speech-asr/models/cnn_ctc_v7/model_spec.json").read_text())
canonical=hashlib.sha256(
    json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
assert canonical == "0282f70168765fba5c1d33d28d47e6db18bc0163d99a1b3a8b15f49312b6401d"
assert spec["id"] == "cnn_ctc_v7"
assert spec["frontend"]["kind"] == "logmel-v1"
assert [x["channels"] for x in spec["network"]["stem"]] == [64, 112]
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [11,19,27,35,43]
assert all(x["channels"] == 112 for x in spec["network"]["residual_blocks"])
assert spec["network"]["estimated_parameters"] == 1818183
assert spec["network"]["estimated_macs_fixed_input"] == 235177984
assert spec["network"]["receptive_field_feature_frames"] == 533
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
PY_V7
grep -q 'V7_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v7 architecture registration missing'
grep -q '"cnn_ctc_v7": "train-cnn-ctc-v7.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v7 training executor registration missing'
grep -q '"cnn_ctc_v7": "evaluate-cnn-ctc-v7.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v7 edge evaluator registration missing'
grep -q '"cnn_ctc_v7"' examples/speech-asr/tools/validate_cnn_ctc_ir.py || fail 'cnn_ctc_v7 IR validation registration missing'
grep -q 'DEFAULT_PARENT = "exp-63fdb8d218673527"' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 model-quality-v4 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "0282f70168765fba5c1d33d28d47e6db18bc0163d99a1b3a8b15f49312b6401d"' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 reviewed model spec changed'
grep -q 'PARENT_CER = 0.7588550365720209' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 parent CER changed'
grep -q 'acceptance-evaluation.json' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 parent acceptance evidence is not canonical'
grep -q 'attempt.get("state") != "AWAIT_REVIEW"' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 parent lifecycle state is not verified'
grep -q 'acceptance.get("status") != "accepted"' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 parent acceptance status is not verified'
if grep -q 'result.get("acceptance")' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py; then
    fail 'cnn_ctc_v7 reads acceptance from compact result.json'
fi
if grep -q 'result.get("model")' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py; then
    fail 'cnn_ctc_v7 reads model identity from compact result.json'
fi
grep -q '"max_cer": PARENT_CER' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 CER decision gate missing'
grep -q 'EXPECTED_PARAMETERS = 1818183' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 parameter count changed'
grep -q 'EXPECTED_MACS = 235177984' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 MAC count changed'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py || fail 'cnn_ctc_v7 checkpoint selection changed'
if grep -q '"augmentation": {' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py; then
    fail 'cnn_ctc_v7 capacity ablation must not add augmentation'
fi
if grep -q '"ctc_objective": {' examples/speech-asr/agent/init_cnn_ctc_v7_wide.py; then
    fail 'cnn_ctc_v7 capacity ablation must use standard CTC'
fi
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v7.py || fail 'cnn_ctc_v7 live training progress missing'
grep -q '"event": "validation_progress"' examples/speech-asr/training/train_cnn_ctc_v7.py || fail 'cnn_ctc_v7 validation progress missing'

# Fast deterministic architecture-screen-v1 on the frozen model-quality-v4 boundary.
for f in \
    examples/speech-asr/tools/prepare_architecture_screen_v1.py \
    examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py \
    examples/speech-asr/docs/adr/model-quality-v4-architecture-screen-v1.md \
    tests/python/test_speech_asr_architecture_screen.py; do
    test -f "$f" || fail "architecture-screen-v1 file missing: $f"
done
for f in \
    scripts/prepare-speech-architecture-screen-v1.sh \
    scripts/init-cnn-ctc-v3-architecture-screen.sh; do
    test -x "$f" || fail "architecture-screen-v1 shell entry point is not executable: $f"
done
grep -q 'RULE_ID = "sha256-id-first-u64-be-mod4-eq0-v1"' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen selection rule changed'
grep -q 'MODULUS = 4' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen fraction changed'
grep -q 'REMAINDER = 0' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen hash bucket changed'
grep -q 'SCREEN_EPOCHS = 12' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen epoch budget changed'
grep -q 'len(selected) / len(train)' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen does not verify selected fraction'
grep -q 'meetings != EXPECTED_MEETINGS' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen does not preserve all train meetings'
grep -q 'VALIDATION_SHA256 = "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen validation boundary changed'
grep -q 'SOURCE_TRAIN_SHA256 = "6025d17f08c1815d1710c3ab56cea51a34365f398e9877b4aefec234e64e4096"' examples/speech-asr/tools/prepare_architecture_screen_v1.py || fail 'architecture-screen source training boundary changed'
grep -q 'DEFAULT_PARENT = "exp-63fdb8d218673527"' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py || fail 'architecture-screen v3 control parent changed'
grep -q '"model_id": "cnn_ctc_v3"' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py || fail 'architecture-screen control changed model'
grep -q '"epochs": SCREEN_EPOCHS' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py || fail 'architecture-screen control does not use screen epoch budget'
grep -q '"checkpoint_selection": "validation_cer"' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py || fail 'architecture-screen checkpoint selection changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py || fail 'architecture-screen v3 control must establish the proxy CER baseline'
if grep -q '"augmentation": {' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py; then
    fail 'architecture-screen v3 control must not add augmentation'
fi
if grep -q '"ctc_objective": {' examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py; then
    fail 'architecture-screen v3 control must use standard CTC'
fi

# cnn_ctc_v8 high-resolution dilated architecture screen.
for f in \
    examples/speech-asr/models/cnn_ctc_v8/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v8/vocab.json \
    examples/speech-asr/models/cnn_ctc_v8/README.md \
    examples/speech-asr/training/cnn_ctc_v8.py \
    examples/speech-asr/training/train_cnn_ctc_v8.py \
    examples/speech-asr/training/export_cnn_ctc_v8.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v8_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v8.py \
    examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v8-screen.md \
    tests/python/test_speech_asr_cnn_ctc_v8.py; do
    test -f "$f" || fail "cnn_ctc_v8 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v8-architecture-screen.sh \
    scripts/train-cnn-ctc-v8.sh \
    scripts/probe-cnn-ctc-v8.sh \
    scripts/prepare-cnn-ctc-v8.sh \
    scripts/evaluate-cnn-ctc-v8.sh; do
    test -x "$f" || fail "cnn_ctc_v8 shell entry point is not executable: $f"
done
python3 - <<'PY_V8'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v8/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
assert canonical == "e9ac743b98591f8c0abe652f3343ecd30d895c044a06949b3d5ee9266a2fcaf4"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,256,39]
assert [x["stride"] for x in spec["network"]["stem"]] == [1,2]
assert [x["channels"] for x in spec["network"]["stem"]] == [64,72]
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,2,2,2]
assert spec["network"]["estimated_parameters"] == 772983
assert spec["network"]["estimated_macs_fixed_input"] == 202897408
assert spec["network"]["receptive_field_feature_frames"] == 509
PY_V8
grep -q 'V8_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v8 architecture registration missing'
grep -q '"cnn_ctc_v8": "train-cnn-ctc-v8.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v8 training registration missing'
grep -q '"cnn_ctc_v8": "evaluate-cnn-ctc-v8.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v8 edge evaluator registration missing'
grep -q 'MODEL_SPEC_SHA256 = "e9ac743b98591f8c0abe652f3343ecd30d895c044a06949b3d5ee9266a2fcaf4"' examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py || fail 'cnn_ctc_v8 model identity changed'
grep -q 'DEFAULT_PARENT = "exp-00e6b1e434d187d8"' examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py || fail 'cnn_ctc_v8 screen parent changed'
grep -q 'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"' examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py || fail 'cnn_ctc_v8 screen manifest changed'
grep -q 'PROMOTION_CER = 0.7744692737430167' examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py || fail 'cnn_ctc_v8 promotion rule changed'
grep -q '"max_cer": PARENT_CER' examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py || fail 'cnn_ctc_v8 screen CER gate missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v8.py || fail 'cnn_ctc_v8 live training progress missing'

# cnn_ctc_v9 deep small-kernel dilated architecture screen.
for f in \
    examples/speech-asr/models/cnn_ctc_v9/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v9/vocab.json \
    examples/speech-asr/models/cnn_ctc_v9/README.md \
    examples/speech-asr/training/cnn_ctc_v9.py \
    examples/speech-asr/training/train_cnn_ctc_v9.py \
    examples/speech-asr/training/export_cnn_ctc_v9.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v9_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v9.py \
    examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v9-screen.md \
    tests/python/test_speech_asr_cnn_ctc_v9.py; do
    test -f "$f" || fail "cnn_ctc_v9 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v9-architecture-screen.sh \
    scripts/train-cnn-ctc-v9.sh \
    scripts/probe-cnn-ctc-v9.sh \
    scripts/prepare-cnn-ctc-v9.sh \
    scripts/evaluate-cnn-ctc-v9.sh; do
    test -x "$f" || fail "cnn_ctc_v9 shell entry point is not executable: $f"
done
python3 - <<'PY_V9'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v9/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
assert canonical == "85253bac6543be6f3d5f81ddd4025e259a23c55aaee972f8549a0368b5c0f37f"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert [x["channels"] for x in spec["network"]["stem"]] == [64,96]
assert len(spec["network"]["residual_blocks"]) == 8
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [7] * 8
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1]
assert spec["network"]["estimated_parameters"] == 646503
assert spec["network"]["estimated_macs_fixed_input"] == 85151744
assert spec["network"]["receptive_field_feature_frames"] == 493
PY_V9
grep -q 'V9_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v9 architecture registration missing'
grep -q '"cnn_ctc_v9": "train-cnn-ctc-v9.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v9 training registration missing'
grep -q '"cnn_ctc_v9": "evaluate-cnn-ctc-v9.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v9 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-00e6b1e434d187d8"' examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py || fail 'cnn_ctc_v9 screen parent changed'
grep -q 'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"' examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py || fail 'cnn_ctc_v9 screen manifest changed'
grep -q 'PROMOTION_CER = 0.7744692737430167' examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py || fail 'cnn_ctc_v9 promotion rule changed'
grep -q '"max_cer": PARENT_CER' examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py || fail 'cnn_ctc_v9 screen CER gate missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v9.py || fail 'cnn_ctc_v9 live training progress missing'

# cnn_ctc_v10 wider deep-dilated architecture screen.
for f in \
    examples/speech-asr/models/cnn_ctc_v10/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v10/vocab.json \
    examples/speech-asr/models/cnn_ctc_v10/README.md \
    examples/speech-asr/training/cnn_ctc_v10.py \
    examples/speech-asr/training/train_cnn_ctc_v10.py \
    examples/speech-asr/training/export_cnn_ctc_v10.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v10_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v10.py \
    examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v10-screen.md \
    tests/python/test_speech_asr_cnn_ctc_v10.py; do
    test -f "$f" || fail "cnn_ctc_v10 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v10-architecture-screen.sh \
    scripts/train-cnn-ctc-v10.sh \
    scripts/probe-cnn-ctc-v10.sh \
    scripts/prepare-cnn-ctc-v10.sh \
    scripts/evaluate-cnn-ctc-v10.sh; do
    test -x "$f" || fail "cnn_ctc_v10 shell entry point is not executable: $f"
done
python3 - <<'PY_V10'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v10/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
assert canonical == "007f9808b11bbf1e84600dd0e5c962486fb468ead81ef476ca8b32f1381e4315"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert [x["channels"] for x in spec["network"]["stem"]] == [64,112]
assert len(spec["network"]["residual_blocks"]) == 8
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [7] * 8
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1]
assert spec["network"]["estimated_parameters"] == 865511
assert spec["network"]["estimated_macs_fixed_input"] == 113149952
assert spec["network"]["receptive_field_feature_frames"] == 493
PY_V10
grep -q 'V10_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v10 architecture registration missing'
grep -q '"cnn_ctc_v10": "train-cnn-ctc-v10.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v10 training registration missing'
grep -q '"cnn_ctc_v10": "evaluate-cnn-ctc-v10.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v10 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-00e6b1e434d187d8"' examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py || fail 'cnn_ctc_v10 screen parent changed'
grep -q 'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"' examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py || fail 'cnn_ctc_v10 screen manifest changed'
grep -q 'PROMOTION_CER = 0.7744692737430167' examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py || fail 'cnn_ctc_v10 promotion rule changed'
grep -q '"max_cer": PARENT_CER' examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py || fail 'cnn_ctc_v10 screen CER gate missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v10.py || fail 'cnn_ctc_v10 live training progress missing'

# cnn_ctc_v11 128-channel deep-dilated architecture screen.
for f in \
    examples/speech-asr/models/cnn_ctc_v11/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v11/vocab.json \
    examples/speech-asr/models/cnn_ctc_v11/README.md \
    examples/speech-asr/training/cnn_ctc_v11.py \
    examples/speech-asr/training/train_cnn_ctc_v11.py \
    examples/speech-asr/training/export_cnn_ctc_v11.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v11_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v11.py \
    examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v11-screen.md \
    tests/python/test_speech_asr_cnn_ctc_v11.py; do
    test -f "$f" || fail "cnn_ctc_v11 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v11-architecture-screen.sh \
    scripts/train-cnn-ctc-v11.sh \
    scripts/probe-cnn-ctc-v11.sh \
    scripts/prepare-cnn-ctc-v11.sh \
    scripts/evaluate-cnn-ctc-v11.sh; do
    test -x "$f" || fail "cnn_ctc_v11 shell entry point is not executable: $f"
done
python3 - <<'PY_V11'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v11/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "0a63c74a2414a26a04bf06da1cc8493b2886257077b451281e0d1da2b107f73a"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert [x["channels"] for x in spec["network"]["stem"]] == [64,128]
assert len(spec["network"]["residual_blocks"]) == 8
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [7] * 8
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1]
assert spec["network"]["estimated_parameters"] == 1117287
assert spec["network"]["estimated_macs_fixed_input"] == 145342464
assert spec["network"]["receptive_field_feature_frames"] == 493
PY_V11
grep -q 'V11_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v11 architecture registration missing'
grep -q '"cnn_ctc_v11": "train-cnn-ctc-v11.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v11 training registration missing'
grep -q '"cnn_ctc_v11": "evaluate-cnn-ctc-v11.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v11 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-00e6b1e434d187d8"' examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py || fail 'cnn_ctc_v11 screen parent changed'
grep -q 'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"' examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py || fail 'cnn_ctc_v11 screen manifest changed'
grep -q 'PROMOTION_CER = 0.7744692737430167' examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py || fail 'cnn_ctc_v11 promotion rule changed'
grep -q '"max_cer": PARENT_CER' examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py || fail 'cnn_ctc_v11 screen CER gate missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v11.py || fail 'cnn_ctc_v11 live training progress missing'


# cnn_ctc_v12 large-capacity MYRIAD probe.
for f in \
    examples/speech-asr/models/cnn_ctc_v12/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v12/vocab.json \
    examples/speech-asr/models/cnn_ctc_v12/README.md \
    examples/speech-asr/training/cnn_ctc_v12.py \
    examples/speech-asr/training/train_cnn_ctc_v12.py \
    examples/speech-asr/training/export_cnn_ctc_v12.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v12_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v12.py \
    examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v12-capacity-probe.md \
    tests/python/test_speech_asr_cnn_ctc_v12.py; do
    test -f "$f" || fail "cnn_ctc_v12 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v12-capacity-probe.sh \
    scripts/train-cnn-ctc-v12.sh \
    scripts/probe-cnn-ctc-v12.sh \
    scripts/prepare-cnn-ctc-v12.sh \
    scripts/evaluate-cnn-ctc-v12.sh; do
    test -x "$f" || fail "cnn_ctc_v12 shell entry point is not executable: $f"
done
python3 - <<'PY_V12'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v12/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "b252cf8f13b86768d99edaa8f74f2397e6eb928d201ab78402dce5603cf4b008"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert [x["channels"] for x in spec["network"]["stem"]] == [128,448]
assert len(spec["network"]["residual_blocks"]) == 16
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [5] * 16
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1] * 2
assert spec["network"]["estimated_parameters"] == 19627687
assert spec["network"]["estimated_macs_fixed_input"] == 2515673088
assert spec["network"]["receptive_field_feature_frames"] == 653
assert spec["training"]["learning_rate"] == 0.003
assert spec["training"]["lr_schedule"] == "onecycle"
assert spec["training"]["min_learning_rate"] == 0.00003
assert spec["training"]["onecycle_pct_start"] == 0.1
assert spec["training"]["onecycle_div_factor"] == 10.0
assert spec["training"]["onecycle_final_div_factor"] == 10.0
PY_V12
grep -q 'V12_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v12 architecture registration missing'
grep -q '"cnn_ctc_v12": "train-cnn-ctc-v12.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v12 training registration missing'
grep -q '"cnn_ctc_v12": "evaluate-cnn-ctc-v12.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v12 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-00e6b1e434d187d8"' examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py || fail 'cnn_ctc_v12 comparison parent changed'
grep -q 'SCREEN_MANIFEST_SHA256 = "0528db59eec36d00d710b3090b0c6404dbdbf548ae7e13d3daf3d5152eacfc8b"' examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py || fail 'cnn_ctc_v12 screen manifest changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py || fail 'cnn_ctc_v12 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py || fail 'cnn_ctc_v12 latency must be measured, not gated'
grep -q 'OneCycleLR' examples/speech-asr/training/train_cnn_ctc_v12.py || fail 'cnn_ctc_v12 OneCycle schedule missing'
grep -q 'scheduler.step()' examples/speech-asr/training/train_cnn_ctc_v12.py || fail 'cnn_ctc_v12 per-step LR schedule missing'
grep -q '"event": "training_progress"' examples/speech-asr/training/train_cnn_ctc_v12.py || fail 'cnn_ctc_v12 live training progress missing'


# cnn_ctc_v13 stabilized large-capacity training retry.
for f in \
    examples/speech-asr/models/cnn_ctc_v13/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v13/vocab.json \
    examples/speech-asr/models/cnn_ctc_v13/README.md \
    examples/speech-asr/training/cnn_ctc_v13.py \
    examples/speech-asr/training/train_cnn_ctc_v13.py \
    examples/speech-asr/training/export_cnn_ctc_v13.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v13_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v13.py \
    examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v13-stabilized-lr.md \
    tests/python/test_speech_asr_cnn_ctc_v13.py; do
    test -f "$f" || fail "cnn_ctc_v13 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v13-stabilized-lr.sh \
    scripts/train-cnn-ctc-v13.sh \
    scripts/probe-cnn-ctc-v13.sh \
    scripts/prepare-cnn-ctc-v13.sh \
    scripts/evaluate-cnn-ctc-v13.sh; do
    test -x "$f" || fail "cnn_ctc_v13 shell entry point is not executable: $f"
done
python3 - <<'PY_V13'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v13/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "44d0def0e6e9e4ba47f0b0af61ece4084e08ee1970055fb441ee0e2c884d4db5"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["channels"] for x in spec["network"]["stem"]] == [128,448]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert len(spec["network"]["residual_blocks"]) == 16
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [5] * 16
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1] * 2
assert spec["network"]["estimated_parameters"] == 19627687
assert spec["network"]["estimated_macs_fixed_input"] == 2515673088
assert spec["network"]["receptive_field_feature_frames"] == 653
assert spec["training"]["learning_rate"] == 0.0012
assert spec["training"]["lr_schedule"] == "onecycle"
assert spec["training"]["min_learning_rate"] == 0.00003
assert spec["training"]["onecycle_pct_start"] == 0.3
assert spec["training"]["onecycle_div_factor"] == 4.0
assert spec["training"]["onecycle_final_div_factor"] == 10.0
PY_V13
grep -q 'V13_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v13 architecture registration missing'
grep -q '"cnn_ctc_v13": "train-cnn-ctc-v13.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v13 training registration missing'
grep -q '"cnn_ctc_v13": "evaluate-cnn-ctc-v13.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v13 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-4e16fd348004571b"' examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py || fail 'cnn_ctc_v13 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "44d0def0e6e9e4ba47f0b0af61ece4084e08ee1970055fb441ee0e2c884d4db5"' examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py || fail 'cnn_ctc_v13 model identity changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py || fail 'cnn_ctc_v13 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py || fail 'cnn_ctc_v13 latency must be measured, not gated'
grep -q 'OneCycleLR' examples/speech-asr/training/train_cnn_ctc_v13.py || fail 'cnn_ctc_v13 OneCycle schedule missing'
grep -q 'scheduler.step()' examples/speech-asr/training/train_cnn_ctc_v13.py || fail 'cnn_ctc_v13 per-step LR schedule missing'


# cnn_ctc_v14 large-capacity cosine-control retry.
for f in \
    examples/speech-asr/models/cnn_ctc_v14/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v14/vocab.json \
    examples/speech-asr/models/cnn_ctc_v14/README.md \
    examples/speech-asr/training/cnn_ctc_v14.py \
    examples/speech-asr/training/train_cnn_ctc_v14.py \
    examples/speech-asr/training/export_cnn_ctc_v14.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v14_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v14.py \
    examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v14-cosine-control.md \
    tests/python/test_speech_asr_cnn_ctc_v14.py; do
    test -f "$f" || fail "cnn_ctc_v14 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v14-cosine-control.sh \
    scripts/train-cnn-ctc-v14.sh \
    scripts/probe-cnn-ctc-v14.sh \
    scripts/prepare-cnn-ctc-v14.sh \
    scripts/evaluate-cnn-ctc-v14.sh; do
    test -x "$f" || fail "cnn_ctc_v14 shell entry point is not executable: $f"
done
python3 - <<'PY_V14'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v14/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "f2897df71bcf54c49febff31a7c5465e8a8653c5d9698276cbf80e5c242d016e"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert [x["channels"] for x in spec["network"]["stem"]] == [128,448]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert len(spec["network"]["residual_blocks"]) == 16
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [5] * 16
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1] * 2
assert spec["network"]["estimated_parameters"] == 19627687
assert spec["network"]["estimated_macs_fixed_input"] == 2515673088
assert spec["network"]["receptive_field_feature_frames"] == 653
assert spec["training"]["learning_rate"] == 0.0003
assert spec["training"]["lr_schedule"] == "cosine"
assert spec["training"]["min_learning_rate"] == 0.00003
assert "onecycle_pct_start" not in spec["training"]
PY_V14
grep -q 'V14_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v14 architecture registration missing'
grep -q '"cnn_ctc_v14": "train-cnn-ctc-v14.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v14 training registration missing'
grep -q '"cnn_ctc_v14": "evaluate-cnn-ctc-v14.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v14 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-968150d44c4479ac"' examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py || fail 'cnn_ctc_v14 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "f2897df71bcf54c49febff31a7c5465e8a8653c5d9698276cbf80e5c242d016e"' examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py || fail 'cnn_ctc_v14 model identity changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py || fail 'cnn_ctc_v14 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py || fail 'cnn_ctc_v14 latency must be measured, not gated'
grep -q 'CosineAnnealingLR' examples/speech-asr/training/train_cnn_ctc_v14.py || fail 'cnn_ctc_v14 cosine schedule missing'

# cnn_ctc_v15 large-capacity BatchNorm conditioning.
for f in \
    examples/speech-asr/models/cnn_ctc_v15/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v15/vocab.json \
    examples/speech-asr/models/cnn_ctc_v15/README.md \
    examples/speech-asr/training/cnn_ctc_v15.py \
    examples/speech-asr/training/train_cnn_ctc_v15.py \
    examples/speech-asr/training/export_cnn_ctc_v15.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v15_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v15.py \
    examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v15-batchnorm-conditioning.md \
    tests/python/test_speech_asr_cnn_ctc_v15.py; do
    test -f "$f" || fail "cnn_ctc_v15 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v15-batchnorm-conditioning.sh \
    scripts/train-cnn-ctc-v15.sh \
    scripts/probe-cnn-ctc-v15.sh \
    scripts/prepare-cnn-ctc-v15.sh \
    scripts/evaluate-cnn-ctc-v15.sh; do
    test -x "$f" || fail "cnn_ctc_v15 shell entry point is not executable: $f"
done
python3 - <<'PY_V15'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v15/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "de7da3031f5d576a6635c4a91e1cb80105e44b65b5ef561e0f93c236b597e145"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,128,39]
assert spec["network"]["kind"] == "residual-temporal-v10"
assert spec["network"]["normalization"] == "batchnorm"
assert spec["network"]["residual_projection_init"] == "kaiming_bn_gamma_0.01"
assert [x["channels"] for x in spec["network"]["stem"]] == [128,448]
assert [x["stride"] for x in spec["network"]["stem"]] == [2,2]
assert len(spec["network"]["residual_blocks"]) == 16
assert [x["kernel"] for x in spec["network"]["residual_blocks"]] == [5] * 16
assert [x["dilation"] for x in spec["network"]["residual_blocks"]] == [1,2,3,4,4,3,2,1] * 2
assert spec["network"]["estimated_parameters"] == 19642599
assert spec["network"]["estimated_macs_fixed_input"] == 2515673088
assert spec["network"]["receptive_field_feature_frames"] == 653
assert spec["training"]["learning_rate"] == 0.0003
assert spec["training"]["lr_schedule"] == "cosine"
assert spec["training"]["min_learning_rate"] == 0.00003
PY_V15
grep -q 'V15_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v15 architecture registration missing'
grep -q '"cnn_ctc_v15": "train-cnn-ctc-v15.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v15 training registration missing'
grep -q '"cnn_ctc_v15": "evaluate-cnn-ctc-v15.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v15 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-1e2478b84317ab02"' examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py || fail 'cnn_ctc_v15 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "de7da3031f5d576a6635c4a91e1cb80105e44b65b5ef561e0f93c236b597e145"' examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py || fail 'cnn_ctc_v15 model identity changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py || fail 'cnn_ctc_v15 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py || fail 'cnn_ctc_v15 latency must be measured, not gated'
grep -q 'BatchNorm1d' examples/speech-asr/training/cnn_ctc_v15.py || fail 'cnn_ctc_v15 BatchNorm conditioning missing'
grep -q 'CosineAnnealingLR' examples/speech-asr/training/train_cnn_ctc_v15.py || fail 'cnn_ctc_v15 cosine schedule missing'

# cnn_ctc_v16 QuartzNet-15x5 architecture reset.
for f in \
    examples/speech-asr/models/cnn_ctc_v16/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v16/vocab.json \
    examples/speech-asr/models/cnn_ctc_v16/README.md \
    examples/speech-asr/training/cnn_ctc_v16.py \
    examples/speech-asr/training/train_cnn_ctc_v16.py \
    examples/speech-asr/training/export_cnn_ctc_v16.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v16_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v16.py \
    examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v16-quartznet.md \
    tests/python/test_speech_asr_cnn_ctc_v16.py; do
    test -f "$f" || fail "cnn_ctc_v16 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v16-quartznet.sh \
    scripts/train-cnn-ctc-v16.sh \
    scripts/probe-cnn-ctc-v16.sh \
    scripts/prepare-cnn-ctc-v16.sh \
    scripts/evaluate-cnn-ctc-v16.sh; do
    test -x "$f" || fail "cnn_ctc_v16 shell entry point is not executable: $f"
done
python3 - <<'PY_V16'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v16/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "70320bc878891f843312f0d8c1ae41ad7f414dbe2a0537e6766af8bf15f70a93"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,256,39]
network = spec["network"]
assert network["kind"] == "quartznet-15x5-v1"
assert network["normalization"] == "batchnorm"
assert network["activation"] == "relu"
assert network["dropout"] == 0.2
assert network["separable_convolution"] == "depthwise-pointwise"
assert network["prologue"] == {
    "name": "C1", "channels": 256, "kernel": 33, "stride": 2,
    "dilation": 1, "padding": 16, "separable": True,
}
assert [g["channels"] for g in network["block_groups"]] == [256,256,512,512,512]
assert [g["kernel"] for g in network["block_groups"]] == [33,39,51,63,75]
assert all(g["block_repeats"] == 3 for g in network["block_groups"])
assert all(g["module_repeats"] == 5 for g in network["block_groups"])
assert network["epilogue"][0] == {
    "name": "C2", "channels": 512, "kernel": 87, "stride": 1,
    "dilation": 2, "padding": 86, "separable": True,
}
assert network["epilogue"][1] == {
    "name": "C3", "channels": 1024, "kernel": 1, "stride": 1,
    "dilation": 1, "padding": 0, "separable": False,
}
assert network["estimated_parameters"] == 18934631
assert network["estimated_macs_fixed_input"] == 4827463680
assert network["receptive_field_feature_frames"] == 8057
assert spec["training"]["learning_rate"] == 0.0003
assert spec["training"]["lr_schedule"] == "cosine"
assert spec["training"]["min_learning_rate"] == 0.00003
assert "CTC logits" in spec["deployment"]["split"]["vpu"]
assert "prefix beam" in spec["deployment"]["split"]["cpu"]
PY_V16
grep -q 'V16_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v16 architecture registration missing'
grep -q '"cnn_ctc_v16": "train-cnn-ctc-v16.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v16 training registration missing'
grep -q '"cnn_ctc_v16": "evaluate-cnn-ctc-v16.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v16 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-5d1209f5120388f8"' examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py || fail 'cnn_ctc_v16 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "70320bc878891f843312f0d8c1ae41ad7f414dbe2a0537e6766af8bf15f70a93"' examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py || fail 'cnn_ctc_v16 model identity changed'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py || fail 'cnn_ctc_v16 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py || fail 'cnn_ctc_v16 latency must be measured, not gated'
grep -q 'groups=in_channels' examples/speech-asr/training/cnn_ctc_v16.py || fail 'cnn_ctc_v16 depthwise convolution missing'
grep -q 'TimeChannelSeparableConv1d' examples/speech-asr/training/cnn_ctc_v16.py || fail 'cnn_ctc_v16 TCS implementation missing'
grep -q 'CosineAnnealingLR' examples/speech-asr/training/train_cnn_ctc_v16.py || fail 'cnn_ctc_v16 cosine screen schedule missing'
grep -q 'MAX_SERVER_RESTARTS = 4' examples/speech-asr/evaluation/evaluate_cnn_ctc_v16.py || fail 'cnn_ctc_v16 persistent-server restart guard missing'

# cnn_ctc_v17 QuartzNet reference-training alignment.
for f in \
    examples/speech-asr/models/cnn_ctc_v17/model_spec.json \
    examples/speech-asr/models/cnn_ctc_v17/vocab.json \
    examples/speech-asr/models/cnn_ctc_v17/README.md \
    examples/speech-asr/training/cnn_ctc_v17.py \
    examples/speech-asr/training/novograd.py \
    examples/speech-asr/training/train_cnn_ctc_v17.py \
    examples/speech-asr/training/export_cnn_ctc_v17.py \
    examples/speech-asr/evaluation/compare_cnn_ctc_v17_onnx.py \
    examples/speech-asr/evaluation/evaluate_cnn_ctc_v17.py \
    examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py \
    examples/speech-asr/docs/adr/model-quality-v4-cnn-ctc-v17-quartznet-reference-training.md \
    tests/python/test_speech_asr_cnn_ctc_v17.py; do
    test -f "$f" || fail "cnn_ctc_v17 file missing: $f"
done
for f in \
    scripts/init-cnn-ctc-v17-quartznet-reference-training.sh \
    scripts/train-cnn-ctc-v17.sh \
    scripts/probe-cnn-ctc-v17.sh \
    scripts/prepare-cnn-ctc-v17.sh \
    scripts/evaluate-cnn-ctc-v17.sh; do
    test -x "$f" || fail "cnn_ctc_v17 shell entry point is not executable: $f"
done
python3 - <<'PY_V17'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v17/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "766a3851160bc4d6d1a5be8f27d45066accce9391683efc99ea5e9833dfb0fa1"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,256,39]
network = spec["network"]
assert network["kind"] == "quartznet-15x5-v1"
assert network["normalization"] == "batchnorm"
assert network["dropout"] == 0
assert network["separable_convolution"] == "depthwise-pointwise"
assert [g["channels"] for g in network["block_groups"]] == [256,256,512,512,512]
assert [g["kernel"] for g in network["block_groups"]] == [33,39,51,63,75]
assert all(g["block_repeats"] == 3 for g in network["block_groups"])
assert all(g["module_repeats"] == 5 for g in network["block_groups"])
assert network["estimated_parameters"] == 18934631
assert network["estimated_macs_fixed_input"] == 4827463680
assert network["receptive_field_feature_frames"] == 8057
training = spec["training"]
assert training["optimizer"] == "novograd"
assert training["learning_rate"] == 0.01
assert training["optimizer_betas"] == [0.8,0.5]
assert training["optimizer_eps"] == 1e-8
assert training["weight_decay"] == 0.001
assert training["lr_schedule"] == "warmup_cosine"
assert training["warmup_ratio"] == 0.12
assert training["min_learning_rate"] == 0.00001
PY_V17
grep -q 'V17_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v17 architecture registration missing'
grep -q '"cnn_ctc_v17": "train-cnn-ctc-v17.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v17 training registration missing'
grep -q '"cnn_ctc_v17": "evaluate-cnn-ctc-v17.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v17 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-4d7f92c301064c45"' examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py || fail 'cnn_ctc_v17 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "766a3851160bc4d6d1a5be8f27d45066accce9391683efc99ea5e9833dfb0fa1"' examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py || fail 'cnn_ctc_v17 model identity changed'
grep -q 'class NovoGrad' examples/speech-asr/training/novograd.py || fail 'cnn_ctc_v17 NovoGrad implementation missing'
grep -q 'warmup_cosine_learning_rate' examples/speech-asr/training/train_cnn_ctc_v17.py || fail 'cnn_ctc_v17 warmup cosine schedule missing'
grep -q 'MAX_SERVER_RESTARTS = 4' examples/speech-asr/evaluation/evaluate_cnn_ctc_v17.py || fail 'cnn_ctc_v17 persistent-server restart guard missing'
grep -q -- '--resume-nonterminal' examples/speech-asr/tools/manage_experiment.py || fail 'experiment manager nonterminal resume flag missing'
grep -q -- '"--resume-nonterminal"' examples/speech-asr/agent/run_experiment.py || fail 'experiment runner does not request nonterminal resume'
grep -q 'load_recorded_local_execution' examples/speech-asr/agent/run_experiment.py || fail 'experiment runner local resume helper missing'
grep -q 'load_recorded_remote_execution' examples/speech-asr/agent/run_experiment.py || fail 'experiment runner remote resume helper missing'
grep -q '"max_cer": None' examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py || fail 'cnn_ctc_v17 CER must be measured, not gated'
grep -q '"max_latency_p95_ms": None' examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py || fail 'cnn_ctc_v17 latency must be measured, not gated'

# cnn_ctc_v18 pretrained QuartzNet transfer.
for f in \
    scripts/prepare-cnn-ctc-v18-pretrained.sh \
    scripts/init-cnn-ctc-v18-pretrained-transfer.sh \
    scripts/train-cnn-ctc-v18.sh \
    scripts/probe-cnn-ctc-v18.sh \
    scripts/prepare-cnn-ctc-v18.sh \
    scripts/evaluate-cnn-ctc-v18.sh; do
    test -x "$f" || fail "cnn_ctc_v18 shell entry point is not executable: $f"
done
python3 - <<'PY_V18'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v18/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "008416f73aa2f90e1eb7b60cb2021175cbc0826b09810aed9aebbe41fe6f422c"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,256,39]
frontend = spec["frontend"]
assert frontend["kind"] == "logmel-v3"
assert frontend["window_samples"] == 320
assert frontend["preemphasis"] == 0.97
assert frontend["mel_scale"] == "slaney"
assert frontend["mel_norm"] == "slaney"
assert frontend["normalization"] == "per_mel_bin_mean_std_valid_zero_pad"
network = spec["network"]
assert network["kind"] == "quartznet-15x5-v1"
assert network["batchnorm_eps"] == 0.001
assert network["dropout"] == 0
assert network["estimated_parameters"] == 18934631
assert network["estimated_macs_fixed_input"] == 4827463680
transfer = spec["transfer"]
assert transfer["source_size_bytes"] == 71083664
assert transfer["source_sha384"] == "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55"
assert len(transfer["source_vocab"]) == 29
assert transfer["fine_tune"] == "all_parameters"
training = spec["training"]
assert training["optimizer"] == "novograd"
assert training["learning_rate"] == 0.001
assert training["optimizer_betas"] == [0.95,0.25]
assert training["weight_decay"] == 0.001
assert training["warmup_ratio"] == 0.12
assert training["min_learning_rate"] == 0.000001
assert training["checkpoint_selection"] == "best_validation_cer"
PY_V18
grep -q 'V18_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v18 architecture registration missing'
grep -q '"cnn_ctc_v18": "train-cnn-ctc-v18.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v18 training registration missing'
grep -q '"cnn_ctc_v18": "evaluate-cnn-ctc-v18.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v18 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-dbcda9a6f7ae7d7f"' examples/speech-asr/agent/init_cnn_ctc_v18_pretrained_transfer.py || fail 'cnn_ctc_v18 parent changed'
grep -q 'PRETRAINED_SHA384' examples/speech-asr/agent/init_cnn_ctc_v18_pretrained_transfer.py || fail 'cnn_ctc_v18 source hash pin missing'
grep -q 'load_pretrained_quartznet' examples/speech-asr/training/train_cnn_ctc_v18.py || fail 'cnn_ctc_v18 trainer does not require pretrained import'
test -f examples/speech-asr/training/verify_cnn_ctc_v18_pretrained.py || fail 'cnn_ctc_v18 pretrained import verifier missing'
grep -q 'shared_decoder_symbol_count' examples/speech-asr/training/verify_cnn_ctc_v18_pretrained.py || fail 'cnn_ctc_v18 pretrained import symbol check missing'
grep -q 'encoder_source_tensors_used' examples/speech-asr/training/verify_cnn_ctc_v18_pretrained.py || fail 'cnn_ctc_v18 pretrained import encoder check missing'
grep -q 'verify_cnn_ctc_v18_pretrained.py' scripts/prepare-cnn-ctc-v18-pretrained.sh || fail 'cnn_ctc_v18 source preparation does not verify tensor import'
grep -q 'PRETRAINED_IMPORT' examples/speech-asr/agent/init_cnn_ctc_v18_pretrained_transfer.py || fail 'cnn_ctc_v18 initializer does not require import verification'
grep -q '"pretrained_initialization"' examples/speech-asr/training/train_cnn_ctc_v18.py || fail 'cnn_ctc_v18 pretrained evidence missing'
grep -q '"initial_validation"' examples/speech-asr/training/train_cnn_ctc_v18.py || fail 'cnn_ctc_v18 zero-step validation evidence missing'
grep -q 'MAX_SERVER_RESTARTS = 4' examples/speech-asr/evaluation/evaluate_cnn_ctc_v18.py || fail 'cnn_ctc_v18 persistent-server restart guard missing'

# cnn_ctc_v19 conservative pretrained fine-tuning.
for f in \
    scripts/prepare-cnn-ctc-v19-pretrained.sh \
    scripts/init-cnn-ctc-v19-conservative-transfer.sh \
    scripts/train-cnn-ctc-v19.sh \
    scripts/probe-cnn-ctc-v19.sh \
    scripts/prepare-cnn-ctc-v19.sh \
    scripts/evaluate-cnn-ctc-v19.sh; do
    test -x "$f" || fail "cnn_ctc_v19 shell entry point is not executable: $f"
done
python3 - <<'PY_V19'
import hashlib
import json
from pathlib import Path
spec = json.loads(Path("examples/speech-asr/models/cnn_ctc_v19/model_spec.json").read_text())
canonical = hashlib.sha256(
    json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
assert canonical == "e8fceef8e30955782efeab6e732319c051998e061f58502bc66d39f71485d330"
assert spec["input_contract"]["shape"] == [1,64,512]
assert spec["output_contract"]["shape"] == [1,256,39]
assert spec["frontend"]["kind"] == "logmel-v3"
network = spec["network"]
assert network["kind"] == "quartznet-15x5-v1"
assert network["batchnorm_eps"] == 0.001
assert network["dropout"] == 0
assert network["estimated_parameters"] == 18934631
assert network["estimated_macs_fixed_input"] == 4827463680
training = spec["training"]
assert training["batch_size"] == 8
assert training["learning_rate"] == 0.0001
assert training["optimizer"] == "novograd"
assert training["optimizer_betas"] == [0.95,0.25]
assert training["weight_decay"] == 0.001
assert training["lr_schedule"] == "cosine"
assert training["warmup_ratio"] == 0
assert training["min_learning_rate"] == 0.000001
assert training["freeze_batchnorm_running_stats"] is True
assert training["initial_validation_checkpoint_candidate"] is True
assert training["checkpoint_selection"] == "best_validation_cer"
PY_V19
grep -q 'V19_ARCHITECTURE' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v19 architecture registration missing'
grep -q '"cnn_ctc_v19": "train-cnn-ctc-v19.sh"' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v19 training registration missing'
grep -q '"cnn_ctc_v19": "evaluate-cnn-ctc-v19.sh"' examples/speech-asr/agent/edge_worker.py || fail 'cnn_ctc_v19 edge evaluator registration missing'
grep -q 'DEFAULT_PARENT = "exp-7c7ac81bc57f8f3b"' examples/speech-asr/agent/init_cnn_ctc_v19_conservative_transfer.py || fail 'cnn_ctc_v19 parent changed'
grep -q 'MODEL_SPEC_SHA256 = "e8fceef8e30955782efeab6e732319c051998e061f58502bc66d39f71485d330"' examples/speech-asr/agent/init_cnn_ctc_v19_conservative_transfer.py || fail 'cnn_ctc_v19 model identity changed'
grep -q 'V18_INITIAL_VALIDATION_CER = 0.5776651500316765' examples/speech-asr/agent/init_cnn_ctc_v19_conservative_transfer.py || fail 'cnn_ctc_v19 parent epoch-zero evidence changed'
test -f tests/python/test_speech_asr_cnn_ctc_v19.py || fail 'cnn_ctc_v19 unit test missing'
grep -q 'def freeze_batchnorm_running_stats' examples/speech-asr/training/train_cnn_ctc_v19.py || fail 'cnn_ctc_v19 BN-stat freeze helper missing'
grep -q 'if warmup_ratio == 0' examples/speech-asr/training/train_cnn_ctc_v19.py || fail 'cnn_ctc_v19 cosine schedule must be warmup-free'
grep -q 'frozen_batchnorm_modules = freeze_batchnorm_running_stats(model)' examples/speech-asr/training/train_cnn_ctc_v19.py || fail 'cnn_ctc_v19 does not freeze BN stats after train mode'
grep -q 'best_epoch = 0' examples/speech-asr/training/train_cnn_ctc_v19.py || fail 'cnn_ctc_v19 epoch-zero checkpoint candidate missing'
grep -q 'persistent MYRIAD server input pipe write failed:' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 pipe-write restart normalization missing'
grep -q 'persistent MYRIAD server output pipe read failed:' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 pipe-read restart normalization missing'
grep -q '"--progress-interval"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 evaluation progress option missing'
grep -q 'default=25' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 evaluation progress cadence changed'
grep -q 'f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 evaluation ETA missing'
grep -q 'def write_f32_file_exact' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'cnn_ctc_v19 exact tensor cache writer missing'
if grep -q '\.tofile(' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py; then fail 'cnn_ctc_v19 evaluator must not use NumPy tofile'; fi
grep -q 'stream_output=True' examples/speech-asr/agent/run_experiment.py || fail 'edge worker progress is buffered by controller'
grep -q '"initial_validation_checkpoint_candidate": True' examples/speech-asr/training/train_cnn_ctc_v19.py || fail 'cnn_ctc_v19 epoch-zero evidence missing'
grep -q 'model_id in {"cnn_ctc_v18", "cnn_ctc_v19"}' examples/speech-asr/python/speech_asr/orchestration.py || fail 'cnn_ctc_v19 pretrained semantic MYRIAD gate missing'
test -x scripts/tune-cnn-ctc-v19-decoder.sh || fail 'cnn_ctc_v19 decoder sweep entry point missing'
test -x scripts/test-speech-asr-ctc-beam.sh || fail 'CTC beam uv/venv test entry point missing'
grep -q 'scripts/python-training.sh' scripts/test-speech-asr-ctc-beam.sh || fail 'CTC beam tests must use uv-managed training venv'
grep -q 'OMP_NUM_THREADS.*:-1' scripts/tune-cnn-ctc-v19-decoder.sh || fail 'decoder sweep must default OMP workers to one thread'
grep -q 'MKL_NUM_THREADS.*:-1' scripts/tune-cnn-ctc-v19-decoder.sh || fail 'decoder sweep must default MKL workers to one thread'
grep -q 'OPENBLAS_NUM_THREADS.*:-1' scripts/tune-cnn-ctc-v19-decoder.sh || fail 'decoder sweep must default OpenBLAS workers to one thread'
test -f tests/python/test_speech_asr_ctc_beam.py || fail 'cnn_ctc_v19 decoder unit test missing'
grep -q 'def prefix_beam_decode' examples/speech-asr/python/speech_asr/ctc_beam.py || fail 'CTC prefix beam decoder missing'
grep -q 'class CharacterNgramLM' examples/speech-asr/python/speech_asr/ctc_beam.py || fail 'character n-gram LM missing'
grep -q 'cache_dir = attempt / "edge" / "evaluation"' examples/speech-asr/evaluation/tune_cnn_ctc_v19_decoder.py || fail 'decoder sweep must use pulled MYRIAD logits'
grep -q '"cached greedy WER' examples/speech-asr/evaluation/tune_cnn_ctc_v19_decoder.py || fail 'decoder sweep greedy WER parity check missing'
grep -q '"schema": "speech-asr/ctc-decoder-sweep"' examples/speech-asr/evaluation/tune_cnn_ctc_v19_decoder.py || fail 'decoder sweep result schema missing'
test -x scripts/freeze-cnn-ctc-v19-decoder.sh || fail 'v19 decoder artifact freezer missing'
test -x scripts/decode-cnn-ctc-v19-logits.sh || fail 'v19 runtime decoder entry point missing'
test -x scripts/benchmark-cnn-ctc-v19-decoder.sh || fail 'v19 CPU decoder benchmark entry point missing'
test -x scripts/evaluate-cnn-ctc-v19-deployed.sh || fail 'v19 deployed decoder evaluator entry point missing'
test -x scripts/run-cnn-ctc-v19-deployed-edge.sh || fail 'v19 deployed decoder edge controller entry point missing'
test -x scripts/run-myriad-tensor.sh || fail 'shared MYRIAD tensor launcher missing'
grep -q 'runtime_exec_openvino' scripts/run-myriad-tensor.sh || fail 'MYRIAD tensor launcher lacks host runtime support'
grep -q 'DOCKER_READY' scripts/run-myriad-tensor.sh || fail 'MYRIAD tensor launcher lacks Docker fallback'
test -f examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder edge controller missing'
grep -q 'flock -n 9' scripts/evaluate-cnn-ctc-v19-deployed.sh || fail 'v19 deployed decoder evaluator lacks exclusive MYRIAD lock'
grep -q 'fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088' scripts/evaluate-cnn-ctc-v19-deployed.sh || fail 'v19 deployed decoder evaluator validation identity changed'
grep -q 'stage_validation_dataset' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller does not stage validation data'
grep -q 'os.link(audio, destination)' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder validation staging is not hard-link first'
grep -q 'edge-speech-preflight.sh' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller lacks edge runtime preflight'
grep -q 'RUNTIME_BACKEND="host"' scripts/evaluate-cnn-ctc-v19-deployed.sh || fail 'v19 deployed decoder evaluator is not pinned to host runtime'
grep -q '"host"' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller is not pinned to host runtime'
grep -q -- '"--preflight-only"' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller lacks dataset/audio preflight'
grep -q 'rsync_push_command' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller does not stage artifacts'
grep -q 'rsync_pull_command' examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py || fail 'v19 deployed decoder controller does not collect evidence'
grep -q 'def build_decoder_artifact' examples/speech-asr/python/speech_asr/ctc_beam.py || fail 'v19 decoder artifact builder missing'
grep -q 'def decode_with_artifact' examples/speech-asr/python/speech_asr/ctc_beam.py || fail 'v19 runtime artifact decode missing'
grep -q 'class FrozenCtcDecoder' examples/speech-asr/python/speech_asr/ctc_beam.py || fail 'v19 reusable runtime decoder missing'
grep -q 'FrozenCtcDecoder.from_artifact' examples/speech-asr/evaluation/benchmark_cnn_ctc_v19_decoder.py || fail 'v19 decoder benchmark must reuse materialized LM'
grep -q '0.5497732671129346' examples/speech-asr/tools/freeze_cnn_ctc_v19_decoder.py || fail 'v19 selected decoder WER evidence changed'
grep -q '0.4634567759027818' examples/speech-asr/tools/freeze_cnn_ctc_v19_decoder.py || fail 'v19 selected decoder CER evidence changed'
grep -q '"--decoder-artifact"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'v19 physical evaluator decoder artifact option missing'
grep -q '"--runtime-backend"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'v19 physical evaluator runtime backend option missing'
grep -q '"launcher_backend"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'v19 physical evaluator runtime backend provenance missing'
grep -q '"acoustic_plus_decoder_latency_p95_ms"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'v19 physical evaluator paired acoustic+decoder latency missing'
grep -q '"acoustic_plus_decoder_realtime_factor"' examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py || fail 'v19 physical evaluator paired acoustic+decoder RTF missing'

# Zero-training NVIDIA QuartzNet / LibriSpeech reference qualification.
test -f examples/speech-asr/models/quartznet15x5_nvidia_ref/model_spec.json || fail 'QuartzNet pretrained reference spec missing'
test -f examples/speech-asr/models/quartznet15x5_nvidia_ref/vocab.json || fail 'QuartzNet pretrained reference vocab missing'
test -x scripts/prepare-librispeech-dev-clean.sh || fail 'LibriSpeech dev-clean preparer missing'
test -x scripts/evaluate-quartznet15x5-reference.sh || fail 'QuartzNet pretrained reference evaluator missing'
test -x scripts/export-quartznet15x5-reference.sh || fail 'QuartzNet pretrained reference ONNX exporter missing'
test -x scripts/compare-quartznet15x5-reference-onnx.sh || fail 'QuartzNet pretrained reference ONNX parity tool missing'
test -x scripts/qualify-quartznet15x5-reference.sh || fail 'QuartzNet pretrained reference qualification entry point missing'
test -x scripts/test-speech-asr-quartznet-reference.sh || fail 'QuartzNet pretrained reference unit-test entry point missing'
grep -q '42e2234ba48799c1f50f24a7926300a1' examples/speech-asr/datasets/librispeech/prepare_dev_clean.py || fail 'LibriSpeech dev-clean archive identity changed'
grep -q '"published_dev_clean_wer": 0.0379' examples/speech-asr/models/quartznet15x5_nvidia_ref/model_spec.json || fail 'published QuartzNet dev-clean WER target changed'
grep -q '"classes": 29' examples/speech-asr/models/quartznet15x5_nvidia_ref/model_spec.json || fail 'QuartzNet reference output width changed'
grep -q '"blank_index": 28' examples/speech-asr/models/quartznet15x5_nvidia_ref/vocab.json || fail 'QuartzNet reference blank index changed'
grep -q 'valid_frames = sample_count // hop' examples/speech-asr/training/quartznet15x5_reference.py || fail 'historical NeMo valid-frame rule changed'
grep -q 'periodic=bool(frontend\["hann_periodic"\])' examples/speech-asr/training/quartznet15x5_reference.py || fail 'historical NeMo Hann window semantics changed'
grep -q 'float(valid_frames - 1)' examples/speech-asr/training/quartznet15x5_reference.py || fail 'historical NeMo ddof=1 normalization changed'
if grep -Eq 'run-myriad-tensor\.sh|evaluate-cnn-ctc-v19-deployed\.sh|run-speech-experiment\.sh' scripts/qualify-quartznet15x5-reference.sh; then
    fail 'QuartzNet reference qualification must remain hardware-free'
fi
grep -q 'eval_status=\$?' scripts/qualify-quartznet15x5-reference.sh || fail 'QuartzNet reference runner must preserve WER gate status'
grep -q 'compare-quartznet15x5-reference-onnx.sh' scripts/qualify-quartznet15x5-reference.sh || fail 'QuartzNet reference runner must collect ONNX evidence after WER evaluation'

# Qualified clean-speech QuartzNet -> OpenVINO FP16 -> MYRIAD path.
test -f examples/speech-asr/python/speech_asr/quartznet_reference_frontend.py || fail 'QuartzNet edge-safe reference frontend missing'
test -f examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD reference evaluator missing'
test -f examples/speech-asr/evaluation/run_quartznet15x5_reference_myriad_edge.py || fail 'QuartzNet MYRIAD edge controller missing'
test -x scripts/qualify-quartznet15x5-reference-numpy.sh || fail 'QuartzNet NumPy reference qualification missing'
test -x scripts/prepare-quartznet15x5-reference-myriad.sh || fail 'QuartzNet MYRIAD artifact preparer missing'
test -x scripts/evaluate-quartznet15x5-reference-myriad.sh || fail 'QuartzNet Pi MYRIAD evaluator missing'
test -x scripts/run-quartznet15x5-reference-myriad-edge.sh || fail 'QuartzNet MYRIAD edge controller entry point missing'
test -x scripts/diagnose-quartznet15x5-reference-myriad-static.sh || fail 'QuartzNet static-shape MYRIAD diagnostic missing'
test -x scripts/analyze-quartznet15x5-reference-myriad-lengths.sh || fail 'QuartzNet MYRIAD length analyzer missing'
test -x scripts/test-speech-asr-quartznet-reference-myriad.sh || fail 'QuartzNet MYRIAD reference tests missing'
grep -q 'EXPECTED_SAMPLES = 2703' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD dev-clean sample boundary changed'
grep -q 'QUALIFIED_WER = 0.037939781625675524' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD source WER evidence changed'
grep -q 'MAX_WER = 0.05' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD WER ceiling changed'
grep -q 'MAX_ABS_WER_DELTA = 0.005' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD WER-delta gate changed'
grep -q 'DIAGNOSTIC_MAX_FRAME_TOTAL_VARIATION = 0.002' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD probability-drift diagnostic reference changed'
grep -q -- '"--tensor-frames"' examples/speech-asr/evaluation/diagnose_quartznet15x5_reference_myriad_static.py || fail 'QuartzNet static diagnostic lacks exact tensor-frame selection'
grep -q -- '"--force-tensor-frames"' examples/speech-asr/evaluation/diagnose_quartznet15x5_reference_myriad_static.py || fail 'QuartzNet static diagnostic lacks same-sample shape isolation'
grep -q 'MYRIAD_CATASTROPHIC_ONSET_FRAMES = 3104' examples/speech-asr/evaluation/analyze_quartznet15x5_reference_myriad_lengths.py || fail 'QuartzNet MYRIAD catastrophic boundary attribution changed'
grep -q 'MYRIAD_LAST_NONCATASTROPHIC_FRAMES = 3088' examples/speech-asr/evaluation/analyze_quartznet15x5_reference_myriad_lengths.py || fail 'QuartzNet MYRIAD last non-catastrophic boundary attribution changed'
grep -q '"gating": False' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD probability drift unexpectedly became a hard gate'
grep -q 'exact-time-runtime-reshape-v1' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD exact-time execution policy changed'
grep -q -- '"--reshape-time"' examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py || fail 'QuartzNet MYRIAD evaluator does not request exact time reshape'
grep -q -- '--backend host' scripts/evaluate-quartznet15x5-reference-myriad.sh || fail 'QuartzNet MYRIAD deployment is not pinned to host runtime'
grep -q 'check-reshape' scripts/evaluate-quartznet15x5-reference-myriad.sh || fail 'QuartzNet MYRIAD deployment does not require reshape-capable runtime'
grep -q 'requestedDevice.usesMyriad && !myriadAvailable' smoke-test/main.cpp || fail 'hello_myriad must allow CPU-only inference without a MYRIAD device'
grep -q 'item.second->setPrecision(Precision::FP32)' smoke-test/main.cpp || fail 'hello_myriad CPU path must keep FP32 input precision'
grep -q 'unsupported input blob width' smoke-test/main.cpp || fail 'hello_myriad tensor loader must support FP16 and FP32 input blobs'
grep -q -- '"--refresh-runtime"' examples/speech-asr/evaluation/run_quartznet15x5_reference_myriad_edge.py || fail 'QuartzNet edge controller lacks runtime refresh path'
grep -q 'rsync_push_command' examples/speech-asr/evaluation/run_quartznet15x5_reference_myriad_edge.py || fail 'QuartzNet edge controller does not stage dataset/artifacts'
grep -q 'rsync_pull_command' examples/speech-asr/evaluation/run_quartznet15x5_reference_myriad_edge.py || fail 'QuartzNet edge controller does not collect evidence'

# CPU/CUDA-only AMI adaptation attribution after source-domain qualification.
test -x scripts/evaluate-quartznet15x5-ami-attribution.sh || fail 'QuartzNet AMI attribution entry point missing'
test -x scripts/test-speech-asr-quartznet-ami-attribution.sh || fail 'QuartzNet AMI attribution unit-test entry point missing'
test -f examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution evaluator missing'
grep -q 'fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution validation identity changed'
grep -q 'EXPECTED_SAMPLES = 1273' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution sample count changed'
grep -q 's0-source-head-source-frontend' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution source stage missing'
grep -q 's1-source-head-fixed-frontend' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution frontend stage missing'
grep -q 's2-ami-head-fixed-frontend-epoch0' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution head stage missing'
grep -q 's3-v19-finetuned' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI attribution fine-tuned stage missing'
grep -q 'EXPECTED_EPOCH0_WER = 0.7393651479162168' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI epoch-zero evidence changed'
grep -q 'EXPECTED_FINETUNED_WER = 0.5983588857698121' examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py || fail 'QuartzNet AMI fine-tuned evidence changed'
if grep -Eqi 'run-myriad-tensor\.sh|run-speech-experiment\.sh|edge-speech|docker run|ssh ' scripts/evaluate-quartznet15x5-ami-attribution.sh; then
    fail 'QuartzNet AMI attribution runner must remain hardware-free'
fi

python3 - <<'PY_CHECK'
from pathlib import Path
for name in [
    "smoke-test/model/make_tiny_ir.py",
    "scripts/reset-stick.py",
    "examples/speech-asr/python/speech_asr/streaming.py",
    "examples/speech-asr/evaluation/replay_streaming.py",
    "examples/speech-asr/tools/make_streaming_updates.py",
    "examples/speech-asr/python/speech_asr/contracts.py",
    "examples/speech-asr/python/speech_asr/cnn_ctc.py",
    "examples/speech-asr/python/speech_asr/cnn_ctc_frontend.py",
    "examples/speech-asr/python/speech_asr/ctc_beam.py",
    "examples/speech-asr/python/speech_asr/cnn_ctc_compare.py",
    "examples/speech-asr/python/speech_asr/quartznet_reference_frontend.py",
    "examples/speech-asr/python/speech_asr/experiment.py",
    "examples/speech-asr/python/speech_asr/orchestration.py",
    "examples/speech-asr/agent/init_cnn_ctc_v1_experiment.py",
    "examples/speech-asr/agent/init_cnn_ctc_v2_experiment.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_experiment.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_quality_baseline.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_cer_selection.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_specaugment.py",
    "examples/speech-asr/agent/init_cnn_ctc_v4_valid_cmvn.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_blank_penalty.py",
    "examples/speech-asr/agent/init_cnn_ctc_v5_interctc.py",
    "examples/speech-asr/agent/init_cnn_ctc_v6_interctc.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_expanded_data.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_scaled_data.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_model_quality_v4.py",
    "examples/speech-asr/agent/init_cnn_ctc_v7_wide.py",
    "examples/speech-asr/agent/init_cnn_ctc_v3_architecture_screen.py",
    "examples/speech-asr/agent/init_cnn_ctc_v8_architecture_screen.py",
    "examples/speech-asr/agent/init_cnn_ctc_v9_architecture_screen.py",
    "examples/speech-asr/agent/init_cnn_ctc_v10_architecture_screen.py",
    "examples/speech-asr/agent/init_cnn_ctc_v11_architecture_screen.py",
    "examples/speech-asr/agent/init_cnn_ctc_v12_capacity_probe.py",
    "examples/speech-asr/agent/init_cnn_ctc_v13_stabilized_lr.py",
    "examples/speech-asr/agent/init_cnn_ctc_v14_cosine_control.py",
    "examples/speech-asr/agent/init_cnn_ctc_v15_batchnorm_conditioning.py",
    "examples/speech-asr/agent/init_cnn_ctc_v16_quartznet.py",
    "examples/speech-asr/agent/init_cnn_ctc_v17_quartznet_reference_training.py",
    "examples/speech-asr/agent/init_cnn_ctc_v18_pretrained_transfer.py",
    "examples/speech-asr/agent/init_cnn_ctc_v19_conservative_transfer.py",
    "examples/speech-asr/agent/run_frozen_heldout_evaluation.py",
    "examples/speech-asr/agent/review_heldout_evaluation.py",
    "examples/speech-asr/agent/review_experiment.py",
    "examples/speech-asr/agent/run_experiment.py",
    "examples/speech-asr/agent/edge_worker.py",
    "examples/speech-asr/tools/manage_experiment.py",
    "examples/speech-asr/tools/validate_contract.py",
    "examples/speech-asr/tools/qualify_model_quality_manifests.py",
    "examples/speech-asr/tools/qualify_expanded_model_quality_manifests.py",
    "examples/speech-asr/tools/qualify_heldout_evaluation_manifest.py",
    "examples/speech-asr/tools/qualify_model_quality_v4.py",
    "examples/speech-asr/tools/prepare_architecture_screen_v1.py",
    "examples/speech-asr/datasets/ami/freeze_model_quality_v4_sources.py",
    "examples/speech-asr/training/cnn_ctc_v1.py",
    "examples/speech-asr/training/train_cnn_ctc_v1.py",
    "examples/speech-asr/training/export_cnn_ctc_v1.py",
    "examples/speech-asr/training/cnn_ctc_v2.py",
    "examples/speech-asr/training/train_cnn_ctc_v2.py",
    "examples/speech-asr/training/export_cnn_ctc_v2.py",
    "examples/speech-asr/training/cnn_ctc_v3.py",
    "examples/speech-asr/training/train_cnn_ctc_v3.py",
    "examples/speech-asr/training/export_cnn_ctc_v3.py",
    "examples/speech-asr/training/cnn_ctc_v5.py",
    "examples/speech-asr/training/train_cnn_ctc_v5.py",
    "examples/speech-asr/training/export_cnn_ctc_v5.py",
    "examples/speech-asr/training/cnn_ctc_v6.py",
    "examples/speech-asr/training/train_cnn_ctc_v6.py",
    "examples/speech-asr/training/export_cnn_ctc_v6.py",
    "examples/speech-asr/training/cnn_ctc_v7.py",
    "examples/speech-asr/training/train_cnn_ctc_v7.py",
    "examples/speech-asr/training/export_cnn_ctc_v7.py",
    "examples/speech-asr/training/cnn_ctc_v8.py",
    "examples/speech-asr/training/train_cnn_ctc_v8.py",
    "examples/speech-asr/training/export_cnn_ctc_v8.py",
    "examples/speech-asr/training/cnn_ctc_v9.py",
    "examples/speech-asr/training/train_cnn_ctc_v9.py",
    "examples/speech-asr/training/export_cnn_ctc_v9.py",
    "examples/speech-asr/training/cnn_ctc_v10.py",
    "examples/speech-asr/training/cnn_ctc_v11.py",
    "examples/speech-asr/training/cnn_ctc_v12.py",
    "examples/speech-asr/training/cnn_ctc_v13.py",
    "examples/speech-asr/training/cnn_ctc_v14.py",
    "examples/speech-asr/training/cnn_ctc_v15.py",
    "examples/speech-asr/training/cnn_ctc_v16.py",
    "examples/speech-asr/training/cnn_ctc_v17.py",
    "examples/speech-asr/training/cnn_ctc_v18.py",
    "examples/speech-asr/training/cnn_ctc_v19.py",
    "examples/speech-asr/training/pretrained_cnn_ctc_v18.py",
    "examples/speech-asr/training/prepare_cnn_ctc_v18_pretrained.py",
    "examples/speech-asr/training/verify_cnn_ctc_v18_pretrained.py",
    "examples/speech-asr/training/quartznet15x5_reference.py",
    "examples/speech-asr/training/export_quartznet15x5_reference.py",
    "examples/speech-asr/datasets/librispeech/prepare_dev_clean.py",
    "examples/speech-asr/evaluation/evaluate_quartznet15x5_reference.py",
    "examples/speech-asr/evaluation/compare_quartznet15x5_reference_onnx.py",
    "examples/speech-asr/evaluation/evaluate_quartznet15x5_ami_attribution.py",
    "examples/speech-asr/evaluation/evaluate_quartznet15x5_reference_myriad.py",
    "examples/speech-asr/evaluation/run_quartznet15x5_reference_myriad_edge.py",
    "examples/speech-asr/evaluation/diagnose_quartznet15x5_reference_myriad_static.py",
    "examples/speech-asr/evaluation/analyze_quartznet15x5_reference_myriad_lengths.py",
    "tests/python/test_speech_asr_quartznet_reference.py",
    "tests/python/test_speech_asr_quartznet_reference_myriad.py",
    "tests/python/test_speech_asr_quartznet_ami_attribution.py",
    "examples/speech-asr/training/novograd.py",
    "examples/speech-asr/training/train_cnn_ctc_v10.py",
    "examples/speech-asr/training/train_cnn_ctc_v11.py",
    "examples/speech-asr/training/train_cnn_ctc_v12.py",
    "examples/speech-asr/training/train_cnn_ctc_v13.py",
    "examples/speech-asr/training/train_cnn_ctc_v14.py",
    "examples/speech-asr/training/train_cnn_ctc_v15.py",
    "examples/speech-asr/training/train_cnn_ctc_v16.py",
    "examples/speech-asr/training/train_cnn_ctc_v17.py",
    "examples/speech-asr/training/train_cnn_ctc_v18.py",
    "examples/speech-asr/training/train_cnn_ctc_v19.py",
    "examples/speech-asr/training/export_cnn_ctc_v10.py",
    "examples/speech-asr/training/export_cnn_ctc_v11.py",
    "examples/speech-asr/training/export_cnn_ctc_v12.py",
    "examples/speech-asr/training/export_cnn_ctc_v13.py",
    "examples/speech-asr/training/export_cnn_ctc_v14.py",
    "examples/speech-asr/training/export_cnn_ctc_v15.py",
    "examples/speech-asr/training/export_cnn_ctc_v16.py",
    "examples/speech-asr/training/export_cnn_ctc_v17.py",
    "examples/speech-asr/training/export_cnn_ctc_v18.py",
    "examples/speech-asr/training/export_cnn_ctc_v19.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v1_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v1_tensor.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v1.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v2_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v2.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v3_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py",
    "examples/speech-asr/training/export_cnn_ctc_v4.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v4_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v4.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v5_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v5.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v6_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v6.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v7_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v7.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v8_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v8.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v9_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v9.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v10_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v11_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v12_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v13_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v14_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v15_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v16_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v17_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v18_onnx.py",
    "examples/speech-asr/evaluation/compare_cnn_ctc_v19_onnx.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v10.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v11.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v12.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v13.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v14.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v15.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v16.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v17.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v18.py",
    "examples/speech-asr/evaluation/evaluate_cnn_ctc_v19.py",
    "examples/speech-asr/evaluation/tune_cnn_ctc_v19_decoder.py",
    "examples/speech-asr/evaluation/benchmark_cnn_ctc_v19_decoder.py",
    "examples/speech-asr/evaluation/run_cnn_ctc_v19_deployed_edge.py",
    "examples/speech-asr/runtime/decode_cnn_ctc_v19_logits.py",
    "examples/speech-asr/tools/freeze_cnn_ctc_v19_decoder.py",
    "examples/speech-asr/evaluation/evaluate_frozen_cnn_ctc_v3_reference.py",
]:
    src = Path(name).read_text()
    compile(src, name, "exec")
    print(f"python syntax: {name}: OK")
PY_CHECK

echo 'static multi-platform checks: PASS'
