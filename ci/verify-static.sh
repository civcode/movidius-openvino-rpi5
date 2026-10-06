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
grep -q 'pretraining_myriad_compatibility' examples/speech-asr/agent/run_experiment.py || fail 'controller lacks numerical pretraining MYRIAD gate'
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
grep -q 'edge-speech-bootstrap.sh' examples/speech-asr/agent/run_frozen_heldout_evaluation.py || fail 'held-out controller does not pin edge revision'
grep -q 'frame_argmax_agreement' examples/speech-asr/evaluation/evaluate_frozen_cnn_ctc_v3_reference.py || fail 'held-out reference lacks PyTorch/ONNX agreement gate'
grep -q 'persistent-tensor-stream-v2' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator lacks persistent execution contract'
grep -q 'class PersistentMyriadServer' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out hardware evaluator lacks persistent server client'
grep -q 'CACHE_VERSION = 3' examples/speech-asr/evaluation/evaluate_cnn_ctc_v3.py || fail 'held-out persistent cache version changed'
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
    "examples/speech-asr/python/speech_asr/cnn_ctc_compare.py",
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
    "examples/speech-asr/evaluation/evaluate_frozen_cnn_ctc_v3_reference.py",
]:
    src = Path(name).read_text()
    compile(src, name, "exec")
    print(f"python syntax: {name}: OK")
PY_CHECK

echo 'static multi-platform checks: PASS'
