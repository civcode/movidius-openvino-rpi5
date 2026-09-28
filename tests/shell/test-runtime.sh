#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/runtime.sh
source "$ROOT/scripts/lib/runtime.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
eq() { [[ "$1" == "$2" ]] || fail "expected '$2', got '$1'"; }

runtime_validate_backend auto
runtime_validate_backend host
runtime_validate_backend docker
if runtime_validate_backend nonsense >/dev/null 2>&1; then
    fail "invalid backend was accepted"
fi

eq "$(runtime_resolve_backend auto 1 missing-image)" host
eq "$(runtime_resolve_backend host 1 missing-image)" host

# Mock Docker readiness so the test is device/daemon independent. This is the
# ARM64 CPU-auto behavior in miniature: no usable host Python backend, but a
# usable Docker image must resolve to docker instead of failing early.
runtime_docker_image_ready() { [[ "$1" == fake-image ]]; }
eq "$(runtime_resolve_backend auto 0 fake-image)" docker
eq "$(runtime_resolve_backend docker 0 fake-image)" docker
if runtime_resolve_backend host 0 fake-image >/dev/null 2>&1; then
    fail "host was selected without host prerequisites"
fi

runtime_device_uses MYRIAD MYRIAD || fail "MYRIAD not detected"
runtime_device_uses HETERO:MYRIAD MYRIAD || fail "HETERO:MYRIAD not detected"
runtime_device_uses HETERO:CPU,MYRIAD CPU || fail "CPU not detected in composite"
runtime_device_uses HETERO:CPU,MYRIAD MYRIAD || fail "MYRIAD not detected in composite"
if runtime_device_uses MYRIAD CPU; then fail "CPU falsely detected"; fi

bash "$ROOT/scripts/run-inference-server.sh" --help >/dev/null
if bash "$ROOT/scripts/run-inference-server.sh" --app mobilenet --backend nonsense >/dev/null 2>&1; then
    fail "common launcher accepted invalid backend"
fi

echo "runtime shell tests: PASS"

tmp_runtime="$(mktemp -d)"
trap 'rm -rf "$tmp_runtime"' EXIT
mkdir -p "$tmp_runtime/openvino/inference_engine/lib/test" "$tmp_runtime/openvino/ngraph/lib"
: > "$tmp_runtime/openvino/inference_engine/lib/test/libinference_engine.so"
cat > "$tmp_runtime/probe.sh" <<'EOF'
#!/usr/bin/env bash
echo runtime-run-ok
EOF
chmod +x "$tmp_runtime/probe.sh"
eq "$(runtime_run_openvino amd64 "$tmp_runtime" "$tmp_runtime/openvino" "$tmp_runtime/probe.sh")" runtime-run-ok
echo after-runtime-run >/dev/null
