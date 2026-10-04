#!/usr/bin/env bash
# Shared source-tree runtime helpers. Safe to source: no set -e and no actions.

runtime_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

runtime_validate_backend() {
    case "${1:-}" in
        auto|host|docker) return 0 ;;
        *) echo "invalid backend '${1:-}'; expected auto|host|docker" >&2; return 2 ;;
    esac
}

runtime_target_can_run_host() {
    local target="$1" machine
    machine="$(uname -m)"
    case "${target}:${machine}" in
        armv7:armv7l|armv7:armv7*|armv7:aarch64|armv7:arm64|arm64:aarch64|arm64:arm64|amd64:x86_64|amd64:amd64)
            return 0 ;;
        *) return 1 ;;
    esac
}

runtime_find_openvino() {
    local root="$1" target="$2"
    local p="$root/work/host-runtime/$target/openvino"
    [[ -d "$p" ]] && printf '%s\n' "$p"
}

runtime_find_ie_libdir() {
    local ov="$1"
    find "$ov/inference_engine/lib" -mindepth 1 -maxdepth 1 -type d \
        -exec test -f '{}/libinference_engine.so' ';' -print -quit 2>/dev/null
}

runtime_make_ld_path() {
    local ov="$1" ie
    ie="$(runtime_find_ie_libdir "$ov")" || return 1
    printf '%s:%s\n' "$ie" "$ov/ngraph/lib"
}

runtime_has_docker() {
    command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

runtime_docker_image_ready() {
    local image="$1"
    [[ -n "$image" ]] && runtime_has_docker &&
        docker image inspect "$image" >/dev/null 2>&1
}

runtime_python_interpreter() {
    local module="$1" root="$2" py
    local -a candidates
    case "$module" in
        tensorflow) candidates=("$root/work/venv-cpu/bin/python") ;;
        *) candidates=("$root/work/venv-apps/bin/python" "$root/work/venv-cpu/bin/python") ;;
    esac
    for py in "${candidates[@]}"; do
        [[ -x "$py" ]] || continue
        "$py" -c "import ${module}" >/dev/null 2>&1 && {
            printf '%s\n' "$py"
            return 0
        }
    done
    return 1
}

runtime_resolve_backend() {
    # Args: requested host_ready(0|1) docker_image
    local requested="$1" host_ready="$2" image="${3:-}"
    runtime_validate_backend "$requested" || return
    case "$requested" in
        host)
            [[ "$host_ready" == 1 ]] || {
                echo "host backend prerequisites are not available" >&2
                return 1
            }
            printf 'host\n'
            ;;
        docker)
            runtime_docker_image_ready "$image" || {
                echo "Docker image '$image' is not available or Docker is not usable" >&2
                return 1
            }
            printf 'docker\n'
            ;;
        auto)
            if [[ "$host_ready" == 1 ]]; then
                printf 'host\n'
            elif runtime_docker_image_ready "$image"; then
                printf 'docker\n'
            else
                echo "auto backend found neither usable host prerequisites nor Docker image '$image'" >&2
                return 1
            fi
            ;;
    esac
}

runtime_device_uses() {
    local requested="${1^^}" plugin="${2^^}"
    requested="${requested// /}"
    [[ ",${requested//:/,}," == *",${plugin},"* ||
       ",${requested//:/,}," == *",${plugin},"*","* ||
       "$requested" == "$plugin" ]]
}

runtime_warn_mvnc_mutex() {
    local mutex="${MVNC_MUTEX:-/tmp/mvnc.mutex}" owner mode other_bit
    [[ -e "$mutex" ]] || return 0
    owner="$(stat -c '%u %a' "$mutex" 2>/dev/null)" || return 0
    mode="${owner##* }"
    other_bit=$(( 8#${mode: -1} & 4 ))
    if [[ "${owner%% *}" != "$(id -u)" && "$other_bit" -eq 0 ]]; then
        echo "WARNING: $mutex is owned by uid ${owner%% *} mode $mode; mvnc may fail to initialize." >&2
        echo "fix: remove $mutex, use one user consistently, or chmod 0666 $mutex" >&2
    fi
}

runtime_run_openvino() {
    # Args: target runtime_root openvino_root executable [args...]
    local target="$1" rt="$2" ov="$3" exe="$4"
    shift 4
    local ie lp loader sysroot
    ie="$(runtime_find_ie_libdir "$ov")" || {
        echo "cannot locate OpenVINO inference-engine libraries under $ov" >&2
        return 1
    }
    lp="$ie:$ov/ngraph/lib"
    if [[ "$target" == armv7 ]]; then
        sysroot="$rt/sysroot"
        loader="$sysroot/lib/ld-linux-armhf.so.3"
        [[ -x "$loader" ]] || {
            echo "ARMHF loader missing: $loader; rerun pull-runtime.sh" >&2
            return 1
        }
        lp="$sysroot/lib/arm-linux-gnueabihf:$sysroot/usr/lib/arm-linux-gnueabihf:$lp"
        "$loader" --library-path "$lp" "$exe" "$@"
        return
    fi
    env LD_LIBRARY_PATH="$lp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$exe" "$@"
}

runtime_exec_openvino() {
    # Args: target runtime_root openvino_root executable [args...]
    local target="$1" rt="$2" ov="$3" exe="$4"
    shift 4
    local ie lp loader sysroot
    ie="$(runtime_find_ie_libdir "$ov")" || {
        echo "cannot locate OpenVINO inference-engine libraries under $ov" >&2
        return 1
    }
    lp="$ie:$ov/ngraph/lib"
    if [[ "$target" == armv7 ]]; then
        sysroot="$rt/sysroot"
        loader="$sysroot/lib/ld-linux-armhf.so.3"
        [[ -x "$loader" ]] || {
            echo "ARMHF loader missing: $loader; rerun pull-runtime.sh" >&2
            return 1
        }
        lp="$sysroot/lib/arm-linux-gnueabihf:$sysroot/usr/lib/arm-linux-gnueabihf:$lp"
        exec "$loader" --library-path "$lp" "$exe" "$@"
    fi
    export LD_LIBRARY_PATH="$lp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec "$exe" "$@"
}
