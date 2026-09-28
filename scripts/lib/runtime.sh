#!/usr/bin/env bash
runtime_repo_root() { cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd; }
runtime_validate_backend() { case "$1" in auto|host|docker) return 0;; *) echo "invalid backend '$1'; expected auto|host|docker" >&2; return 2;; esac; }
runtime_find_openvino() { local root="$1" target="$2"; local p="$root/work/host-runtime/$target/openvino"; [[ -d "$p" ]] && printf '%s\n' "$p"; }
runtime_find_ie_libdir() { local ov="$1"; find "$ov/inference_engine/lib" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/libinference_engine.so' ';' -print -quit; }
runtime_make_ld_path() { local ov="$1" ie; ie="$(runtime_find_ie_libdir "$ov")" || return 1; printf '%s:%s\n' "$ie" "$ov/ngraph/lib"; }
runtime_has_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
runtime_resolve_backend() {
  local requested="$1" ov="${2:-}" image="${3:-}"
  runtime_validate_backend "$requested" || return
  if [[ "$requested" != auto ]]; then printf '%s\n' "$requested"; return; fi
  if [[ -n "$ov" && -d "$ov" ]]; then printf 'host\n'; return; fi
  if [[ -n "$image" ]] && runtime_has_docker && docker image inspect "$image" >/dev/null 2>&1; then printf 'docker\n'; return; fi
  echo "auto backend found neither an extracted host runtime nor a usable Docker image" >&2; return 1
}
