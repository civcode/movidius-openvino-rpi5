# OpenVINO 2020.3.2 patch set

The project applies these patches to the pinned OpenVINO source before both
`armv7`, `arm64` and `amd64` builds. `build.sh --no-patches` remains available for
reproducing upstream failures.

## `0001-cross-compile-skip-host-protoc.patch`

The upstream cross-compile dependency logic tries to fetch/use a host `protoc`
archive for the nGraph ONNX importer. This project disables that importer. The
patch only changes behavior when `CMAKE_CROSSCOMPILING` is true, so it affects
the ARMv7 build and is inert during native arm64 and amd64 configures.

## `0002-ade-gcc10-no-error-warnings.patch`

ADE is compiled with `-Werror`; modern GCC emits several diagnostics in this old
source revision. The patch keeps `-Werror` globally while downgrading the known
warnings. It is safe for all three targets and keeps the two builds on one source
configuration.

## `0003-gcc10-no-error-deprecated.patch`

The 2020.3.2 inference-engine headers mark `IShapeInferImpl` and other legacy
extension APIs with `__attribute__((deprecated))`; the engine's own sources
reference those markers. On a clean x86_64 configure, `TREAT_WARNING_AS_ERROR`
is `ON` by default (see `cmake/features.cmake`), so GCC 10 turns the
`-Wdeprecated-declarations` notes into hard errors. The patch keeps `-Werror`
globally while downgrading only that warning class, matching the approach of
`0002`. It is safe for all three targets (the flag is a no-op where `-Werror`
is not enabled).
