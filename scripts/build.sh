#!/usr/bin/env bash
# Configure (first time only) and build. Incremental: after editing ck_kernel/* only the
# CK translation unit is recompiled and the benchmark relinked.
#
#   scripts/build.sh            # fast tuning build: only the bf16 unpadded CK kernel
#   scripts/build.sh --full     # every dtype/padding instance (for full runs, fp16)
#   scripts/build.sh --clean    # wipe build/ and reconfigure
#
# Env overrides: CK_ROOT (CK source checkout), GPU_TARGETS (default gfx950), BUILD_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"   # BUILD_DIR: e.g. a separate full build next to the tuning build

tuning=ON   # default: fast tuning build (bf16 unpadded CK kernel only)
for arg in "$@"; do
  case "$arg" in
    --clean) rm -rf "$BUILD" ;;
    --full) tuning=OFF ;;
  esac
done

if [[ ! -f "$BUILD/build.ninja" ]]; then
  cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
    -DCK_ROOT="${CK_ROOT:-/workspace/rocm-libraries/projects/composablekernel}" \
    -DGPU_TARGETS="${GPU_TARGETS:-gfx950}"
fi

cmake -S "$ROOT" -B "$BUILD" -DTUNEMAX_TUNING_BUILD=$tuning > /dev/null
echo "build mode: $([[ $tuning == ON ]] && echo 'tuning (bf16 unpadded only)' || echo full)"
start=$(date +%s)
cmake --build "$BUILD"
echo "build took $(( $(date +%s) - start ))s"

"$ROOT/scripts/check_ck_headers.sh"
