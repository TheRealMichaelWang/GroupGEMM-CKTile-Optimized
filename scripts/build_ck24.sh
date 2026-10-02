#!/usr/bin/env bash
# Compile ONLY the CK kernel with upstream clang-24 (apt.llvm.org nightly) and relink the
# benchmark as build/tunemax_bench24. Everything else stays built by ROCm's amdclang.
# -fdelayed-template-parsing: CK has a gfx908-only body upstream clang rejects at definition time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; B="$ROOT/build"
CK="$(grep -E '^CK_ROOT:' "$B/CMakeCache.txt" | cut -d= -f2-)"
start=$(date +%s)
clang++-24 -x hip --offload-arch=gfx950 --no-offload-new-driver -std=c++20 -O3 -fPIC \
  --rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode \
  -fdelayed-template-parsing -fno-offload-uniform-block -DNDEBUG $* \
  -I "$CK/include" -I "$ROOT/common" -I "$ROOT/ck_kernel" \
  -c "$ROOT/ck_kernel/ck_grouped_gemm.cpp" -o "$B/ck24.o"
cd "$B" && /opt/rocm/bin/amdclang++ -O3 --offload-arch=gfx950 --hip-link --rtlib=compiler-rt -unwindlib=libgcc \
  CMakeFiles/tunemax_bench.dir/bench/bench_main.cpp.o ck24.o libtunemax_baselines.a \
  /opt/rocm/lib/libamdhip64.so /opt/rocm/lib/libhipblaslt.so /opt/rocm/core-10.0/lib/libhipblas.so \
  -o tunemax_bench24
echo "clang-24 CK build + link took $(( $(date +%s) - start ))s -> build/tunemax_bench24"
