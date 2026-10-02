cmake_minimum_required(VERSION 3.25)
project(cktile_tunemax LANGUAGES CXX HIP)

# CK Tile is header-only. Its headers come from this source checkout, never from
# /opt/rocm (ROCm ships an older copy there). scripts/check_ck_headers.sh verifies it.
set(CK_ROOT "/workspace/rocm-libraries/projects/composablekernel"
    CACHE PATH "Composable Kernel source checkout (the folder that contains include/ck_tile)")
if(NOT EXISTS "${CK_ROOT}/include/ck_tile/core.hpp")
  message(FATAL_ERROR "CK_ROOT=${CK_ROOT} has no include/ck_tile/core.hpp")
endif()

set(GPU_TARGETS "gfx950" CACHE STRING "GPU architecture(s) to compile for")
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# hipBLASLt and hipBLAS come from the installed ROCm (/opt/rocm). Do not
# find_package(composable_kernel): that would pull in the ROCm copy of CK.
find_package(hipblaslt REQUIRED)
find_package(hipblas REQUIRED)

# --- The kernel being tuned. Editing ck_kernel/* rebuilds only this target. ------------
add_library(tunemax_ck STATIC ck_kernel/ck_grouped_gemm.cpp)
set_source_files_properties(ck_kernel/ck_grouped_gemm.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(tunemax_ck BEFORE PRIVATE
  ${CK_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/ck_kernel)
# Same device flags CK's own CMake and Primus-Turbo use for CK kernels
# (-amdgpu-coerce-illegal-types is left out: this compiler doesn't accept it).
# kernel-resource-usage prints VGPRs / spills / occupancy for every CK kernel.
target_compile_options(tunemax_ck PRIVATE -O3 -fno-offload-uniform-block
  -Rpass-analysis=kernel-resource-usage)
# Extra device flags for experiments, e.g. -DCK_EXTRA_FLAGS="-mllvm -amdgpu-mfma-vgpr-form=0"
set(CK_EXTRA_FLAGS "" CACHE STRING "Extra compile flags for the CK kernel only")
separate_arguments(CK_EXTRA_FLAGS_LIST UNIX_COMMAND "${CK_EXTRA_FLAGS}")
target_compile_options(tunemax_ck PRIVATE ${CK_EXTRA_FLAGS_LIST})
# Tuning build: compile only the bf16, unpadded CK instance (~4x faster rebuilds).
option(TUNEMAX_TUNING_BUILD "Compile only the bf16 unpadded CK kernel" OFF)
if(TUNEMAX_TUNING_BUILD)
  target_compile_definitions(tunemax_ck PRIVATE TUNEMAX_TUNING_BUILD)
endif()

# --- Baselines: hipBLASLt GroupedGemm and Primus-Turbo's per-group loop. ---------------
set(BASELINE_SOURCES
  backends/hipblaslt_grouped_backend.cpp
  backends/hipblaslt_loop_backend.cpp
  third_party/primus_turbo/kernels/hipblaslt_grouped_gemm.cu
  third_party/primus_turbo/kernels/hipblaslt_gemm.cu)
add_library(tunemax_baselines STATIC ${BASELINE_SOURCES})
set_source_files_properties(${BASELINE_SOURCES} PROPERTIES LANGUAGE HIP)
target_include_directories(tunemax_baselines PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/primus_turbo/include)
target_link_libraries(tunemax_baselines PUBLIC roc::hipblaslt roc::hipblas)
target_compile_options(tunemax_baselines PRIVATE -O3)

# --- Benchmark driver. -----------------------------------------------------------------
add_executable(tunemax_bench bench/bench_main.cpp)
set_source_files_properties(bench/bench_main.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(tunemax_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(tunemax_bench PRIVATE tunemax_ck tunemax_baselines)
target_compile_options(tunemax_bench PRIVATE -O3)
