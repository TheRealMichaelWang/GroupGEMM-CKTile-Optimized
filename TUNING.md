# Tuning log

Quick loop: `scripts/quick.sh` (rebuild, print spills/occupancy, CK only on top3, 20 iters).
Sweep: `echo "MT NT KT MW NW MWT NWT KWT PIPE [Sched] [BlocksPerCU] [Persistent]" | scripts/sweep.sh`
(PIPE = V3 | V4 | MEM | ASYNC | EW). Appends to `results/sweep.log`. It edits
`ck_kernel/ck_grouped_gemm_config.hpp` in place, so reset it to the best afterwards.

Target: beat hipblaslt_loop on top3 bf16 (~1430 TFLOPS avg, 100 iters). Goal 1500-1800.

## Results so far (top3 bf16, mean TFLOPS, CK only)

| Config (tile / warps / MFMA / pipeline / mode) | TFLOPS | Note |
|---|---|---|
| 256x256x64 / 2x2 / 32x32x16 / V3 / persistent | ~100 | spills 386-728 VGPRs |
| 128x128x64 / 2x2 / 32x32x16 / V3 / persistent | ~900 | CK example default |
| same, 2 blocks per CU | ~600 | worse |
| 128x128x64 / 2x2 / 16x16x32 / V3 / persistent | ~940 | |
| 256x256x64 / 2x4 / 16x16x32 / V3 / persistent | ~250 | spills, **wrong results** |
| 256x128x64 / 2x2 / 16x16x32 / V3 / persistent | ~860 | |
| 256x256x32 / 2x2 / 32x32x16 / V4 (CK ComputeV4) | ~170 | spills |
| 256x256x32 / 2x2 / 16x16x32 / V4 / persistent | ~925 | spills 64 |
| 128x128x64 + `Async=true` on V3 | ~500-900 | **wrong results**, don't use |
| CompAsyncEightWaves 192x256x64 / 4x2 | n/a | doesn't compile with GroupedGemmKernel (needs HasHotLoop, GetVectorSizeC) |
| 128x128x64 / 2x2 / 16x16x32 / V3 / **non-persistent** | ~1100 | non-persistent is a big win |
| 256x256x32 / 2x2 / 16x16x32 / V3 / non-persistent | **~1170** | **current best** |
| 256x256x64 / 2x2 / 16x16x32 / V3 / non-persistent | ~955 | |
| 256x192x32, 256x128x32, 128x256x64 (V3 non-persistent) | 855-930 | |
| V4 / ASYNC variants non-persistent | 680-1050 | |
| TilePartitioner GroupNum/M01 (4/4, 16/4, 8/8, 8/2) | 1106-1118 | default 8/4 best |
| dropping CK's -mllvm flags | 1170 vs 1142 | kept only -O3 -fno-offload-uniform-block |

## What hipBLASLt runs (TestID 59)
`MT256x320x64_MI16x16x1`, 256 threads (4 waves, 1 per SIMD), LDS 144 KB,
128 VGPR + 384 AGPR, no spills. Tiles over (N=6144) x (tokens). So: bigger tile than CK
can currently fit, accumulators held in AGPRs, deep LDS buffering.

## Ideas not yet tried
- Custom grouped kernel around CompAsyncEightWaves (8 waves, 192x256) - it's CK's
  highest-throughput 16-bit path but not wired into GroupedGemmKernel.
- Custom pipeline policy (our own Policy class for V3): LDS layout/padding, larger
  prefetch depth, so a 256x256x64 or 256x320 tile fits in registers.
- Swap A/B roles (compute out^T, tile tokens x N like hipBLASLt).

## Round 2 (2026-10-02) - LOCKED CONFIG: use this for every shape

`ck_kernel/ck_grouped_gemm_config.hpp`: CK eight-wave async ping-pong pipeline
(`tunemax::GroupedEightWavePipeline`, adapter in `ck_kernel/eight_wave_pipeline.hpp`),
256x256x64, 4x2 warps, 16x16x32, CShuffle epilogue, non-persistent, GroupNum 8 / M01 4.

Top3 bf16, 100 iters, hipblaslt_loop run first: ck_tile 1405 vs hipblaslt_loop 1419 mean TFLOPS
(TestID 59/71: CK ties or wins; 323 Kimi N=4096: hipBLASLt ~1480 vs CK ~1375).

Key facts found:
- Both kernels hit the MI355X 1400 W power cap (CK ~1725 MHz, hipBLASLt ~1650 MHz), so
  TFLOPS = energy efficiency. Check with `python3 scripts/power.py` while a run is going
  (pick the busiest GPU; HIP device 0 is amd-smi GPU 3 here).
- CK's eight-wave kernel uses ~45% more LDS cycles than hipBLASLt (per-wave tile 64x128 vs
  128x128/128x160). That is the remaining structural gap; 256x256x64 is the largest tile the
  eight-wave pipeline fits (320x256 spills 1100 VGPRs, 256x384 violates its layout asserts).
- Any 8-warp config with CompV3/Mem gives wrong results on gfx9 (CK's CShuffle epilogue and
  V3 hard-code the eight-wave layout when M_Warp*N_Warp == 8). Use only the eight-wave pipeline.
- Did not help: compiler flags (MFMA VGPR/AGPR form, CK's -mllvm set), XCD remap partitioner
  (`XcdRemap`, no change in L2 misses), VectorSize, DataCachePrefetch, persistent mode,
  K_Tile 32/128, 2 blocks/CU, weight-preshuffle pipeline (~700 TF), 4-wave CompAsync
  (~1060; 50% LDS bank conflicts, XOR policy in `xor_async_policy.hpp` didn't remove them).

Remaining idea (large effort): a 4-wave pipeline with 128x128+ per-wave tiles that keeps
the MFMA pipes busy at 1 wave/SIMD - i.e. what hipBLASLt's hand-scheduled kernel does.

Tools: `scripts/quick.sh` (bf16-only tuning build, ~15 s + ~5 s run), `QUICK_ONLY=323`,
`scripts/build.sh --full` before full runs or fp16, `scripts/pmc.sh BACKEND` (counters),
`scripts/asm_loop.py` (hot-loop instruction mix), `scripts/flags.sh "<flags>"`.
- amdgpu_num_vgpr(N) on our own entry kernel: caps the TOTAL VGPR+AGPR budget (128 -> 256 VGPR/0 AGPR, occupancy 2, scratch spills, ~800 TF); 192 ignored. Cannot force hipBLASLt-style 120 VGPR / 384 AGPR split this way.
- Non-temporal C stores (ck_kernel/coherence_epilogue.hpp, CStoreMode=kStoreNT): 1394 -> ~1412 top3 mean (59: 1438). Enum values differ host vs device, so the mode is a plain int.

## Round 3 (2026-10-02): diagnosis with thread trace (scripts/att_summary.py)
- gfx950 LDS ground truth (microbenchmark): ds_read_b128 is conflict-free iff each 8-lane group
  covers distinct 16B slots of a 128B window. Plain 64B bf16 rows -> 2-way (50%); 128B rows -> 4-way.
  Conflict-free: chunk ^ ((row/2)%4) for 64B rows, chunk ^ (row%8) for 128B rows.
  `swizzled_lds_policy.hpp` (V3/V4 register-staged) and `xor_async_policy.hpp` (async, swizzles the
  global side) implement it; the async kernel went to 0% conflicts - but no speedup.
- LDS read bandwidth measured ~180-210 B/clk/CU, so the 8-wave kernel is NOT LDS-bound.
- Thread trace, Kimi shape: CK 8-wave stalls = 33% MFMA, 28% s_waitcnt vmcnt (async loads),
  18% barriers. hipBLASLt = 71% MFMA, <6% waits. The 8-wave ping-pong issues each async load only
  one phase (~1000 cycles) before use; global latency at the power-capped clock exceeds that.
  Deeper prefetch needs more LDS buffers than fit at K=64 (3x64 KB > 160 KB).
- 4-wave 256x256 kernels (CompV3/CompAsync, with/without swizzle): register-starved (all 512 regs,
  ~50 AGPR<->VGPR moves per 128 MFMAs) -> 47-62% MFMA utilization, 1000-1180 TFLOPS.
- 8-warp CompAsync (4x2, Default epilogue) is correct but ~1000-1050. Persistent + XCD remap: 1379.
- Our copy of the eight-wave loop (ck_kernel/eight_wave_custom.hpp, generated from CK) reproduces CK exactly (RelaxA=0). Relaxing the end-of-phase A wait (RelaxA=4/8) stays correct but gains <=1% (1397-1409): the wait just moves to the next vmcnt(0) barrier. Deeper prefetch needs more LDS stages than fit at 256x256x64; at K=32 (4 stages fit) the 2x barriers cost more (EW K32 = 1145).
- RelaxA has no effect in the binary: the compiler re-tightens the end-of-phase wait to vmcnt(4) (no vmcnt(8) emitted). The wait is required: each wave group async-loads HALF of every A tile and the other group reads it right after the barrier. Deeper prefetch needs a 3rd LDS stage (~192 KB at 256x256x64 > 160 KB); duplicating A per group fits only at K=32, where the load-to-use distance stays ~1000 cycles.
- Newer compiler: upstream clang-24 nightly (apt.llvm.org, LLVM main 2026-09-11),
  `scripts/build_ck24.sh` builds only the CK kernel with it (-fdelayed-template-parsing needed for a
  gfx908-only CK body) and links build/tunemax_bench24. Same register allocation (4-wave kernels
  still split accumulators); best config ~1387 vs ~1400 with amdclang. No gain.
- 8-warp CompAsync + XOR at K=64: spills 547 VGPRs (~130 TF). Eight-wave pipeline with 2x2 warps:
  wrong results, ~800 TF.
- Eight-wave + 32x32x16 MFMA: 48% LDS bank conflicts with CK's policy (swizzle factor 2). Our policy copy (eight_wave_policy_swizzled.hpp, factor 8) removes them (0%) but gives wrong results and no speedup (~1337 either way) -> conflicts are not its limiter. Not used.

## Round 4 (2026-10-02): hand-written pipeline (HIP intrinsics + CK primitives, CK unmodified)
`ck_kernel/hand_pipeline_k32.hpp` (pipeline HAND32, 256x256x32, 2x2 warps, 4 LDS stages) and
`hand_pipeline.hpp` (HAND, 256x256x64, 2 stages). Both pass the correctness check.
- HAND (K64, 2 stages): ~1110 (loads in a burst stall the vector-memory queue, 35%); interleaving
  the loads leaves too little slack before use (vmcnt waits 38%) -> ~1035-1076.
- HAND32 (K32, 4 stages, loads 3 tiles ahead spread over the MFMAs): ~1180-1200.
  LDS traffic = hipBLASLt level (1.55e10 vs 1.47e10 active), bank conflicts 3%, but MFMA busy/CU
  busy = 2.40 (60%) vs hipBLASLt 3.40 (85%); runs at 2064 MHz at the 1400 W cap.
  Remaining stalls: load issue ~9-15%, waits ~7%, barriers ~5%, non-MFMA issue overhead.
- Lessons: `using Base::operator()` silently selected CK's V3 loop; generic->int->LDS pointer casts
  give flat loads (use a C-style address-space cast); amd_async_buffer_load lowered to load+ds_write
  here (use m0_set_with_memory + async_buffer_load_dwordxn_v); branches around MFMA blocks cause
  massive spills (keep the loop body branch-free, clamp the tail).
- HAND32 + sched_group_barrier (MFMA, DS read alternation): ~1227 TF, MFMA busy 63% at 2021 MHz.
  Trace: async load issue ~18% of wave time (~37 cycles/load): K=32 rows are 64 B, so every
  load touches 16 half-used cache lines (K=64 rows: 8 full lines).
- RING (K=64, 2 LDS stages, A fragment ring + double-buffered B = hipBLASLt's register-as-third-
  buffer scheme): correct, but needs ~448 registers; compiler uses all 512 + 104 B scratch and 84
  AGPR<->VGPR moves per loop -> ~600 TF. Single-buffered B fits but frees the LDS stage only
  mid-tile (= HAND, ~1035-1110).
