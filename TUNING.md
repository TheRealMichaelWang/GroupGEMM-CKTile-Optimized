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
- RING with raw float4 accumulators + direct __builtin_amdgcn_mfma (layout verified correct): worse, 1464 B scratch / ~92 TF; -amdgpu-mfma-vgpr-form=0/1 no change.

## Round 5 (2026-10-02): HandPipelineAsmS -> ~1555 TFLOPS top3 (hipblaslt_loop ~1427)
Pipeline `ck_kernel/hand_pipeline_asm_s.hpp` (config key ASMS) + `ck_kernel/fast_epilogue.hpp`,
persistent GroupedGemmKernel. Metric used while tuning: `scripts/att.sh` (thread trace of one CU,
Kimi-K2) prints cycles per wave-tile (ideal 229376 = 14336 MFMAs x 16) - far less noisy than the
+-10 TFLOPS run-to-run spread of the benchmark. Steps (mean top3 TFLOPS / cycles per tile):
- Explicit schedule: sched_barrier after every MFMA and at most one memory op per MFMA gap;
  end-of-tile barrier moved into row 7 with the next tile's A row 0 read right after it: 1393 -> 1431.
  Barrier position in row 7 (MFMA 11) -> ~1443.
- FastEpilogue (C via LDS: ds_write_b64 + full-row buffer_store_dwordx4 NT, one barrier) instead of
  CShuffle (2-byte LDS writes, a barrier per round, ~22k cycles/tile): -> ~1505-1513.
  16-byte-aligned LDS row stride (528 B; 520 B made every ds_read_b128 misaligned, 136 cycles).
- K step via the buffer instruction's soffset (fixed resources, no per-iteration rebuild, no spill).
- Swapped MFMA operands in unpadded bf16 instances (C^T per tile): each lane holds 4 consecutive
  columns of a row, so the epilogue needs no DPP transpose: ~1535.
- Peeled tile 0 initializes the accumulators with C=0 MFMAs (no 256 v_accvgpr_write).
- kB3: B gets a third LDS buffer (A 2 x 32 KB + B 3 x 32 KB = 160 KB), so B(kt+3) can be loaded
  anywhere in iteration kt and only one barrier per iteration remains; loop unrolled x6 so all
  stage/buffer offsets are constants (a dynamic kt%3 cost ~70 cycles/iteration): 251.5k cycles/tile,
  ~1541.
- Persistent kernel: removes ~10k cycles/tile of workgroup turnaround: ~1557. Volatile-asm lane id
  keeps lane-derived values from being hoisted out of the tile loop and spilled.
- Early A: the first 2 loads of A(kt+2) go right after the row-7 barrier of iteration kt (the
  earliest point their stage is free): ~251.0k cycles/tile. 3 early loads, front-loading the rest,
  or an even 1-load-per-row spread were all worse (A latency to the row-7 wait dominates).
- Prefetches past the last K tile get a buffer resource with num_records = 0 (no memory traffic)
  instead of re-loading a clamped tile; checked only in the tail (main loop runs while every
  prefetch is in range, tail of up to 8 iterations): ~250.2k cycles/tile.
- Counters (DS-V4 48 groups): MFMA busy / CU busy = 3.62 (90.6%), LDS bank conflicts 1.7%,
  ~1695 MHz at 1400 W (hipBLASLt: ~90% at ~1560 MHz).
Where the remaining ~22k cycles/tile (9%) go: main loop 16.6-16.7 cycles/MFMA (row-7 barrier skew
~25 cycles/iteration, A-load issue stalls ~35/iteration); tile boundary ~12k: 32 C stores + the
next tile's 24 prologue loads serialize on the CU's vector-memory path (~35 B/clk/CU, ~100-150
cycles per 1 KB instruction), plus accumulator reads/conversion ~2k.
Tried without gain: 32x32x16 MFMA variant (1228), triple-buffered A (1371), B loads at other
slots (collisions with LDS reads in the same MFMA gap cost up to 50 TFLOPS), XCD remap / partitioner
GroupNum 4/16 (CK's spatially-local order already gives 12 unique blocks per 32 tiles per XCD, the
optimum), default vs NT C stores (same), direct 8-byte C stores without LDS (+22k cycles), epilogue
halves interleaving stores with staging (same), staggering workgroups (stores are CU-bound, not
GPU-bound), moving B1/B2 prologue loads into tile 0 (same).
