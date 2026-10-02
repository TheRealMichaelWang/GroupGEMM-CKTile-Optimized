// Tuning knobs for the CK Tile grouped GEMM. This is the first file to edit.
//
// After changing anything here or in ck_grouped_gemm.cpp:
//   ./scripts/build.sh          # rebuilds only the CK translation unit + relinks
//   ./scripts/run.sh top3       # 3 largest shapes, quick
//   ./scripts/run.sh full       # all 360 Primus-Turbo shapes
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "eight_wave_pipeline.hpp"
#include "xor_async_policy.hpp"
#include "swizzled_lds_policy.hpp"
#include "coherence_epilogue.hpp"

namespace tunemax {

struct CkTileConfig {
    // Block tile: each workgroup computes an M_Tile x N_Tile block of the output and
    // walks K in K_Tile steps.
    // Best found (top3 bf16 ~1390-1400 TFLOPS): CK's eight-wave async ping-pong pipeline via
    // our GroupedEightWavePipeline adapter, 256x256x64, 4x2 warps, 16x16x32, CShuffle
    // epilogue, non-persistent. The eight-wave pipeline needs N_Warp == 2 and fits nothing
    // bigger than 256x256x64 (LDS / registers). See TUNING.md for everything tried.
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    // Warps per block in M, N, K (block size = product * 64 threads).
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // Per-warp MFMA tile. Must be a combination the CK warp-GEMM dispatcher supports
    // for the dtype on gfx950 (e.g. 32x32x16 or 16x16x32 for bf16/fp16).
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    // Pipeline and its scheduler. Other options in ck_tile/ops/gemm.hpp include
    // GemmPipelineAgBgCrCompV4 (set DoubleSmemBuffer = true) and GemmPipelineAgBgCrMem.
    template <typename Problem>
    using Pipeline = tunemax::GroupedEightWavePipeline<Problem>;
    static constexpr auto Scheduler        = ck_tile::GemmPipelineScheduler::Intrawave;
    // Preshuffle: B (weights) pre-arranged so the pipeline loads it straight to registers.
    // Needs Pipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2 and shuffled weights.
    static constexpr bool Preshuffle       = false;
    static constexpr bool DoubleSmemBuffer = false;

    // Epilogue: true = CShuffle (via LDS), false = Default (direct from registers).
    static constexpr bool CShuffleEpilogue = true;
    // Cache policy for C stores (coherence_epilogue.hpp): kStoreDefault, kStoreNT (non-temporal,
    // C is never re-read, like hipBLASLt's NTC/NTD), kStoreDeviceNT, kStoreSystemNT.
    static constexpr int CStoreMode = kStoreNT;

    // Async: global->LDS loads that bypass registers (gfx950). VectorSize: max global load
    // width in elements. DataCachePrefetch: ck_tile::DataCachePrefetchKind::None or others.
    static constexpr bool             Async      = false;
    static constexpr ck_tile::index_t VectorSize = 16;
    static constexpr auto DataCachePrefetchA      = ck_tile::DataCachePrefetchKind::None;
    static constexpr auto DataCachePrefetchB      = ck_tile::DataCachePrefetchKind::None;

    // Tile ordering: groups of TilePartitionerGroupNum tile rows, M01-wide swizzle, for
    // L2 reuse.
    static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    static constexpr ck_tile::index_t TilePartitionerM01      = 4;
    // Remap tile ids so each XCD (8 on MI355X, separate L2s) works on a contiguous patch.
    static constexpr bool XcdRemap = false;

    // Persistent: grid = CUs * occupancy (capped at CUs * kBlockPerCu), each workgroup loops
    // over tiles. Non-persistent: one workgroup per tile, hardware schedules them.
    static constexpr bool Persistent = false;
    // kBlockPerCu is also the kernel's launch-bounds min-blocks-per-CU.
    static constexpr int kBlockPerCu = 1;
};

} // namespace tunemax
