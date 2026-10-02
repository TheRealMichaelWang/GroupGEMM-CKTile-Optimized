// Hand-written GEMM main loop, K=32 tiles with 4 LDS stages (deep global prefetch).
// GemmPipeline for ck_tile::GroupedGemmKernel; built from CK Tile primitives, CK unmodified.
//
// 4 waves (2x2), block tile 256x256x32, warp tile 128x128 in 16x16x32 bf16 MFMAs, laid out
// like CK's block GEMM so the C tile feeds CK's epilogue unchanged.
//
// Why: a 256x256 tile needs 64 KB of global->LDS traffic per 64 K, about half the vector-
// memory unit's time; loads must be spread over a whole tile AND issued well before use.
// With 4 LDS stages, tile kt+3 is loaded (spread across tile kt's MFMAs) while kt+1, kt+2 are
// already resident, giving ~1 tile of slack after the last load is issued.
//
// Iteration kt:
//   wait until tile kt+1 has landed (tile kt+2 may still be in flight); barrier
//   MFMAs on tile kt (64), interleaved with: LDS->regs fragments of tile kt+1, and the 4+4
//   async loads of tile kt+3 into the stage tile kt-1 used (its fragments were read 2 iters ago)
// LDS rows are 64 B (32 bf16); 16-byte chunk c of row r is stored at c ^ ((r/2) % 4), which is
// bank-conflict free for 16x16x32 fragment reads on gfx950 (measured).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile::tunemax_hand {

template <typename Problem>
struct HandPipelineK32 : public GemmPipelineAgBgCrCompV3<Problem> {
    using Base      = GemmPipelineAgBgCrCompV3<Problem>;
    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemm =
        remove_cvref_t<decltype(UniversalGemmPipelineAgBgCrPolicy::template GetBlockGemm<Problem>())>;
    using WarpGemm    = typename BlockGemm::WarpGemm;
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;
    using CWarpDstr   = typename WarpGemm::CWarpDstr;

    static constexpr index_t kM = 256, kN = 256, kK = 32, kStages = 4;
    static constexpr index_t kMIter = 8, kNIter = 8;
    static_assert(Problem::BlockGemmShape::kM == kM && Problem::BlockGemmShape::kN == kN &&
                      Problem::BlockGemmShape::kK == kK,
                  "HandPipelineK32 is written for 256x256x32");
    static_assert(Problem::kBlockSize == 256, "4 waves");
    static_assert(sizeof(ADataType) == 2 && sizeof(BDataType) == 2, "16-bit inputs only");

    static constexpr index_t kRowBytes   = kK * 2;            // 64
    static constexpr index_t kTileBytes  = kM * kRowBytes;    // 16 KB per operand
    static constexpr index_t kStageBytes = 2 * kTileBytes;    // 32 KB
    static constexpr index_t kLoads      = kTileBytes / (4 * 1024); // 4 per wave per operand

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kStages * kStageBytes; }

    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
    using Vec8 = ext_vector_t<ADataType, 8>;

    template <typename Window>
    CK_TILE_DEVICE static auto base_and_ld(const Window &win, index_t &ld) {
        const auto &view = win.get_bottom_tensor_view();
        const auto &desc = view.get_tensor_descriptor();
        ld = desc.calculate_offset(make_multi_index(1, 0)) -
             desc.calculate_offset(make_multi_index(0, 0));
        return view.get_buffer_view().p_data_ + desc.calculate_offset(win.get_window_origin());
    }

    template <typename ADramWindow, typename BDramWindow,
              std::enable_if_t<!is_detected<is_tuple, ADramWindow>::value, bool> = true>
    CK_TILE_DEVICE auto operator()(const ADramWindow &a_win, const BDramWindow &b_win,
                                   index_t num_loop, void *p_smem) const {
        const index_t lane = threadIdx.x % 64;
        const index_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
        const index_t wm = wave / 2, wn = wave % 2;

        index_t lda, ldb;
        const ADataType *a_ptr = base_and_ld(a_win, lda);
        const BDataType *b_ptr = base_and_ld(b_win, ldb);

        auto *lds              = (CK_TILE_LDS_ADDR uint8_t *)p_smem;
        const index_t lds_base = __builtin_amdgcn_readfirstlane(
            static_cast<index_t>(reinterpret_cast<uintptr_t>(lds)));

        // Async load j (0..3) of this wave fills rows (j*4+wave)*16 .. +16 (1 KB): lane l ->
        // row l/4, physical chunk l%4, logical chunk (l%4) ^ ((l/8)%4).
        const index_t ld_row   = lane / 4;
        const index_t ld_chunk = (lane % 4) ^ ((lane / 8) % 4);
        const index_t voff_a   = ((wave * 16 + ld_row) * lda + ld_chunk * 8) * 2;
        const index_t voff_b   = ((wave * 16 + ld_row) * ldb + ld_chunk * 8) * 2;
        auto issue_load_pair = [&](index_t kt, index_t stage, auto j) {
            __builtin_amdgcn_sched_barrier(0);
            const auto ra = make_wave_buffer_resource(a_ptr + kt * kK + j * 64 * lda, 0x7ffff000);
            const auto rb = make_wave_buffer_resource(b_ptr + kt * kK + j * 64 * ldb, 0x7ffff000);
            const index_t dst = lds_base + stage * kStageBytes + (j * 4 + wave) * 1024;
            m0_set_with_memory(dst);
            async_buffer_load_dwordxn_v<4>(p_smem, ra, voff_a, 0, 0);
            m0_set_with_memory(dst + kTileBytes);
            async_buffer_load_dwordxn_v<4>(p_smem, rb, voff_b, 0, 0);
            __builtin_amdgcn_sched_barrier(0);
        };
        auto issue_tile = [&](index_t kt, index_t stage) {
            static_for<0, kLoads, 1>{}([&](auto j) { issue_load_pair(kt, stage, j); });
        };

        // Fragment of iter i: row (i*2+w)*16 + lane%16, logical chunk lane/16, stored at chunk
        // (lane/16) ^ (((lane%16)/2) % 4).
        const index_t fr_off = (lane % 16) * kRowBytes + (((lane / 16) ^ (((lane % 16) / 2) % 4)) * 16);
        const index_t a_off  = wm * 16 * kRowBytes + fr_off;
        const index_t b_off  = wn * 16 * kRowBytes + fr_off;

        AWarpTensor fa[2][kMIter];
        BWarpTensor fb[2][kNIter];
        auto read_a = [&](index_t stage, auto buf, auto i) {
            fa[buf][i].get_thread_buffer().template set_as<Vec8>(
                number<0>{}, *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                                 lds + stage * kStageBytes + a_off + i * 32 * kRowBytes));
        };
        auto read_b = [&](index_t stage, auto buf, auto i) {
            fb[buf][i].get_thread_buffer().template set_as<Vec8>(
                number<0>{}, *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                                 lds + stage * kStageBytes + kTileBytes + b_off + i * 32 * kRowBytes));
        };
        auto read_frags = [&](index_t stage, auto buf) {
            static_for<0, kMIter, 1>{}([&](auto i) { read_a(stage, buf, i); });
            static_for<0, kNIter, 1>{}([&](auto i) { read_b(stage, buf, i); });
        };

        auto c_block = BlockGemm::MakeCBlockTile();
        clear_tile(c_block);
        auto mfma_row = [&](auto buf, auto mi) {
            static_for<0, kNIter, 1>{}([&](auto ni) {
                CWarpTensor c_w;
                c_w.get_thread_buffer() = c_block.get_y_sliced_thread_data(
                    merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                WarpGemm{}(c_w, fa[buf][mi], fb[buf][ni]);
                c_block.set_y_sliced_thread_data(
                    merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths), c_w.get_thread_buffer());
            });
        };

        // --- prologue: tiles 0,1,2 in flight; fragments of tile 0 in registers
        static_for<0, kStages - 1, 1>{}([&](auto t) { issue_tile(min(index_t{t}, num_loop - 1), t); });
        block_sync_lds_direct_load<2 * 2 * kLoads>(); // tile 0 landed
        read_frags(0, number<0>{});

        // --- main loop: two iterations per trip so register buffers are compile-time
        auto iteration = [&](index_t kt, auto buf) {
            block_sync_lds_direct_load<2 * kLoads>(); // tile kt+1 landed (kt+2 may be in flight)
            const index_t s_next = (kt + 1) % kStages;
            const index_t s_load = (kt + 3) % kStages;
            const index_t kt_load = min(kt + 3, num_loop - 1);
            static_for<0, kMIter, 1>{}([&](auto mi) {
                // spread next-tile fragment reads and the 4+4 loads of tile kt+3 over the rows
                read_a(s_next, number<1 - buf>{}, mi);
                read_b(s_next, number<1 - buf>{}, mi);
                if constexpr (mi % 2 == 0)
                    issue_load_pair(kt_load, s_load, number<mi / 2>{});
                mfma_row(buf, mi);
            });
        };
        index_t kt = 0;
        for (; kt + 1 < num_loop; kt += 2) {
            iteration(kt, number<0>{});
            iteration(kt + 1, number<1>{});
        }
        if (kt < num_loop)
            iteration(kt, number<0>{});
        block_sync_lds_direct_load<0>(); // drain clamped loads before the epilogue reuses LDS
        return c_block;
    }

    template <typename AsWindows, typename AElementFunction, typename BsWindows,
              typename BElementFunction,
              std::enable_if_t<is_detected<is_tuple, AsWindows>::value, bool> = true>
    CK_TILE_DEVICE auto operator()(const AsWindows &as, const AElementFunction &,
                                   const BsWindows &bs, const BElementFunction &,
                                   index_t num_loop, void *p_smem) const {
        return (*this)(as[number<0>{}], bs[number<0>{}], num_loop, p_smem);
    }
};

} // namespace ck_tile::tunemax_hand

namespace tunemax {
template <typename Problem>
using HandPipelineK32 = ck_tile::tunemax_hand::HandPipelineK32<Problem>;
} // namespace tunemax
