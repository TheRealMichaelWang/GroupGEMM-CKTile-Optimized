// Hand-written GEMM main loop, used as the GemmPipeline of ck_tile::GroupedGemmKernel.
//
// Built from CK Tile primitives (amd_async_buffer_load, make_wave_buffer_resource, the
// pipeline's own WarpGemm and C block tile) plus a hand-arranged schedule. CK is not modified.
//
// Shape: 4 waves (2x2), block tile 256x256x64, warp tile 128x128 in 16x16x32 bf16 MFMAs,
// laid out exactly like CK's block GEMM (wave w = wm*2+wn owns rows (mIter*2+wm)*16.. and
// cols (nIter*2+wn)*16..), so the returned C tile feeds CK's epilogue unchanged.
//
// Schedule per K tile kt (stage s = kt % 2, two 64 KB LDS stages):
//   a) LDS -> regs: fragments of k-half 1 of tile kt
//   b) 64 MFMAs on k-half 0 (registers already loaded)
//   c) wait for tile kt+1's async loads; one barrier (stage s fully read, stage s^1 landed)
//   d) issue async global->LDS loads of tile kt+2 into stage s (a full tile ahead of use)
//   e) LDS -> regs: fragments of k-half 0 of tile kt+1
//   f) 64 MFMAs on k-half 1
// LDS rows are 128 B (64 bf16); 16-byte chunk c of row r is stored at chunk c ^ (r % 8), which
// is bank-conflict free for the 16x16x32 fragment reads on gfx950 (measured).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile::tunemax_hand {

template <typename Problem>
struct HandPipeline : public GemmPipelineAgBgCrCompV3<Problem> {
    using Base = GemmPipelineAgBgCrCompV3<Problem>;
    // Note: no `using Base::operator()` -- V3's plain-window overload would win overload
    // resolution and silently run V3's loop instead of this one.

    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemm =
        remove_cvref_t<decltype(UniversalGemmPipelineAgBgCrPolicy::template GetBlockGemm<Problem>())>;
    using WarpGemm    = typename BlockGemm::WarpGemm;
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;
    using CWarpDstr   = typename WarpGemm::CWarpDstr;

    static constexpr index_t kM = 256, kN = 256, kK = 64;
    static constexpr index_t kMIter = 8, kNIter = 8, kKIter = 2; // per wave, per K tile
    static_assert(Problem::BlockGemmShape::kM == kM && Problem::BlockGemmShape::kN == kN &&
                      Problem::BlockGemmShape::kK == kK,
                  "HandPipeline is written for 256x256x64");
    static_assert(Problem::kBlockSize == 256, "HandPipeline expects 4 waves");
    static_assert(sizeof(ADataType) == 2 && sizeof(BDataType) == 2, "16-bit inputs only");

    static constexpr index_t kRowBytes   = kK * 2;           // 128
    static constexpr index_t kTileBytes  = kM * kRowBytes;   // 32 KB per operand
    static constexpr index_t kStageBytes = 2 * kTileBytes;   // A + B
    static constexpr index_t kLoadsPerOp = kTileBytes / (4 * 64 * 16); // 8 per wave per operand
    static_assert(kLoadsPerOp == kMIter, "one A+B load pair is issued per MFMA row");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 2 * kStageBytes; }

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

        // LDS pointer via an address-space cast (keeps ds_read codegen), plus the same address
        // as a provably wave-uniform 32-bit value for M0.
        auto *lds = (CK_TILE_LDS_ADDR uint8_t *)p_smem;
        const index_t lds_base = __builtin_amdgcn_readfirstlane(
            static_cast<index_t>(reinterpret_cast<uintptr_t>(lds)));

        // --- global -> LDS (async). Load j of this wave fills LDS rows (j*4+wave)*8 .. +8;
        // lane l writes 16 B at physical chunk l%8 of row l/8, i.e. logical chunk (l%8)^(l/8).
        const index_t ld_row   = lane / 8;
        const index_t ld_chunk = (lane % 8) ^ ld_row;
        // One A and one B load (index j) of tile kt into stage. CK's direct global->LDS load
        // (buffer_load ... lds): M0 = wave-uniform LDS address, lane l lands at M0 + l*16. The
        // compiler does not see these loads; the explicit vmcnt waits are the synchronization.
        // Row offset of load j lives in the (scalar) buffer descriptor, so every load of an
        // operand shares one per-lane VGPR offset: lane l -> row (wave*8 + l/8) of the 32-row
        // group, 16-byte chunk (l%8)^(l/8).
        const index_t voff_a = ((wave * 8 + ld_row) * lda + ld_chunk * 8) * 2;
        const index_t voff_b = ((wave * 8 + ld_row) * ldb + ld_chunk * 8) * 2;
        auto issue_load_pair = [&](index_t kt, index_t stage, auto j) {
            __builtin_amdgcn_sched_barrier(0); // keep each pair where it is placed in the MFMA stream
            const auto ra = make_wave_buffer_resource(a_ptr + kt * kK + j * 32 * lda, 0x7ffff000);
            const auto rb = make_wave_buffer_resource(b_ptr + kt * kK + j * 32 * ldb, 0x7ffff000);
            const index_t stage_base = lds_base + stage * kStageBytes + (j * 4 + wave) * 1024;
            m0_set_with_memory(stage_base);
            async_buffer_load_dwordxn_v<4>(p_smem, ra, voff_a, 0, 0);
            m0_set_with_memory(stage_base + kTileBytes);
            async_buffer_load_dwordxn_v<4>(p_smem, rb, voff_b, 0, 0);
            __builtin_amdgcn_sched_barrier(0);
        };
        auto issue_tile = [&](index_t kt, index_t stage) {
            static_for<0, kLoadsPerOp, 1>{}([&](auto j) { issue_load_pair(kt, stage, j); });
        };

        // --- LDS -> registers. Fragment (iter i, k-half kh): row (i*2+w)*16 + lane%16, logical
        // chunk kh*4 + lane/16, stored at chunk (that ^ row%8) with row%8 == lane%8.
        const index_t fr_row = lane % 16;
        auto frag_off = [&](index_t w, index_t kh) {
            return (w * 16 + fr_row) * kRowBytes + (((kh * 4 + lane / 16) ^ (lane % 8)) * 16);
        };
        const index_t a_off[2] = {frag_off(wm, 0), frag_off(wm, 1)};
        const index_t b_off[2] = {frag_off(wn, 0), frag_off(wn, 1)};

        AWarpTensor fa[2][kMIter];
        BWarpTensor fb[2][kNIter];
        auto read_frags = [&](index_t stage, index_t kh, auto buf) {
            const auto *sa = lds + stage * kStageBytes;
            const auto *sb = sa + kTileBytes;
            static_for<0, kMIter, 1>{}([&](auto i) {
                fa[buf][i].get_thread_buffer().template set_as<Vec8>(
                    number<0>{},
                    *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(sa + a_off[kh] + i * 32 * kRowBytes));
            });
            static_for<0, kNIter, 1>{}([&](auto i) {
                fb[buf][i].get_thread_buffer().template set_as<Vec8>(
                    number<0>{},
                    *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(sb + b_off[kh] + i * 32 * kRowBytes));
            });
        };

        auto c_block = BlockGemm::MakeCBlockTile();
        clear_tile(c_block);
        auto mfma = [&](auto buf, auto &&per_row) {
            static_for<0, kMIter, 1>{}([&](auto mi) {
                per_row(mi);
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
            });
        };

        // --- prologue
        issue_tile(0, 0);
        if (num_loop > 1)
            issue_tile(1, 1);
        if (num_loop > 1)
            block_sync_lds_direct_load<2 * kLoadsPerOp>(); // tile 0 landed, tile 1 may be in flight
        else
            block_sync_lds_direct_load<0>();
        read_frags(0, 0, number<0>{});

        // --- main loop (branch-free body: branches around the MFMA blocks wreck register
        // allocation). In the last two iterations the "next" loads/reads are clamped to the
        // last tile; they hit a stage that is never read again, so they are harmless.
        for (index_t kt = 0; kt < num_loop; ++kt) {
            const index_t s       = kt & 1;
            const index_t kt_load = min(kt + 2, num_loop - 1);
            read_frags(s, 1, number<1>{});          // a)
            mfma(number<0>{}, [](auto) {});         // b)
            block_sync_lds_direct_load<0>();        // c) tile kt+1 landed; stage s fully read
            read_frags(s ^ 1, 0, number<0>{});      // e)
            // f)+d): two load pairs before each of the first 4 MFMA rows -- early enough to land
            // before the next barrier, spread enough not to clog the vector-memory queue.
            mfma(number<1>{}, [&](auto mi) {
                if constexpr (mi < kLoadsPerOp / 2) {
                    issue_load_pair(kt_load, s, number<2 * mi>{});
                    issue_load_pair(kt_load, s, number<2 * mi + 1>{});
                }
            });
        }
        // Drain the clamped loads before the epilogue reuses LDS.
        block_sync_lds_direct_load<0>();
        return c_block;
    }

    // Tuple-of-windows + element-function form used by UniversalGemmKernel::RunGemm
    // (instantiated by the grouped kernel's base class); forwards to the loop above.
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
using HandPipeline = ck_tile::tunemax_hand::HandPipeline<Problem>;
} // namespace tunemax
