// Hand-written GEMM main loop, K=64 tiles, 2 LDS stages + registers as a third buffer.
// GemmPipeline for ck_tile::GroupedGemmKernel; built from CK Tile primitives, CK unmodified.
//
// 4 waves (2x2), block tile 256x256x64, warp tile 128x128 in 16x16x32 bf16 MFMAs, laid out
// like CK's block GEMM so the C tile feeds CK's epilogue unchanged.
//
// Why K=64: LDS rows of 128 B make every async load cover whole 128-byte cache lines (K=32
// rows are 64 B, half lines, ~2x the load cost). Why registers: only 2 K=64 stages fit in LDS,
// so the next tile's fragments are pulled into registers during the current tile, which frees
// its LDS stage a tile early:
//   - B fragments are double-buffered (every MFMA row needs all of B): next tile's B is read
//     during rows 0-3 of the current tile.
//   - A fragments are a ring: right after MFMA row i is done with A[i], next tile's A[i] is
//     read into the same registers.
// Iteration kt (stage s = kt % 2 holds tile kt, already fully read into registers):
//   wait: tile kt+1 landed in stage s^1, all LDS reads of stage s done; barrier
//   rows 0-7: 16 MFMAs each; rows 0-3 also issue the async loads of tile kt+2 into stage s and
//   read next-tile B frags; every row then reads next-tile A[row].
// LDS: 128-byte rows, 16-byte chunk c of row r stored at c ^ (r % 8) (conflict free, measured).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile::tunemax_hand {

template <typename Problem>
struct HandPipelineRing : public GemmPipelineAgBgCrCompV3<Problem> {
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

    static constexpr index_t kM = 256, kN = 256, kK = 64;
    static constexpr index_t kMIter = 8, kNIter = 8, kKH = 2;
    static_assert(Problem::BlockGemmShape::kM == kM && Problem::BlockGemmShape::kN == kN &&
                      Problem::BlockGemmShape::kK == kK,
                  "HandPipelineRing is written for 256x256x64");
    static_assert(Problem::kBlockSize == 256, "4 waves");
    static_assert(sizeof(ADataType) == 2 && sizeof(BDataType) == 2, "16-bit inputs only");

    static constexpr index_t kRowBytes   = kK * 2;            // 128
    static constexpr index_t kTileBytes  = kM * kRowBytes;    // 32 KB per operand
    static constexpr index_t kStageBytes = 2 * kTileBytes;    // 64 KB
    static constexpr index_t kLoads      = kTileBytes / (4 * 1024); // 8 per wave per operand

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

        auto *lds              = (CK_TILE_LDS_ADDR uint8_t *)p_smem;
        const index_t lds_base = __builtin_amdgcn_readfirstlane(
            static_cast<index_t>(reinterpret_cast<uintptr_t>(lds)));

        // Async load j (0..7) of an operand: rows (j*4+wave)*8 .. +8 (1 KB); lane l -> row l/8,
        // physical chunk l%8, logical chunk (l%8) ^ (l/8).
        const index_t ld_row   = lane / 8;
        const index_t ld_chunk = (lane % 8) ^ ld_row;
        const index_t voff_a   = ((wave * 8 + ld_row) * lda + ld_chunk * 8) * 2;
        const index_t voff_b   = ((wave * 8 + ld_row) * ldb + ld_chunk * 8) * 2;
        auto make_rsrc = [&](index_t kt) {
            return make_tuple(make_wave_buffer_resource(a_ptr + kt * kK, 0x7ffff000),
                              make_wave_buffer_resource(b_ptr + kt * kK, 0x7ffff000));
        };
        auto issue_load = [&](const auto &rsrc, index_t stage, auto j, auto op) {
            __builtin_amdgcn_sched_barrier(0);
            const index_t dst =
                lds_base + stage * kStageBytes + op * kTileBytes + (j * 4 + wave) * 1024;
            m0_set_with_memory(dst);
            if constexpr (op == 0)
                async_buffer_load_dwordxn_v<4>(p_smem, rsrc[number<0>{}], voff_a + j * 32 * lda * 2, 0, 0);
            else
                async_buffer_load_dwordxn_v<4>(p_smem, rsrc[number<1>{}], voff_b + j * 32 * ldb * 2, 0, 0);
            __builtin_amdgcn_sched_barrier(0);
        };
        auto issue_tile = [&](index_t kt, index_t stage) {
            const auto rsrc = make_rsrc(kt);
            static_for<0, kLoads, 1>{}([&](auto j) {
                issue_load(rsrc, stage, j, number<0>{});
                issue_load(rsrc, stage, j, number<1>{});
            });
        };

        // Fragment (iter i, k-half kh): row (i*2+w)*16 + lane%16 (row%8 == lane%8), logical
        // chunk kh*4 + lane/16.
        auto frag_off = [&](index_t w, index_t kh) {
            return (w * 16 + lane % 16) * kRowBytes + (((kh * 4 + lane / 16) ^ (lane % 8)) * 16);
        };
        const index_t a_off[2] = {frag_off(wm, 0), frag_off(wm, 1)};
        const index_t b_off[2] = {frag_off(wn, 0), frag_off(wn, 1)};

        AWarpTensor fa[kMIter][kKH];      // ring
        BWarpTensor fb[2][kNIter][kKH];   // double-buffered
        auto read_a = [&](index_t stage, auto i) {
            static_for<0, kKH, 1>{}([&](auto kh) {
                fa[i][kh].get_thread_buffer().template set_as<Vec8>(
                    number<0>{}, *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                                     lds + stage * kStageBytes + a_off[kh] + i * 32 * kRowBytes));
            });
        };
        auto read_b = [&](index_t stage, auto buf, auto i) {
            static_for<0, kKH, 1>{}([&](auto kh) {
                fb[buf][i][kh].get_thread_buffer().template set_as<Vec8>(
                    number<0>{}, *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                                     lds + stage * kStageBytes + kTileBytes + b_off[kh] + i * 32 * kRowBytes));
            });
        };

        auto c_block = BlockGemm::MakeCBlockTile();
        clear_tile(c_block);
        auto mfma_one = [&](auto buf, auto mi, auto ni, auto kh) {
            CWarpTensor c_w;
            c_w.get_thread_buffer() = c_block.get_y_sliced_thread_data(
                merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
            WarpGemm{}(c_w, fa[mi][kh], fb[buf][ni][kh]);
            c_block.set_y_sliced_thread_data(
                merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths), c_w.get_thread_buffer());
        };

        // --- prologue: tiles 0 and 1 in flight; tile 0 fragments in registers
        issue_tile(0, 0);
        issue_tile(min(index_t{1}, num_loop - 1), 1);
        block_sync_lds_direct_load<2 * kLoads>(); // tile 0 landed
        static_for<0, kMIter, 1>{}([&](auto i) { read_a(0, i); });
        static_for<0, kNIter, 1>{}([&](auto i) { read_b(0, number<0>{}, i); });

        auto iteration = [&](index_t kt, auto buf) {
            // tile kt+1 landed; every wave's LDS reads of stage kt%2 (tile kt) are done
            s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();
            const index_t s      = kt & 1;
            const auto rsrc_load = make_rsrc(min(kt + 2, num_loop - 1));
            static_for<0, kMIter, 1>{}([&](auto mi) {
                static_for<0, kNIter, 1>{}([&](auto ni) {
                    static_for<0, kKH, 1>{}([&](auto kh) {
                        mfma_one(buf, mi, ni, kh);
                        // rows 0-3: one async load of tile kt+2 after every 4th MFMA
                        constexpr index_t idx = ni * kKH + kh;
                        if constexpr (mi < 4 && idx % 4 == 0) {
                            constexpr index_t ld = mi * 4 + idx / 4; // 0..15
                            issue_load(rsrc_load, s, number<ld / 2>{}, number<ld % 2>{});
                        }
                    });
                });
                if constexpr (mi < 4) { // next tile's B frags, 2 per row
                    read_b(s ^ 1, number<1 - buf>{}, number<2 * mi>{});
                    read_b(s ^ 1, number<1 - buf>{}, number<2 * mi + 1>{});
                }
                read_a(s ^ 1, mi); // ring: next tile's A[mi] replaces the one just used
            });
        };
        index_t kt = 0;
        for (; kt + 1 < num_loop; kt += 2) {
            iteration(kt, number<0>{});
            iteration(kt + 1, number<1>{});
        }
        if (kt < num_loop)
            iteration(kt, number<0>{});
        s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>(); // drain before the epilogue uses LDS
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
using HandPipelineRing = ck_tile::tunemax_hand::HandPipelineRing<Problem>;
} // namespace tunemax
