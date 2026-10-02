// Hand-written GEMM main loop with inline-asm MFMAs that pin the register split:
// accumulators in AGPRs ("+a"), fragments in VGPRs ("v"). Same schedule as
// hand_pipeline_ring.hpp (K=64 tiles, 2 LDS stages, next tile's fragments held in registers:
// A as a ring, B double-buffered), which the compiler alone could not allocate without spills.
// Everything else is HIP + CK Tile primitives inside ck_tile::GroupedGemmKernel; CK unmodified.
//
// Register budget per lane: AGPR 256 = accumulators (8x8 MFMA tiles x 4 floats);
// VGPR: A ring 16x4 + B 2x16x4 = 192, plus addresses.
// MFMA order per row: k-half outer, column inner, so the two updates of each accumulator are
// 8 MFMAs apart (the compiler cannot see MFMA hazards inside inline asm).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

// Lane id via volatile asm: values derived from it are recomputed in each call instead of
// being hoisted out of the persistent kernel's tile loop (where they would be spilled).
CK_TILE_DEVICE int tunemax_lane_id() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(l));
    return l;
}

namespace ck_tile::tunemax_hand {

template <typename Problem>
struct HandPipelineAsmS : public GemmPipelineAgBgCrCompV3<Problem> {
    using Base      = GemmPipelineAgBgCrCompV3<Problem>;
    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemm =
        remove_cvref_t<decltype(UniversalGemmPipelineAgBgCrPolicy::template GetBlockGemm<Problem>())>;
    using WarpGemm    = typename BlockGemm::WarpGemm;
    using CWarpTensor = typename WarpGemm::CWarpTensor;
    using CWarpDstr   = typename WarpGemm::CWarpDstr;

    static constexpr index_t kM = 256, kN = 256, kK = 64;
    static constexpr index_t kMIter = 8, kNIter = 8, kKH = 2;
#ifndef TUNEMAX_BAR_IDX
#define TUNEMAX_BAR_IDX 11
#endif
    // MFMA index in row 7 before which the end-of-tile barrier sits
    static constexpr index_t kBarIdx = TUNEMAX_BAR_IDX;
#ifndef TUNEMAX_A_PER_ROW
#define TUNEMAX_A_PER_ROW 4
#endif
    static constexpr index_t kAPerRow = TUNEMAX_A_PER_ROW;
    // Unpadded instances (whose C goes through tunemax::FastEpilogue) compute C^T per MFMA tile
    // (operands swapped): each lane then holds 4 consecutive columns of one row, which the
    // epilogue stores without a transpose. Padded instances keep CK's C layout.
    static constexpr bool kTransposedAcc = !Problem::kPadM && !Problem::kPadN && !Problem::kPadK &&
                                           std::is_same_v<ADataType, bf16_t>;
#ifndef TUNEMAX_B_OFF
#define TUNEMAX_B_OFF 1
#endif
    static constexpr index_t kBOff    = TUNEMAX_B_OFF; // B loads after MFMA kBOff and 8 + kBOff
#ifndef TUNEMAX_B3
#define TUNEMAX_B3 1
#endif
    // kB3: B gets a third LDS buffer (A 2 x 32 KB + B 3 x 32 KB = all 160 KB of gfx950 LDS), so
    // B(kt+3) can overwrite B(kt)'s buffer anywhere in iteration kt: one barrier per iteration.
    static constexpr bool kB3 = TUNEMAX_B3;
#ifndef TUNEMAX_B_ROW0
#define TUNEMAX_B_ROW0 4
#endif
    // B(kt+3) loads go in rows kBRow0 .. kBRow0+3 (kB3 only; otherwise rows 4-7)
    static constexpr index_t kBRow0 = kB3 ? TUNEMAX_B_ROW0 : 4;
    // B loads issued before the row-7 barrier, i.e. after A(kt+1)'s loads
#ifndef TUNEMAX_LOAD_MODE
#define TUNEMAX_LOAD_MODE 0
#endif
    // kLoadMode 1 (kB3 only): A(kt+1) 2 per row in rows 0-3 (after MFMAs 0, 8); B(kt+3) 1 per
    // row in all 8 rows (after MFMA 11 in rows 0-3, MFMA 9 in rows 4-7): one memory op per gap
    // and an even stream of loads. kLoadMode 0: A per kAPerRow, B 2 per row from kBRow0.
    static constexpr index_t kLoadMode = kB3 ? TUNEMAX_LOAD_MODE : 0;
#ifndef TUNEMAX_EARLY_A
#define TUNEMAX_EARLY_A 2
#endif
    // kEarlyA (kLoadMode 0, kAPerRow 4): the first kEarlyA loads of A(kt+2) are issued in row 7
    // of iteration kt, right after the barrier that frees their stage (after MFMAs 12, 14, 15).
    static constexpr index_t kEarlyA = (kB3 && kAPerRow == 4) ? TUNEMAX_EARLY_A : 0;
    // B load slot (MFMA index) in row r for kLoadMode 1
    static constexpr index_t b1_idx(index_t r) { return r < 4 ? 11 : 9; }
    static constexpr index_t kBLoadsBeforeBar = [] {
        index_t n = 0;
        if (kLoadMode == 1) { // B loads after the last A load (row 3, MFMA 8) and before the barrier
            for (index_t r = 3; r < 8; ++r)
                n += r < 7 ? 1 : (kBarIdx > b1_idx(7));
            return n;
        }
        for (index_t r = kBRow0; r < kBRow0 + 4; ++r)
            n += r < 7 ? 2 : (kBarIdx > kBOff) + (kBarIdx > 8 + kBOff);
        return n;
    }();
    static_assert(Problem::BlockGemmShape::kM == kM && Problem::BlockGemmShape::kN == kN &&
                      Problem::BlockGemmShape::kK == kK,
                  "HandPipelineAsmS is written for 256x256x64");
    static_assert(Problem::kBlockSize == 256, "4 waves");
    static_assert(std::is_same_v<ADataType, BDataType> &&
                      (std::is_same_v<ADataType, bf16_t> || std::is_same_v<ADataType, half_t>),
                  "bf16 or fp16 inputs");
    // K tails are not masked (columns past K are the next row's data): the host only launches
    // this pipeline when K % kK == 0.

    static constexpr index_t kRowBytes   = kK * 2;
    static constexpr index_t kTileBytes  = kM * kRowBytes;
    static constexpr index_t kStageBytes = 2 * kTileBytes;
    static constexpr index_t kLoads      = kTileBytes / (4 * 1024); // 8 per wave per operand

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() {
        return kB3 ? 5 * kTileBytes : 2 * kStageBytes;
    }
    // byte offsets of A stage s and B buffer t in LDS
    CK_TILE_HOST_DEVICE static constexpr index_t a_base(index_t s) {
        return kB3 ? s * kTileBytes : s * kStageBytes;
    }
    CK_TILE_HOST_DEVICE static constexpr index_t b_base(index_t t) {
        return kB3 ? (2 + t) * kTileBytes : t * kStageBytes + kTileBytes;
    }

    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
    // Fragments as raw 128-bit bit patterns (always exactly 4 VGPRs); the MFMA reads them as bf16x8.
    using Vec8 = ext_vector_t<int32_t, 4>;
    using F4   = ext_vector_t<float, 4>;

    // Window -> (pointer at the window origin, leading dimension, bytes from there to the end of
    // the tensor). The byte count bounds the buffer resource, so rows past the matrix (partial
    // M/N tiles) read as zeros.
    template <typename Window>
    CK_TILE_DEVICE static auto base_and_ld(const Window &win, index_t &ld, index_t &bytes) {
        const auto &view = win.get_bottom_tensor_view();
        const auto &desc = view.get_tensor_descriptor();
        ld = desc.calculate_offset(make_multi_index(1, 0)) -
             desc.calculate_offset(make_multi_index(0, 0));
        const index_t origin = desc.calculate_offset(win.get_window_origin());
        bytes = (static_cast<index_t>(view.get_buffer_view().buffer_size_) - origin) *
                static_cast<index_t>(sizeof(ADataType));
        return view.get_buffer_view().p_data_ + origin;
    }

    // acc = a * b (C operand 0): the first K step initializes the accumulators
    CK_TILE_DEVICE static void mfma_init(F4 &acc, const Vec8 &a, const Vec8 &b) {
        if constexpr (std::is_same_v<ADataType, bf16_t>)
            asm("v_mfma_f32_16x16x32_bf16 %0, %1, %2, 0" : "=a"(acc) : "v"(a), "v"(b));
        else
            asm("v_mfma_f32_16x16x32_f16 %0, %1, %2, 0" : "=a"(acc) : "v"(a), "v"(b));
    }
    CK_TILE_DEVICE static void mfma(F4 &acc, const Vec8 &a, const Vec8 &b) {
        if constexpr (std::is_same_v<ADataType, bf16_t>)
            asm("v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0" : "+a"(acc) : "v"(a), "v"(b));
        else
            asm("v_mfma_f32_16x16x32_f16 %0, %1, %2, %0" : "+a"(acc) : "v"(a), "v"(b));
    }

    template <typename ADramWindow, typename BDramWindow,
              std::enable_if_t<!is_detected<is_tuple, ADramWindow>::value, bool> = true>
    CK_TILE_DEVICE auto operator()(const ADramWindow &a_win, const BDramWindow &b_win,
                                   index_t num_loop, void *p_smem) const {
        const index_t lane = tunemax_lane_id(); // volatile: recomputed per tile, not spilled
        const index_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
        const index_t wm = wave / 2, wn = wave % 2;

        index_t lda, ldb, a_bytes, b_bytes;
        const ADataType *a_ptr = base_and_ld(a_win, lda, a_bytes);
        const BDataType *b_ptr = base_and_ld(b_win, ldb, b_bytes);

        auto *lds              = (CK_TILE_LDS_ADDR uint8_t *)p_smem;
        const index_t lds_base = __builtin_amdgcn_readfirstlane(
            static_cast<index_t>(reinterpret_cast<uintptr_t>(lds)));

        // Async load j (0..7) of an operand: rows (j*4+wave)*8 .. +8; lane l -> row l/8,
        // physical chunk l%8, logical chunk (l%8)^(l/8). 128-byte rows = whole cache lines.
        const index_t ld_row   = lane / 8;
        const index_t ld_chunk = (lane % 8) ^ ld_row;
        const index_t voff_a   = ((wave * 8 + ld_row) * lda + ld_chunk * 8) * 2;
        const index_t voff_b   = ((wave * 8 + ld_row) * ldb + ld_chunk * 8) * 2;
        // Fixed buffer resources at k = 0; the K tile goes in the instruction's soffset, so
        // stepping K costs 2 SALU ops instead of rebuilding two resources.
        const int32x4_t rsrc_a0 = make_wave_buffer_resource(a_ptr, a_bytes);
        const int32x4_t rsrc_b0 = make_wave_buffer_resource(b_ptr, b_bytes);
        auto make_rsrc = [&](index_t kt) { return kt * (kK * 2); }; // soffset of K tile kt
        // buf_off: byte offset of the destination tile buffer (a_base / b_base)
        auto issue_load = [&](index_t koff, index_t buf_off, auto j, auto op) {
            __builtin_amdgcn_sched_barrier(0);
            const index_t dst = lds_base + buf_off + (j * 4 + wave) * 1024;
            m0_set_with_memory(dst);
            const index_t voff = op == 0 ? voff_a + j * 32 * lda * 2 : voff_b + j * 32 * ldb * 2;
            asm volatile("buffer_load_dwordx4 %1, %2, %3 offen lds"
                         : "=r"(p_smem) /* dummy dependency for smem */
                         : "v"(voff), "s"(op == 0 ? rsrc_a0 : rsrc_b0), "s"(koff)
                         : "memory");
            __builtin_amdgcn_sched_barrier(0);
        };

        // Fragment (iter i, k-half kh): row (i*2+w)*16 + lane%16, logical chunk kh*4 + lane/16,
        // stored at chunk (that ^ lane%8).
        auto frag_off = [&](index_t w, index_t kh) {
            return (w * 16 + lane % 16) * kRowBytes + (((kh * 4 + lane / 16) ^ (lane % 8)) * 16);
        };
        const index_t a_off[2] = {frag_off(wm, 0), frag_off(wm, 1)};
        const index_t b_off[2] = {frag_off(wn, 0), frag_off(wn, 1)};

        Vec8 fa[2][kKH];         // A rows, read just in time (2-row ring)
        Vec8 fb[2][kNIter][kKH]; // double-buffered
        F4 acc[kMIter][kNIter]; // initialized by the first K step
        auto read_a = [&](index_t stage, auto i) {
            static_for<0, kKH, 1>{}([&](auto kh) {
                fa[i % 2][kh] = *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                    lds + a_base(stage) + a_off[kh] + i * 32 * kRowBytes);
            });
        };
        auto read_b = [&](index_t stage, auto buf, auto i) {
            static_for<0, kKH, 1>{}([&](auto kh) {
                fb[buf][i][kh] = *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(
                    lds + b_base(stage) + b_off[kh] + i * 32 * kRowBytes);
            });
        };

        // --- prologue: A0, B0 -> stage 0; B1 -> stage 1; pre-read B0 and A0 row 0; B2 -> stage 0.
        {
            const auto r0 = make_rsrc(0), r1 = make_rsrc(min(index_t{1}, num_loop - 1));
            static_for<0, kLoads, 1>{}([&](auto j) { issue_load(r0, a_base(0), j, number<0>{}); });
            static_for<0, kLoads, 1>{}([&](auto j) { issue_load(r0, b_base(0), j, number<1>{}); });
            static_for<0, kLoads, 1>{}([&](auto j) { issue_load(r1, b_base(1), j, number<1>{}); });
        }
        block_sync_lds_direct_load<kLoads>(); // A0, B0 landed
        static_for<0, kNIter, 1>{}([&](auto i) { read_b(0, number<0>{}, i); });
        read_a(0, number<0>{});
        s_waitcnt_barrier<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>(); // B0 read everywhere
        {
            const auto r2 = make_rsrc(min(index_t{2}, num_loop - 1));
            static_for<0, kLoads, 1>{}([&](auto j) { issue_load(r2, b_base(kB3 ? 2 : 0), j, number<1>{}); });
        }

        // One LDS read of a 16x32 fragment.
        auto rd = [&](Vec8 &dst, index_t off) {
            dst = *reinterpret_cast<const CK_TILE_LDS_ADDR Vec8 *>(lds + off);
        };
        // Iteration kt, stage s = kt % 2. Every MFMA is followed by a sched_barrier and at most
        // one memory op, so issue costs hide under the MFMA in flight.
        //   entry    : A(kt) row 0 and B(kt) frags already in registers
        //   rows 0-3 : A(kt+1) -> stage s^1; pre-read B(kt+1) from stage s^1
        //   row 4    : barrier (all waves done reading B(kt+1)); rows 4-7: B(kt+3) -> stage s^1
        //   row 7    : wait A(kt+1) landed + barrier (also: all waves done reading A(kt) in s),
        //              then read A(kt+1) row 0 from stage s^1
        // t3 = kt % 3 (kB3 B buffer rotation), a compile-time constant like buf (loop unrolled x6)
        auto iteration = [&](index_t kt, auto buf, auto first, auto t3) {
            constexpr index_t s = buf; // stage of tile kt (buf == kt % 2)
            const auto rsrc_a = make_rsrc(min(kt + 1, num_loop - 1));
            const auto rsrc_a2 = make_rsrc(min(kt + 2, num_loop - 1)); // early A(kt+2) loads
            const auto rsrc_b = make_rsrc(min(kt + 3, num_loop - 1));
            const index_t a_cur = a_base(s), a_nxt = a_base(s ^ 1);
            // B(kt+1) is pre-read from b_nxt; B(kt+3) is loaded into b_wr
            constexpr index_t b_wr  = kB3 ? b_base(t3) : b_base(s ^ 1);
            constexpr index_t b_nxt = kB3 ? b_base(t3 == 2 ? 0 : t3 + 1) : b_base(s ^ 1);
            static_for<0, kMIter, 1>{}([&](auto mi) {
                static_for<0, kKH * kNIter, 1>{}([&](auto idx) {
                    constexpr index_t kh = idx / kNIter, ni = idx % kNIter;
                    if constexpr (!kB3 && mi == 4 && idx == kBOff) // all waves done reading B(kt+1)
                        s_waitcnt_barrier<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                    // loads issued after A(kt+1)'s may fly: B(kt+3) of rows 4-6 and row 7 so far
                    if constexpr (mi == 7 && idx == kBarIdx)
                        s_waitcnt_barrier<kBLoadsBeforeBar, waitcnt_arg::kMaxExpCnt, 0>();
                    const Vec8 &x = kTransposedAcc ? fb[buf][ni][kh] : fa[mi % 2][kh];
                    const Vec8 &y = kTransposedAcc ? fa[mi % 2][kh] : fb[buf][ni][kh];
                    if constexpr (first && kh == 0)
                        mfma_init(acc[mi][ni], x, y);
                    else
                        mfma(acc[mi][ni], x, y);
                    __builtin_amdgcn_sched_barrier(0);
                    // loads after MFMA 0 and 8
                    // A(kt+1): kAPerRow loads per row from row 0; B(kt+3): 2 per row in rows 4-7
                    // (4 per row: after MFMAs 0, 5, 8, 13, which hold no LDS read)
                    constexpr index_t a_slot =
                        kAPerRow == 2 ? (idx == 0 ? 0 : idx == 8 ? 1 : -1)
                                      : (idx == 0 ? 0 : idx == 5 ? 1 : idx == 8 ? 2 : idx == 13 ? 3 : -1);
                    if constexpr (kLoadMode == 1) {
                        if constexpr (mi < 4 && idx % 8 == 0)
                            issue_load(rsrc_a, a_nxt, number<mi * 2 + idx / 8>{}, number<0>{});
                        if constexpr (idx == b1_idx(mi))
                            issue_load(rsrc_b, b_wr, number<mi>{}, number<1>{});
                    } else {
                    // loads j < kEarlyA were issued by the previous iteration (not by tile 0)
                    constexpr index_t a_j = mi * kAPerRow + a_slot - (first ? 0 : kEarlyA);
                    if constexpr (a_slot >= 0 && a_j >= 0 && a_j + (first ? 0 : kEarlyA) < kLoads)
                        issue_load(rsrc_a, a_nxt, number<a_j + (first ? 0 : kEarlyA)>{}, number<0>{});
                    constexpr index_t e_slot = idx == 12 ? 0 : idx == 14 ? 1 : idx == 15 ? 2 : -1;
                    if constexpr (mi == 7 && e_slot >= 0 && e_slot < kEarlyA)
                        issue_load(rsrc_a2, a_cur, number<e_slot>{}, number<0>{});
                    if constexpr (mi >= kBRow0 && mi < kBRow0 + 4 && idx % 8 == kBOff)
                        issue_load(rsrc_b, b_wr, number<(mi - kBRow0) * 2 + idx / 8>{}, number<1>{});
                    }
                    // next A row (or next tile's row 0 in row 7) after MFMA 2 and 4
                    // memory ops: loads after even MFMAs, LDS reads after odd ones
                    if constexpr (mi + 1 < kMIter && (idx == 2 || idx == 4)) {
                        constexpr index_t kk = idx / 4;
                        rd(fa[(mi + 1) % 2][kk], a_cur + a_off[kk] + (mi + 1) * 32 * kRowBytes);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    if constexpr (mi == 7 && (idx == kBarIdx || idx == kBarIdx + 2)) {
                        constexpr index_t kk = (idx - kBarIdx) / 2;
                        rd(fa[0][kk], a_nxt + a_off[kk]);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    // rows 0-3: next tile's B, columns 2mi and 2mi+1, after MFMA 6, 10, 12, 14
                    // (row 3 earlier, so they are done by the barrier at the start of row 4)
                    constexpr index_t q = mi < 3 ? (idx == 6 ? 0 : idx == 10 ? 1 : idx == 12 ? 2 : idx == 14 ? 3 : -1)
                                                 : (idx == 3 ? 0 : idx == 5 ? 1 : idx == 6 ? 2 : idx == 7 ? 3 : -1);
                    if constexpr (mi < 4 && q >= 0) {
                        constexpr index_t c = 2 * mi + q / 2, kk = q % 2;
                        rd(fb[1 - buf][c][kk], b_nxt + b_off[kk] + c * 32 * kRowBytes);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                });
            });
        };
        // tile 0 peeled: its first K step initializes the accumulators (no zero fill)
        // tile kt uses buf = kt % 2 and t3 = kt % 3: unroll by 6 so both are constants
        iteration(0, number<0>{}, true_type{}, number<0>{});
        index_t kt = 1;
        for (; kt + 5 < num_loop; kt += 6) {
            static_for<1, 7, 1>{}([&](auto u) {
                iteration(kt + u - 1, number<u % 2>{}, false_type{}, number<u % 3>{});
            });
        }
        static_for<1, 6, 1>{}([&](auto u) { // up to 5 remaining tiles
            if (kt + u - 1 < num_loop)
                iteration(kt + u - 1, number<u % 2>{}, false_type{}, number<u % 3>{});
        });
        s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>(); // drain before the epilogue uses LDS

        // Accumulators -> CK's C tile (same per-lane layout as CK's warp GEMM, verified).
        auto c_block = BlockGemm::MakeCBlockTile();
        static_for<0, kMIter, 1>{}([&](auto mi) {
            static_for<0, kNIter, 1>{}([&](auto ni) {
                CWarpTensor c_w;
                c_w.get_thread_buffer().template set_as<F4>(number<0>{}, acc[mi][ni]);
                c_block.set_y_sliced_thread_data(
                    merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths), c_w.get_thread_buffer());
            });
        });
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
using HandPipelineAsmS = ck_tile::tunemax_hand::HandPipelineAsmS<Problem>;
} // namespace tunemax
