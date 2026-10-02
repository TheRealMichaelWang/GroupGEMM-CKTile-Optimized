// C epilogue for the hand pipelines (4 waves 2x2, 16x16 MFMA accumulators, full tiles).
//
// CK's CShuffleEpilogue moves the 256x256 C tile through LDS in many small rounds (2-byte
// LDS writes, a barrier and an LDS wait per round), ~8% of the kernel time. Here:
//   1. each lane's 4 accumulator rows (one column) are converted to bf16 and transposed
//      inside each lane quad with DPP, so lane j of a quad holds one row x 4 columns;
//   2. one ds_write_b64 per 16x16 MFMA tile stages the whole C tile in LDS, row major
//      (row stride 528 B);
//   3. one barrier, then each wave streams 64 full rows (512 B) out with ds_read_b128 +
//      buffer_store_dwordx4, with the chosen cache policy.
// Padded instances, atomic (split-K) windows and other dtypes fall back to the base epilogue.
#pragma once

#include "ck_tile/core.hpp"
#include "coherence_epilogue.hpp"

namespace tunemax {

template <typename BaseEpilogue, int Mode, bool Pad>
struct FastEpilogue : public BaseEpilogue {
    static constexpr ck_tile::index_t kM = 256, kN = 256;
    static constexpr ck_tile::index_t kRowStride = kN * 2 + 16; // bytes, keeps 16 B alignment
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
        return ck_tile::max(BaseEpilogue::GetSmemSize(), kM * kRowStride);
    }

    template <typename CWindow, typename CTile, typename DWindows>
    CK_TILE_DEVICE auto operator()(CWindow &c_window, const CTile &c_tile,
                                   const DWindows &d_windows, void *smem) {
        using namespace ck_tile;
        using View = remove_cvref_t<decltype(c_window.get_bottom_tensor_view())>;
        using CData = remove_cvref_t<typename View::DataType>;
        if constexpr (Pad || View::DstInMemOp != memory_operation_enum::set ||
                      !std::is_same_v<CData, bf16_t>) {
            auto c_win = with_store_mode<Mode>(c_window);
            return BaseEpilogue::operator()(c_win, c_tile, d_windows, smem);
        } else {
            // C tile Y dims: (MIter, NIter, per-MFMA-tile dims...), as CK's block GEMM lays it out
            constexpr auto y_lengths = to_sequence(
                CTile::get_tile_distribution().get_ys_to_d_descriptor().get_lengths());
            static_assert(y_lengths[number<0>{}] == 8 && y_lengths[number<1>{}] == 8,
                          "FastEpilogue expects 8x8 16x16 MFMA tiles per wave");
            constexpr auto c_warp_y_lengths     = y_lengths.pop_front().pop_front();
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<c_warp_y_lengths.size(), 0>{};
            using F4 = ext_vector_t<float, 4>;

            const index_t lane = threadIdx.x % 64;
            const index_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
            const index_t wm = wave / 2, wn = wave % 2;
            const index_t j  = lane % 4; // position in the lane quad
            auto *lds = (CK_TILE_LDS_ADDR uint8_t *)smem;

            // after the transpose lane holds row (lane/16)*4 + j, columns (lane%16 & ~3) .. +3
            const index_t w_off = (wm * 16 + (lane / 16) * 4 + j) * kRowStride +
                                  (wn * 16 + (lane % 16) - j) * 2;
            // byte selectors for the 16-bit interleave of step 2
            const uint32_t sel = (j & 1) ? 0x03020706u : 0x05040100u;

            static_for<0, 8, 1>{}([&](auto mi) {
                static_for<0, 8, 1>{}([&](auto ni) {
                    const auto v = c_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    const F4 f{v[number<0>{}], v[number<1>{}], v[number<2>{}], v[number<3>{}]};
                    // d0 = rows 0,1; d1 = rows 2,3 of this lane's column
                    uint32_t d0 = bit_cast<uint32_t>(
                        ext_vector_t<bf16_t, 2>{type_convert<bf16_t>(f[0]), type_convert<bf16_t>(f[1])});
                    uint32_t d1 = bit_cast<uint32_t>(
                        ext_vector_t<bf16_t, 2>{type_convert<bf16_t>(f[2]), type_convert<bf16_t>(f[3])});
                    // step 1: lanes 0,1 swap their d1 with the d0 of lanes 2,3
                    const uint32_t send = (j < 2) ? d1 : d0;
                    const uint32_t recv = __builtin_amdgcn_mov_dpp(send, 0x4e, 0xf, 0xf, true); // [2,3,0,1]
                    if (j < 2) d1 = recv; else d0 = recv;
                    // step 2: 16-bit interleave with the pair partner (lane ^ 1)
                    const uint32_t o0 = __builtin_amdgcn_mov_dpp(d0, 0xb1, 0xf, 0xf, true); // [1,0,3,2]
                    const uint32_t o1 = __builtin_amdgcn_mov_dpp(d1, 0xb1, 0xf, 0xf, true);
                    // even: (lo(d), lo(o)); odd: (hi(o), hi(d))
                    const uint32_t r0 = __builtin_amdgcn_perm(o0, d0, sel);
                    const uint32_t r1 = __builtin_amdgcn_perm(o1, d1, sel);
                    *reinterpret_cast<CK_TILE_LDS_ADDR ext_vector_t<uint32_t, 2> *>(
                        lds + w_off + mi * 32 * kRowStride + ni * 32 * 2) =
                        ext_vector_t<uint32_t, 2>{r0, r1};
                });
            });
            block_sync_lds();

            // each wave stores rows wave*64 .. +64; 2 rows per instruction, 16 B per lane
            const auto &view = c_window.get_bottom_tensor_view();
            const auto &desc = view.get_tensor_descriptor();
            const index_t ldc = desc.calculate_offset(make_multi_index(1, 0)) -
                                desc.calculate_offset(make_multi_index(0, 0));
            const CData *c_ptr =
                view.get_buffer_view().p_data_ + desc.calculate_offset(c_window.get_window_origin());
            const auto rsrc = cast_to_amdgpu_buffer_rsrc_t(
                make_wave_buffer_resource(c_ptr, 0x7ffff000));
            const index_t r_row = wave * 64 + lane / 32, chunk = lane % 32;
            const index_t g_off = (r_row * ldc + chunk * 8) * 2;
            const index_t l_off = r_row * kRowStride + chunk * 16;
            // batches of 16 LDS reads, then 16 stores (registers are free after the main loop)
            using U4 = ext_vector_t<uint32_t, 4>;
            static_for<0, 32, 16>{}([&](auto i0) {
                U4 d[16];
                static_for<0, 16, 1>{}([&](auto i) {
                    d[i] = *reinterpret_cast<const CK_TILE_LDS_ADDR U4 *>(
                        lds + l_off + (i0 + i) * 2 * kRowStride);
                });
                static_for<0, 16, 1>{}([&](auto i) {
                    __builtin_amdgcn_raw_buffer_store_b128(
                        d[i], rsrc, g_off + (i0 + i) * 2 * ldc * 2, 0,
                        static_cast<int>(store_coherence<Mode>()));
                });
            });
        }
    }
};

} // namespace tunemax
