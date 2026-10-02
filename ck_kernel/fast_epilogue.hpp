// C epilogue for the hand pipelines (4 waves 2x2, 16x16 MFMA accumulators holding C^T, full
// tiles; see HandPipelineAsmS::kTransposedAcc).
//
// CK's CShuffleEpilogue moves the 256x256 C tile through LDS in many small rounds (2-byte
// LDS writes, a barrier and an LDS wait per round), ~8% of the kernel time. Here:
//   1. the hand pipelines compute C^T per 16x16 MFMA tile (swapped operands), so each lane
//      holds 4 consecutive columns of one row; converted to bf16 they are 8 contiguous bytes;
//   2. one ds_write_b64 per 16x16 MFMA tile stages the whole C tile in LDS, row major
//      (row stride 528 B, conflict free);
//   3. one barrier, then each wave streams 64 full rows (512 B) out with ds_read_b128 +
//      buffer_store_dwordx4, with the chosen cache policy.
// Padded instances mask rows past M (bounded buffer resource) and columns past N (CK's
// padded-view validity check). bf16 and fp16; split-K (atomic) windows are unsupported.
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
        if constexpr (!std::is_same_v<CData, bf16_t> && !std::is_same_v<CData, half_t>) {
            auto c_win = with_store_mode<Mode>(c_window);
            return BaseEpilogue::operator()(c_win, c_tile, d_windows, smem);
        } else if constexpr (View::DstInMemOp != memory_operation_enum::set) {
            // The accumulators hold C^T tiles, which CK's (split-K atomic) epilogue cannot take.
            // The tunemax backend always launches k_batch = 1.
            __builtin_trap();
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

            const index_t lane = tunemax_lane_id(); // volatile: recomputed per tile, not spilled
            const index_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
            const index_t wm = wave / 2, wn = wave % 2;
            const index_t j  = lane % 4; // position in the lane quad
            auto *lds = (CK_TILE_LDS_ADDR uint8_t *)smem;

            // Accumulators hold C^T per 16x16 MFMA tile (hand pipelines swap the MFMA operands
            // in unpadded instances): lane l has row l%16, columns (l/16)*4 .. +3.
            // Row stride 528 B: the 32 lanes of a half wave hit 64 distinct banks.
            const index_t w_off = (wm * 16 + lane % 16) * kRowStride + (wn * 16 + (lane / 16) * 4) * 2;
            static_for<0, 8, 1>{}([&](auto mi) {
                static_for<0, 8, 1>{}([&](auto ni) {
                    const auto v = c_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mi, ni>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    const uint32_t d0 = bit_cast<uint32_t>(ext_vector_t<CData, 2>{
                        type_convert<CData>(v[number<0>{}]), type_convert<CData>(v[number<1>{}])});
                    const uint32_t d1 = bit_cast<uint32_t>(ext_vector_t<CData, 2>{
                        type_convert<CData>(v[number<2>{}]), type_convert<CData>(v[number<3>{}])});
                    *reinterpret_cast<CK_TILE_LDS_ADDR ext_vector_t<uint32_t, 2> *>(
                        lds + w_off + mi * 32 * kRowStride + ni * 32 * 2) =
                        ext_vector_t<uint32_t, 2>{d0, d1};
                });
            });
            block_sync_lds();

            // each wave stores rows wave*64 .. +64; 2 rows per instruction, 16 B per lane
            const auto &view = c_window.get_bottom_tensor_view();
            const auto &desc = view.get_tensor_descriptor();
            const index_t ldc = desc.calculate_offset(make_multi_index(1, 0)) -
                                desc.calculate_offset(make_multi_index(0, 0));
            const auto origin  = c_window.get_window_origin();
            const index_t o_off = desc.calculate_offset(origin);
            const CData *c_ptr  = view.get_buffer_view().p_data_ + o_off;
            // Bounded to the end of C: rows past M fall outside and are dropped.
            const index_t c_bytes =
                (static_cast<index_t>(view.get_buffer_view().buffer_size_) - o_off) * 2;
            const auto rsrc = cast_to_amdgpu_buffer_rsrc_t(make_wave_buffer_resource(c_ptr, c_bytes));
            const index_t r_row = wave * 64 + lane / 32, chunk = lane % 32;
            // Columns past N (padded instances): a lane's 8 columns are fixed for the whole tile; a
            // lane whose chunk is outside C (CK's padded-view validity check) stores to an offset
            // past the buffer, which is dropped. Chunks never straddle N: CK's IsSupportedArgument
            // requires N % 8 == 0 for this C.
            bool chunk_ok = true;
            if constexpr (Pad)
                chunk_ok = coordinate_has_valid_offset_assuming_top_index_is_valid(
                    desc, make_tensor_coordinate(
                              desc, make_multi_index(origin[0], origin[1] + chunk * 8 + 7)));
            const index_t g_off = chunk_ok ? (r_row * ldc + chunk * 8) * 2 : 0x7f000000;
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
