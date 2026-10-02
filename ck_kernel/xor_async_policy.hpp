// Policy for ck_tile::GemmPipelineAgBgCrCompAsync that uses CK's XOR-swizzled LDS layout
// for 16-bit types too.
//
// CK's GemmPipelineAgBgCrCompAsyncDefaultPolicy only enables its XOR swizzle for fp8/bf8
// (IsSupportedXorSwizzleDataType). For bf16/fp16 it falls back to a plain [K/KPack][M][KPack]
// layout whose 64-byte rows make ds_read_b128 from different rows hit the same banks: on
// MI355X the 4-wave 256x256x32 kernel spent ~50% of its LDS cycles in bank conflicts. The
// swizzle code itself works in element counts, so this policy reuses CK's own swizzle
// helpers and only drops the data-type gate. Everything the gate controls is overridden
// here, including the LDS size (CK's GetSmemSize* are CRTP'd on the default policy and would
// size the buffers for the unswizzled layout).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace tunemax {

struct XorAsyncPolicy : public ck_tile::GemmPipelineAgBgCrCompAsyncDefaultPolicy<false> {
    using Def = ck_tile::GemmPipelineAgBgCrCompAsyncDefaultPolicy<false>;
    using ck_tile::GemmPipelineAgBgCrCompAsyncDefaultPolicy<false>::I0;
    using ck_tile::GemmPipelineAgBgCrCompAsyncDefaultPolicy<false>::I1;

    // Same as CK's UseXorSwizzle without the fp8/bf8-only data-type check.
    template <typename Problem>
    static constexpr bool UseXor = !Def::template is_a_load_tr<Problem> &&
                                   !Def::template is_b_load_tr<Problem> &&
                                   Def::template IsSupportedXorSwizzleAsyncWidth<Problem>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution() {
        if constexpr (UseXor<Problem>)
            return Def::template MakeXorSwizzleABDramTileDistribution<
                Problem, Problem::BlockGemmShape::kM, Def::template GetSmemPackA<Problem>()>();
        else
            return Def::template MakeADramTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution() {
        if constexpr (UseXor<Problem>)
            return Def::template MakeXorSwizzleABDramTileDistribution<
                Problem, Problem::BlockGemmShape::kN, Def::template GetSmemPackB<Problem>()>();
        else
            return Def::template MakeBDramTileDistribution<Problem>();
    }

    // Row split used by the bf16 swizzle: a load instruction covers M4 rows x K1 chunks. Rows are
    // 64 B (K1 = 4 chunks of 16 B) and a ds_read_b128 lane group spans 128 B, so the chunk is
    // XORed with (row / M4L) % K1 where M4L = 128 B / row bytes = 2 (measured conflict-free).
    template <typename Problem>
    static constexpr ck_tile::index_t M4L =
        128 / (Problem::BlockGemmShape::WarpTile::at(ck_tile::number<2>{}) * 2) > 0
            ? 128 / (Problem::BlockGemmShape::WarpTile::at(ck_tile::number<2>{}) * 2)
            : 1;

    template <typename Problem, ck_tile::index_t MNPerBlock, ck_tile::index_t WarpTileMN>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBf16SwizzledLdsDescriptor() {
        using namespace ck_tile;
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using WarpTile       = typename BlockGemmShape::WarpTile;
        constexpr index_t K2        = Def::template GetSmemPackA<Problem>();
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t K1        = WarpTile::at(number<2>{}) / K2;
        constexpr index_t K0        = KPerBlock / (K1 * K2);
        constexpr index_t warp_num  = Problem::kBlockSize / get_warp_size();
        constexpr index_t M4        = get_warp_size() / K1;      // rows per load instruction
        constexpr index_t M4l       = M4L<Problem>;
        constexpr index_t M4h       = M4 / M4l;
        constexpr index_t M2        = WarpTileMN / M4;
        constexpr index_t M1        = warp_num / M2;
        constexpr index_t M0        = MNPerBlock / M1 / M2 / M4;
        static_assert(M0 * M1 * M2 * M4 == MNPerBlock && M4h * M4l == M4 && K0 * K1 * K2 == KPerBlock);
        // Same physical order as CK's XOR layout: [M2][M1][M0][K0][M4h][M4l][K1][K2] (+16 pad / M2)
        constexpr index_t PadSize = 16;
        constexpr auto desc_0 = make_naive_tensor_descriptor(
            number_tuple<M2, M1, M0, K0, M4h, M4l, K1, K2>{},
            number_tuple<M1 * M0 * K0 * M4 * K1 * K2 + PadSize, M0 * K0 * M4 * K1 * K2,
                         K0 * M4 * K1 * K2, M4 * K1 * K2, M4l * K1 * K2, K1 * K2, K2, 1>{},
            number<K2>{}, number<1>{});
        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M2>{}), make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M0>{}), make_pass_through_transform(number<K0>{}),
                       make_xor_transform(make_tuple(number<M4h>{}, number<K1>{})),
                       make_pass_through_transform(number<M4l>{}), make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4, 6>{},
                       sequence<5>{}, sequence<7>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4, 6>{},
                       sequence<5>{}, sequence<7>{}));
        return transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M4h, M4l>{}),
                       make_merge_transform_v3_division_mod(number_tuple<K0, K1, K2>{})),
            make_tuple(sequence<2, 1, 0, 4, 5>{}, sequence<3, 6, 7>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor() {
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        return MakeBf16SwizzledLdsDescriptor<Problem, Problem::BlockGemmShape::kM, WarpTile::at(I0)>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor() {
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        return MakeBf16SwizzledLdsDescriptor<Problem, Problem::BlockGemmShape::kN, WarpTile::at(I1)>();
    }

    // Global side: lane (row, chunk') fetches chunk' ^ ((row / M4L) % K1) so that the contiguous
    // async LDS writes land in the layout above.
    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeBf16SwizzledAsyncWindow(const Window &window) {
        using namespace ck_tile;
        using WarpTile             = typename Problem::BlockGemmShape::WarpTile;
        constexpr index_t K2       = Def::template GetSmemPackA<Problem>();
        constexpr index_t K1       = WarpTile::at(number<2>{}) / K2;
        constexpr index_t M4       = get_warp_size() / K1;
        constexpr index_t M4l      = M4L<Problem>;
        constexpr index_t M4h      = M4 / M4l;
        auto &&tensor_view         = window.get_bottom_tensor_view();
        const auto [rows, cols]    = tensor_view.get_tensor_descriptor().get_lengths();
        const index_t k_tiles      = cols / (K1 * K2);
        const index_t M0           = integer_divide_ceil(rows, M4);
        const auto row_lens        = make_tuple(M0, number<M4h>{}, number<M4l>{});
        const auto col_lens        = make_tuple(k_tiles, number<K1>{}, number<K2>{});
        const auto desc_0 = transform_tensor_descriptor(
            tensor_view.get_tensor_descriptor(),
            make_tuple(make_unmerge_transform(row_lens), make_unmerge_transform(col_lens)),
            make_tuple(sequence<0>{}, sequence<1>{}), make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}));
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M4h>{}, number<K1>{})),
                       make_pass_through_transform(number<M4l>{}), make_pass_through_transform(k_tiles),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}));
        const auto desc = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(row_lens),
                       make_merge_transform_v3_division_mod(col_lens)),
            make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}), make_tuple(sequence<0>{}, sequence<1>{}));
        return make_tile_window(
            make_tensor_view<address_space_enum::global>(&tensor_view.get_buffer_view()(0), desc),
            window.get_window_lengths(), window.get_window_origin());
    }
    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const Window &window) {
        return MakeBf16SwizzledAsyncWindow<Problem>(window);
    }
    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadBDramWindow(const Window &window) {
        return MakeBf16SwizzledAsyncWindow<Problem>(window);
    }

    // LDS size from *this* policy's descriptors (the swizzled layout is padded).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA() {
        using T = typename Problem::ADataType;
        return ck_tile::integer_least_multiple(
            MakeALdsBlockDescriptor<Problem>().get_element_space_size() * sizeof(T), 16);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeB() {
        using T = typename Problem::BDataType;
        return ck_tile::integer_least_multiple(
            MakeBLdsBlockDescriptor<Problem>().get_element_space_size() * sizeof(T), 16);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }
};

// The async pipeline with the policy above.
template <typename Problem>
using AsyncXorPipeline = ck_tile::GemmPipelineAgBgCrCompAsync<Problem, XorAsyncPolicy>;

} // namespace tunemax
