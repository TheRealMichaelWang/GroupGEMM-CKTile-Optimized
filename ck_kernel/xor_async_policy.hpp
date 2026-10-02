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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor() {
        if constexpr (UseXor<Problem>) {
            using WarpTile = typename Problem::BlockGemmShape::WarpTile;
            return Def::template MakeXorSwizzledABLdsBlockDescriptor<
                Problem, Problem::BlockGemmShape::kM, WarpTile::at(I0),
                Def::template GetSmemPackA<Problem>(), Def::template GetWGAttrNumAccess<Problem>()>();
        } else {
            return Def::template MakeALdsBlockDescriptor<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor() {
        if constexpr (UseXor<Problem>) {
            using WarpTile = typename Problem::BlockGemmShape::WarpTile;
            return Def::template MakeXorSwizzledABLdsBlockDescriptor<
                Problem, Problem::BlockGemmShape::kN, WarpTile::at(I1),
                Def::template GetSmemPackB<Problem>(), Def::template GetWGAttrNumAccess<Problem>()>();
        } else {
            return Def::template MakeBLdsBlockDescriptor<Problem>();
        }
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const Window &window) {
        if constexpr (UseXor<Problem>)
            return Def::template MakeAsyncLoadABDramWindow<
                Problem, Def::template GetSmemPackA<Problem>(),
                Def::template GetWGAttrNumAccess<Problem>()>(window);
        else
            return Def::template MakeAsyncLoadADramWindow<Problem>(window);
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadBDramWindow(const Window &window) {
        if constexpr (UseXor<Problem>)
            return Def::template MakeAsyncLoadABDramWindow<
                Problem, Def::template GetSmemPackB<Problem>(),
                Def::template GetWGAttrNumAccess<Problem>()>(window);
        else
            return Def::template MakeAsyncLoadBDramWindow<Problem>(window);
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
