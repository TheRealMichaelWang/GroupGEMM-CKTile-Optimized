// Bank-conflict-free LDS layout for 16-bit A/B tiles in CK Tile's 4-wave pipelines.
//
// With CK's default UniversalGemmPipelineAgBgCrPolicy, bf16 tiles are stored as plain
// K-contiguous rows. For KPerBlock = 32 a row is 64 B, so rows m and m+4 start on the same
// bank; a 16x16x32 MFMA fragment read (16 lanes = rows 0..15 at one 16-byte K chunk) then
// hits only 4 distinct bank groups. Measured on MI355X: ~50% of all LDS cycles were bank
// conflicts in CompV3 (and 75% in CompAsync).
//
// Layout here: a row m holds C = KPerBlock*2B/16B chunks of 16 B. Split m into
// (M0 = m/16, Mx = (m/Mr) % C, Mr = m % Mr) with Mr = 16/C rows per 256 B, and store chunk
// c at chunk position c ^ Mx. For the 16 rows of a fragment, (Mr, c ^ Mx) then covers all 16
// 16-byte slots of a 256-byte window exactly once -> conflict free. Row-wise writes still
// fill whole 256-byte windows. Same element count as the default layout (no padding).
//
// Only the LDS descriptors change; everything else is CK's policy.
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace tunemax {

template <ck_tile::index_t WindowBytes = 256>
struct SwizzledLdsPolicyT : public ck_tile::UniversalGemmPipelineAgBgCrPolicy {
    using Default = ck_tile::UniversalGemmPipelineAgBgCrPolicy;

    // MN x K tile, K contiguous, 16-byte chunks XOR-swizzled as described above.
    template <typename DataType, ck_tile::index_t MN, ck_tile::index_t K>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSwizzledKContiguousDescriptor() {
        using ck_tile::number;
        constexpr ck_tile::index_t K2 = 16 / sizeof(DataType); // elements per 16-byte chunk
        constexpr ck_tile::index_t C  = K / K2;                // chunks per row
        constexpr ck_tile::index_t S  = WindowBytes / 16;      // 16-byte slots per window
        static_assert(K % K2 == 0 && C >= 1 && S % C == 0, "row must fit the bank window");
        constexpr ck_tile::index_t Mr = S / C; // rows per bank window
        constexpr ck_tile::index_t Mx = C;     // swizzle group size
        static_assert(MN % S == 0, "MN tile must be a multiple of the swizzle group");
        constexpr ck_tile::index_t M0 = MN / S;

        constexpr auto desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(number<M0>{}, number<Mx>{}, number<Mr>{}, number<C>{},
                                number<K2>{}),
            ck_tile::make_tuple(number<S * K>{}, number<Mr * K>{}, number<K>{}, number<K2>{},
                                number<1>{}),
            number<K2>{}, number<1>{});

        constexpr auto desc_1 = ck_tile::transform_tensor_descriptor(
            desc_0,
            ck_tile::make_tuple(ck_tile::make_pass_through_transform(number<M0>{}),
                                ck_tile::make_pass_through_transform(number<Mr>{}),
                                ck_tile::make_xor_transform(
                                    ck_tile::make_tuple(number<Mx>{}, number<C>{})),
                                ck_tile::make_pass_through_transform(number<K2>{})),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{},
                                ck_tile::sequence<1, 3>{}, ck_tile::sequence<4>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{},
                                ck_tile::sequence<1, 3>{}, ck_tile::sequence<4>{}));

        return ck_tile::transform_tensor_descriptor(
            desc_1,
            ck_tile::make_tuple(
                ck_tile::make_merge_transform_v3_division_mod(
                    ck_tile::make_tuple(number<M0>{}, number<Mx>{}, number<Mr>{})),
                ck_tile::make_merge_transform_v3_division_mod(
                    ck_tile::make_tuple(number<C>{}, number<K2>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0, 1, 2>{}, ck_tile::sequence<3, 4>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
    }

    template <typename Problem>
    static constexpr bool UseSwizzle =
        std::is_same_v<typename Problem::ALayout, ck_tile::tensor_layout::gemm::RowMajor> &&
        std::is_same_v<typename Problem::BLayout, ck_tile::tensor_layout::gemm::ColumnMajor> &&
        sizeof(typename Problem::ADataType) == 2 && sizeof(typename Problem::BDataType) == 2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor() {
        if constexpr (UseSwizzle<Problem>)
            return MakeSwizzledKContiguousDescriptor<typename Problem::ADataType,
                                                     Problem::BlockGemmShape::kM,
                                                     Problem::BlockGemmShape::kK>();
        else
            return Default::template MakeALdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor() {
        if constexpr (UseSwizzle<Problem>)
            return MakeSwizzledKContiguousDescriptor<typename Problem::BDataType,
                                                     Problem::BlockGemmShape::kN,
                                                     Problem::BlockGemmShape::kK>();
        else
            return Default::template MakeBLdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA() {
        return ck_tile::integer_least_multiple(
            MakeALdsBlockDescriptor<Problem>().get_element_space_size() *
                sizeof(typename Problem::ADataType),
            16);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeB() {
        return ck_tile::integer_least_multiple(
            MakeBLdsBlockDescriptor<Problem>().get_element_space_size() *
                sizeof(typename Problem::BDataType),
            16);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }
};

using SwizzledLdsPolicy = SwizzledLdsPolicyT<256>;

template <typename Problem>
using CompV3Swizzled = ck_tile::GemmPipelineAgBgCrCompV3<Problem, SwizzledLdsPolicyT<256>>;
template <typename Problem>
using CompV3Swizzled128 = ck_tile::GemmPipelineAgBgCrCompV3<Problem, SwizzledLdsPolicyT<128>>;
template <typename Problem>
using CompV4Swizzled = ck_tile::GemmPipelineAgBgCrCompV4<Problem, SwizzledLdsPolicyT<256>>;
template <typename Problem>
using CompV4Swizzled128 = ck_tile::GemmPipelineAgBgCrCompV4<Problem, SwizzledLdsPolicyT<128>>;

} // namespace tunemax
