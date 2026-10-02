// Tile partitioner that keeps neighbouring output tiles on the same XCD.
//
// MI3xx dispatches consecutive workgroups round-robin over its 8 XCDs, each with its own
// L2. With CK's default order, tiles that share A rows / B columns land on different XCDs
// and don't reuse each other's L2 lines. This partitioner first applies CK's own
// RemapXCD (a permutation of the group's tile ids so each XCD gets a contiguous run),
// then CK's spatially-local 1D->2D mapping. It is a bijection per group, so correctness
// does not depend on tile counts; the locality benefit is best when a group's tile count
// is a multiple of NumXcds (then a tile's id mod 8 really is its XCD).
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace tunemax {

template <typename BlockGemmShape, ck_tile::index_t GroupNum, ck_tile::index_t M01,
          ck_tile::index_t NumXcds = 8>
struct XcdRemapTilePartitioner
    : public ck_tile::GemmSpatiallyLocalTilePartitioner<BlockGemmShape, GroupNum, M01> {
    using Base = ck_tile::GemmSpatiallyLocalTilePartitioner<BlockGemmShape, GroupNum, M01>;

    CK_TILE_HOST_DEVICE XcdRemapTilePartitioner(ck_tile::index_t M, ck_tile::index_t N) noexcept
        : Base(M, N), total_tiles_(Base::GridSize(M, N)) {}

    CK_TILE_DEVICE auto GetOutputTileIndex(ck_tile::index_t block_1d_id) noexcept
        -> const ck_tile::tuple<ck_tile::index_t, ck_tile::index_t> {
        return Base::GetOutputTileIndex(Base::RemapXCD(block_1d_id, total_tiles_, NumXcds));
    }

private:
    ck_tile::index_t total_tiles_; // the base keeps M and N private
};

} // namespace tunemax
