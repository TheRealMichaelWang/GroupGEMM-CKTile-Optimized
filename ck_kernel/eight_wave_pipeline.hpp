// Adapter that lets CK Tile's eight-wave async pipeline
// (ck_tile::GemmPipelineAgBgCrCompAsyncEightWaves) run inside ck_tile::GroupedGemmKernel.
// No CK code is changed; this only fills the two gaps between them:
//
//  1. GroupedGemmKernel calls pipeline(a_window, b_window, num_loop, smem) with plain
//     windows. In the eight-wave pipeline that overload is the quant entry point, which
//     wants HasHotLoop/TailNum baked into the Problem. We instead forward to its tuple
//     overload, which picks the hot-loop / tail variant at run time from num_loop.
//  2. GroupedGemmKernel::GetName() asks for GetVectorSizeC(), which this pipeline lacks
//     (it is only used in the kernel name).
//
// Pair it with the CShuffle epilogue and M_Warp * N_Warp == 8: on gfx9 that epilogue
// expects exactly this pipeline's eight-wave C layout.
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "coherence_epilogue.hpp"

namespace tunemax {

template <typename Problem, int AMode = kStoreDefault, int BMode = kStoreDefault>
struct GroupedEightWavePipeline : public ck_tile::GemmPipelineAgBgCrCompAsyncEightWaves<Problem> {
    using Base = ck_tile::GemmPipelineAgBgCrCompAsyncEightWaves<Problem>;

    static constexpr ck_tile::index_t GetVectorSizeC() { return 8; } // kernel name only

    // Keep the base overloads (tuple windows + element functions) visible too; the
    // grouped kernel's base class calls those.
    using Base::operator();

    template <typename ADramWindow, typename BDramWindow,
              std::enable_if_t<!ck_tile::is_detected<ck_tile::is_tuple, ADramWindow>::value,
                               bool> = true>
    CK_TILE_DEVICE auto operator()(const ADramWindow &a_window, const BDramWindow &b_window,
                                   ck_tile::index_t num_loop, void *p_smem) const {
        // Optional cache policy for the A/B loads (same helper as the C stores).
        return Base::operator()(ck_tile::make_tuple(with_store_mode<AMode>(a_window)),
                                ck_tile::element_wise::PassThrough{},
                                ck_tile::make_tuple(with_store_mode<BMode>(b_window)),
                                ck_tile::element_wise::PassThrough{}, num_loop, p_smem);
    }
};

} // namespace tunemax
