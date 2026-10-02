// CK Tile grouped GEMM: the kernel being tuned.
//
// Forward MoE layout (see common/backend.hpp):
//   A = x   row-major    [M_g, K], stride K
//   B = w^T column-major [K, N],   stride K   (w[g] is [N, K] row-major)
//   C = out row-major    [M_g, N], stride N
//
// Uses CK Tile's persistent ("tileloop") GroupedGemmKernel: the grid is sized to the
// GPU (not to the work), every workgroup strides through the tiles of all groups, and
// the per-group arguments are read from device memory. Two instances are compiled per
// dtype: an unpadded fast path and a padded fallback for shapes that are not multiples
// of the tile.
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.hpp"
#include "ck_grouped_gemm_config.hpp"
#include "xcd_partitioner.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace tunemax {
namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename Cfg, typename DataType, bool Pad> struct CkGroupedGemm {
    using ALayout     = Row;
    using BLayout     = Col;
    using CLayout     = Row;
    using AccDataType = float;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Cfg::M_Tile, Cfg::N_Tile, Cfg::K_Tile>,
        ck_tile::sequence<Cfg::M_Warp, Cfg::N_Warp, Cfg::K_Warp>,
        ck_tile::sequence<Cfg::M_Warp_Tile, Cfg::N_Warp_Tile, Cfg::K_Warp_Tile>>;

    using TilePartitioner = std::conditional_t<
        Cfg::XcdRemap,
        XcdRemapTilePartitioner<GemmShape, Cfg::TilePartitionerGroupNum, Cfg::TilePartitionerM01>,
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, Cfg::TilePartitionerGroupNum,
                                                   Cfg::TilePartitionerM01>>;

    // Same as ck_tile::PersistentTileGemmUniversalTraits, spelled out so the config can
    // also set the async-load, cache-prefetch and vector-size knobs.
    using Traits = ck_tile::TileGemmUniversalTraits<Pad, Pad, Pad, Cfg::DoubleSmemBuffer, ALayout,
                                                    BLayout, CLayout,
                                                    /*TransposeC=*/false,
                                                    /*UseStructuredSparsity=*/false,
                                                    /*UsePersistentKernel=*/Cfg::Persistent,
                                                    /*NumWaveGroups=*/1,
                                                    Cfg::Preshuffle, Cfg::VectorSize,
                                                    Cfg::DataCachePrefetchA,
                                                    Cfg::DataCachePrefetchB, Cfg::Async>;

    using Problem = ck_tile::UniversalGemmPipelineProblem<DataType, DataType, AccDataType,
                                                          GemmShape, Traits, Cfg::Scheduler>;

    using Pipeline = typename Cfg::template Pipeline<Problem>;

    // CShuffle: stage the C tile through LDS for wide, coalesced stores. Note: on gfx9 CK's
    // CShuffleEpilogue assumes the eight-wave pipeline's layout whenever M_Warp*N_Warp == 8,
    // so 8-warp configs with other pipelines need the Default epilogue.
    using CShuffleEpi = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
        DataType, DataType, ck_tile::tuple<>, AccDataType, DataType, ck_tile::tuple<>, CLayout,
        ck_tile::element_wise::PassThrough, TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock, Cfg::M_Warp, Cfg::N_Warp, Cfg::M_Warp_Tile,
        Cfg::N_Warp_Tile, Cfg::K_Warp_Tile, Problem::TransposeC>>;

    // Default: store straight from the accumulator registers, no LDS.
    using DefaultEpi = ck_tile::DefaultGemm2DEpilogue<ck_tile::DefaultGemm2DEpilogueProblem<
        DataType, DataType, ck_tile::tuple<>, AccDataType, DataType, ck_tile::tuple<>, CLayout,
        ck_tile::element_wise::PassThrough, TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock, Pad, Pad, Cfg::M_Warp_Tile, Cfg::N_Warp_Tile,
        Cfg::K_Warp_Tile, Problem::TransposeC>>;

    using Epilogue = std::conditional_t<Cfg::CShuffleEpilogue, CShuffleEpi, DefaultEpi>;

    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, Pipeline, Epilogue>;

    static bool is_supported(const std::vector<ck_tile::GemmTransKernelArg<>> &kargs) {
        return Kernel::IsSupportedArgument(kargs);
    }

    // Persistent: one wave of workgroups sized to the GPU. Otherwise one workgroup per tile.
    static dim3 grid(hipStream_t stream, int total_tiles) {
        if constexpr (!Cfg::Persistent)
            return dim3(total_tiles, 1, 1);
        dim3      g      = Kernel::MaxOccupancyGridSize(ck_tile::stream_config{stream});
        const int max_wg = ck_tile::get_available_compute_units(ck_tile::stream_config{stream}) *
                           Cfg::kBlockPerCu;
        g.x = std::min<unsigned>(g.x, max_wg);
        return g;
    }

    static void launch(const void *dev_kargs, int group_num, dim3 grid, hipStream_t stream) {
        // time_kernel=false: launch_kernel just enqueues, no events or syncs.
        ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<Cfg::kBlockPerCu>(
                Kernel{}, grid, Kernel::BlockSize(), 0,
                ck_tile::cast_pointer_to_constant_address_space(dev_kargs), group_num));
    }

    static std::string name() { return Kernel::GetName(); }
};

class CkTileBackend final : public Backend {
public:
    std::string name() const override { return "ck_tile"; }

    std::string prepare(const GroupedGemmProblem &p, hipStream_t stream) override {
        if (p.dtype == DType::BF16)
            return prepare_typed<ck_tile::bf16_t>(p, stream);
#ifdef TUNEMAX_TUNING_BUILD
        return "tuning build: only bf16 compiled (scripts/build.sh --full for everything)";
#else
        return prepare_typed<ck_tile::half_t>(p, stream);
#endif
    }

    void run(hipStream_t stream) override { launch_(dev_kargs_, group_num_, grid_, stream); }

    void release() override {
        if (dev_kargs_ != nullptr)
            TUNEMAX_HIP_CHECK(hipFree(dev_kargs_));
        dev_kargs_ = nullptr;
    }

    std::string detail() const override { return detail_; }

    ~CkTileBackend() override { release(); }

private:
    template <typename DataType>
    std::string prepare_typed(const GroupedGemmProblem &p, hipStream_t stream) {
        using Cfg = CkTileConfig;
        std::vector<ck_tile::GemmTransKernelArg<>> kargs;
        kargs.reserve(p.group_num);

        const auto *x   = static_cast<const DataType *>(p.x);
        const auto *w   = static_cast<const DataType *>(p.w);
        auto       *out = static_cast<DataType *>(p.out);
        bool divisible  = p.n % Cfg::N_Tile == 0 && p.k % Cfg::K_Tile == 0;
        int64_t row     = 0;
        int     tiles   = 0;
        const int n_tiles = int((p.n + Cfg::N_Tile - 1) / Cfg::N_Tile);
        for (int g = 0; g < p.group_num; ++g) {
            const int64_t m = p.group_m[g];
            divisible       = divisible && m % Cfg::M_Tile == 0;
            // block_start/block_end: this group's tile range. Only the non-persistent kernel
            // reads them (to find its group); the persistent one recomputes them.
            const int group_tiles = int((m + Cfg::M_Tile - 1) / Cfg::M_Tile) * n_tiles;
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{
                {x + row * p.k},
                {w + int64_t(g) * p.n * p.k},
                {},
                out + row * p.n,
                static_cast<ck_tile::index_t>(m),
                static_cast<ck_tile::index_t>(p.n),
                static_cast<ck_tile::index_t>(p.k),
                {static_cast<ck_tile::index_t>(p.k)}, // stride A (row-major M x K)
                {static_cast<ck_tile::index_t>(p.k)}, // stride B (col-major K x N)
                {},
                static_cast<ck_tile::index_t>(p.n), // stride C (row-major M x N)
                1},                                  // k_batch (no split-K)
                tiles, tiles + group_tiles);
            tiles += group_tiles;
            row += m;
        }
        total_tiles_ = tiles;

        if (divisible)
            return finish<CkGroupedGemm<Cfg, DataType, false>>(kargs, p, stream);
#ifdef TUNEMAX_TUNING_BUILD
        return "tuning build: padded instance not compiled (shape not a tile multiple)";
#else
        return finish<CkGroupedGemm<Cfg, DataType, true>>(kargs, p, stream);
#endif
    }

    template <typename Instance>
    std::string finish(const std::vector<ck_tile::GemmTransKernelArg<>> &kargs,
                       const GroupedGemmProblem &p, hipStream_t stream) {
        if (!Instance::is_supported(kargs))
            return "CK IsSupportedArgument rejected the shape";
        const size_t bytes = kargs.size() * sizeof(ck_tile::GemmTransKernelArg<>);
        TUNEMAX_HIP_CHECK(hipMalloc(&dev_kargs_, bytes));
        TUNEMAX_HIP_CHECK(
            hipMemcpyAsync(dev_kargs_, kargs.data(), bytes, hipMemcpyHostToDevice, stream));
        TUNEMAX_HIP_CHECK(hipStreamSynchronize(stream));
        group_num_ = p.group_num;
        grid_      = Instance::grid(stream, total_tiles_);
        launch_    = &Instance::launch;
        detail_    = Instance::name() + " grid=" + std::to_string(grid_.x);
        return "";
    }

    void       *dev_kargs_ = nullptr;
    int         group_num_ = 0;
    int         total_tiles_ = 0;
    dim3        grid_{};
    void       (*launch_)(const void *, int, dim3, hipStream_t) = nullptr;
    std::string detail_;
};

} // namespace

std::unique_ptr<Backend> make_ck_tile_backend() { return std::make_unique<CkTileBackend>(); }

} // namespace tunemax
