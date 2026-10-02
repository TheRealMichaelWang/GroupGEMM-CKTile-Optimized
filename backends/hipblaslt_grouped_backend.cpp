// hipBLASLt's own grouped GEMM: hipblaslt_ext::GroupedGemm. One kernel launch covers the
// tiles of every group; per-group arguments are read from a device buffer
// (run(deviceUserArgs, stream)), like the CK backend's device kargs.
//
// hipBLASLt is column-major, so the row-major forward problem is solved transposed:
//   out_g^T [N, M_g] = w_g [N, K] (stored K x N col-major -> op T)
//                    * x_g^T [K, M_g] (stored K x M_g col-major -> op N)
#include <hipblaslt/hipblaslt-ext.hpp>

#include <string>
#include <vector>

#include "backend.hpp"

#define TUNEMAX_HIPBLAS_CHECK(expr)                                                           \
    do {                                                                                      \
        hipblasStatus_t st_ = (expr);                                                         \
        if (st_ != HIPBLAS_STATUS_SUCCESS) {                                                  \
            std::fprintf(stderr, "hipBLASLt error %d at %s:%d: %s\n", int(st_), __FILE__,     \
                         __LINE__, #expr);                                                    \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

namespace tunemax {
namespace {

constexpr size_t kWorkspaceBytes = size_t(128) << 20;

class HipblasltGroupedBackend final : public Backend {
public:
    HipblasltGroupedBackend() {
        TUNEMAX_HIPBLAS_CHECK(hipblasLtCreate(&handle_));
        TUNEMAX_HIP_CHECK(hipMalloc(&workspace_, kWorkspaceBytes));
    }

    ~HipblasltGroupedBackend() override {
        release();
        (void) hipFree(workspace_);
        (void) hipblasLtDestroy(handle_);
    }

    std::string name() const override { return "hipblaslt_grouped"; }

    std::string prepare(const GroupedGemmProblem &p, hipStream_t stream) override {
        const hipDataType t = p.dtype == DType::BF16 ? HIP_R_16BF : HIP_R_16F;
        gemm_ = std::make_unique<hipblaslt_ext::GroupedGemm>(handle_, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                                             t, t, t, t, HIPBLAS_COMPUTE_32F);

        const size_t es = dtype_bytes(p.dtype);
        std::vector<int64_t> m(p.group_num, p.n), n(p.group_m), k(p.group_num, p.k),
            batch(p.group_num, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(p.group_num);
        std::vector<hipblaslt_ext::GemmInputs>   inputs(p.group_num);
        alpha_.assign(p.group_num, 1.0f);
        beta_.assign(p.group_num, 0.0f);
        int64_t row = 0;
        for (int g = 0; g < p.group_num; ++g) {
            const char *w   = static_cast<const char *>(p.w) + size_t(g) * p.n * p.k * es;
            const char *x   = static_cast<const char *>(p.x) + size_t(row) * p.k * es;
            char       *out = static_cast<char *>(p.out) + size_t(row) * p.n * es;
            inputs[g].setA(w);
            inputs[g].setB(x);
            inputs[g].setC(out);
            inputs[g].setD(out);
            inputs[g].setAlpha(&alpha_[g]);
            inputs[g].setBeta(&beta_[g]);
            row += p.group_m[g];
        }
        TUNEMAX_HIPBLAS_CHECK(gemm_->setProblem(m, n, k, batch, epilogue, inputs));

        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kWorkspaceBytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
        (void) gemm_->algoGetHeuristic(1, pref, candidates);
        if (candidates.empty())
            return std::string("no hipBLASLt grouped kernel for ") + dtype_name(p.dtype) +
                   " in the installed library";

        // Host template of the per-group device arguments, filled from setProblem().
        std::vector<hipblaslt_ext::UserArguments> host_args(p.group_num);
        TUNEMAX_HIPBLAS_CHECK(gemm_->getDefaultValueForDeviceUserArguments(host_args.data()));
        const size_t bytes = host_args.size() * sizeof(hipblaslt_ext::UserArguments);
        TUNEMAX_HIP_CHECK(hipMalloc(&dev_args_, bytes));
        TUNEMAX_HIP_CHECK(
            hipMemcpyAsync(dev_args_, host_args.data(), bytes, hipMemcpyHostToDevice, stream));

        // Use the heuristic's first choice, as a normal hipBLASLt caller would.
        TUNEMAX_HIPBLAS_CHECK(gemm_->initialize(candidates[0].algo, workspace_));
        TUNEMAX_HIP_CHECK(hipStreamSynchronize(stream));

        detail_ = gemm_->getKernelName();
        return "";
    }

    void run(hipStream_t stream) override {
        TUNEMAX_HIPBLAS_CHECK(gemm_->run(dev_args_, stream));
    }

    void release() override {
        gemm_.reset();
        if (dev_args_ != nullptr)
            TUNEMAX_HIP_CHECK(hipFree(dev_args_));
        dev_args_ = nullptr;
    }

    std::string detail() const override { return detail_; }

private:
    hipblasLtHandle_t                           handle_    = nullptr;
    void                                       *workspace_ = nullptr;
    void                                       *dev_args_  = nullptr;
    std::unique_ptr<hipblaslt_ext::GroupedGemm> gemm_;
    std::vector<float>                          alpha_, beta_;
    std::string                                 detail_;
};

} // namespace

std::unique_ptr<Backend> make_hipblaslt_grouped_backend() {
    return std::make_unique<HipblasltGroupedBackend>();
}

} // namespace tunemax
