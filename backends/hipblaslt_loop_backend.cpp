// Primus-Turbo's "hipBLASLt grouped GEMM": the unmodified code in
// third_party/primus_turbo/kernels/hipblaslt_grouped_gemm.cu, called with the same
// parameters Primus-Turbo's PyTorch binding passes for the forward pass
// (csrc/pytorch/grouped_gemm/hipblaslt_grouped_gemm.cpp: transA=false, transB=true,
// pre_sync=false, beta=0).
//
// Each run() is one hipblasLtMatmul per group (with a heuristic query each), round-robin
// over 4 streams, plus the host-side argument setup. All of that is in the timed region,
// as it is in Primus-Turbo.
#include <hipblaslt/hipblaslt.h>

#include "backend.hpp"
#include "primus_turbo/grouped_gemm.h"

namespace tunemax {
namespace {

class HipblasltLoopBackend final : public Backend {
public:
    HipblasltLoopBackend() {
        if (hipblasLtCreate(&handle_) != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "hipblasLtCreate failed\n");
            std::exit(1);
        }
        TUNEMAX_HIP_CHECK(
            hipMalloc(&workspace_, primus_turbo::get_hipblaslt_grouped_gemm_workspace_size()));
    }

    ~HipblasltLoopBackend() override {
        (void) hipFree(workspace_);
        (void) hipblasLtDestroy(handle_);
    }

    std::string name() const override { return "hipblaslt_loop"; }

    std::string prepare(const GroupedGemmProblem &p, hipStream_t stream) override {
        const hipDataType t = p.dtype == DType::BF16 ? HIP_R_16BF : HIP_R_16F;
        params_             = {};
        params_.a_ptr       = p.x;
        params_.a_type      = t;
        params_.a_shape     = {p.total_m(), p.k};
        params_.b_ptr       = p.w;
        params_.b_type      = t;
        params_.b_shape     = {p.group_num, p.n, p.k};
        params_.c_ptr       = p.out;
        params_.c_type      = t;
        params_.c_shape     = {p.total_m(), p.n};
        params_.beta        = 0.0f;
        // Primus passes a device tensor here and its code reads it on the host; pinned
        // host memory gives the same values without relying on that.
        params_.group_lens_ptr = p.group_lens_pinned;
        params_.group_offs_ptr = nullptr; // not read by the hipBLASLt path
        params_.transA         = false;
        params_.transB         = true;
        params_.group_num      = p.group_num;
        params_.stream         = stream;
        params_.workspace      = workspace_;
        params_.handle         = handle_;
        return "";
    }

    void run(hipStream_t stream) override {
        params_.stream = stream;
        primus_turbo::hipblaslt_grouped_gemm(params_, /*pre_sync=*/false);
    }

    std::string detail() const override { return "1 hipblasLtMatmul per group, 4 streams"; }

private:
    hipblasLtHandle_t                        handle_    = nullptr;
    void                                    *workspace_ = nullptr;
    primus_turbo::HipblasltGroupedGemmParams params_;
};

} // namespace

std::unique_ptr<Backend> make_hipblaslt_loop_backend() {
    return std::make_unique<HipblasltLoopBackend>();
}

} // namespace tunemax
