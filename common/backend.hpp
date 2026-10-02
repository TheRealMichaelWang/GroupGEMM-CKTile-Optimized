// Shared problem description and backend interface for every grouped GEMM implementation
// the benchmark compares. Keep this header small: the CK core and the baselines all
// include it, so changing it rebuilds everything.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#define TUNEMAX_HIP_CHECK(expr)                                                               \
    do {                                                                                      \
        hipError_t err_ = (expr);                                                             \
        if (err_ != hipSuccess) {                                                             \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(err_),      \
                         __FILE__, __LINE__, #expr);                                          \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

namespace tunemax {

enum class DType { BF16, FP16 };

inline const char *dtype_name(DType t) { return t == DType::BF16 ? "bf16" : "fp16"; }
inline size_t      dtype_bytes(DType) { return 2; }

// Forward MoE grouped GEMM, the same op as Primus-Turbo's
// `grouped_gemm(x, w, group_lens, trans_b=True)`:
//   x   [sum(group_m), k]  row-major   (tokens of all experts, back to back)
//   w   [group_num, n, k]  row-major   (one weight matrix per expert)
//   out [sum(group_m), n]  row-major
//   out[rows of group g] = x[rows of group g] @ w[g]^T
// fp32 accumulation, output in the input dtype.
struct GroupedGemmProblem {
    DType                dtype     = DType::BF16;
    int                  group_num = 0;
    int64_t              n         = 0;
    int64_t              k         = 0;
    std::vector<int64_t> group_m; // host copy, one entry per group

    const void *x   = nullptr; // device
    const void *w   = nullptr; // device
    void       *out = nullptr; // device

    // Same values as group_m, in pinned host memory that the GPU can also read. The
    // Primus-Turbo loop backend reads group lengths on the host each call.
    const int64_t *group_lens_pinned = nullptr;

    int64_t total_m() const {
        int64_t s = 0;
        for (int64_t m : group_m) s += m;
        return s;
    }
    double flops() const { return 2.0 * double(total_m()) * double(n) * double(k); }
};

// One grouped GEMM implementation.
//   prepare(): untimed. Allocate scratch, pick kernels/algorithms, build argument
//              buffers. Return "" when ready, or a short reason the backend can't run
//              this problem (reported as N/A in the results).
//   run():     timed. Enqueue exactly one grouped GEMM on `stream`. Called back to back
//              for warmup and timing, so it must not synchronize unless that is part of
//              what the backend really costs.
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const                                               = 0;
    virtual std::string prepare(const GroupedGemmProblem &problem, hipStream_t stream) = 0;
    virtual void        run(hipStream_t stream)                                    = 0;
    // Free whatever prepare() allocated. Called after each problem.
    virtual void release() {}
    // Free-form note recorded in the results, e.g. the kernel or algorithm chosen.
    virtual std::string detail() const { return ""; }
};

std::unique_ptr<Backend> make_ck_tile_backend();                  // ck_kernel/
std::unique_ptr<Backend> make_hipblaslt_grouped_backend();        // backends/ (heuristic top-1)
std::unique_ptr<Backend> make_hipblaslt_loop_backend();           // backends/ (Primus-Turbo)

} // namespace tunemax
