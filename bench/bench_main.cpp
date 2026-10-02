// Grouped GEMM benchmark: CK Tile vs hipBLASLt GroupedGemm vs Primus-Turbo's hipBLASLt loop.
//
// Reads test cases (TestID,Case,B,M,N,K) from a CSV, runs every backend on each case
// (forward pass, balanced groups: B groups of M rows, like Primus-Turbo's benchmark),
// checks the output against a sampled fp32 reference, times it with HIP events and
// writes one results row per (case, backend).
//
//   tunemax_bench --shapes shapes/top3.csv [--dtype bf16|fp16] [--backends a,b,c]
//                 [--only 59,71] [--warmup 20] [--iters 100]
//                 [--no-check] [--out results/x.csv]
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"

namespace tunemax {
namespace {

// ---------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------
struct Options {
    std::string              shapes;
    std::string              out;
    DType                    dtype = DType::BF16;
    std::vector<std::string> backends{"ck_tile", "hipblaslt_grouped", "hipblaslt_loop"};
    std::set<int>            only;
    int                      warmup          = 20;
    int                      iters           = 100;
    bool                     check           = true;
};

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> parts;
    std::stringstream        ss(s);
    std::string              item;
    while (std::getline(ss, item, sep)) {
        while (!item.empty() && (item.back() == '\r' || item.back() == ' '))
            item.pop_back();
        if (!item.empty())
            parts.push_back(item);
    }
    return parts;
}

[[noreturn]] void usage(const char *msg) {
    std::cerr << "error: " << msg << "\n"
              << "usage: tunemax_bench --shapes FILE.csv [--dtype bf16|fp16]\n"
              << "         [--backends ck_tile,hipblaslt_grouped,hipblaslt_loop]\n"
              << "         [--only TESTID,...] [--warmup N] [--iters N]\n"
              << "         [--no-check] [--out FILE.csv]\n";
    std::exit(2);
}

Options parse_args(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a    = argv[i];
        auto              next = [&]() -> std::string {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--shapes")
            o.shapes = next();
        else if (a == "--out")
            o.out = next();
        else if (a == "--dtype") {
            const std::string d = next();
            if (d == "bf16")
                o.dtype = DType::BF16;
            else if (d == "fp16")
                o.dtype = DType::FP16;
            else
                usage("--dtype must be bf16 or fp16");
        } else if (a == "--backends")
            o.backends = split(next(), ',');
        else if (a == "--only")
            for (const auto &id : split(next(), ','))
                o.only.insert(std::stoi(id));
        else if (a == "--warmup")
            o.warmup = std::stoi(next());
        else if (a == "--iters")
            o.iters = std::stoi(next());
        else if (a == "--no-check")
            o.check = false;
        else if (a == "-h" || a == "--help")
            usage("help requested");
        else
            usage(("unknown argument " + a).c_str());
    }
    if (o.shapes.empty())
        usage("--shapes is required");
    if (o.iters < 1 || o.warmup < 0)
        usage("--iters must be >= 1 and --warmup >= 0");
    return o;
}

// ---------------------------------------------------------------------------------------
// Shapes CSV
// ---------------------------------------------------------------------------------------
struct TestCase {
    int         id;
    std::string name;
    int64_t     b, m, n, k;
};

std::vector<TestCase> read_cases(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        usage(("cannot open " + path).c_str());
    std::string header;
    std::getline(f, header);
    std::map<std::string, int> col;
    const auto                 names = split(header, ',');
    for (int i = 0; i < int(names.size()); ++i)
        col[names[i]] = i;
    for (const char *need : {"TestID", "Case", "B", "M", "N", "K"})
        if (!col.count(need))
            usage((path + ": missing column " + need).c_str());

    std::vector<TestCase> cases;
    std::string           line;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        const auto v = split(line, ',');
        cases.push_back({std::stoi(v[col["TestID"]]), v[col["Case"]], std::stoll(v[col["B"]]),
                         std::stoll(v[col["M"]]), std::stoll(v[col["N"]]),
                         std::stoll(v[col["K"]])});
    }
    return cases;
}

// ---------------------------------------------------------------------------------------
// Device helpers: input fill and sampled reference check
// ---------------------------------------------------------------------------------------
template <typename T> __device__ T    from_float(float v);
template <> __device__ __hip_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __half         from_float(float v) { return __float2half(v); }
__device__ float                      to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ float                      to_float(__half v) { return __half2float(v); }

__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Standard normal values (like torch.randn), deterministic in (seed, index).
template <typename T> __global__ void fill_randn(T *p, size_t n, uint64_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
        const uint64_t r  = splitmix64(seed ^ (i * 0x2545F4914F6CDD1Dull));
        const float    u1 = (float(r >> 40) + 1.0f) * (1.0f / 16777217.0f); // (0, 1]
        const float    u2 = float((r >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f);
        p[i] = from_float<T>(sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2));
    }
}

struct Sample {
    int64_t row; // global row in x / out
    int64_t col; // column in out (row of w[g])
    int64_t group;
};

// One block per sampled output element: fp32 dot product of x[row, :] and w[g][col, :].
template <typename T>
__global__ void sampled_reference(const T *x, const T *w, const T *out, int64_t n, int64_t k,
                                  const Sample *samples, float *ref, float *got) {
    const Sample s  = samples[blockIdx.x];
    const T     *xr = x + s.row * k;
    const T     *wr = w + (s.group * n + s.col) * k;
    float        acc = 0.f;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x)
        acc += to_float(xr[i]) * to_float(wr[i]);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (int(threadIdx.x) < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ref[blockIdx.x] = partial[0];
        got[blockIdx.x] = to_float(out[s.row * n + s.col]);
    }
}

// Which output elements to check: per group, the four corners (catches padding and
// group-offset bugs) plus random positions.
std::vector<Sample> make_samples(const GroupedGemmProblem &p) {
    constexpr int kPerGroup = 256;
    std::mt19937_64 rng(1234);
    std::vector<Sample> s;
    int64_t row0 = 0;
    for (int g = 0; g < p.group_num; ++g) {
        const int64_t m = p.group_m[g];
        if (m > 0) {
            for (int64_t r : {row0, row0 + m - 1})
                for (int64_t c : {int64_t(0), p.n - 1})
                    s.push_back({r, c, g});
            for (int i = 0; i < kPerGroup; ++i)
                s.push_back({row0 + int64_t(rng() % m), int64_t(rng() % p.n), g});
        }
        row0 += m;
    }
    return s;
}

struct CheckResult {
    bool   pass       = false;
    double max_abs    = 0.0;
    int    mismatches = 0;
};

// Same tolerance as Primus-Turbo's check_allclose for bf16/fp16 (torch.allclose with
// rtol = atol = 1e-2): |got - ref| <= atol + rtol * |ref|.
template <typename T>
CheckResult check_output(const GroupedGemmProblem &p, const std::vector<Sample> &samples,
                         hipStream_t stream) {
    const size_t ns = samples.size();
    Sample      *d_samples;
    float       *d_ref, *d_got;
    TUNEMAX_HIP_CHECK(hipMalloc(&d_samples, ns * sizeof(Sample)));
    TUNEMAX_HIP_CHECK(hipMalloc(&d_ref, ns * sizeof(float)));
    TUNEMAX_HIP_CHECK(hipMalloc(&d_got, ns * sizeof(float)));
    TUNEMAX_HIP_CHECK(hipMemcpyAsync(d_samples, samples.data(), ns * sizeof(Sample),
                                     hipMemcpyHostToDevice, stream));
    sampled_reference<T><<<ns, 256, 0, stream>>>(static_cast<const T *>(p.x),
                                                 static_cast<const T *>(p.w),
                                                 static_cast<const T *>(p.out), p.n, p.k,
                                                 d_samples, d_ref, d_got);
    TUNEMAX_HIP_CHECK(hipGetLastError());
    std::vector<float> ref(ns), got(ns);
    TUNEMAX_HIP_CHECK(
        hipMemcpyAsync(ref.data(), d_ref, ns * sizeof(float), hipMemcpyDeviceToHost, stream));
    TUNEMAX_HIP_CHECK(
        hipMemcpyAsync(got.data(), d_got, ns * sizeof(float), hipMemcpyDeviceToHost, stream));
    TUNEMAX_HIP_CHECK(hipStreamSynchronize(stream));
    TUNEMAX_HIP_CHECK(hipFree(d_samples));
    TUNEMAX_HIP_CHECK(hipFree(d_ref));
    TUNEMAX_HIP_CHECK(hipFree(d_got));

    CheckResult r;
    for (size_t i = 0; i < ns; ++i) {
        const double err = std::fabs(double(got[i]) - double(ref[i]));
        r.max_abs        = std::max(r.max_abs, std::isnan(err) ? INFINITY : err);
        if (!(err <= 1e-2 + 1e-2 * std::fabs(double(ref[i]))))
            ++r.mismatches;
    }
    r.pass = r.mismatches == 0;
    return r;
}

// ---------------------------------------------------------------------------------------
// Device buffers, grown as needed and reused across cases
// ---------------------------------------------------------------------------------------
struct DeviceBuffer {
    void  *ptr   = nullptr;
    size_t bytes = 0;
    void  *get(size_t need) {
        if (need > bytes) {
            if (ptr)
                TUNEMAX_HIP_CHECK(hipFree(ptr));
            TUNEMAX_HIP_CHECK(hipMalloc(&ptr, need));
            bytes = need;
        }
        return ptr;
    }
    ~DeviceBuffer() {
        if (ptr)
            (void) hipFree(ptr);
    }
};

template <typename T> void fill(void *p, size_t n, uint64_t seed, hipStream_t stream) {
    fill_randn<T><<<4096, 256, 0, stream>>>(static_cast<T *>(p), n, seed);
    TUNEMAX_HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------------------
struct Row {
    TestCase    tc;
    std::string backend, check, detail;
    double      ms = NAN, tflops = NAN, max_abs = NAN;
};

std::string csv_quote(const std::string &s) {
    std::string q = "\"";
    for (char c : s)
        q += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

void write_csv(const std::string &path, const std::vector<Row> &rows, const std::string &gpu,
               DType dtype) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "error: cannot write " << path << "\n";
        std::exit(1);
    }
    f << "TestID,Case,B,M,N,K,Dtype,GPU,Backend,Check,Time_ms,TFLOPS,MaxAbsErr,Detail\n";
    f << std::fixed;
    for (const auto &r : rows) {
        f << r.tc.id << ',' << r.tc.name << ',' << r.tc.b << ',' << r.tc.m << ',' << r.tc.n
          << ',' << r.tc.k << ',' << dtype_name(dtype) << ',' << csv_quote(gpu) << ','
          << r.backend << ',' << r.check << ',';
        if (std::isnan(r.ms))
            f << ",,";
        else
            f << std::setprecision(4) << r.ms << ',' << std::setprecision(2) << r.tflops << ',';
        if (!std::isnan(r.max_abs))
            f << std::setprecision(4) << r.max_abs;
        f << ',' << csv_quote(r.detail) << '\n';
    }
}

void print_summary(const std::vector<Row> &rows, const std::vector<std::string> &backends) {
    // Per backend: mean time and TFLOPS over the cases where *every* backend passed, so
    // the averages compare the same set of shapes.
    std::map<int, std::map<std::string, const Row *>> by_case;
    for (const auto &r : rows)
        by_case[r.tc.id][r.backend] = &r;
    std::vector<int> common;
    for (const auto &[id, m] : by_case) {
        bool all = m.size() == backends.size();
        for (const auto &[b, r] : m)
            all = all && r->check == "PASS";
        if (all)
            common.push_back(id);
    }

    std::cout << "\n==== Summary over " << common.size() << " of " << by_case.size()
              << " cases where all backends passed ====\n";
    if (common.empty()) {
        std::cout << "(no common passing cases; see per-backend rows above)\n";
        return;
    }
    const std::string ref = backends.front();
    std::cout << std::left << std::setw(20) << "backend" << std::right << std::setw(14)
              << "sum ms" << std::setw(14) << "mean TFLOPS" << std::setw(22)
              << ("geomean vs " + ref) << '\n';
    for (const auto &b : backends) {
        double sum_ms = 0, sum_tf = 0, log_ratio = 0;
        for (int id : common) {
            const Row *r = by_case[id][b];
            sum_ms += r->ms;
            sum_tf += r->tflops;
            log_ratio += std::log(r->ms / by_case[id][ref]->ms);
        }
        // > 1 means this backend is slower than the reference backend.
        const double geomean = std::exp(log_ratio / common.size());
        std::cout << std::left << std::setw(20) << b << std::right << std::fixed
                  << std::setprecision(3) << std::setw(14) << sum_ms << std::setprecision(1)
                  << std::setw(14) << sum_tf / common.size() << std::setprecision(3)
                  << std::setw(21) << geomean << "x\n";
    }
    std::cout << "(geomean column = time / " << ref << " time; >1 means slower than " << ref
              << ")\n";
}

std::string default_out_path(const Options &o) {
    std::string stem = o.shapes.substr(o.shapes.find_last_of('/') + 1);
    stem             = stem.substr(0, stem.find_last_of('.'));
    char        ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&now));
    return "results/" + stem + "_" + dtype_name(o.dtype) + "_" + ts + ".csv";
}

int run(const Options &opt) {
    hipDeviceProp_t prop;
    TUNEMAX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const std::string gpu = std::string(prop.name) + " (" + prop.gcnArchName + ")";

    std::vector<std::unique_ptr<Backend>> backends;
    for (const auto &b : opt.backends) {
        if (b == "ck_tile")
            backends.push_back(make_ck_tile_backend());
        else if (b == "hipblaslt_grouped")
            backends.push_back(make_hipblaslt_grouped_backend());
        else if (b == "hipblaslt_loop")
            backends.push_back(make_hipblaslt_loop_backend());
        else
            usage(("unknown backend " + b).c_str());
    }

    std::vector<TestCase> cases;
    for (const auto &c : read_cases(opt.shapes))
        if (opt.only.empty() || opt.only.count(c.id))
            cases.push_back(c);
    const std::string out_path = opt.out.empty() ? default_out_path(opt) : opt.out;

    std::cout << "GPU: " << gpu << "\nshapes: " << opt.shapes << " (" << cases.size()
              << " cases), dtype " << dtype_name(opt.dtype) << ", warmup " << opt.warmup
              << ", iters " << opt.iters << "\n";

    hipStream_t stream;
    TUNEMAX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipEvent_t start, stop;
    TUNEMAX_HIP_CHECK(hipEventCreate(&start));
    TUNEMAX_HIP_CHECK(hipEventCreate(&stop));

    DeviceBuffer x_buf, w_buf, out_buf;
    int64_t     *group_lens_pinned = nullptr;
    size_t       pinned_groups     = 0;
    std::vector<Row> rows;

    for (const auto &tc : cases) {
        GroupedGemmProblem p;
        p.dtype     = opt.dtype;
        p.group_num = int(tc.b);
        p.n         = tc.n;
        p.k         = tc.k;
        p.group_m.assign(tc.b, tc.m); // balanced, like gen_grouped_gemm_group_lens(balance=True)
        const size_t es = dtype_bytes(p.dtype);
        p.x   = x_buf.get(size_t(p.total_m()) * p.k * es);
        p.w   = w_buf.get(size_t(p.group_num) * p.n * p.k * es);
        p.out = out_buf.get(size_t(p.total_m()) * p.n * es);
        if (pinned_groups < size_t(p.group_num)) {
            if (group_lens_pinned)
                TUNEMAX_HIP_CHECK(hipHostFree(group_lens_pinned));
            TUNEMAX_HIP_CHECK(hipHostMalloc(&group_lens_pinned, p.group_num * sizeof(int64_t)));
            pinned_groups = p.group_num;
        }
        std::copy(p.group_m.begin(), p.group_m.end(), group_lens_pinned);
        p.group_lens_pinned = group_lens_pinned;

        if (p.dtype == DType::BF16) {
            fill<__hip_bfloat16>(const_cast<void *>(p.x), size_t(p.total_m()) * p.k, 1, stream);
            fill<__hip_bfloat16>(const_cast<void *>(p.w), size_t(p.group_num) * p.n * p.k, 2,
                                 stream);
        } else {
            fill<__half>(const_cast<void *>(p.x), size_t(p.total_m()) * p.k, 1, stream);
            fill<__half>(const_cast<void *>(p.w), size_t(p.group_num) * p.n * p.k, 2, stream);
        }
        const std::vector<Sample> samples = make_samples(p);

        std::cout << "\n[" << tc.id << "] " << tc.name << "  B=" << tc.b << " M=" << tc.m
                  << " N=" << tc.n << " K=" << tc.k << "  (" << std::setprecision(3)
                  << p.flops() / 1e12 << " TFLOP)\n";

        for (auto &backend : backends) {
            Row row{tc, backend->name(), "", "", NAN, NAN, NAN};
            try {
                // Zero the output so a backend that writes nothing fails the check.
                TUNEMAX_HIP_CHECK(hipMemsetAsync(p.out, 0, size_t(p.total_m()) * p.n * es, stream));
                const std::string why_not = backend->prepare(p, stream);
                if (!why_not.empty()) {
                    row.check  = "N/A";
                    row.detail = why_not;
                } else {
                    row.detail = backend->detail();
                    backend->run(stream);
                    TUNEMAX_HIP_CHECK(hipGetLastError());
                    TUNEMAX_HIP_CHECK(hipStreamSynchronize(stream));
                    if (opt.check) {
                        const CheckResult c = p.dtype == DType::BF16
                                                  ? check_output<__hip_bfloat16>(p, samples, stream)
                                                  : check_output<__half>(p, samples, stream);
                        row.check   = c.pass ? "PASS" : "FAIL";
                        row.max_abs = c.max_abs;
                        if (!c.pass)
                            row.detail += " | " + std::to_string(c.mismatches) + "/" +
                                          std::to_string(samples.size()) + " samples off";
                    } else {
                        row.check = "SKIP";
                    }
                    for (int i = 0; i < opt.warmup; ++i)
                        backend->run(stream);
                    TUNEMAX_HIP_CHECK(hipEventRecord(start, stream));
                    for (int i = 0; i < opt.iters; ++i)
                        backend->run(stream);
                    TUNEMAX_HIP_CHECK(hipEventRecord(stop, stream));
                    TUNEMAX_HIP_CHECK(hipEventSynchronize(stop));
                    TUNEMAX_HIP_CHECK(hipGetLastError());
                    float total_ms = 0.f;
                    TUNEMAX_HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
                    row.ms     = double(total_ms) / opt.iters;
                    row.tflops = p.flops() / (row.ms * 1e-3) / 1e12;
                }
            } catch (const std::exception &e) {
                (void) hipStreamSynchronize(stream);
                row.check  = "ERROR";
                row.detail = e.what();
            }
            backend->release();

            std::cout << "  " << std::left << std::setw(18) << row.backend << std::setw(6)
                      << row.check << std::right << std::fixed;
            if (std::isnan(row.ms))
                std::cout << std::setw(12) << "-" << std::setw(14) << "-";
            else
                std::cout << std::setprecision(3) << std::setw(10) << row.ms << " ms"
                          << std::setprecision(1) << std::setw(9) << row.tflops << " TFLOPS";
            std::cout << "  " << row.detail.substr(0, 90) << "\n";
            std::cout.unsetf(std::ios::fixed);
            rows.push_back(row);
        }
        // Write after every case so a crash or Ctrl-C keeps what finished.
        write_csv(out_path, rows, gpu, opt.dtype);
    }

    print_summary(rows, opt.backends);
    std::cout << "\nresults: " << out_path << "\n";

    if (group_lens_pinned)
        TUNEMAX_HIP_CHECK(hipHostFree(group_lens_pinned));
    TUNEMAX_HIP_CHECK(hipEventDestroy(start));
    TUNEMAX_HIP_CHECK(hipEventDestroy(stop));
    TUNEMAX_HIP_CHECK(hipStreamDestroy(stream));

    for (const auto &r : rows)
        if (r.check == "FAIL" || r.check == "ERROR")
            return 1;
    return 0;
}

} // namespace
} // namespace tunemax

int main(int argc, char **argv) { return tunemax::run(tunemax::parse_args(argc, argv)); }
