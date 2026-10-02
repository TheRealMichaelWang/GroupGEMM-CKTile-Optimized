# Vendored from Primus-Turbo

Copied **unmodified** from https://github.com/AMD-AGI/Primus-Turbo at commit
`8cda13c` (2026-09-30). License: MIT, see `LICENSE` in this folder.

| Here | Upstream |
|---|---|
| `kernels/hipblaslt_grouped_gemm.cu` | `csrc/kernels/grouped_gemm/hipblaslt_grouped_gemm.cu` |
| `kernels/hipblaslt_gemm.cu` | `csrc/kernels/gemm/hipblaslt_gemm.cu` |
| `include/primus_turbo/*.h` | `csrc/include/primus_turbo/*.h` (only the headers the two files above need) |

This is the "hipBLASLt grouped GEMM" backend Primus-Turbo ships. It is **not** a
grouped kernel: it calls one `hipblasLtMatmul` per group (with a heuristic query per
call), spread round-robin over 4 streams. Do not edit these files; they are the
baseline. To refresh them, re-copy from a newer Primus-Turbo checkout.
