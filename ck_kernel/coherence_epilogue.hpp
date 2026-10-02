// Epilogue wrapper that stores C with a chosen cache policy (e.g. non-temporal).
//
// ck_tile::UniversalGemmKernel builds the C window with coherence_default, so the epilogue
// always writes C through L2 with normal caching. For these GEMMs C is written once and never
// re-read, so caching it only evicts A/B lines that neighbouring tiles still need. hipBLASLt's
// kernels use non-temporal C/D stores (NTC4/NTD4). This wrapper rebuilds the same C view (same
// pointer, same descriptor, same memory op) with another amd_buffer_coherence_enum and then
// runs the wrapped CK epilogue unchanged.
#pragma once

#include "ck_tile/core.hpp"

namespace tunemax {

// Store modes (plain ints so the kernel's template name is the same in the host and device
// passes; amd_buffer_coherence_enum has different values in each).
enum : int { kStoreDefault = 0, kStoreNT = 1, kStoreDeviceNT = 2, kStoreSystemNT = 3 };

template <int Mode> CK_TILE_DEVICE constexpr ck_tile::amd_buffer_coherence_enum store_coherence() {
#if defined(__gfx950__) || defined(__gfx942__)
    using E = ck_tile::amd_buffer_coherence_enum;
    if constexpr (Mode == kStoreNT)
        return E::WAVE_NT1; // nt=1: no temporal reuse expected
    else if constexpr (Mode == kStoreDeviceNT)
        return E::DEVICE_NT1;
    else if constexpr (Mode == kStoreSystemNT)
        return E::SYSTEM_NT1;
#endif
    return ck_tile::amd_buffer_coherence_enum::coherence_default;
}

// Same window (pointer, descriptor, lengths, origin) with another cache policy. Works for
// plain (undistributed) tile windows such as the A/B/C block windows the GEMM kernel builds.
template <int Mode, typename Window> CK_TILE_DEVICE auto with_store_mode(const Window &window) {
    if constexpr (Mode == kStoreDefault) {
        return window;
    } else {
        const auto &view = window.get_bottom_tensor_view();
        using View       = ck_tile::remove_cvref_t<decltype(view)>;
        auto new_view    = ck_tile::make_tensor_view<ck_tile::address_space_enum::global,
                                                     View::DstInMemOp, store_coherence<Mode>(),
                                                     View::LargeTensor>(
            view.get_buffer_view().p_data_, view.get_tensor_descriptor());
        return ck_tile::make_tile_window(new_view, window.get_window_lengths(),
                                         window.get_window_origin());
    }
}

template <typename BaseEpilogue, int Mode>
struct CoherenceEpilogue : public BaseEpilogue {
    template <typename CWindow, typename CTile, typename DWindows>
    CK_TILE_DEVICE auto operator()(CWindow &c_window, const CTile &c_tile,
                                   const DWindows &d_windows, void *smem) {
        auto c_win = with_store_mode<Mode>(c_window);
        return BaseEpilogue::operator()(c_win, c_tile, d_windows, smem);
    }
};

} // namespace tunemax
