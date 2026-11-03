// include/basalt/kernels/pdl.hpp - programmatic dependent launch.
//
// A kernel started through launch_pdl inside a PdlScope may begin while the kernel before it on the stream is still
// finishing.  Every such kernel calls pdl_wait() on every path before it reads memory an earlier kernel writes and
// before it writes anything (only constant data such as weights may be read before it): pdl_wait() returns once the
// kernels before it on the stream have completed and their writes are visible, so the results are those of the plain
// launch.  Only the launch overlaps.
//
// launch_pdl is the plain <<<>>> launch outside a PdlScope, before CUDA 12.3, and on a device or a kernel build below
// compute capability 9.0 (where pdl_wait() compiles to nothing).  pdl_wait() in a
// kernel launched the plain way returns at once.
#pragma once

namespace basalt::kernels {

inline constexpr bool pdl_requested() { return true; }

/// The decode chain's kernels start their dependents early (pdl_launch_dependents) and stage their first weights
/// before pdl_wait, so each stage's weight reads begin during the previous stage's tail.
inline constexpr bool pdl2_requested() { return true; }
struct PdlState {
    bool on = false;
    int plain = 0;   // launches still to go the plain way
};
inline PdlState& pdl_state() {
    thread_local PdlState s;
    return s;
}
/// Launches this thread made with the attribute (the caller resets it).
inline long& pdl_count() {
    thread_local long n = 0;
    return n;
}

/// The launch_pdl calls of this thread while it lives may use the attribute; `first_plain`: the first one does not
/// (it follows something other than a kernel of this kind: a copy, a wait, another stream).
class PdlScope {
public:
    explicit PdlScope(bool on, bool first_plain = false) : prev_(pdl_state()) {
        pdl_state().on = on && pdl_requested();
        pdl_state().plain = first_plain ? 1 : 0;
    }
    ~PdlScope() { pdl_state() = prev_; }
    PdlScope(const PdlScope&) = delete;
    PdlScope& operator=(const PdlScope&) = delete;

private:
    PdlState prev_;
};

}  // namespace basalt::kernels

#if defined(__CUDACC__)
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define PDL_LAUNCH 1
#include <map>
#include <mutex>
#include <utility>
#endif

namespace basalt::kernels {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

/// Lets the next kernel on the stream (launched as a programmatic dependent) start once every block of this grid has
/// called it or exited.  Scheduling only: the dependent still reads this grid's results after its pdl_wait().
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

#if defined(PDL_LAUNCH)
/// The current device and the kernel's build both have griddepcontrol (compute capability 9.0+), per device.
inline bool pdl_kernel_ok(const void* func) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    static std::mutex mu;
    static std::map<std::pair<const void*, int>, bool> known;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_pair(func, dev);
    const auto it = known.find(key);
    if (it != known.end()) return it->second;
    int major = 0;
    cudaFuncAttributes fa{};
    bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 9;
    if (ok) {
        if (cudaFuncGetAttributes(&fa, func) != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
        } else {
            ok = fa.ptxVersion >= 90 && fa.binaryVersion >= 90;
        }
    }
    known[key] = ok;
    return ok;
}
#endif

/// For a launch made by hand (cudaLaunchKernelEx): true when the scope asks for the programmatic attribute here
/// (consumes the scope's plain count as launch_pdl does).
inline bool pdl_launch_wanted() {
#if defined(PDL_LAUNCH)
    PdlState& ps = pdl_state();
    if (!ps.on) return false;
    if (ps.plain > 0) {
        --ps.plain;
        return false;
    }
    ++pdl_count();
    return true;
#else
    return false;
#endif
}

/// `kernel<<<grid, block, smem, stream>>>(args...)`, or the same launch with programmatic stream serialization
/// inside a PdlScope (see above).  The kernel must call pdl_wait() first.
template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args... args) {
#if defined(PDL_LAUNCH)
    PdlState& ps = pdl_state();
    if (ps.on) {
        if (ps.plain > 0) {
            --ps.plain;
        } else if (pdl_kernel_ok((const void*) kernel)) {
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = grid;
            cfg.blockDim = block;
            cfg.dynamicSmemBytes = smem;
            cfg.stream = stream;
            cfg.attrs = at;
            cfg.numAttrs = 1;
            // an error stays for the caller's launch check
            if (cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess) ++pdl_count();
            return;
        }
    }
#endif
    kernel<<<grid, block, smem, stream>>>(args...);
}

}  // namespace basalt::kernels
#endif
