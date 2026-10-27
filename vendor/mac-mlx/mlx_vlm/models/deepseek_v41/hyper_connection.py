"""Opt-in V4.1 hyper-connection kernels and reference-compatible fallbacks.

For sub-block t, let (p_t, q_t, C_t) = HC_t(h_t). The sublayer input is
sum_i p_{t-1,i} h_{t,i}, NOT sum_i p_{t,i} h_{t,i}. The initial p is
one-hot lane zero; the final collapse consumes the last FFN's p.
Expansion is h_{t+1,j} = q_{t,j} * sublayer_output + sum_i C_{t,i,j} h_{t,i}.

A fused sinkhorn/collapse kernel must accept incoming_pre separately and
return (collapsed, next_pre, post, comb). Its collapse reads incoming_pre,
while its sigmoid writes next_pre. V4's kernel reads its own new sigmoid
and does not return pre, so neither its inputs nor outputs fit this contract.
DSV41_METAL_SINKHORN fuses only the split/normalization loop, returning the
next pre-mix separately; the existing staggered collapse remains unchanged.

DSV41_HC_FUSED fuses projection-first RMS scaling, Sinkhorn/gates and lagged
collapse; DSV41_HC_FUSED_POST fuses expansion without a 5-D temporary. Both
default off pending consolidated end-to-end validation. Metal reductions/FMA
can change rounding: see docs/dsv41_f4_report.md and the numerical gates.
The older DSV41_HC_COMPILE_POST flag is read at module import.
"""

import os
from functools import lru_cache

import mlx.core as mx


COMPILE_POST = os.environ.get("DSV41_HC_COMPILE_POST", "0") == "1"
SINKHORN_CALLS = 0
FUSED_CALLS = 0
POST_CALLS = 0


@lru_cache(maxsize=1)
def _staggered_kernel():
    # One 256-thread group per token. Projection is deliberately computed before
    # normalization, as in the reference. IncomingPre never aliases NextPre.
    return mx.fast.metal_kernel(
        name="dsv41_staggered_hc4",
        input_names=["X", "IncomingPre", "Projection", "Scale", "Base", "Eps", "NormEps"],
        output_names=["Collapsed", "NextPre", "Post", "Comb"],
        source="""
        typedef metal::vec<T, 4> T4;
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg = tid / 32;
        threadgroup float partial[8];
        threadgroup float inv_rms;
        float ss = 0.0f;
        float4 p = float4(IncomingPre[row*4], IncomingPre[row*4+1],
                          IncomingPre[row*4+2], IncomingPre[row*4+3]);
        for (uint d = tid * 4; d < D; d += 256 * 4) {
            float4 a = float4(*(const device T4*)(X + (row*4+0)*D+d));
            float4 b = float4(*(const device T4*)(X + (row*4+1)*D+d));
            float4 c = float4(*(const device T4*)(X + (row*4+2)*D+d));
            float4 e = float4(*(const device T4*)(X + (row*4+3)*D+d));
            float4 squares = a*a + b*b + c*c + e*e;
            ss += (squares.x + squares.y) + (squares.z + squares.w);
            float4 y = metal::fma(e, p.w, metal::fma(c, p.z, metal::fma(b, p.y, a*p.x)));
            *(device T4*)(Collapsed + row*D+d) = T4(y);
        }
        ss = simd_sum(ss);
        if (lane == 0) partial[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
            if (lane == 0) inv_rms = metal::rsqrt(total / float(4*D) + NormEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            uint i = metal::min(lane, 3u);
            float active = lane < 4 ? 1.0f : 0.0f;
            float pre_z = (Projection[row*24+i] * inv_rms) * Scale[0] + Base[i];
            float post_z = (Projection[row*24+4+i] * inv_rms) * Scale[1] + Base[4+i];
            if (lane < 4) {
                NextPre[row*4+i] = 1.0f/(1.0f+metal::precise::exp(-pre_z)) + Eps;
                Post[row*4+i] = 2.0f/(1.0f+metal::precise::exp(-post_z));
            }
            uint off = row*24+8+i*4;
            float4 z = (float4(Projection[off],Projection[off+1],Projection[off+2],Projection[off+3]) * inv_rms)
                       * Scale[2] + float4(Base[8+i*4],Base[9+i*4],Base[10+i*4],Base[11+i*4]);
            float m = metal::max(metal::max(z.x,z.y),metal::max(z.z,z.w));
            float4 a = metal::precise::exp(z-m);
            // No epsilon in the initial softmax denominator (unlike V4).
            a = (a / ((a.x+a.y)+(a.z+a.w)) + Eps) * active;
            for (uint iter=0; iter<ITERS; ++iter) {
                if (iter) a /= ((a.x+a.y)+(a.z+a.w)) + Eps;
                float4 sums = float4(simd_sum(a.x),simd_sum(a.y),simd_sum(a.z),simd_sum(a.w));
                a /= sums + Eps;
            }
            if (lane < 4) *(device float4*)(Comb+row*16+i*4) = a;
        }
        """,
        ensure_row_contiguous=True,
    )


def fused_staggered_hc(x, incoming_pre, projection, scale, base, iters, eps, norm_eps):
    """Return (collapsed, next_pre, post, comb), preserving V4.1's lagged mix.

    projection must be float(x.flatten(2)) @ float(fn).T, BEFORE RMS scaling.
    Supported inference contract: [B,S,4,D], D divisible by four, Metal.
    Unsupported opt-in calls fail loudly instead of bypassing measurement.
    """
    global FUSED_CALLS
    if (x.ndim != 4 or x.size == 0 or x.shape[2] != 4 or x.shape[-1] % 4 or iters < 1
            or not mx.metal.is_available() or mx.default_device() != mx.gpu):
        raise ValueError("Fused V4.1 HC requires Metal, [B,S,4,D], D%4=0, iterations>=1")
    if x.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError("Fused V4.1 HC requires fp32/fp16/bf16 residuals")
    shape = x.shape[:2]
    if projection.shape != (*shape, 24) or scale.shape != (3,) or base.shape != (24,):
        raise ValueError("Invalid V4.1 HC projection/scale/base shape")
    incoming_pre = mx.broadcast_to(incoming_pre, (*shape, 4)).astype(mx.float32)
    FUSED_CALLS += 1
    return _staggered_kernel()(
        inputs=[x, incoming_pre, projection.astype(mx.float32), scale.astype(mx.float32),
                base.astype(mx.float32), mx.array(eps, mx.float32), mx.array(norm_eps, mx.float32)],
        template=[("T", x.dtype), ("D", x.shape[-1]), ("ITERS", iters)],
        grid=(shape[0]*shape[1]*256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*shape, x.shape[-1]), (*shape, 4), (*shape, 4), (*shape, 4, 4)],
        output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32],
    )


@lru_cache(maxsize=1)
def _post_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_hc_post4", input_names=["X", "Residual", "Post", "Comb"],
        output_names=["Y"], source="""
        uint d = thread_position_in_grid.x;
        uint j = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;
        if (d >= D) return;
        float a = float(Residual[(row*4+0)*D+d]) * Comb[row*16+j];
        float b = float(Residual[(row*4+1)*D+d]) * Comb[row*16+4+j];
        float c = float(Residual[(row*4+2)*D+d]) * Comb[row*16+8+j];
        float e = float(Residual[(row*4+3)*D+d]) * Comb[row*16+12+j];
        Y[(row*4+j)*D+d] = T((a+b+c+e) + Post[row*4+j]*float(X[row*D+d]));
        """,
        ensure_row_contiguous=True,
    )


def metal_hc_post(x, residual, post, comb):
    """Expand C[input_lane,output_lane] without a [B,S,4,4,D] temporary."""
    global POST_CALLS
    if (residual.shape != (*x.shape[:2], 4, x.shape[-1])
            or post.shape != (*x.shape[:2], 4) or comb.shape != (*x.shape[:2], 4, 4)
            or not mx.metal.is_available() or mx.default_device() != mx.gpu):
        raise ValueError("Fused V4.1 post requires Metal and four residual lanes")
    POST_CALLS += 1
    return _post_kernel()(
        inputs=[x, residual, post.astype(mx.float32), comb.astype(mx.float32)],
        template=[("D", x.shape[-1]), ("T", x.dtype)],
        grid=(x.shape[-1], 4, x.shape[0]*x.shape[1]), threadgroup=(256, 1, 1),
        output_shapes=[residual.shape], output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _sinkhorn_kernel():
    return mx.fast.metal_kernel(
        name="dsv41_sinkhorn4", input_names=["Mix", "Scale", "Base", "Eps"],
        output_names=["Pre", "Post", "Comb"], source="""
        uint row = thread_position_in_grid.x;
        if (row >= N) return;
        float a[16];
        for (uint i=0; i<4; ++i) {
            float p = Mix[row*24+i]*Scale[0]+Base[i];
            float q = Mix[row*24+4+i]*Scale[1]+Base[4+i];
            Pre[row*4+i] = 1.0f/(1.0f+metal::precise::exp(-p))+Eps;
            Post[row*4+i] = 2.0f/(1.0f+metal::precise::exp(-q));
            for (uint j=0; j<4; ++j)
                a[i*4+j] = Mix[row*24+8+i*4+j]*Scale[2]+Base[8+i*4+j];
            float m = metal::max(metal::max(a[i*4],a[i*4+1]),metal::max(a[i*4+2],a[i*4+3]));
            for (uint j=0; j<4; ++j) a[i*4+j] = metal::precise::exp(a[i*4+j]-m);
            float s = (a[i*4]+a[i*4+1])+(a[i*4+2]+a[i*4+3]);
            for (uint j=0; j<4; ++j) a[i*4+j] = a[i*4+j]/s+Eps;
        }
        for (uint iter=0; iter<ITERS; ++iter) {
            if (iter) {
                for (uint i=0; i<4; ++i) {
                    float s = (a[i*4]+a[i*4+1])+(a[i*4+2]+a[i*4+3])+Eps;
                    for (uint j=0; j<4; ++j) a[i*4+j] /= s;
                }
            }
            for (uint j=0; j<4; ++j) {
                float s = (a[j]+a[4+j])+(a[8+j]+a[12+j])+Eps;
                for (uint i=0; i<4; ++i) a[i*4+j] /= s;
            }
        }
        for (uint i=0; i<16; ++i) Comb[row*16+i] = a[i];
        """
    )


def metal_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    """Keep the staggered collapse outside; fuse only the 4x4 normalization loop."""
    global SINKHORN_CALLS
    if hc_mult != 4 or sinkhorn_iters < 1 or not mx.metal.is_available():
        raise ValueError("V4.1 Metal Sinkhorn requires HC=4, iterations>=1 and Metal")
    SINKHORN_CALLS += 1
    shape = mixes.shape[:-1]
    n = mixes.size // 24
    return _sinkhorn_kernel()(
        inputs=[mixes.astype(mx.float32), scale.astype(mx.float32), base.astype(mx.float32),
                mx.array(eps, mx.float32)], template=[("N", n), ("ITERS", sinkhorn_iters)],
        grid=(n, 1, 1), threadgroup=(min(32, n), 1, 1),
        output_shapes=[(*shape, 4), (*shape, 4), (*shape, 4, 4)],
        output_dtypes=[mx.float32]*3)


@mx.compile
def compiled_hc_post(x, residual, post, comb):
    """Compile the reference FP32 expression without substituting a GEMM."""
    prod = comb.astype(mx.float32)[..., None] * residual.astype(mx.float32)[..., :, None, :]
    out = post.astype(mx.float32)[..., None] * x.astype(mx.float32)[..., None, :]
    return (out + mx.sum(prod, axis=2)).astype(x.dtype)
