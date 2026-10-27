"""Opt-in V4.1 fp32 index scoring and exact 512-position selection.

No native GLM dependency. Metal failures propagate. Ties select lower positions;
MLX argpartition does not specify which equal-score positions it retains.
"""
from functools import lru_cache

import mlx.core as mx


@lru_cache(None)
def _projection():
    return mx.fast.metal_kernel(
        name="v41_index_projection", input_names=["x", "p", "scale"],
        output_names=["w"], source=r"""
        uint lane = thread_index_in_simdgroup;
        uint head = thread_position_in_grid.x / 32;
        if (head >= B*S*H) return;
        uint row = head / H, h = head % H;
        float sum = 0;
        for (uint d = lane; d < X; d += 32)
            sum += float(x[row*X+d]) * float(p[h*X+d]);
        sum = simd_sum(sum);
        if (lane == 0) w[head] = sum * scale[0];
        """)


@lru_cache(None)
def _reduce():
    return mx.fast.metal_kernel(
        name="v41_index_reduce", input_names=["scores", "weights"],
        output_names=["out"], source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= B*S*N) return;
        uint row = i / N, key = i % N;
        float acc = 0;
        for (uint h = 0; h < H; ++h)
            acc += max(float(scores[(row*H+h)*N+key]), 0.0f)
                   * float(weights[row*H+h]);
        out[i] = acc;
        """)


def index_scores(x, projection, q, index_k, scale):
    """FP32 accumulation, including when inputs/parameters are bf16 or half."""
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_METAL_INDEXER requires Metal")
    b, s, h, d = q.shape
    n = index_k.shape[1]
    if (x.shape[:2] != (b, s) or projection.shape != (h, x.shape[-1])
            or index_k.shape != (b, n, d) or min(b, s, h, d, n) <= 0):
        raise ValueError("invalid V4.1 indexer shapes")
    allowed = (mx.float32, mx.float16, mx.bfloat16)
    if any(a.dtype not in allowed for a in (x, projection, q, index_k)):
        raise ValueError("indexer expects floating point inputs")
    w = _projection()(inputs=[x, projection, mx.array([scale], mx.float32)],
        template=[("B", b), ("S", s), ("H", h), ("X", x.shape[-1])],
        grid=(b*s*h*32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, s, h)], output_dtypes=[mx.float32])[0]
    scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                       index_k.astype(mx.float32))
    return _reduce()(inputs=[scores, w],
        template=[("B", b), ("S", s), ("H", h), ("N", n)],
        grid=(b*s*n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, s, n)], output_dtypes=[mx.float32])[0]


@lru_cache(None)
def _topk():
    # One group per row. Find the kth ordered float with radix selection, then
    # resolve boundary ties by position. A prefix scan emits position-sorted
    # indices directly; no negation, argpartition, cast, sort or gather kernels.
    return mx.fast.metal_kernel(
        name="v41_index_topk", input_names=["scores", "lens", "offset"],
        output_names=["out"], source=r"""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;
        threadgroup uint scratch[GROUPS];
        threadgroup uint total;
        uint keys[C];
        for (uint j=0; j<C; ++j) {
            uint i=tid*C+j;
            float v = i<N ? scores[row*N+i] : -INFINITY;
            uint u=as_type<uint>(v == 0.0f ? 0.0f : v);
            keys[j]=(u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
        }
        uint prefix=0;
        for (int bit=31; bit>=0; --bit) {
            uint pivot=prefix | (1u<<bit), count=0;
            for (uint j=0; j<C; ++j)
                count += (tid*C+j<N && keys[j]>=pivot);
            uint sum=simd_sum(count);
            if (lane==0) scratch[sg]=sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid==0) {
                uint s=0; for (uint g=0; g<GROUPS; ++g) s+=scratch[g];
                total=s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (total>=TOPK) prefix=pivot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint greater=0, equal=0;
        for (uint j=0; j<C; ++j) if (tid*C+j<N) {
            greater += keys[j]>prefix; equal += keys[j]==prefix;
        }
        uint sum=simd_sum(greater);
        if (lane==0) scratch[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid==0) { uint s=0; for (uint g=0; g<GROUPS; ++g) s+=scratch[g]; total=s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint needed=TOPK-total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before=simd_prefix_exclusive_sum(equal);
        sum=simd_sum(equal);
        if (lane==0) scratch[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint g=0; g<sg; ++g) before+=scratch[g];
        uint tie_start=before, count=0;
        for (uint j=0; j<C; ++j) {
            bool tie=keys[j]==prefix;
            count+=tid*C+j<N && (keys[j]>prefix || (tie && before<needed));
            before+=tie;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint dest=simd_prefix_exclusive_sum(count);
        sum=simd_sum(count);
        if (lane==0) scratch[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint g=0; g<sg; ++g) dest+=scratch[g];
        before=tie_start;
        for (uint j=0; j<C; ++j) {
            uint i=tid*C+j;
            bool tie=keys[j]==prefix;
            bool selected=i<N && (keys[j]>prefix || (tie && before<needed));
            before+=tie;
            if (selected) out[row*TOPK+dest++] = i<uint(lens[row%S]) && scores[row*N+i]>-INFINITY
                                  ? int(i)+offset[0] : -1;
        }
        """)


def index_topk(scores, k, lens, offset):
    """Exact rank selection for finite scores/-inf, with stable boundary ties."""
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_METAL_INDEXER requires Metal")
    b, s, n = scores.shape
    if scores.dtype != mx.float32 or not 0 < k <= n or lens.size != s:
        raise ValueError("expected fp32 [B,S,N], 0<k<=N, S visibility lengths")
    threads = 1024 if n > 16384 else 256
    return _topk()(inputs=[scores, lens.astype(mx.int32), mx.array([offset], mx.int32)],
        template=[("S", s), ("N", n), ("TOPK", k), ("C", (n+threads-1)//threads),
                  ("GROUPS", threads//32)],
        grid=(b*s*threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(b, s, k)], output_dtypes=[mx.int32])[0]
