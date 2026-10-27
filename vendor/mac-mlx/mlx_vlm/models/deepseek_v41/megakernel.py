"""Residual/norm megakernels; decode DSV41_MEGAKERNEL=1, default off.

Prefill uses DSV41_PREFILL_MEGAKERNEL=1 (also default off), row kernels and
Metal Sinkhorn independently of the decode/compiled/fused-HC switches.

Three launches cover both collapse/norms and both residual expansions. The
middle launch writes the expanded residual AND the normalized FFN input.
Attention, HC coefficient generation, cache/gather and MoE remain ordinary calls.
No weights are captured: cached kernels receive the current arrays every call.
"""
from functools import lru_cache
import os

import mlx.core as mx


# Explicit products and casts preserve the reference's FP32 reduction and its
# activation rounding between expand, collapse and RMSNorm. Disable contraction
# so a compiler FMA cannot silently remove a product-rounding boundary.
_EXPAND = """
    float lanes[4];
    for (uint j = 0; j < 4; ++j) {
        float p0 = float(Comb[j]) * float(H[d]);
        float p1 = float(Comb[4+j]) * float(H[D+d]);
        float p2 = float(Comb[8+j]) * float(H[2*D+d]);
        float p3 = float(Comb[12+j]) * float(H[3*D+d]);
        float add = float(Post[j]) * float(X[d]);
        lanes[j] = float(T(add + (((p0 + p1) + p2) + p3)));
        Residual[j*D+d] = T(lanes[j]);
    }
"""


@lru_cache(maxsize=64)
def _kernel(kind, eps=0.0):
    if kind not in ("pre", "bridge", "post"):
        raise ValueError(kind)
    inputs = ["H", "Pre", "Weight"] if kind == "pre" else ["H", "X", "Post", "Comb"]
    if kind == "bridge":
        inputs += ["Pre", "Weight"]
    if kind == "post":
        source = "uint d = thread_position_in_grid.x; if (d >= D) return;" + _EXPAND
        outputs = ["Residual"]
    else:
        values = (_EXPAND if kind == "bridge" else """
            float lanes[4];
            for (uint j = 0; j < 4; ++j) lanes[j] = float(H[j*D+d]);
        """)
        source = """
        uint tid = thread_index_in_threadgroup;
        threadgroup float sums[32];
        // Match MLX's axis=-1 row reduction (not its all-axis reduction).
        // <=64 values accumulate serially; wider rows use four reads/thread.
        constexpr uint N = D <= 64 ? D : ((D+TG*4-1)/(TG*4))*4;
        float collapsed[N];
        float ss = 0.0f;
        for (uint k = 0; k < N; ++k) {
            uint d = D <= 64 ? tid*D+k : (k/4)*TG*4+tid*4+k%4;
            float v = 0.0f;
            if (d < D) {
        """ + values + """
                float p0 = float(Pre[0]) * lanes[0];
                float p1 = float(Pre[1]) * lanes[1];
                float p2 = float(Pre[2]) * lanes[2];
                float p3 = float(Pre[3]) * lanes[3];
                v = float(T(((p0 + p1) + p2) + p3));
            }
            collapsed[k] = v;
            ss += v*v;
        }
        ss = simd_sum(ss);
        if (tid%32 == 0) sums[tid/32] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(tid < TG/32 ? sums[tid] : 0.0f);
        if (tid == 0) sums[0] = metal::precise::rsqrt(total/float(D) + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = sums[0];
        for (uint k = 0; k < N; ++k) {
            uint d = D <= 64 ? tid*D+k : (k/4)*TG*4+tid*4+k%4;
            if (d < D)
                Norm[d] = T((collapsed[k]*inv) * float(Weight[d]));
        }
        """
        outputs = ["Norm"] if kind == "pre" else ["Residual", "Norm"]
    return mx.fast.metal_kernel(
        name="dsv41_mega_" + kind, input_names=inputs, output_names=outputs,
        header="#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n",
        source=f"const float EPS = {float(eps)!r}f;\n" + source)


def _check(h):
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_MEGAKERNEL requires Metal; no silent fallback")
    if h.ndim != 4 or h.shape[:3] != (1, 1, 4) or not 0 < h.shape[-1] <= 8192:
        raise ValueError("Megakernel requires [1,1,4,D], 0 < D <= 8192")
    if h.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError("Megakernel requires floating activations")


def _norm_threads(d):
    return 32 if d <= 512 else 128 if d <= 1024 else min(1024, ((d+127)//128)*32)


def collapse_norm(h, pre, norm):
    _check(h)
    d = h.shape[-1]
    tg = _norm_threads(d)
    return _kernel("pre", norm.eps)(
        inputs=[h, pre, norm.weight],
        template=[("T", h.dtype), ("D", d), ("TG", tg)],
        grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(1, 1, d)], output_dtypes=[h.dtype])[0]


def expand_collapse_norm(x, h, post, comb, pre, norm):
    _check(h)
    d = h.shape[-1]
    tg = _norm_threads(d)
    return _kernel("bridge", norm.eps)(
        inputs=[h, x, post, comb, pre, norm.weight],
        template=[("T", x.dtype), ("D", d), ("TG", tg)],
        grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[h.shape, (1, 1, d)], output_dtypes=[x.dtype, x.dtype])


def expand(x, h, post, comb):
    _check(h)
    d = h.shape[-1]
    return _kernel("post")(
        inputs=[h, x, post, comb], template=[("T", x.dtype), ("D", d)],
        grid=(d, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[h.shape], output_dtypes=[x.dtype])[0]


def decode_block(layer, h, pre_mix, start_pos, cache, shared, image_mask=None):
    """Use incoming pre for attention, attention's pre for FFN (staggered HC)."""
    _check(h)
    if os.environ.get("DSV41_COMPILE_BLOCK", "0") == "1":
        graph = getattr(layer, "_megakernel_graph", None)
        if graph is None:
            graph = _MegakernelGraphs()
            object.__setattr__(layer, "_megakernel_graph", graph)
        graph.configure()
        return graph(layer, h, pre_mix, start_pos, cache, shared, image_mask)
    attn_pre, post, comb = layer.attn_hc(h)
    x = collapse_norm(h, pre_mix, layer.attn_norm)
    attention = layer.attn(x, start_pos, cache, shared)
    h, x = expand_collapse_norm(attention, h, post, comb, attn_pre, layer.ffn_norm)
    ffn_pre, post, comb = layer.ffn_hc(h)
    return expand(layer.ffn(x, image_mask), h, post, comb), ffn_pre


# Reuse the existing graph registry and its explicit, current-parameter binding.
# Custom residual kernels compose with compilation of HC and MoE arithmetic;
# selecting the megakernel must not discard those independent launch savings.
from .block_graph import BlockGraphs, _bind


class _MegakernelGraphs(BlockGraphs):
    def _make(self, template):
        @mx.compile
        def pre(h, pre_mix, parameters):
            self.counts["pre_traces"] += 1
            with _bind(template, parameters):
                attn_pre, post, comb = template.attn_hc(h)
                return collapse_norm(h, pre_mix, template.attn_norm), attn_pre, post, comb

        @mx.compile
        def post(attention, h, attn_pre, post_mix, comb, parameters, image_mask):
            self.counts["post_traces"] += 1
            with _bind(template, parameters):
                h, x = expand_collapse_norm(attention, h, post_mix, comb, attn_pre, template.ffn_norm)
                ffn_pre, post_mix, comb = template.ffn_hc(h)
                return expand(template.ffn(x, image_mask), h, post_mix, comb), ffn_pre

        return pre, post


# Prefill kernels are separate so decode source, entry points and grids stay fixed.
@lru_cache(maxsize=64)
def _rows_kernel(kind, eps=0.0):
    if kind not in ("pre", "bridge", "post"):
        raise ValueError(kind)
    inputs = ["H", "Pre", "Weight"] if kind == "pre" else ["H", "X", "Post", "Comb"]
    if kind == "bridge":
        inputs += ["Pre", "Weight"]
    if kind == "post":
        source = "uint d = thread_position_in_grid.x; if (d >= D) return;" + _EXPAND
        outputs = ["Residual"]
    else:
        values = (_EXPAND if kind == "bridge" else """
            float lanes[4];
            for (uint j = 0; j < 4; ++j) lanes[j] = float(H[j*D+d]);
        """)
        source = """
        uint tid = thread_index_in_threadgroup;
        threadgroup float sums[32];
        // Match MLX's axis=-1 row reduction (not its all-axis reduction).
        // <=64 values accumulate serially; wider rows use four reads/thread.
        constexpr uint N = D <= 64 ? D : ((D+TG*4-1)/(TG*4))*4;
        float collapsed[N];
        float ss = 0.0f;
        for (uint k = 0; k < N; ++k) {
            uint d = D <= 64 ? tid*D+k : (k/4)*TG*4+tid*4+k%4;
            float v = 0.0f;
            if (d < D) {
        """ + values + """
                float p0 = float(Pre[0]) * lanes[0];
                float p1 = float(Pre[1]) * lanes[1];
                float p2 = float(Pre[2]) * lanes[2];
                float p3 = float(Pre[3]) * lanes[3];
                v = float(T(((p0 + p1) + p2) + p3));
            }
            collapsed[k] = v;
            ss += v*v;
        }
        ss = simd_sum(ss);
        if (tid%32 == 0) sums[tid/32] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(tid < TG/32 ? sums[tid] : 0.0f);
        if (tid == 0) sums[0] = metal::precise::rsqrt(total/float(D) + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = sums[0];
        for (uint k = 0; k < N; ++k) {
            uint d = D <= 64 ? tid*D+k : (k/4)*TG*4+tid*4+k%4;
            if (d < D)
                Norm[d] = T((collapsed[k]*inv) * float(Weight[d]));
        }
        """
        outputs = ["Norm"] if kind == "pre" else ["Residual", "Norm"]
    # The y dimension selects a flattened [batch, token] row. Local pointer
    # aliases let the arithmetic use exactly the decode kernel's row offsets.
    # Weight is shared; every other input/output advances independently.
    strides = {"H": "4*D", "X": "D", "Pre": "4", "Post": "4",
               "Comb": "16", "Residual": "4*D", "Norm": "D"}
    aliases = "uint row = threadgroup_position_in_grid.y;\n"
    for name in inputs:
        if name != "Weight":
            aliases += f"const auto {name} = Rows{name} + row*({strides[name]});\n"
    for name in outputs:
        aliases += f"auto {name} = Rows{name} + row*({strides[name]});\n"
    return mx.fast.metal_kernel(
        name="dsv41_mega_rows_" + kind,
        input_names=[name if name == "Weight" else "Rows" + name for name in inputs],
        output_names=["Rows" + name for name in outputs],
        header="#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n",
        source=f"const float EPS = {float(eps)!r}f;\n" + aliases + source,
        ensure_row_contiguous=True)


def _check_rows(h):
    if not mx.metal.is_available():
        raise RuntimeError("DSV41_PREFILL_MEGAKERNEL requires Metal; no silent fallback")
    if (h.ndim != 4 or h.shape[2] != 4 or min(h.shape[:2]) < 1
            or not 0 < h.shape[-1] <= 8192):
        raise ValueError("Row megakernels require [B,S,4,D], B,S > 0, 0 < D <= 8192")
    if h.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError("Row megakernels require floating activations")


def _row_pre(h, pre, norm):
    if norm.weight.shape != (h.shape[-1],):
        raise ValueError("Row megakernel norm weight must have shape [D]")
    # Initial lane-zero mixes and sliced inputs may have zero/non-unit strides.
    return mx.contiguous(mx.broadcast_to(pre, (*h.shape[:2], 4)))


def _check_row_post(x, h, post, comb):
    shape = h.shape[:2]
    if (x.shape != (*shape, h.shape[-1]) or post.shape != (*shape, 4)
            or comb.shape != (*shape, 4, 4) or x.dtype != h.dtype):
        raise ValueError("Row expansion requires matching X/H dtype and [B,S,D]/[B,S,4]/[B,S,4,4]")


def collapse_norm_rows(h, pre, norm):
    """Collapse incoming HC lanes and normalize, one threadgroup per token."""
    _check_rows(h)
    pre = _row_pre(h, pre, norm)
    b, n, _, d = h.shape
    tg = _norm_threads(d)
    return _rows_kernel("pre", norm.eps)(
        inputs=[mx.contiguous(h), pre, mx.contiguous(norm.weight)],
        template=[("T", h.dtype), ("D", d), ("TG", tg)],
        grid=(tg, b*n, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(b, n, d)], output_dtypes=[h.dtype])[0]


def expand_collapse_norm_rows(x, h, post, comb, pre, norm):
    """Expand attention residual and collapse/normalize the next FFN input."""
    _check_rows(h)
    _check_row_post(x, h, post, comb)
    pre = _row_pre(h, pre, norm)
    b, n, _, d = h.shape
    tg = _norm_threads(d)
    return _rows_kernel("bridge", norm.eps)(
        inputs=[mx.contiguous(a) for a in (h, x, post, comb, pre, norm.weight)],
        template=[("T", x.dtype), ("D", d), ("TG", tg)],
        grid=(tg, b*n, 1), threadgroup=(tg, 1, 1),
        output_shapes=[h.shape, (b, n, d)], output_dtypes=[x.dtype, x.dtype])


def expand_rows(x, h, post, comb):
    """Expand FFN residual with independent token rows on grid.y."""
    _check_rows(h)
    _check_row_post(x, h, post, comb)
    b, n, _, d = h.shape
    return _rows_kernel("post")(
        inputs=[mx.contiguous(a) for a in (h, x, post, comb)],
        template=[("T", x.dtype), ("D", d)],
        grid=(d, b*n, 1), threadgroup=(256, 1, 1),
        output_shapes=[h.shape], output_dtypes=[x.dtype])[0]


def _prefill_hc(hc, h):
    """Use the existing row Sinkhorn without changing process-wide HC flags."""
    from .hyper_connection import metal_sinkhorn

    xf = h.reshape(*h.shape[:2], -1).astype(mx.float32)
    if hc.fast_norm:
        mixes = mx.fast.rms_norm(xf, None, hc.norm_eps) @ hc.fn.astype(mx.float32).T
    else:
        rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + hc.norm_eps)
        mixes = (xf @ hc.fn.astype(mx.float32).T) * rsqrt
    return metal_sinkhorn(mixes, hc.scale, hc.base, hc.hc, hc.iters, hc.eps)


def prefill_block(layer, h, pre_mix, start_pos, cache, shared, image_mask=None):
    """Staggered HC for n>1; attention/cache and MoE remain ordinary calls.

    Ledger boundaries match the eager path. The fused bridge also computes
    FFN collapse/norm at the attention cut, so that work is charged to attn_ms.
    Attention's level-2 cuts remain active inside its ordinary __call__.
    """
    import time
    from .prefill_ledger import _PREFILL_LEDGER_CUTS, _prefill_layer_add

    _check_rows(h)
    if h.shape[1] <= 1:
        raise ValueError("prefill_block requires more than one token")
    attn_pre, post, comb = _prefill_hc(layer.attn_hc, h)
    x = collapse_norm_rows(h, pre_mix, layer.attn_norm)
    attention = layer.attn(x, start_pos, cache, shared)
    h, x = expand_collapse_norm_rows(attention, h, post, comb, attn_pre, layer.ffn_norm)
    if _PREFILL_LEDGER_CUTS:
        t0 = time.perf_counter()
        mx.eval(h, x, attn_pre)
        mx.synchronize()
        _prefill_layer_add(layer.attn.layer_id, "attn_ms", (time.perf_counter() - t0) * 1000.0)
    ffn_pre, post, comb = _prefill_hc(layer.ffn_hc, h)
    h = expand_rows(layer.ffn(x, image_mask), h, post, comb)
    if _PREFILL_LEDGER_CUTS:
        t0 = time.perf_counter()
        mx.eval(h, ffn_pre)
        mx.synchronize()
        _prefill_layer_add(layer.attn.layer_id, "ffn_ms", (time.perf_counter() - t0) * 1000.0)
    return h, ffn_pre


