"""Opt-in decode block graphs, with an explicit eager attention boundary.

Attention owns Python cache lengths, capacity growth, compressor parity and
shared-source pointers. Capturing it today would freeze those side effects.
The two pure regions here share traces across layers by passing weights as
inputs. Only batch-one, single-token decode is admitted: existing HC/MoE
reshapes embed Python sequence dimensions even with shapeless compilation.
"""
from collections import Counter
from contextlib import contextmanager
import os

import mlx.core as mx


_NAMES = ("attn_hc", "attn_norm", "ffn_hc", "ffn_norm", "ffn")


def _structure(value):
    if isinstance(value, mx.array):
        return (tuple(value.shape), str(value.dtype))
    if isinstance(value, dict):
        return (type(value).__name__, tuple((k, _structure(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_structure(v) for v in value)
    return repr(value)


def _parameters(layer):
    return {name: getattr(layer, name).parameters() for name in _NAMES}


@contextmanager
def _bind(template, parameters):
    # MLX's functional module-update pattern. Restore real parameters even if
    # tracing fails; tracers must never escape into the live model.
    original = _parameters(template)
    try:
        for name in _NAMES:
            getattr(template, name).update(parameters[name])
        yield
    finally:
        for name in _NAMES:
            getattr(template, name).update(original[name])


class BlockGraphs:
    """One registry per model, never per conversation or token."""

    def __init__(self):
        self.counts = Counter()
        self.graphs = {}
        self.bindings = {}
        self.flags = None
        self.configure()

    def configure(self):
        """Check math switches once per forward, outside the 40-layer loop.

        Replace this registry after structural module/quantization changes.
        Ordinary weight replacement is supported without rebuilding it.
        """
        flags = tuple(sorted((k, v) for k, v in os.environ.items()
                             if k.startswith('DSV41_') and k not in
                             ('DSV41_COMPILE_BLOCK', 'DSV41_ASYNC_PREPARE',
                              'DSV41_COALESCE_HASH_INPUTS')))
        if self.flags != flags:
            self.bindings.clear()
            self.flags = flags

    def stats(self):
        return dict(self.counts, variants=len(self.graphs),
                    trace_count=self.counts["pre_traces"] + self.counts["post_traces"])

    def report(self, layers):
        if self.counts['block_calls'] == layers or self.counts['block_calls'] % (100 * layers) == 0:
            import json
            print('[dsv41-block-graph] ' + json.dumps(self.stats()), flush=True)

    def _variant(self, layer):
        bound = self.bindings.get(id(layer))
        if bound is not None:
            return bound
        # Include scalar module configuration and quantization layout, never
        # weight values. Compute once per layer, not on every decode token.
        signature = (tuple(_structure(getattr(layer, n)) for n in _NAMES),
                     layer.attn_hc.fast_norm, layer.attn_hc.iters,
                     layer.attn_hc.eps, layer.attn_hc.norm_eps,
                     layer.attn_norm.eps, layer.ffn_norm.eps, self.flags)
        if signature not in self.graphs:
            self.graphs[signature] = self._make(layer)
        self.bindings[id(layer)] = self.graphs[signature]
        return self.graphs[signature]

    def _make(self, template):
        from .language import hc_pre, hc_post

        # MLX 0.32.2 rejects HC's Slice under shapeless=True. These regions
        # admit only [1,1,hc,dim]; KV growth is outside their argument trees.
        @mx.compile
        def pre(h, pre_mix, parameters):
            self.counts["pre_traces"] += 1  # Python executes only when traced.
            with _bind(template, parameters):
                attn_pre, post, comb = template.attn_hc(h)
                x = template.attn_norm(hc_pre(h, pre_mix))
                return x, attn_pre, post, comb

        @mx.compile
        def post(attention, h, attn_pre, post_mix, comb, parameters, image_mask):
            self.counts["post_traces"] += 1
            with _bind(template, parameters):
                h = hc_post(attention, h, post_mix, comb)
                ffn_pre, post_mix, comb = template.ffn_hc(h)
                x = template.ffn_norm(hc_pre(h, attn_pre))
                return hc_post(template.ffn(x, image_mask), h, post_mix, comb), ffn_pre

        return pre, post

    def __call__(self, layer, h, pre_mix, start_pos, cache, shared, image_mask):
        if h.shape[:2] != (1, 1):
            raise ValueError("BlockGraphs accepts only batch-one single-token decode")
        pre, post = self._variant(layer)
        parameters = _parameters(layer)  # current weights, including replacements
        self.counts["block_calls"] += 1
        self.counts["pre_calls"] += 1
        x, attn_pre, post_mix, comb = pre(h, pre_mix, parameters)
        attention = layer.attn(x, start_pos, cache, shared)
        self.counts["attention_eager_calls"] += 1
        self.counts["post_calls"] += 1
        return post(attention, h, attn_pre, post_mix, comb, parameters, image_mask)
