"""Mac numerical gates and glue timing, no checkpoint required.

pytest -q -s mlx_vlm/models/deepseek_v41/test_megakernel_prefill.py
python mlx_vlm/models/deepseek_v41/test_megakernel_prefill.py --timing
"""

import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal required", allow_module_level=True)

# Also support the standalone timing command from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mlx_vlm.models.deepseek_v41 import megakernel as mega
from mlx_vlm.models.deepseek_v41 import hyper_connection as hc_kernels
from mlx_vlm.models.deepseek_v41 import prefill_ledger as ledger
from mlx_vlm.models.deepseek_v41.language import DeepseekV41Block, HyperConnection
from mlx_vlm.models.deepseek_v4.hyper_connection import _hc_split_sinkhorn_ops


def _pre(h, pre):
    return mx.sum(pre[..., None].astype(mx.float32) * h.astype(mx.float32), axis=2).astype(h.dtype)


def _post(x, h, post, comb):
    prod = comb.astype(mx.float32)[..., None] * h.astype(mx.float32)[..., :, None, :]
    out = post.astype(mx.float32)[..., None] * x.astype(mx.float32)[..., None, :]
    return (out + mx.sum(prod, axis=2)).astype(x.dtype)


def _norm(x, norm):
    xf = x.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + norm.eps)
    return (norm.weight * (xf * inv)).astype(x.dtype)


def _data(n, d=5120, b=1):
    mx.random.seed(4100 + n + d)
    shapes = ((b, n, 4, d), (b, n, d), (b, n, 4), (b, n, 4), (b, n, 4, 4))
    h, x, pre, post, comb = [mx.random.normal(s).astype(mx.bfloat16) for s in shapes]
    norm = SimpleNamespace(eps=1e-6, weight=1 + 0.1 * mx.random.normal((d,)))
    mx.eval(h, x, pre, post, comb, norm.weight)
    return h, x, pre, post, comb, norm


def _run(kind, data, mode):
    h, x, pre, post, comb, norm = data
    if mode == "eager":
        if kind == "pre":
            return [_norm(_pre(h, pre), norm)]
        residual = _post(x, h, post, comb)
        if kind == "post":
            return [residual]
        return [residual, _norm(_pre(residual, pre), norm)]
    if kind == "pre":
        fn = mega.collapse_norm_rows if mode == "rows" else mega.collapse_norm
        return [fn(h, pre, norm)]
    if kind == "post":
        fn = mega.expand_rows if mode == "rows" else mega.expand
        return [fn(x, h, post, comb)]
    fn = mega.expand_collapse_norm_rows if mode == "rows" else mega.expand_collapse_norm
    return fn(x, h, post, comb, pre, norm)


def _stats(actual, expected):
    a = np.asarray(actual.astype(mx.float32)).astype(np.float64)
    e = np.asarray(expected.astype(mx.float32)).astype(np.float64)
    assert np.isfinite(a).all() and np.isfinite(e).all()
    delta = a - e
    return float(np.max(np.abs(delta))), float(np.sum(delta * delta)), float(np.sum(e * e))


def _metrics(stats):
    return stats[0], (stats[1] / max(stats[2], 1e-30)) ** 0.5


def _assert_gate(label, rows, decode):
    row_abs, row_rms = _metrics(rows)
    dec_abs, dec_rms = _metrics(decode)
    print(f"{label}: rows max_abs={row_abs:.9g} rms_relative={row_rms:.9g}; "
          f"decode max_abs={dec_abs:.9g} rms_relative={dec_rms:.9g}")
    # No arbitrary tolerance floor: an exact decode baseline requires exact rows.
    assert row_abs <= 2 * dec_abs, label
    assert row_rms <= 2 * dec_rms, label


@pytest.mark.parametrize("n", [2, 16, 512, 2048])
def test_rows_against_eager_and_decode(n):
    data = _data(n)
    for kind in ("pre", "bridge", "post"):
        expected = _run(kind, data, "eager")
        actual = _run(kind, data, "rows")
        mx.eval(*expected, *actual)
        row_stats = []
        for a, e in zip(actual, expected):
            assert a.shape == e.shape and a.dtype == mx.bfloat16
            row_stats.append(_stats(a, e))
        decode_stats = [[0.0, 0.0, 0.0] for _ in expected]
        # Every token contributes to the one-token baseline, against the SAME
        # eager tensors. Bound graph size by evaluating 32 decode rows at a time.
        for start in range(0, n, 32):
            stop = min(n, start + 32)
            outputs = []
            for token in range(start, stop):
                one = tuple(mx.contiguous(a[:, token:token+1]) for a in data[:-1]) + (data[-1],)
                outputs.append(_run(kind, one, "decode"))
            batched = [mx.concatenate([o[j] for o in outputs], axis=1) for j in range(len(expected))]
            mx.eval(*batched)
            for j, result in enumerate(batched):
                s = _stats(result, expected[j][:, start:stop])
                decode_stats[j][0] = max(decode_stats[j][0], s[0])
                decode_stats[j][1] += s[1]
                decode_stats[j][2] += s[2]
        for j in range(len(expected)):
            _assert_gate(f"n={n} {kind}[{j}]", row_stats[j], decode_stats[j])


@pytest.mark.parametrize("d", [7, 64, 513, 1025, 8192])
def test_batched_broadcast_and_tail_rows(d):
    h, x, _, post, comb, norm = _data(2, d, b=2)
    # Broadcast residuals match the first model layer; coefficients also exercise
    # strided row inputs. Distinct batches catch incorrect flattening offsets.
    h = mx.broadcast_to(h[:, :, :1], h.shape)
    pre = mx.array([1, 0, 0, 0], dtype=mx.bfloat16)
    data = (h, x, mx.broadcast_to(pre, (2, 2, 4)), post[:, ::-1], comb[:, ::-1], norm)
    for kind in ("pre", "bridge", "post"):
        actual = _run(kind, data, "rows")
        # Compare each row to decode exactly: indexing should change no arithmetic.
        for b in range(2):
            for t in range(2):
                one = tuple(mx.contiguous(a[b:b+1, t:t+1]) for a in data[:-1]) + (norm,)
                decode = _run(kind, one, "decode")
                for a, e in zip(actual, decode):
                    assert mx.array_equal(a[b:b+1, t:t+1], e).item()
    assert mx.array_equal(mega.collapse_norm_rows(h, pre, norm),
                          mega.collapse_norm_rows(h, data[2], norm)).item()


@pytest.mark.parametrize("fast_norm", [False, True])
def test_prefill_coefficients_use_metal_sinkhorn(fast_norm):
    h, _, _, _, _, _ = _data(16, 64)
    hc = _hc(64)
    hc.fast_norm = fast_norm
    xf = h.reshape(1, 16, -1).astype(mx.float32)
    if fast_norm:
        mixes = mx.fast.rms_norm(xf, None, hc.norm_eps) @ hc.fn.T
    else:
        mixes = (xf @ hc.fn.T) * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + hc.norm_eps)
    expected = _hc_split_sinkhorn_ops(mixes, hc.scale, hc.base, 4, hc.iters, hc.eps)
    count = hc_kernels.SINKHORN_CALLS
    actual = mega._prefill_hc(hc, h)
    assert hc_kernels.SINKHORN_CALLS == count + 1
    for a, e in zip(actual, expected):
        assert mx.allclose(a, e, atol=1e-6, rtol=1e-6).item()


def _hc(d):
    hc = HyperConnection(SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                                         rms_norm_eps=1e-6, hidden_size=d))
    hc.fn = mx.random.normal((24, 4*d)) / (4*d) ** 0.5
    hc.base = mx.random.normal((24,))
    hc.scale = mx.array([0.5, 0.7, 0.9])
    return hc


@pytest.mark.parametrize("cuts", [False, True])
def test_prefill_block_routing_stagger_and_ledger(monkeypatch, cuts):
    h, _, pre, _, _, norm = _data(3, 64)
    calls = []
    cache, shared, mask = object(), object(), object()

    class Attention:
        layer_id = 7

        def __call__(self, x, start_pos, actual_cache, actual_shared):
            assert (start_pos, actual_cache, actual_shared) == (19, cache, shared)
            calls.append("attn")
            return (x * 0.25).astype(x.dtype)

    def ffn(x, actual_mask):
        assert actual_mask is mask
        calls.append("ffn")
        return (x * 0.5).astype(x.dtype)

    layer = SimpleNamespace(attn=Attention(), ffn=ffn, attn_hc=_hc(64), ffn_hc=_hc(64),
                            attn_norm=norm, ffn_norm=norm)
    # An explicit row-kernel chain is the exact routing/stagger oracle.
    attn_pre, post, comb = mega._prefill_hc(layer.attn_hc, h)
    x = mega.collapse_norm_rows(h, pre, norm)
    residual, x = mega.expand_collapse_norm_rows((x * 0.25).astype(x.dtype), h, post, comb, attn_pre, norm)
    ffn_pre, post, comb = mega._prefill_hc(layer.ffn_hc, residual)
    expected = mega.expand_rows((x * 0.5).astype(x.dtype), residual, post, comb)
    monkeypatch.setattr(ledger, "_PREFILL_LEDGER_CUTS", cuts)
    monkeypatch.setattr(ledger, "PREFILL_LAYER_MS", {})
    monkeypatch.setenv("DSV41_PREFILL_MEGAKERNEL", "1")
    # Existing switches must not send a multi-token call into decode/compile or
    # demand incoming_pre from HyperConnection's separate fused-HC entry point.
    for flag in ("DSV41_MEGAKERNEL", "DSV41_COMPILE_BLOCK", "DSV41_HC_FUSED"):
        monkeypatch.setenv(flag, "1")
    count = hc_kernels.SINKHORN_CALLS
    actual, next_pre = DeepseekV41Block.__call__(layer, h, pre, 19, cache, shared, mask)
    assert hc_kernels.SINKHORN_CALLS == count + 2
    assert calls == ["attn", "ffn"]
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(next_pre, ffn_pre).item()
    assert set(ledger.PREFILL_LAYER_MS) == ({(7, "attn_ms"), (7, "ffn_ms")} if cuts else set())
    assert all(ms >= 0 for ms in ledger.PREFILL_LAYER_MS.values())


def timing_glue(repeats=10, warmup=3):
    """Two collapses/norms + two expansions, synthetic sublayer outputs.

    Includes eager dispatch and eval/synchronize; excludes HC projection and
    Sinkhorn, attention, MoE, and ledger cuts. All inputs are materialized first.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats >= 1 and warmup >= 0 required")
    h, attn, incoming, post, comb, norm = _data(2048)
    shapes = (attn.shape, incoming.shape, post.shape, comb.shape)
    ffn, attn_pre, ffn_post, ffn_comb = [mx.random.normal(s).astype(mx.bfloat16) for s in shapes]
    mx.eval(ffn, attn_pre, ffn_post, ffn_comb)

    def eager():
        attn_x = _norm(_pre(h, incoming), norm)
        residual = _post(attn, h, post, comb)
        ffn_x = _norm(_pre(residual, attn_pre), norm)
        return attn_x, ffn_x, _post(ffn, residual, ffn_post, ffn_comb)

    def fused():
        attn_x = mega.collapse_norm_rows(h, incoming, norm)
        residual, ffn_x = mega.expand_collapse_norm_rows(attn, h, post, comb, attn_pre, norm)
        return attn_x, ffn_x, mega.expand_rows(ffn, residual, ffn_post, ffn_comb)

    for _ in range(warmup):
        for fn in (eager, fused):
            mx.eval(*fn())
            mx.synchronize()
    samples = {"eager": [], "fused": []}
    for trial in range(repeats):
        order = (("eager", eager), ("fused", fused))
        for name, fn in (order if trial % 2 == 0 else order[::-1]):
            mx.synchronize()
            t0 = perf_counter()
            outputs = fn()
            mx.eval(*outputs)
            mx.synchronize()
            samples[name].append((perf_counter() - t0) * 1000)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print(f"n=2048 D=5120 bf16 glue ms/layer (median of {repeats}): "
          f"eager={medians['eager']:.3f} fused={medians['fused']:.3f} "
          f"speedup={medians['eager']/medians['fused']:.2f}x")
    print("Scope: hc_pre/hc_post + RMSNorm only; excludes coefficients, attention and FFN.")
    return medians


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.timing:
        timing_glue(args.repeats, args.warmup)
    else:
        parser.error("Use --timing, or run this file with pytest for numerical gates")
