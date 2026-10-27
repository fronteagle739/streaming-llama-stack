"""Compare the fused normalization with the existing precise FP32 implementation."""
import pytest

mx = pytest.importorskip("mlx.core")
from .hyper_connection import metal_sinkhorn
from ..deepseek_v4.hyper_connection import _hc_split_sinkhorn_ops


@pytest.mark.parametrize("tokens", [1, 7, 256])
@pytest.mark.parametrize("iterations", [1, 20])
def test_sinkhorn_matches_reference(tokens, iterations):
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    mx.random.seed(41)
    mixes = mx.random.normal((1, tokens, 24)) * 4
    scale = mx.array([0.5, 0.7, 0.9])
    base = mx.random.normal((24,))
    actual = metal_sinkhorn(mixes, scale, base, 4, iterations, 1e-6)
    expected = _hc_split_sinkhorn_ops(mixes, scale, base, 4, iterations, 1e-6)
    mx.eval(actual, expected)
    for a, b in zip(actual, expected):
        assert mx.allclose(a, b, atol=1e-6, rtol=1e-6).item()
    assert mx.all(mx.isfinite(actual[2])).item()


@pytest.mark.parametrize("width", [64, 5120])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_fast_rms_preserves_reference_precision(monkeypatch, width, dtype):
    from .language import RMSNorm
    mx.random.seed(41)
    norm = RMSNorm(width)
    norm.weight = 1 + mx.random.normal((width,)) * 0.1
    x = mx.random.normal((1, 17, width)).astype(dtype)
    monkeypatch.setenv("DSV41_FAST_RMS_NORM", "0")
    expected = norm(x)
    monkeypatch.setenv("DSV41_FAST_RMS_NORM", "1")
    actual = norm(x)
    mx.eval(actual, expected)
    # Native reduction order can put a BF16 result on the adjacent rounding
    # value. The real QAT model is separately held to the long retrieval gate.
    tolerance = 0.008 if dtype == mx.bfloat16 else 1e-5
    assert mx.allclose(actual, expected, atol=1e-6, rtol=tolerance).item()
