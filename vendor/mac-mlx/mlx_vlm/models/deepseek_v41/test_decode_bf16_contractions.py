"""Lever 2a: the bf16 attention-output contraction and the bf16 head agree with the fp32 ones to
bf16 rounding (the GEMM accumulates in fp32), and the bf16 head refuses without the fp32-logits kernel."""
import os
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_bf16_wo_a_contraction_matches_fp32_to_bf16_rounding():
    g, r, d = 8, 1024, 4096
    out = mx.random.normal((1, 1, g, d)).astype(mx.bfloat16)
    w = (mx.random.normal((g, r, d)) * 0.02).astype(mx.bfloat16)
    ref = mx.einsum("bsgd,grd->bsgr", out.astype(mx.float32), w.astype(mx.float32))
    got = mx.einsum("bsgd,grd->bsgr", out, w).astype(mx.float32)
    mx.eval(ref, got)
    ref_n, got_n = np.array(ref), np.array(got)
    scale = np.abs(ref_n).max()
    assert np.abs(ref_n - got_n).max() <= 1.6e-2 * scale, np.abs(ref_n - got_n).max() / scale


def test_bf16_head_through_the_omlx_kernel_gives_fp32_logits_close_to_the_fp32_gemm():
    from mlx_vlm.models.deepseek_v41.omlx_head import project_logits
    vocab, width = 8192, 512
    h = mx.random.normal((1, 1, width)).astype(mx.bfloat16)
    w = (mx.random.normal((vocab, width)) * 0.05).astype(mx.bfloat16)
    ref = h.astype(mx.float32) @ w.astype(mx.float32).T
    got = project_logits(h, w)
    mx.eval(ref, got)
    assert got.dtype == mx.float32
    ref_n, got_n = np.array(ref), np.array(got)
    assert np.abs(ref_n - got_n).max() <= 1e-3 * np.abs(ref_n).max()
    assert int(mx.argmax(ref)) == int(mx.argmax(got))


def test_head_bf16_without_the_kernel_is_refused(monkeypatch):
    monkeypatch.setenv("DSV41_HEAD_BF16", "1")
    monkeypatch.delenv("DSV41_OMLX_HEAD", raising=False)
    import importlib
    from mlx_vlm.models.deepseek_v41 import language as lang
    importlib.reload(lang)
    try:
        obj = lang.LanguageModel.__new__(lang.LanguageModel)
        with pytest.raises(RuntimeError, match="DSV41_OMLX_HEAD"):
            lang.LanguageModel.attach_engram_storage(obj, "/nonexistent")
    finally:
        monkeypatch.delenv("DSV41_HEAD_BF16", raising=False)
        importlib.reload(lang)
