"""Optional native-kernel acceleration for GLM-5.2 (`glm_moe_dsa`) DSA.

This is the glue layer between our pure-MLX GLM-5.2 forward (``glm_moe_dsa.py``)
and the vendored oMLX GLM sparse-MLA / DSA Metal kernels
(``mlx_vlm.custom_kernels.glm_moe_dsa``, from github.com/jundot/omlx PR #1984).

Design contract
---------------
* The pure-MLX path in ``glm_moe_dsa.py`` is the SSOT and stays byte-exact.
* Every accelerator entry point here returns ``None`` whenever the native
  extension is unavailable, force-disabled, or the input shapes/dtypes do not
  match the kernel's strict contract. The caller then runs the existing
  pure-MLX code, so behaviour is identical when kernels are off.
* ``MLX_GLM_DSA_NATIVE_KERNELS=0`` force-disables the native path regardless of
  whether the extension is built. Any other value (or unset) leaves it enabled;
  when unset the path is still a no-op unless the compiled ``_ext`` is present,
  so it is safe-by-default.

The kernel source itself is vendored near-verbatim; only this thin glue does
the shape mapping between our forward and the oMLX op contracts.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# Lazily import the vendored kernel package so that a partial / missing native
# build never breaks model import. All accelerator calls degrade to None.
try:
    from ...custom_kernels.glm_moe_dsa import sparse_mla as _omlx_sparse_mla
    from ...custom_kernels.glm_moe_dsa.kernels import fast as _glm_fast
except Exception as exc:  # pragma: no cover - depends on the vendored package
    _omlx_sparse_mla = None
    _glm_fast = None
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


def _native_forced_off() -> bool:
    return str(os.environ.get("MLX_GLM_DSA_NATIVE_KERNELS", "")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def native_available() -> bool:
    """True only when the compiled ``_ext`` is importable AND not force-disabled."""
    if _native_forced_off() or _glm_fast is None:
        return False
    try:
        return bool(_glm_fast.native_available())
    except Exception:  # pragma: no cover - defensive
        return False


def import_error() -> Optional[Exception]:
    return _IMPORT_ERROR


def status() -> dict:
    """Human-readable status for logging / diagnostics."""
    return {
        "forced_off": _native_forced_off(),
        "package_imported": _omlx_sparse_mla is not None,
        "native_ext_available": native_available(),
        "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
    }


def fused_indexer_scores(
    q_scaled: mx.array,
    k: mx.array,
    weights_hlast: mx.array,
) -> Optional[mx.array]:
    """Native fused ``sum_h relu(q_h @ k.T) * weight_h`` -> ``[B, 1, L, K]``.

    Maps our lightning-indexer math onto oMLX's Steel indexer kernel. The kernel
    applies NO internal scale, so the caller must pass ``q`` already multiplied
    by ``softmax_scale`` and ``weights`` already multiplied by ``n_heads**-0.5``
    (this mirrors the pure-MLX path exactly).

    Args:
        q_scaled: ``[B, H=32, L, D=128]`` queries pre-scaled by softmax_scale.
        k:        ``[B, 1, K, D=128]`` indexer keys (K = cache length).
        weights_hlast: ``[B, L, H=32]`` per-head weights (already n_heads**-0.5).

    Returns ``[B, 1, L, K]`` scores, or ``None`` to fall back to pure MLX.
    """
    if not native_available() or _omlx_sparse_mla is None:
        return None
    try:
        return _omlx_sparse_mla.fused_indexer_scores(
            q_scaled,
            k,
            weights_hlast,
            causal=False,
        )
    except Exception as exc:  # pragma: no cover - kernel contract mismatch
        logger.debug("native fused_indexer_scores fell back to MLX: %r", exc)
        return None


def dsa_topk_indices(scores: mx.array, topk: int) -> Optional[mx.array]:
    """Native DSA top-k over ``[B, 1, L, K]`` scores -> ``[B, 1, L, topk]``.

    Returns ``None`` (fall back to ``mx.argpartition``) when the native op is
    unavailable. The native op is only queried when ``_glm_fast`` reports it as
    a native symbol, so we never accidentally dispatch to a non-existent
    ``mx.fast.dsa_topk_indices`` on stock MLX.
    """
    if not native_available() or _glm_fast is None:
        return None
    try:
        if not _glm_fast.has("dsa_topk_indices"):
            return None
        return _glm_fast.dsa_topk_indices(scores, topk)
    except Exception as exc:  # pragma: no cover - kernel contract mismatch
        logger.debug("native dsa_topk_indices fell back to MLX: %r", exc)
        return None
