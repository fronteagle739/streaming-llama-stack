"""Observable, strict native-kernel admission for V4.1.

Both model classes import NATIVE_STATUS: one warning per module load if native
acceleration is absent/disabled. Call native_status() for a fresh diagnostic and
require_native(required_symbols=(... ,)) before opting into native kernels.
For native_ops functions returning None on failure, use call_native(...): merely
loading the extension does NOT prove its nanobind ABI accepts MLX arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
from importlib import metadata
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeStatus:
    loaded: bool
    available: bool
    forced_off: bool
    version: str | None
    binary_sha256: str | None
    mlx_version: str | None
    symbols: tuple[str, ...]
    error: str | None


@lru_cache(maxsize=8)
def _binary_sha256(filename):
    # The current vendored extension exports no version. Identify the actual
    # binary rather than presenting the Python package version as its version.
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def native_status() -> NativeStatus:
    """Report extension presence separately from environment-controlled dispatch.

    version='unreported' means the loaded binary exports no __version__; its
    SHA256 is the build identity. No compilation or kernel invocation is done.
    """
    loaded = available = forced_off = False
    version = binary_sha256 = error = None
    symbols = ()
    try:
        mlx_version = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        mlx_version = None
    try:
        ops = importlib.import_module("mlx_vlm.models.glm_moe_dsa.native_ops")
        detail = ops.status()
        forced_off = bool(detail["forced_off"])
        available = bool(ops.native_available())
        fast = importlib.import_module("mlx_vlm.custom_kernels.glm_moe_dsa.fast")
        loaded = bool(fast.is_native_available())
        symbols = tuple(fast.native_symbols())
        cause = fast.import_error()
        error = repr(cause) if cause is not None else detail.get("import_error")
        if loaded:
            ext = importlib.import_module("mlx_vlm.custom_kernels.glm_moe_dsa._ext")
            version = str(getattr(ext, "__version__", "unreported"))
            filename = getattr(ext, "__file__", None)
            if filename:
                binary_sha256 = _binary_sha256(filename)
    except Exception as exc:
        available = False
        error = repr(exc)
    return NativeStatus(loaded, available, forced_off, version, binary_sha256,
                        mlx_version, symbols, error)


def require_native(*, required_symbols=(), expected_version=None) -> NativeStatus:
    """Fail closed, including under python -O; recheck force-disable each call."""
    state = native_status()
    missing = sorted(set(required_symbols) - set(state.symbols))
    if not state.loaded or not state.available or missing:
        raise RuntimeError(f"DeepSeek-V4.1 native kernels required: {state}; missing={missing}")
    if expected_version is not None and state.version != expected_version:
        raise RuntimeError(f"DeepSeek-V4.1 native version mismatch: expected "
                           f"{expected_version!r}, got {state.version!r}")
    return state


def call_native(operation, *args, required_symbols=(), **kwargs):
    """Call a None-on-failure native_ops entry point without silent MLX fallback.

    Callers must pass the extension symbols the operation needs. Evaluate the
    returned MLX array before stopping a benchmark timer; this wrapper is not a
    synchronization point. Do not pass helpers that fall back internally to MLX.
    """
    if not required_symbols:
        raise ValueError("call_native requires explicit native extension symbols")
    require_native(required_symbols=required_symbols)
    name = getattr(operation, "__name__", type(operation).__name__)
    try:
        result = operation(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"DeepSeek-V4.1 native call {name} failed (ABI/contract): {exc}") from exc
    if result is None:
        raise RuntimeError(f"DeepSeek-V4.1 native call {name} returned None "
                           "(ABI/contract failure); silent fallback refused")
    return result


NATIVE_STATUS = native_status()
if not NATIVE_STATUS.available:
    logger.warning("DeepSeek-V4.1 NATIVE KERNELS UNAVAILABLE: %s. "
                   "Native optimisation measurements must use require_native/call_native.",
                   NATIVE_STATUS)
else:
    logger.info("DeepSeek-V4.1 native kernels: %s", NATIVE_STATUS)
