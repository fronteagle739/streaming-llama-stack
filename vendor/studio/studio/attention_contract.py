# Derived from Studio backend/studio/attention_contract.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""Runtime contract for production attention backends.

Some model families can run with plain Torch SDPA, xFormers, or compatibility
stubs, but those paths are not acceptable proof for Streaming Studio's native
streaming implementations. This module answers one narrow question: is a real
FlashAttention backend available for CUDA production runs?
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from functools import lru_cache


FLASH_ATTENTION_REQUIRED_PREFIXES: tuple[str, ...] = (
    "cosmos3",
    "wan",
    "bernini",
    "scail",
    "domainshuttle",
    "domain-shuttle",
    "krea2-native",
    "boogu",
    "anima",
    "z-image",
    "zimage",
    "ltx",
    "joyai",
    "glm",
)


def arch_requires_flash_attention(arch: str | None) -> bool:
    a = (arch or "").strip().lower()
    return bool(a) and any(a.startswith(prefix) for prefix in FLASH_ATTENTION_REQUIRED_PREFIXES)


def _callable_attr(module_name: str, *attrs: str) -> tuple[str, str] | None:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    for attr in attrs:
        if callable(getattr(mod, attr, None)):
            return module_name, attr
    return None


def _probe_external_flash_attention(module_name: str, attr: str) -> tuple[bool, str]:
    """Run a real FA CUDA call in a subprocess.

    Some FA3 wheels import successfully on RTX 5090/SM120 but abort the process
    with "no kernel image is available" at first launch. A subprocess keeps the
    daemon alive while still proving the backend is executable.
    """
    code = f"""
import importlib
import torch
mod = importlib.import_module({module_name!r})
fn = getattr(mod, {attr!r})
torch.cuda.init()
q = torch.randn((32, 4, 64), device='cuda', dtype=torch.float16)
k = torch.randn((32, 4, 64), device='cuda', dtype=torch.float16)
v = torch.randn((32, 4, 64), device='cuda', dtype=torch.float16)
cu = torch.tensor([0, 32], device='cuda', dtype=torch.int32)
if {attr!r}.endswith('varlen_func'):
    out = fn(q, k, v, cu, cu, 32, 32)
else:
    out = fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
if isinstance(out, tuple):
    out = out[0]
torch.cuda.synchronize()
print(tuple(out.shape), out.dtype)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:
        return False, f"{module_name}.{attr} subprocess probe failed: {type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, f"{module_name}.{attr}: {proc.stdout.strip()}"
    detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " | ")
    return False, f"{module_name}.{attr} probe exited {proc.returncode}: {detail[:700]}"


def _probe_torch_flash_attention() -> tuple[bool, str]:
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        return False, f"torch probe unavailable: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        device = torch.device("cuda")
        q = torch.randn((1, 2, 8, 64), device=device, dtype=torch.float16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize(device)
        return True, "torch.nn.functional.scaled_dot_product_attention FLASH_ATTENTION"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if "caught" in locals() and caught:
            messages = [str(w.message) for w in caught if "flash" in str(w.message).lower()]
            if messages:
                detail = f"{detail}; " + " | ".join(messages[:3])
        return False, detail


@lru_cache(maxsize=1)
def flash_attention_status() -> dict:
    """Return a cached, lightweight status dict for real FlashAttention availability."""
    fa3 = _callable_attr("flash_attn_interface", "flash_attn_func", "flash_attn_varlen_func")
    if fa3:
        ok, detail = _probe_external_flash_attention(*fa3)
        if ok:
            return {"available": True, "backend": "flash_attn_interface", "detail": detail}
        fa3_detail = detail
    else:
        fa3_detail = "flash_attn_interface is not importable"
    fa2 = _callable_attr("flash_attn", "flash_attn_func", "flash_attn_varlen_func")
    if fa2:
        ok, detail = _probe_external_flash_attention(*fa2)
        if ok:
            return {"available": True, "backend": "flash_attn", "detail": detail}
        fa2_detail = detail
    else:
        fa2_detail = "flash_attn is not importable"
    ok, detail = _probe_torch_flash_attention()
    if ok:
        return {"available": True, "backend": "torch_flash_attention", "detail": detail}
    return {
        "available": False,
        "backend": "none",
        "detail": (
            "No executable Dao flash_attn/flash_attn_interface backend is available "
            f"({fa3_detail}; {fa2_detail}) and the Torch fused FLASH_ATTENTION probe failed: {detail}"
        ),
    }


def attention_note(arch: str | None) -> str:
    if arch_requires_flash_attention(arch):
        return "attention: FlashAttention required"
    return ""


_TRUTHY = {"1", "true", "yes", "on"}

# Opt-IN strict mode: HARD-FAIL a required-FA arch when no real FlashAttention kernel is present. This is
# for validated production-proof runs. The DEFAULT (unset) is a clean SDPA fallback with a warning, because
# on SM120/Blackwell (RTX 5090) NO FlashAttention kernel exists (the FA3 wheel is Hopper-only and torch's
# fused FLASH is not compiled for SM120) — hard-failing there would make every Wan-family model unrunnable.
STRICT_FLASH_ENV = "STUDIO_REQUIRE_FLASH_ATTENTION"
# Legacy: STUDIO_ALLOW_ATTENTION_FALLBACK=1 forces the fallback. It is now the default behavior, so this
# only matters to explicitly override a STRICT setting back to fallback.
ALLOW_FALLBACK_ENV = "STUDIO_ALLOW_ATTENTION_FALLBACK"


def strict_flash_required() -> bool:
    """True only when the operator opted into a strict production-proof run (STUDIO_REQUIRE_FLASH_ATTENTION=1)
    AND did not also set STUDIO_ALLOW_ATTENTION_FALLBACK. Default is the clean SDPA fallback."""
    if os.environ.get(ALLOW_FALLBACK_ENV, "").strip().lower() in _TRUTHY:
        return False
    return os.environ.get(STRICT_FLASH_ENV, "").strip().lower() in _TRUTHY


def require_flash_attention_for_arch(arch: str | None) -> dict:
    """Resolve the attention backend for a required-FA arch, preferring real FlashAttention but falling
    back CLEANLY instead of crashing.

    - real FlashAttention available -> use it.
    - no real FA + strict opt-in (STUDIO_REQUIRE_FLASH_ATTENTION=1) -> raise (production-proof gate).
    - no real FA + default -> WARN and return a fallback status; the backend then runs on PyTorch SDPA
      (cuDNN/memory-efficient/math), which is correct (just slower) and IS reachable on SM120. This is a
      slow-but-correct run, NOT a crash; it does not count as a FlashAttention production proof.
    """
    if not arch_requires_flash_attention(arch):
        return {"available": True, "backend": "not-required", "detail": "arch does not require FlashAttention"}
    st = flash_attention_status()
    if st["available"]:
        return st
    if strict_flash_required():
        raise RuntimeError(
            f"'{arch or 'model'}' requires real FlashAttention for a strict production run "
            f"({STRICT_FLASH_ENV}=1). {st['detail']}. Install a FlashAttention backend with SM120/Blackwell "
            f"kernels, or unset {STRICT_FLASH_ENV} to run on the correct-but-slower SDPA fallback."
        )
    print(
        f"[attention] {arch or 'model'}: no real FlashAttention kernel for this GPU; using the backend's "
        "PyTorch SDPA fallback (cuDNN/memory-efficient/math) — slower but correct. This run does NOT count "
        f"as a FlashAttention production proof. {st['detail']}",
        flush=True,
    )
    return {**st, "fallback": True}
