# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/native_quant.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import ctypes
import os
from pathlib import Path
from typing import Final
import numpy as np
QUANT_GEOMETRY: Final[dict[str, tuple[int, int]]] = {'F32': (1, 4), 'F16': (1, 2), 'BF16': (1, 2), 'Q4_0': (32, 18), 'Q8_0': (32, 34), 'Q3_K': (256, 110), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210)}
_DLL = None
_DLL_DIR_HANDLE = None

def _candidate_dlls() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    explicit = os.environ.get('GGML_QUANT_DLL', '').strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([root / 'prefill_llama.cpp' / 'build' / 'bin' / 'ggml-base.dll', root / 'prefill_llama.cpp' / 'build4' / 'bin' / 'ggml-base.dll', root / 'prefill_llama.cpp' / 'build3' / 'bin' / 'ggml-base.dll'])
    return candidates

def _load_dll():
    global _DLL, _DLL_DIR_HANDLE
    if _DLL is not None:
        return _DLL
    errors = []
    for path in _candidate_dlls():
        if not path.is_file():
            continue
        try:
            if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
                _DLL_DIR_HANDLE = os.add_dll_directory(str(path.parent))
            _DLL = ctypes.CDLL(str(path))
            return _DLL
        except OSError as exc:
            errors.append(f'{path}: {exc}')
    detail = '; '.join(errors) if errors else 'no candidate DLL exists'
    raise RuntimeError(f'llama.cpp ggml-base codec library is unavailable; build prefill_llama.cpp or set GGML_QUANT_DLL ({detail})')

def _functions(quant: str):
    if quant not in QUANT_GEOMETRY:
        raise NotImplementedError(f'native block codec does not support {quant!r}')
    suffix = {'Q4_0': 'q4_0', 'Q8_0': 'q8_0', 'Q3_K': 'q3_K', 'Q4_K': 'q4_K', 'Q5_K': 'q5_K', 'Q6_K': 'q6_K'}[quant]
    dll = _load_dll()
    qfn = getattr(dll, f'quantize_row_{suffix}_ref')
    dfn = getattr(dll, f'dequantize_row_{suffix}')
    qfn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
    qfn.restype = None
    dfn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    dfn.restype = None
    return (qfn, dfn)

def quantize_native(values: np.ndarray, quant: str) -> np.ndarray:
    block_values, block_bytes = QUANT_GEOMETRY[quant]
    src = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if not src.size or src.size % block_values:
        raise ValueError(f'{quant} value count must be a non-zero multiple of {block_values}')
    if quant == 'F32':
        return src.view(np.uint8).copy()
    if quant == 'F16':
        return src.astype(np.float16).view(np.uint8)
    if quant == 'BF16':
        bits = src.view(np.uint32)
        rounded = bits + np.uint32(32767) + (bits >> 16 & 1)
        return (rounded >> 16).astype(np.uint16).view(np.uint8)
    dst = np.empty(src.size // block_values * block_bytes, dtype=np.uint8)
    qfn, _ = _functions(quant)
    qfn(src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p(dst.ctypes.data), src.size)
    return dst

def dequantize_native(packed: bytes | bytearray | memoryview | np.ndarray, quant: str) -> np.ndarray:
    block_values, block_bytes = QUANT_GEOMETRY[quant]
    if isinstance(packed, np.ndarray):
        src = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    else:
        src = np.frombuffer(packed, dtype=np.uint8)
    if not src.size or src.size % block_bytes:
        raise ValueError(f'{quant} byte count must be a non-zero multiple of {block_bytes}')
    if quant == 'F32':
        return src.view(np.float32).copy()
    if quant == 'F16':
        return src.view(np.float16).astype(np.float32)
    if quant == 'BF16':
        return (src.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    dst = np.empty(src.size // block_bytes * block_values, dtype=np.float32)
    _, dfn = _functions(quant)
    dfn(ctypes.c_void_p(src.ctypes.data), dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), dst.size)
    return dst
