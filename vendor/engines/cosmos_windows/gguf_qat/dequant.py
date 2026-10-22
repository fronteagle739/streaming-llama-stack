# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/dequant.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Tuple
import torch
QK_K = 256

def dequantize_q4_0(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1
    n_blocks = raw.numel() // 18
    blocks = raw.view(n_blocks, 18)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    qs = blocks[:, 2:]
    lo = (qs & 15).to(torch.int16)
    hi = (qs >> 4).to(torch.int16)
    q = torch.cat([lo, hi], dim=1)
    q = q - 8
    return (d * q.to(torch.float32)).reshape(-1)

def dequantize_q8_0(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1
    n_blocks = raw.numel() // 34
    blocks = raw.view(n_blocks, 34)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    x = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    return (x * d).reshape(-1)

def dequantize_q4_k(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1 and (raw.numel() % 144 == 0)
    n_blocks = raw.numel() // 144
    blocks = raw.view(n_blocks, 144)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
    packed_scales = blocks[:, 4:16].reshape(n_blocks, 3, 4)
    qs = blocks[:, 16:]
    ds, ms, high = (packed_scales[:, 0], packed_scales[:, 1], packed_scales[:, 2])
    scales = torch.cat([ds & 63, high & 15 | ds >> 2 & 48], dim=1)
    mins = torch.cat([ms & 63, high >> 4 | ms >> 2 & 48], dim=1)
    scales = (d * scales.to(torch.float32)).reshape(n_blocks, 8, 1)
    mins = (dmin * mins.to(torch.float32)).reshape(n_blocks, 8, 1)
    shifts = torch.tensor([0, 4], dtype=torch.uint8, device=raw.device).reshape(1, 1, 2, 1)
    q = qs.reshape(n_blocks, -1, 1, 32) >> shifts & 15
    q = q.reshape(n_blocks, 8, 32).to(torch.float32)
    return (scales * q - mins).reshape(-1)

def dequantize_q5_k(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1 and (raw.numel() % 176 == 0)
    n_blocks = raw.numel() // 176
    blocks = raw.view(n_blocks, 176)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
    packed_scales = blocks[:, 4:16].reshape(n_blocks, 3, 4)
    qh, qs = (blocks[:, 16:48], blocks[:, 48:])
    ds, ms, high = (packed_scales[:, 0], packed_scales[:, 1], packed_scales[:, 2])
    scales = torch.cat([ds & 63, high & 15 | ds >> 2 & 48], dim=1)
    mins = torch.cat([ms & 63, high >> 4 | ms >> 2 & 48], dim=1)
    scales = (d * scales.to(torch.float32)).reshape(n_blocks, 8, 1)
    mins = (dmin * mins.to(torch.float32)).reshape(n_blocks, 8, 1)
    low_shifts = torch.tensor([0, 4], dtype=torch.uint8, device=raw.device).reshape(1, 1, 2, 1)
    high_shifts = torch.arange(8, dtype=torch.uint8, device=raw.device).reshape(1, 1, 8, 1)
    low = (qs.reshape(n_blocks, -1, 1, 32) >> low_shifts & 15).reshape(n_blocks, 8, 32)
    high_bits = (qh.reshape(n_blocks, -1, 1, 32) >> high_shifts & 1).reshape(n_blocks, 8, 32)
    q = (low | high_bits << 4).to(torch.float32)
    return (scales * q - mins).reshape(-1)

def dequantize_q6_k(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1
    n_blocks = raw.numel() // 210
    blocks = raw.view(n_blocks, 210)
    ql = blocks[:, :128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(torch.int8).to(torch.float32)
    d = blocks[:, 208:210].contiguous().view(torch.float16).to(torch.float32)
    dscale = (d * scales).reshape(n_blocks, QK_K // 16, 1)
    ql = ql.reshape(n_blocks, -1, 1, 64)
    shifts2 = torch.tensor([0, 4], dtype=torch.uint8, device=raw.device).reshape(1, 1, 2, 1)
    ql = ql >> shifts2 & 15
    ql = ql.reshape(n_blocks, -1, 32)
    qh = qh.reshape(n_blocks, -1, 1, 32)
    shifts4 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=raw.device).reshape(1, 1, 4, 1)
    qh = qh >> shifts4 & 3
    qh = qh.reshape(n_blocks, -1, 32)
    q = (ql | qh << 4).to(torch.int8).to(torch.int16) - 32
    q = q.reshape(n_blocks, QK_K // 16, -1).to(torch.float32)
    return (dscale * q).reshape(n_blocks, QK_K).reshape(-1)

def dequantize_q3_k(raw: torch.Tensor) -> torch.Tensor:
    assert raw.dtype == torch.uint8 and raw.ndim == 1
    n_blocks = raw.numel() // 110
    blocks = raw.view(n_blocks, 110)
    hmask = blocks[:, :32]
    qs = blocks[:, 32:96]
    scales_b = blocks[:, 96:108]
    d = blocks[:, 108:110].contiguous().view(torch.float16).to(torch.float32)
    lscales = scales_b[:, :8]
    hscales = scales_b[:, 8:12]
    sh2 = torch.tensor([0, 4], dtype=torch.uint8, device=raw.device).reshape(1, 2, 1)
    lscales = (lscales.reshape(n_blocks, 1, 8) >> sh2).reshape(n_blocks, 16)
    sh4 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=raw.device).reshape(1, 4, 1)
    hscales = (hscales.reshape(n_blocks, 1, 4) >> sh4).reshape(n_blocks, 16)
    scales = lscales & 15 | (hscales & 3) << 4
    scales = (scales.to(torch.int8).to(torch.int16) - 32).to(torch.float32)
    dl = (d * scales).reshape(n_blocks, 16, 1)
    qshift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=raw.device).reshape(1, 1, 4, 1)
    ql = qs.reshape(n_blocks, -1, 1, 32) >> qshift
    ql = ql.reshape(n_blocks, 16, QK_K // 16) & 3
    hshift = torch.arange(8, dtype=torch.uint8, device=raw.device).reshape(1, 1, 8, 1)
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> hshift
    qh = qh.reshape(n_blocks, 16, QK_K // 16) & 1 ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8)).to(torch.float32)
    return (dl * q).reshape(n_blocks, QK_K).reshape(-1)
_DEQUANT = {'Q4_0': dequantize_q4_0, 'Q4_K': dequantize_q4_k, 'Q5_K': dequantize_q5_k, 'Q8_0': dequantize_q8_0, 'Q6_K': dequantize_q6_k, 'Q3_K': dequantize_q3_k}

def dequantize(raw: torch.Tensor, quant: str, shape) -> torch.Tensor:
    if quant in ('F32',):
        out = raw.view(torch.float32)
    elif quant in ('F16',):
        out = raw.view(torch.float16).to(torch.float32)
    elif quant in ('BF16',):
        out = raw.view(torch.bfloat16).to(torch.float32)
    elif quant in _DEQUANT:
        out = _DEQUANT[quant](raw)
    else:
        raise NotImplementedError(f'dequantize: quant {quant!r} not implemented')
    return out.reshape(tuple((int(s) for s in shape)))

def supported_quants() -> Tuple[str, ...]:
    return tuple(_DEQUANT.keys()) + ('F32', 'F16', 'BF16')
if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader, _ensure_gguf_py
    start_rss_guard(limit_gb=6.0, label='dequant_selftest')
    _ensure_gguf_py()
    from gguf import quants as gq
    from gguf.constants import GGMLQuantizationType
    cases = [('', 'blk.0.attn_q.weight', 'Q4_0'), ('', 'blk.5.ffn_down.weight', 'Q4_0'), ('', 'token_embd.weight', 'Q8_0'), ('', 'token_embd.weight', 'Q6_K'), ('', 'blk.0.attn_q.weight', 'Q4_0'), ('', 'blk.1.ffn_gate_exps.weight', 'Q3_K'), ('', 'blk.1.attn_q_b.weight', 'Q8_0')]
    qtype_map = {'Q4_0': GGMLQuantizationType.Q4_0, 'Q4_K': GGMLQuantizationType.Q4_K, 'Q5_K': GGMLQuantizationType.Q5_K, 'Q8_0': GGMLQuantizationType.Q8_0, 'Q6_K': GGMLQuantizationType.Q6_K, 'Q3_K': GGMLQuantizationType.Q3_K}
    type_size = {'Q4_0': 18, 'Q4_K': 144, 'Q5_K': 176, 'Q8_0': 34, 'Q6_K': 210, 'Q3_K': 110}
    block_size = {'Q4_0': 32, 'Q4_K': 256, 'Q5_K': 256, 'Q8_0': 32, 'Q6_K': 256, 'Q3_K': 256}
    MAX_BLOCKS = 8192
    all_ok = True
    for path, name, quant in cases:
        r = GGUFTensorReader(path, device='cuda')
        info = r.info(name)
        assert info.quant == quant, (name, info.quant, quant)
        ts, bs = (type_size[quant], block_size[quant])
        nb_full = info.n_bytes // ts
        nb = min(nb_full, MAX_BLOCKS)
        nbytes = nb * ts
        host_raw = np.array(r._window(info)[:nbytes], dtype=np.uint8)
        gpu_raw = torch.from_numpy(host_raw.copy()).to('cuda')
        gpu_out = dequantize(gpu_raw, quant, (nb * bs,)).reshape(-1).detach().cpu().numpy()
        ref = gq.dequantize(host_raw, qtype_map[quant]).reshape(-1).astype(np.float32)
        n = min(gpu_out.size, ref.size)
        maxdiff = float(np.max(np.abs(gpu_out[:n] - ref[:n]))) if n else float('nan')
        exact = bool(np.array_equal(gpu_out[:n], ref[:n]))
        ok = gpu_out.size == ref.size and exact
        all_ok = all_ok and ok
        print(f"  [{quant}] {name[:28]:28s} blocks={nb:>6d}/{nb_full:<9d} n={ref.size:>9d} max|gpu-ref|={maxdiff:.3e} bit-exact={exact} -> {('OK' if ok else 'FAIL')}")
        r.close()
        del gpu_raw, gpu_out, host_raw, ref
        torch.cuda.empty_cache()
    print(f'peak RSS {peak_rss_gb():.3f} GB')
    print('DEQUANT BIT-EXACT:', 'ALL PASS' if all_ok else 'FAILURES PRESENT')
    sys.exit(0 if all_ok else 1)
