# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_attention_adapter.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
Segment = Tuple[int, int]
_SINK_LSE_QCHUNK = int(os.environ.get('MAGI2_SINK_LSE_QCHUNK', '1024'))
_SINK_LSE_BUDGET_MB = int(os.environ.get('MAGI2_SINK_LSE_BUDGET_MB', '1024'))

def varlen_segments(cu_seqlens: Sequence[int]) -> List[Segment]:
    cu = list(cu_seqlens)
    if len(cu) < 2:
        raise ValueError('cu_seqlens must have >= 2 entries')
    if cu[0] != 0:
        raise ValueError(f'cu_seqlens must start at 0, got {cu[0]}')
    segs: List[Segment] = []
    for a, b in zip(cu[:-1], cu[1:]):
        if b < a:
            raise ValueError(f'cu_seqlens not non-decreasing: {a} -> {b}')
        segs.append((a, b))
    return segs

def segment_lengths(cu_seqlens: Sequence[int]) -> List[int]:
    return [b - a for a, b in varlen_segments(cu_seqlens)]

def max_seqlen(cu_seqlens: Sequence[int]) -> int:
    lens = segment_lengths(cu_seqlens)
    return max(lens) if lens else 0

def normalize_ranges(q_ranges: Sequence[Sequence[int]], k_ranges: Sequence[Sequence[int]]) -> List[Tuple[Segment, Segment]]:
    if len(q_ranges) != len(k_ranges):
        raise ValueError(f'q_ranges ({len(q_ranges)}) != k_ranges ({len(k_ranges)})')
    out: List[Tuple[Segment, Segment]] = []
    for (q0, q1), (k0, k1) in zip(q_ranges, k_ranges):
        if q1 < q0 or k1 < k0:
            raise ValueError(f'empty/negative range q=({q0},{q1}) k=({k0},{k1})')
        out.append(((int(q0), int(q1)), (int(k0), int(k1))))
    return out

def ranges_partition_queries(q_ranges: Sequence[Sequence[int]], total_q: int) -> bool:
    spans = sorted(((int(a), int(b)) for a, b in q_ranges))
    cursor = 0
    for a, b in spans:
        if a != cursor:
            return False
        cursor = b
    return cursor == total_q

def is_hopper_major(major: int) -> bool:
    return major >= 9

def sm120_takes_hopper_path(major: int) -> bool:
    return major != 9 and is_hopper_major(major)

def split_q_range_no_overlap(q_ranges: Sequence[Sequence[int]], k_ranges: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[List[List[int]]]]:
    boundaries = sorted({int(a) for a, _ in q_ranges} | {int(b) for _, b in q_ranges})
    candidates = [[boundaries[i], boundaries[i + 1], []] for i in range(len(boundaries) - 1)]
    for (q0, q1), (k0, k1) in zip(q_ranges, k_ranges):
        for cand in candidates:
            if q0 <= cand[0] and cand[1] <= q1:
                cand[2].append([int(k0), int(k1)])
    q_out: List[List[int]] = []
    k_out: List[List[List[int]]] = []
    for cand in candidates:
        if cand[2]:
            q_out.append([cand[0], cand[1]])
            k_out.append(cand[2])
    return (q_out, k_out)

def sink_denominator_scale(lse: float, sink: float) -> float:
    return 1.0 / (1.0 + math.exp(sink - lse))

@dataclass
class Magi2AttentionReport:
    memeff_enabled: bool
    magi_compile_disabled: bool
    cudnn_available: Optional[bool]
    notes: List[str]

def enable_magi2_sdpa(*, disable_magi_compile: bool=True) -> Magi2AttentionReport:
    notes: List[str] = []
    if disable_magi_compile:
        os.environ.setdefault('MAGI2_DISABLE_MAGI_COMPILE', '1')
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
        notes.append('magi_compile disabled via env')
    memeff = False
    cudnn: Optional[bool] = None
    try:
        from cosmos_windows.attention_memeff import enable_memeff_attention
        memeff = enable_memeff_attention(allow_math_fallback=False)
        notes.append('cuDNN-first SDPA priority installed (attention_memeff)')
        try:
            from torch.nn.attention import SDPBackend
            cudnn = getattr(SDPBackend, 'CUDNN_ATTENTION', None) is not None
        except Exception:
            cudnn = None
    except Exception as exc:
        notes.append(f'attention_memeff unavailable: {exc}')
    return Magi2AttentionReport(memeff_enabled=memeff, magi_compile_disabled=disable_magi_compile, cudnn_available=cudnn, notes=notes)

def preview_bidirectional_sdpa(q, k, v, cu_seqlens: Sequence[int], *, sink=None, gate=None, scale: Optional[float]=None):
    import torch
    import torch.nn.functional as F
    compute_dtype = v.dtype
    if q.dtype != compute_dtype:
        q = q.to(compute_dtype)
    if k.dtype != compute_dtype:
        k = k.to(compute_dtype)
    S, H, D = q.shape
    scale = scale if scale is not None else D ** (-0.5)
    segs = varlen_segments(cu_seqlens)
    out = torch.empty_like(q)
    sink_vec = None
    if sink is not None:
        sink_vec = sink.reshape(-1).to(torch.float32)
    for a, b in segs:
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False, scale=scale)
        if sink_vec is not None:
            L = b - a
            lse = torch.empty(1, H, L, device=q.device, dtype=torch.float32)
            ksf = ks.float()
            budget_b = _SINK_LSE_BUDGET_MB * 1024 * 1024
            chunk = max(32, min(_SINK_LSE_QCHUNK, int(budget_b // max(1, H * L * 4))))
            for s in range(0, L, chunk):
                e = min(s + chunk, L)
                logits = torch.matmul(qs[:, :, s:e].float(), ksf.transpose(-1, -2)) * scale
                lse[:, :, s:e] = torch.logsumexp(logits, dim=-1)
                del logits
            factor = torch.sigmoid(lse - sink_vec.view(1, H, 1))
            o = o * factor.unsqueeze(-1).to(o.dtype)
        out[a:b] = o.squeeze(0).transpose(0, 1)
    if gate is not None:
        g = gate.reshape(S, H, 1) if gate.dim() == 2 else gate
        out = out * torch.sigmoid(g).to(out.dtype)
    return out
_SINK_PROBE_DONE = False
_FA2_STATE: dict = {'fn': None, 'checked': False, 'note': None}

def fa2_varlen_fn():
    if _FA2_STATE['checked']:
        return _FA2_STATE['fn']
    _FA2_STATE['checked'] = True
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        _FA2_STATE['fn'] = flash_attn_varlen_func
        _FA2_STATE['note'] = 'FA2 varlen (flash_attn.flash_attn_interface), native sm_120'
    except Exception as exc:
        _FA2_STATE['fn'] = None
        _FA2_STATE['note'] = f'FA2 unavailable ({type(exc).__name__}: {exc}) -> SDPA tier-A'
    return _FA2_STATE['fn']

def preview_varlen_fa2(q, k, v, cu_seqlens, *, sink=None, gate=None, scale=None):
    import torch
    fn = fa2_varlen_fn()
    if fn is None:
        raise RuntimeError(f"FA2 varlen requested but not importable: {_FA2_STATE['note']}")
    compute_dtype = v.dtype
    if q.dtype != compute_dtype:
        q = q.to(compute_dtype)
    if k.dtype != compute_dtype:
        k = k.to(compute_dtype)
    S, H, D = q.shape
    scale = scale if scale is not None else D ** (-0.5)
    lens = segment_lengths(cu_seqlens)
    max_seqlen = max(lens) if lens else 0
    cu = torch.tensor(list(cu_seqlens), device=q.device, dtype=torch.int32)
    need_lse = sink is not None
    res = fn(q, k, v, cu, cu, max_seqlen, max_seqlen, dropout_p=0.0, softmax_scale=scale, causal=False, return_attn_probs=need_lse)
    if isinstance(res, tuple):
        out, lse = (res[0], res[1] if len(res) > 1 else None)
    else:
        out, lse = (res, None)
    if need_lse:
        if lse is None:
            raise RuntimeError('FA2 did not return softmax_lse but an attention sink is present (the sink correction REQUIRES the LSE); refusing to drop the sink.')
        sink_vec = sink.reshape(-1).to(torch.float32)
        factor = torch.sigmoid(lse.float() - sink_vec.view(-1, 1))
        out = out * factor.transpose(0, 1).unsqueeze(-1).to(out.dtype)
    if gate is not None:
        g = gate.reshape(S, H, 1) if gate.dim() == 2 else gate
        out = out * torch.sigmoid(g).to(out.dtype)
    return out

def refiner_local_sdpa(q, k, v, q_ranges: Sequence[Sequence[int]], k_ranges: Sequence[Sequence[int]], *, gate=None, scale: Optional[float]=None):
    import torch
    import torch.nn.functional as F
    S, Hq, D = q.shape
    Hkv = k.shape[1]
    scale = scale if scale is not None else D ** (-0.5)
    pairs = normalize_ranges(q_ranges, k_ranges)
    if not ranges_partition_queries(q_ranges, S):
        raise NotImplementedError('refiner q_ranges overlap (auto_range_merge); tier-A needs the LSE-merge path (port magi2_refiner._custom_flex_flash_attn_func). Non-overlapping tiling only.')
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = torch.zeros_like(q)
    for (q0, q1), (k0, k1) in pairs:
        if q1 == q0 or k1 == k0:
            continue
        qs = q[q0:q1].transpose(0, 1).unsqueeze(0)
        ks = k[k0:k1].transpose(0, 1).unsqueeze(0)
        vs = v[k0:k1].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False, scale=scale)
        out[q0:q1] = o.squeeze(0).transpose(0, 1)
    if gate is not None:
        g = gate.reshape(S, Hq, 1) if gate.dim() == 2 else gate
        out = out * torch.sigmoid(g).to(out.dtype)
    return out

def refiner_flex_sdpa(query, key, value, q_ranges, k_ranges, *, scale: Optional[float]=None):
    import torch
    import torch.nn.functional as F
    S, H, D = query.shape
    scale = scale if scale is not None else D ** (-0.5)
    q_out_ranges, k_out_list = split_q_range_no_overlap([list(map(int, r)) for r in q_ranges], [list(map(int, r)) for r in k_ranges])
    output = torch.zeros_like(query)
    output_lse = torch.zeros((S, H), dtype=torch.float32, device=query.device)
    for (q0, q1), kwins in zip(q_out_ranges, k_out_list):
        acc_out = None
        acc_lse = None
        qs = query[q0:q1].transpose(0, 1).unsqueeze(0)
        for k0, k1 in kwins:
            if k1 <= k0:
                continue
            ks = key[k0:k1].transpose(0, 1).unsqueeze(0)
            vs = value[k0:k1].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False, scale=scale)
            o = o.squeeze(0).transpose(0, 1)
            logits = torch.matmul(qs.float(), ks.float().transpose(-1, -2)) * scale
            lse = torch.logsumexp(logits, dim=-1).squeeze(0).transpose(0, 1)
            if acc_out is None:
                acc_out, acc_lse = (o, lse)
            else:
                max_lse = torch.max(acc_lse, lse)
                a_se, c_se = (torch.exp(acc_lse - max_lse), torch.exp(lse - max_lse))
                sum_se = a_se + c_se
                a_sc, c_sc = ((a_se / sum_se).unsqueeze(-1), (c_se / sum_se).unsqueeze(-1))
                acc_out = acc_out * a_sc.to(acc_out.dtype) + o * c_sc.to(o.dtype)
                acc_lse = torch.log(sum_se) + max_lse
        if acc_out is not None:
            output[q0:q1] = acc_out
            output_lse[q0:q1] = acc_lse
    return (output, output_lse)

def _make_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__['__shim__'] = True
    return m

def _identity_decorator(*a, **k):
    if len(a) == 1 and (not k) and callable(a[0]):
        return a[0]

    def _wrap(fn):
        return fn
    return _wrap

def install_dependency_shims(force: bool=False) -> List[str]:
    import importlib.util
    shimmed: List[str] = []

    def _ensure(name: str, build: Callable[[], types.ModuleType]) -> None:
        if not force:
            try:
                if importlib.util.find_spec(name) is not None:
                    return
            except (ImportError, ValueError):
                pass
        if name in sys.modules and (not force):
            return
        sys.modules[name] = build()
        shimmed.append(name)

    def _magi_compiler():
        m = _make_module('magi_compiler')
        m.magi_compile = _identity_decorator
        m.magi_register_custom_op = _identity_decorator
        return m

    def _magi_compiler_api():
        m = _make_module('magi_compiler.api')
        m.magi_register_custom_op = _identity_decorator
        m.magi_compile = _identity_decorator
        return m

    def _magi_compiler_config():
        m = _make_module('magi_compiler.config')

        class CompileConfig:

            def __init__(self, *a, **k):
                self.__dict__.update(k)
        m.CompileConfig = CompileConfig
        return m

    def _flash_attn_interface():
        m = _make_module('flash_attn_interface')

        class _FA3:

            @staticmethod
            def fwd(*a, **k):
                raise RuntimeError('flash_attn_3_cuda.fwd invoked on SM120 — the SDPA intercept was not installed before the model ran (this must never happen; fail loud).')
        m.flash_attn_3_cuda = _FA3()
        m.flash_attn_func = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('flash_attn_interface.flash_attn_func called on SM120 (no FA3)'))
        return m

    def _magi_attn_extensions():
        return _make_module('magi_attn_extensions')

    def _fa3_iface_with_sink():
        m = _make_module('magi_attn_extensions.fa3_interface_with_sink')

        class FA3VarlenFuncWithSink:

            @staticmethod
            def correct_out_lse_with_sink(out, lse, sink, sink_layout=None):
                return (out, lse)
        m.FA3VarlenFuncWithSink = FA3VarlenFuncWithSink
        m.fa3_func_with_sink = lambda *a, **k: None
        m.fa3_varlen_func_with_sink = lambda *a, **k: None
        return m
    _ensure('magi_compiler', _magi_compiler)
    _ensure('magi_compiler.api', _magi_compiler_api)
    _ensure('magi_compiler.config', _magi_compiler_config)
    existing = sys.modules.get('flash_attn_interface')
    if existing is None and (not force):
        try:
            if importlib.util.find_spec('flash_attn_interface') is not None:
                import flash_attn_interface as existing
        except (ImportError, ValueError):
            existing = None
    if force or existing is None:
        _ensure('flash_attn_interface', _flash_attn_interface)
    elif not hasattr(existing, 'flash_attn_3_cuda'):
        existing.flash_attn_3_cuda = _flash_attn_interface().flash_attn_3_cuda
        shimmed.append('flash_attn_interface.flash_attn_3_cuda (injected)')
    _ensure('magi_attn_extensions', _magi_attn_extensions)
    _ensure('magi_attn_extensions.fa3_interface_with_sink', _fa3_iface_with_sink)
    return shimmed

@dataclass
class Magi2InterceptReport:
    shimmed_modules: List[str] = field(default_factory=list)
    patched_is_hopper: List[str] = field(default_factory=list)
    patched_attention: List[str] = field(default_factory=list)
    device_major: Optional[int] = None
    would_take_hopper_path: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {'shimmed_modules': list(self.shimmed_modules), 'patched_is_hopper': list(self.patched_is_hopper), 'patched_attention': list(self.patched_attention), 'device_major': self.device_major, 'would_take_hopper_path': self.would_take_hopper_path, 'notes': list(self.notes)}

def install_magi2_attention_intercept(*, install_shims: bool=True, force_shims: bool=False) -> Magi2InterceptReport:
    report = Magi2InterceptReport()
    if install_shims:
        report.shimmed_modules = install_dependency_shims(force=force_shims)
    try:
        import torch
        major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
    except Exception as exc:
        major = None
        report.notes.append(f'torch/device probe failed: {exc}')
    report.device_major = major
    report.would_take_hopper_path = sm120_takes_hopper_path(major) if major is not None else None

    def _force_false():
        return False
    for modname in ('inference.common.arch', 'inference.model.magi2_preview', 'inference.model.magi2_refiner'):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, 'is_hopper_arch'):
            setattr(mod, 'is_hopper_arch', _force_false)
            report.patched_is_hopper.append(modname)
    prev = sys.modules.get('inference.model.magi2_preview')
    if prev is not None and hasattr(prev, 'flash_attn_with_sink'):
        setattr(prev, 'flash_attn_with_sink', _preview_flash_attn_with_sink_sdpa)
        report.patched_attention.append('magi2_preview.flash_attn_with_sink')
    refn = sys.modules.get('inference.model.magi2_refiner')
    if refn is not None:
        if hasattr(refn, 'flex_flash_attn_func'):
            setattr(refn, 'flex_flash_attn_func', _refiner_flex_flash_attn_func_sdpa)
            report.patched_attention.append('magi2_refiner.flex_flash_attn_func')
        if hasattr(refn, 'flash_attn_func'):
            setattr(refn, 'flash_attn_func', _refiner_flash_attn_func_sdpa)
            report.patched_attention.append('magi2_refiner.flash_attn_func')
    if not report.patched_attention:
        report.notes.append('reference model modules not imported yet — call install_magi2_attention_intercept() AFTER importing inference.model.magi2_preview/refiner so the attention swap takes effect (shims + is_hopper patch still applied).')
    return report

def _preview_flash_attn_with_sink_sdpa(q, k, v, varlen_handler, cp_split_sizes, attn_softcap, attn_sinks):
    cu = getattr(varlen_handler, 'cu_seqlens_q', None)
    if cu is None:
        cu = getattr(varlen_handler, 'cu_seqlens', None)
    q3 = q.squeeze(0) if q.dim() == 4 else q
    k3 = k.squeeze(0) if k.dim() == 4 else k
    v3 = v.squeeze(0) if v.dim() == 4 else v
    cu_list = cu.tolist() if hasattr(cu, 'tolist') else list(cu) if cu is not None else [0, q3.shape[0]]
    sink = attn_sinks if attn_sinks is not None and getattr(attn_sinks, 'numel', lambda: 1)() else None
    global _SINK_PROBE_DONE
    if not _SINK_PROBE_DONE:
        _SINK_PROBE_DONE = True
        print(f'[MAGI2-ATTN-PROBE] q={tuple(q3.shape)} k={tuple(k3.shape)} sink={(None if sink is None else tuple(sink.shape))} softcap={attn_softcap} n_segments={max(0, len(cu_list) - 1)} first_segs={cu_list[:5]}', flush=True)
    if attn_softcap is not None and float(attn_softcap) > 0:
        raise RuntimeError(f'MAGI-2 attention requested softcap={attn_softcap}, but the SM120 SDPA/FA2 tiers do not implement logit soft-capping — output would be silently wrong. Implement it or run a backend that supports it.')
    import os as _os
    tier = _os.environ.get('MAGI2_ATTN_TIER', 'auto').strip().lower()
    if tier in ('auto', 'fa2') and fa2_varlen_fn() is not None:
        return preview_varlen_fa2(q3, k3, v3, cu_list, sink=sink)
    if tier == 'fa2':
        raise RuntimeError(f"MAGI2_ATTN_TIER=fa2 but FA2 is unavailable: {_FA2_STATE['note']}")
    return preview_bidirectional_sdpa(q3, k3, v3, cu_list, sink=sink)

def _refiner_flex_flash_attn_func_sdpa(query, key, value, q_ranges, k_ranges, attn_type_map=None, max_seqlen_q=None, auto_range_merge=False, sparse_load=False):
    qr = q_ranges.tolist() if hasattr(q_ranges, 'tolist') else list(q_ranges)
    kr = k_ranges.tolist() if hasattr(k_ranges, 'tolist') else list(k_ranges)
    return refiner_flex_sdpa(query, key, value, qr, kr)

def _refiner_flash_attn_func_sdpa(query, key, value):
    import torch.nn.functional as F
    q = query.transpose(1, 2) if query.dim() == 4 else query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(1, 2) if key.dim() == 4 else key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(1, 2) if value.dim() == 4 else value.transpose(0, 1).unsqueeze(0)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return o.transpose(1, 2) if query.dim() == 4 else o.squeeze(0).transpose(0, 1)
