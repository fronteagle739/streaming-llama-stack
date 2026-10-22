# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_compute_precision.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Optional
COMPUTE_DTYPES = ('bf16', 'fp8', 'fp4')

def _build_fp8_kernel():
    import torch
    import triton
    import triton.language as tl
    from inference.flash_mh_moe.triton.mh_moe_fwd import _binary_search_expert_id, _select_block_config, _swiglu7_fwd

    @triton.jit
    def _kernel(x_ptr, wg_ptr, wu_ptr, wd_ptr, y_ptr, gather_ids_ptr, probs_ptr, expert_offsets_ptr, cu_expert_tile_counts_ptr, stride_x_s, stride_x_h, stride_x_dh, stride_wg_ne, stride_wg_dh, stride_wg_de, stride_wu_ne, stride_wu_dh, stride_wu_de, stride_wd_ne, stride_wd_de, stride_wd_dh, stride_y_s, stride_y_h, stride_y_dh, D_HEAD: tl.constexpr, D_EXPERT: tl.constexpr, NUM_HEADS: tl.constexpr, NUM_EXPERTS: tl.constexpr, LOG2_NUM_EXPERTS: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_DH: tl.constexpr, BLOCK_DE: tl.constexpr, ACC_DTYPE: tl.constexpr=tl.float32):
        tile_id = tl.program_id(0)
        total_tiles = tl.load(cu_expert_tile_counts_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return
        expert_id = _binary_search_expert_id(cu_expert_tile_counts_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS)
        expert_id_i64 = expert_id.to(tl.int64)
        head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)
        tile_in_expert = tile_id - tl.load(cu_expert_tile_counts_ptr + expert_id)
        token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start
        block_dh_offs = tl.arange(0, BLOCK_DH)
        block_de_offs = tl.arange(0, BLOCK_DE)
        block_t_offs = tl.arange(0, BLOCK_T)
        dh_offs = tl.arange(0, D_HEAD)
        block_token_offs = token_start + block_t_offs
        block_token_mask = block_t_offs < n
        gather_ids = tl.load(gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0)
        probs = tl.load(probs_ptr + block_token_offs, mask=block_token_mask, other=0.0)
        x_offs = gather_ids * stride_x_s + head_id * stride_x_h
        y_acc = tl.zeros([BLOCK_T, D_HEAD], dtype=ACC_DTYPE)
        for dexpert_start in tl.range(0, D_EXPERT, BLOCK_DE):
            dexpert_block_offs = dexpert_start + block_de_offs
            gate_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=ACC_DTYPE)
            up_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=ACC_DTYPE)
            for dhead_start in tl.static_range(0, D_HEAD, BLOCK_DH):
                dhead_block_offs = dhead_start + block_dh_offs
                x_block = tl.load(x_ptr + x_offs[:, None] + dhead_block_offs[None, :] * stride_x_dh, mask=block_token_mask[:, None], other=0.0)
                w_gate = tl.load(wg_ptr + expert_id_i64 * stride_wg_ne + dhead_block_offs[:, None] * stride_wg_dh + dexpert_block_offs[None, :] * stride_wg_de)
                w_up = tl.load(wu_ptr + expert_id_i64 * stride_wu_ne + dhead_block_offs[:, None] * stride_wu_dh + dexpert_block_offs[None, :] * stride_wu_de)
                xb = x_block.to(tl.float8e4nv)
                gate_acc += tl.dot(xb, w_gate.to(tl.float8e4nv))
                up_acc += tl.dot(xb, w_up.to(tl.float8e4nv))
            hidden = _swiglu7_fwd(gate_acc, up_acc, tl.bfloat16)
            w_down = tl.load(wd_ptr + expert_id_i64 * stride_wd_ne + dexpert_block_offs[:, None] * stride_wd_de + dh_offs[None, :] * stride_wd_dh)
            y_acc += tl.dot(hidden.to(tl.float8e4nv), w_down.to(tl.float8e4nv))
        y_acc = y_acc * probs[:, None]
        y_offs = gather_ids * stride_y_s + head_id * stride_y_h
        y_ptrs = y_ptr + y_offs[:, None] + dh_offs[None, :] * stride_y_dh
        tl.atomic_add(y_ptrs, y_acc.to(y_ptr.dtype.element_ty), mask=block_token_mask[:, None])
    import math

    def func(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down):
        T = gather_ids.size(0)
        if T == 0:
            return torch.zeros_like(x)
        num_heads = x.size(1)
        d_head, d_expert = (x.size(2), W_down.size(1))
        bt, bdh, bde, ns, nw = _select_block_config()
        y = torch.zeros_like(x)
        nfe = expert_offsets.size(0) - 1
        nfe_log2 = max(1, math.ceil(math.log2(max(nfe, 1) + 1)))
        etc = expert_offsets.diff()
        tile_counts = (etc + bt - 1) // bt
        cu_tiles = torch.cat([torch.zeros(1, dtype=torch.int32, device=expert_offsets.device), torch.cumsum(tile_counts, dim=0, dtype=torch.int32)])
        grid = (T + bt - 1) // bt + nfe
        _kernel[grid,](x, W_gate, W_up, W_down, y, gather_ids, probs, expert_offsets, cu_tiles, x.stride(0), x.stride(1), x.stride(2), W_gate.stride(0), W_gate.stride(1), W_gate.stride(2), W_up.stride(0), W_up.stride(1), W_up.stride(2), W_down.stride(0), W_down.stride(1), W_down.stride(2), y.stride(0), y.stride(1), y.stride(2), D_HEAD=d_head, D_EXPERT=d_expert, NUM_HEADS=num_heads, NUM_EXPERTS=nfe, LOG2_NUM_EXPERTS=nfe_log2, BLOCK_T=bt, BLOCK_DH=bdh, BLOCK_DE=bde, ACC_DTYPE=tl.float32, num_stages=ns, num_warps=nw)
        return y
    return (_kernel, func)
_STATE = {'installed': None, 'orig': None, 'func': None}

def install_moe_compute_dtype(compute_dtype: str, emit=None) -> dict:
    report = {'compute_dtype': compute_dtype, 'installed': None, 'note': None}
    cd = (compute_dtype or 'bf16').lower()
    if cd not in COMPUTE_DTYPES:
        raise ValueError(f'compute_dtype must be one of {COMPUTE_DTYPES}; got {compute_dtype!r}')
    import inference.model.magi2_preview as prev
    if _STATE['orig'] is None:
        _STATE['orig'] = prev._mh_moe_fwd
    if cd == 'bf16':
        prev._mh_moe_fwd = _STATE['orig']
        _STATE['installed'] = 'bf16'
        report['installed'] = 'bf16'
        report['note'] = 'lossless bf16 (reference kernel)'
        return report
    if cd == 'fp4':
        report['note'] = 'fp4 has no portable Triton tl.dot on this build (cutlass #3096 / compute_120f grouped-GEMM path required) -> FALLING BACK to fp8 (LOUD).'
        if emit is not None:
            emit.emit('stage_log', stage='compute_precision', level='warn', message=report['note'])
        else:
            print(f"[MAGI2-COMPUTE] WARNING: {report['note']}", flush=True)
    if _STATE['func'] is None:
        _STATE['func'] = _build_fp8_kernel()[1]
    _fp8_func = _STATE['func']

    def _flash_mh_moe_fwd_fp8(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down, deterministic: bool=False):
        return _fp8_func(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down)
    prev._mh_moe_fwd = _flash_mh_moe_fwd_fp8
    _STATE['installed'] = 'fp8'
    report['installed'] = 'fp8'
    report['note'] = report['note'] or 'fp8 cast-on-tile expert GEMM (weights bf16 in VRAM, fp32 accumulate)'
    if emit is not None:
        emit.emit('stage_log', stage='compute_precision', message=report['note'], compute_dtype=cd)
    return report
