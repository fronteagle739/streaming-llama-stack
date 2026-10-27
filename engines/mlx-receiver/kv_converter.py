from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple
import mlx.core as mx
import numpy as np
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30
_Q8_0_BLOCK_ELEMENTS = 32
_Q8_0_BLOCK_BYTES = 2 + _Q8_0_BLOCK_ELEMENTS
GGML_TYPE_TO_NP = {GGML_TYPE_F32: np.float32, GGML_TYPE_F16: np.float16}
GGML_TYPE_ELEMENT_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2}

def _dequant_q8_0_to_f16(raw: bytes, n_rows: int, row_elements: int) -> np.ndarray:
    if row_elements % _Q8_0_BLOCK_ELEMENTS != 0:
        raise ValueError(f'Q8_0 row_elements {row_elements} not a multiple of {_Q8_0_BLOCK_ELEMENTS}')
    blocks_per_row = row_elements // _Q8_0_BLOCK_ELEMENTS
    row_bytes = blocks_per_row * _Q8_0_BLOCK_BYTES
    expected = n_rows * row_bytes
    if len(raw) != expected:
        raise ValueError(f'Q8_0 byte count mismatch: got {len(raw)} expected {expected} (rows={n_rows} blocks_per_row={blocks_per_row})')
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, blocks_per_row, _Q8_0_BLOCK_BYTES)
    scale_bytes = np.ascontiguousarray(blocks[:, :, :2])
    scales = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16).reshape(n_rows, blocks_per_row)
    quant_bytes = np.ascontiguousarray(blocks[:, :, 2:])
    quants = np.frombuffer(quant_bytes.tobytes(), dtype=np.int8).reshape(n_rows, blocks_per_row, _Q8_0_BLOCK_ELEMENTS)
    deq = quants.astype(np.float32) * scales.astype(np.float32)[:, :, None]
    return deq.reshape(n_rows, row_elements).astype(np.float16)

def _read_k_rows(raw: bytes, k_type: int, n_rows: int, row_elements: int) -> np.ndarray:
    if k_type == GGML_TYPE_Q8_0:
        return _dequant_q8_0_to_f16(raw, n_rows, row_elements)
    if k_type == GGML_TYPE_BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(n_rows, row_elements)
        return mx.array(u16, dtype=mx.bfloat16).astype(mx.float16).numpy()
    np_dtype = _ggml_type_to_numpy(k_type)
    return np.frombuffer(raw, dtype=np_dtype).reshape(n_rows, row_elements).astype(np.float16, copy=False)

@dataclass
class KVArtifactHeader:
    magic: bytes
    format_major: int
    format_minor: int
    n_layers: int
    n_ctx: int
    token_count: int
    type_k: int
    type_v: int
    flags: int
    payload_bytes: int
    payload_crc32: int
    last_token_id: int = -1
    v_trans: int = 0
    is_mla: int = 0
    model_fingerprint: int = 0

@dataclass
class KVLayerData:
    keys: mx.array
    values: mx.array

def parse_artifact_header(data: bytes) -> Tuple[KVArtifactHeader, int]:
    if len(data) < 44:
        raise ValueError(f'Data too short for KVARTIF1 header: {len(data)} bytes')
    magic = data[:8]
    if magic != b'KVARTIF1':
        raise ValueError(f"Bad magic: {magic!r}, expected b'KVARTIF1'")
    fmt_major, fmt_minor, n_layers, n_ctx, token_count, type_k, type_v, flags = struct.unpack_from('<HH III HH I', data, 8)
    payload_bytes, payload_crc32 = struct.unpack_from('<QI', data, 32)
    hdr = KVArtifactHeader(magic=magic, format_major=fmt_major, format_minor=fmt_minor, n_layers=n_layers, n_ctx=n_ctx, token_count=token_count, type_k=type_k, type_v=type_v, flags=flags, payload_bytes=payload_bytes, payload_crc32=payload_crc32)
    offset = 44
    if fmt_minor >= 1:
        hdr.last_token_id, hdr.v_trans, hdr.is_mla, _reserved, hdr.model_fingerprint = struct.unpack_from('<i BB H Q', data, offset)
        offset += 16
    if fmt_minor >= 3:
        offset += 84
    if fmt_minor >= 4:
        offset += 4
    if fmt_minor >= 5:
        offset += 12
    if fmt_minor >= 6:
        offset += 8
    if fmt_minor >= 7:
        offset += 4
    if fmt_minor >= 8:
        offset += 16
    if fmt_minor >= 9:
        offset += 8
    if fmt_minor >= 10:
        offset += 8
    return (hdr, offset)

def parse_kv_payload(payload: bytes, n_kv_heads: int, head_dim: int, n_kv_heads_v: Optional[int]=None, head_dim_v: Optional[int]=None) -> Tuple[List[KVLayerData], dict]:
    if n_kv_heads_v is None:
        n_kv_heads_v = n_kv_heads
    if head_dim_v is None:
        head_dim_v = head_dim
    off = 0

    def read(fmt):
        nonlocal off
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, payload, off)
        off += size
        return vals if len(vals) > 1 else vals[0]
    n_stream = read('<I')
    if n_stream != 1:
        raise ValueError(f'Expected n_stream=1, got {n_stream} (multi-stream not supported yet)')
    cell_count = read('<I')
    if cell_count == 0:
        raise ValueError('Empty KV state (cell_count=0)')
    for _cell_idx in range(cell_count):
        _pos = read('<i')
        n_seq_id = read('<I')
        off += n_seq_id * 4
    v_trans = read('<I')
    n_layer = read('<I')
    layers: List[KVLayerData] = []
    row_elements_dense = n_kv_heads * head_dim
    k_arrays = []
    for _il in range(n_layer):
        k_type_i = read('<i')
        k_size_row = read('<Q')
        k_nbytes = cell_count * k_size_row
        k_raw = payload[off:off + k_nbytes]
        off += k_nbytes
        k_np = _read_k_rows(k_raw, k_type_i, cell_count, row_elements_dense)
        k_arrays.append(k_np)
    v_arrays = []
    for _il in range(n_layer):
        if v_trans == 0:
            v_type_i = read('<i')
            v_size_row = read('<Q')
            v_nbytes = cell_count * v_size_row
            v_raw = payload[off:off + v_nbytes]
            off += v_nbytes
            np_dtype = _ggml_type_to_numpy(v_type_i)
            v_np = np.frombuffer(v_raw, dtype=np_dtype).reshape(cell_count, -1)
        else:
            v_type_i = read('<i')
            v_size_el = read('<I')
            n_embd_v_gqa = read('<I')
            v_nbytes = n_embd_v_gqa * cell_count * v_size_el
            v_raw = payload[off:off + v_nbytes]
            off += v_nbytes
            np_dtype = _ggml_type_to_numpy(v_type_i)
            v_np = np.frombuffer(v_raw, dtype=np_dtype).reshape(n_embd_v_gqa, cell_count)
            v_np = v_np.T.copy()
        v_arrays.append(v_np)
    for il in range(n_layer):
        k_np = k_arrays[il]
        v_np = v_arrays[il]
        k_reshaped = k_np.reshape(1, cell_count, n_kv_heads, head_dim)
        k_reshaped = k_reshaped.transpose(0, 2, 1, 3)
        v_reshaped = v_np.reshape(1, cell_count, n_kv_heads_v, head_dim_v)
        v_reshaped = v_reshaped.transpose(0, 2, 1, 3)
        k_mx = mx.array(k_reshaped.astype(np.float16))
        v_mx = mx.array(v_reshaped.astype(np.float16))
        layers.append(KVLayerData(keys=k_mx, values=v_mx))
    metadata = {'v_trans': v_trans, 'n_layer': n_layer, 'cell_count': cell_count, 'n_stream': n_stream}
    return (layers, metadata)

def parse_artifact_file(path: str, n_kv_heads: int, head_dim: int, n_kv_heads_v: Optional[int]=None, head_dim_v: Optional[int]=None) -> Tuple[KVArtifactHeader, List[KVLayerData], dict]:
    with open(path, 'rb') as f:
        data = f.read()
    header, payload_offset = parse_artifact_header(data)
    payload = data[payload_offset:payload_offset + header.payload_bytes]
    layers, meta = parse_kv_payload(payload, n_kv_heads, head_dim, n_kv_heads_v, head_dim_v)
    meta['header'] = header
    return (header, layers, meta)

def inject_kv_into_cache(cache_list: list, layers: List[KVLayerData]) -> None:
    if len(cache_list) != len(layers):
        raise ValueError(f'Cache has {len(cache_list)} layers but payload has {len(layers)}')
    for i, (cache, layer_data) in enumerate(zip(cache_list, layers)):
        cache.state = (layer_data.keys, layer_data.values)

def parse_kv_payload_mla(payload: bytes, kv_lora_rank: int, qk_rope_head_dim: int) -> Tuple[List[Tuple['mx.array', 'mx.array']], dict]:
    import struct as _struct
    off = 0

    def read(fmt):
        nonlocal off
        size = _struct.calcsize(fmt)
        vals = _struct.unpack_from(fmt, payload, off)
        off += size
        return vals if len(vals) > 1 else vals[0]
    n_stream = read('<I')
    if n_stream != 1:
        raise ValueError(f'Expected n_stream=1, got {n_stream}')
    cell_count = read('<I')
    if cell_count == 0:
        raise ValueError('Empty KV state (cell_count=0)')
    for _ in range(cell_count):
        read('<i')
        n_seq_id = read('<I')
        off += n_seq_id * 4
    v_trans = read('<I')
    n_layer_word = read('<I')
    has_k_idx = n_layer_word & 2147483648 != 0
    n_layer = n_layer_word & 2147483647
    if has_k_idx:
        read('<I')
    expected_k_row = (kv_lora_rank + qk_rope_head_dim) * 2
    layers_out = []
    row_elements_mla = kv_lora_rank + qk_rope_head_dim
    import os as _os
    k_pe_undo_scale = float(_os.environ.get('LLAMA_MLX_K_PE_SCALE', '1.0'))
    mv = payload if isinstance(payload, memoryview) else memoryview(payload)
    for _il in range(n_layer):
        k_type = read('<i')
        k_size_row = read('<Q')
        k_nbytes = cell_count * k_size_row
        k_np = _read_k_rows(mv[off:off + k_nbytes], k_type, cell_count, row_elements_mla)
        off += k_nbytes
        k_all = mx.array(k_np.reshape(1, 1, cell_count, row_elements_mla))
        del k_np
        kv_latent_mx = k_all[..., :kv_lora_rank].astype(mx.bfloat16)
        k_pe_mx = k_all[..., kv_lora_rank:]
        if k_pe_undo_scale != 1.0:
            k_pe_mx = k_pe_mx.astype(mx.float32) * k_pe_undo_scale
        k_pe_mx = k_pe_mx.astype(mx.bfloat16)
        mx.eval(kv_latent_mx, k_pe_mx)
        del k_all
        layers_out.append((kv_latent_mx, k_pe_mx))
    idx_k_arrays = []
    idx_head_size = 0
    if has_k_idx:
        for _il in range(n_layer):
            present = read('<I')
            if not present:
                idx_k_arrays.append(None)
                continue
            idx_type = read('<i')
            idx_size_row = read('<Q')
            idx_nbytes = cell_count * idx_size_row
            idx_head_size = int(idx_size_row // 2)
            idx_np = _read_k_rows(mv[off:off + idx_nbytes], idx_type, cell_count, idx_head_size)
            off += idx_nbytes
            idx_mx = mx.array(idx_np.reshape(1, 1, cell_count, idx_head_size)).astype(mx.bfloat16)
            mx.eval(idx_mx)
            del idx_np
            idx_k_arrays.append(idx_mx)
    if off < len(payload):
        for _il in range(n_layer):
            if off >= len(payload):
                break
            if v_trans == 0:
                v_type = read('<i')
                v_size_row = read('<Q')
                off += cell_count * v_size_row
            else:
                v_type = read('<i')
                v_size_el = read('<I')
                n_embd_v_gqa = read('<I')
                off += n_embd_v_gqa * cell_count * v_size_el
    idx_k_mx = idx_k_arrays if has_k_idx else None
    meta = {'v_trans': v_trans, 'n_layer': n_layer, 'cell_count': cell_count, 'has_k_idx': has_k_idx, 'idx_head_size': idx_head_size, 'idx_k_mx': idx_k_mx}
    return (layers_out, meta)

def inject_kv_mla(cache_list: list, layers_mla: List[Tuple]) -> None:
    if len(cache_list) != len(layers_mla):
        raise ValueError(f'Cache has {len(cache_list)} layers but payload has {len(layers_mla)}')
    for cache, (kv_latent, k_pe) in zip(cache_list, layers_mla):
        target = cache.caches[0] if hasattr(cache, 'caches') else cache
        target.state = (kv_latent, k_pe)

def inject_kv_idx(cache_list: list, idx_k_mx: list, n_cells: int) -> None:
    if idx_k_mx is None:
        return
    for cache, idx_k in zip(cache_list, idx_k_mx):
        if idx_k is None or not hasattr(cache, 'caches') or len(cache.caches) < 2:
            continue
        idx_trim = idx_k[:, :, :n_cells, :]
        cache.caches[1].state = (idx_trim, mx.zeros([1, 1, n_cells, 0], dtype=idx_trim.dtype))

def append_kv_mla(cache_list: list, layers_mla: List[Tuple], idx_k_mx, n_new: int) -> None:
    if len(cache_list) != len(layers_mla):
        raise ValueError(f'Cache has {len(cache_list)} layers but payload has {len(layers_mla)}')
    for cache, (kv_latent, k_pe) in zip(cache_list, layers_mla):
        target = cache.caches[0] if hasattr(cache, 'caches') else cache
        res_k, res_v = target.state
        if n_new <= 0:
            continue
        new_k = kv_latent[:, :, :n_new, :].astype(res_k.dtype)
        new_v = k_pe[:, :, :n_new, :].astype(res_v.dtype)
        target.state = (mx.concatenate([res_k, new_k], axis=-2), mx.concatenate([res_v, new_v], axis=-2))
    if idx_k_mx is None or n_new <= 0:
        return
    for cache, idx_k in zip(cache_list, idx_k_mx):
        if idx_k is None or not hasattr(cache, 'caches') or len(cache.caches) < 2:
            continue
        c1 = cache.caches[1]
        res_ik, _res_iv = c1.state
        new_ik = idx_k[:, :, :n_new, :].astype(res_ik.dtype)
        all_ik = mx.concatenate([res_ik, new_ik], axis=-2)
        c1.state = (all_ik, mx.zeros([1, 1, all_ik.shape[2], 0], dtype=all_ik.dtype))

def _ggml_type_to_numpy(ggml_type: int) -> np.dtype:
    if ggml_type == GGML_TYPE_F32:
        return np.float32
    elif ggml_type == GGML_TYPE_F16:
        return np.float16
    elif ggml_type == GGML_TYPE_BF16:
        return np.uint16
    else:
        raise ValueError(f'Unsupported ggml_type for KV cache: {ggml_type}')
