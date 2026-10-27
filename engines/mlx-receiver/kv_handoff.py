from __future__ import annotations
import os
import socket
import struct
import sys
import time
from typing import Any, Dict, List, Optional

def _sampling_profile_from_kva_header(data: bytes, off: int) -> Dict[str, Any]:
    values = struct.unpack_from('<Qiiiii' + 'f' * 14, data, off)
    keys = ('user_sampling_config', 'seed', 'min_keep', 'top_k', 'penalty_last_n', 'mirostat', 'top_p', 'min_p', 'typ_p', 'temp', 'dynatemp_range', 'dynatemp_exponent', 'penalty_repeat', 'penalty_freq', 'penalty_present', 'xtc_probability', 'xtc_threshold', 'top_n_sigma', 'mirostat_tau', 'mirostat_eta')
    if not any(values):
        return {}
    return dict(zip(keys, values))

def _parse_kva_prompt_payload(path: str) -> dict:
    with open(path, 'rb') as f:
        data = f.read()
    return _parse_kva_prompt_payload_bytes(data)

def _parse_kva_prompt_payload_bytes(data: bytes) -> dict:
    if len(data) < 44:
        raise ValueError(f'kva too short: {len(data)} bytes')
    if data[:8] != b'KVARTIF1':
        raise ValueError(f'bad magic: {data[:8]!r}')
    off = 8
    format_major, format_minor, n_layers, n_ctx, token_count, type_k, type_v, flags = struct.unpack_from('<HH III HH I', data, off)
    off += 2 + 2 + 4 + 4 + 4 + 2 + 2 + 4
    payload_bytes, payload_crc32 = struct.unpack_from('<QI', data, off)
    off += 8 + 4
    last_token_id = -1
    v_trans = 0
    is_mla = 0
    model_fingerprint = 0
    requested_n_predict = -1
    sampling_profile = None
    if format_minor >= 1:
        last_token_id, v_trans, is_mla, _reserved, model_fingerprint = struct.unpack_from('<i BB H Q', data, off)
        off += 4 + 1 + 1 + 2 + 8
    if format_minor >= 3:
        sampling_profile = _sampling_profile_from_kva_header(data, off)
        off += 84
    if format_minor >= 4:
        requested_n_predict, = struct.unpack_from('<i', data, off)
        off += 4
    prompt_token_count = 0
    prompt_text_bytes = 0
    streamed_prefix_token_count = 0
    if format_minor >= 5:
        prompt_token_count, prompt_text_bytes, streamed_prefix_token_count = struct.unpack_from('<III', data, off)
        off += 12
    if format_minor >= 6:
        off += 8
    source_slot_id = -1
    if format_minor >= 7:
        source_slot_id, = struct.unpack_from('<i', data, off)
        off += 4
    img_cell_count = 0
    img_n_embd = 0
    img_pos_start = 0
    img_raw_bytes = 0
    if format_minor >= 8:
        img_cell_count, img_n_embd, img_pos_start, img_raw_bytes = struct.unpack_from('<IIII', data, off)
        off += 16
    mtp_hidden_n_tokens = 0
    mtp_hidden_n_embd = 0
    if format_minor >= 9:
        mtp_hidden_n_tokens, mtp_hidden_n_embd = struct.unpack_from('<II', data, off)
        off += 8
    base_prefix_cells = 0
    if format_minor >= 10:
        base_prefix_cells, _reserved_delta = struct.unpack_from('<II', data, off)
        off += 8
    has_image_embd = bool(flags & 1 << 4)
    has_mtp_hidden = bool(flags & 1 << 6)
    has_raw_image = bool(flags & 1 << 5)
    if not has_raw_image:
        img_raw_bytes = 0
    payload_offset = off
    payload_end = payload_offset + payload_bytes
    if payload_end > len(data):
        raise ValueError(f'kva payload truncated: header says {payload_bytes} bytes from {payload_offset}, file is {len(data)}')
    img_embd_count = img_cell_count * img_n_embd
    img_embd_bytes = img_embd_count * 4
    mtp_hidden_count = mtp_hidden_n_tokens * mtp_hidden_n_embd
    mtp_hidden_bytes = mtp_hidden_count * 2
    aux_tail_bytes = img_embd_bytes + img_raw_bytes + mtp_hidden_bytes
    prompt_text_off = payload_offset
    if format_minor >= 5:
        prompt_text_off = payload_end - aux_tail_bytes - prompt_token_count * 4 - prompt_text_bytes
    prompt_token_off = prompt_text_off + prompt_text_bytes
    img_embd_off = prompt_token_off + prompt_token_count * 4
    raw_image_off = img_embd_off + img_embd_bytes
    mtp_hidden_off = raw_image_off + img_raw_bytes
    kv_state_bytes = memoryview(data)[payload_offset:prompt_text_off]
    prompt_text = ''
    token_ids: List[int] = []
    if format_minor >= 5:
        if prompt_text_bytes > 0:
            prompt_text = bytes(data[prompt_text_off:prompt_text_off + prompt_text_bytes]).decode('utf-8', errors='replace')
        if prompt_token_count > 0:
            token_ids = list(struct.unpack(f'<{prompt_token_count}i', data[prompt_token_off:prompt_token_off + prompt_token_count * 4]))
    image_embd = None
    if format_minor >= 8 and img_embd_count > 0:
        import numpy as _np
        image_embd = _np.frombuffer(data[img_embd_off:img_embd_off + img_embd_bytes], dtype=_np.float32, count=img_embd_count).reshape(img_cell_count, img_n_embd).copy()
    raw_image = None
    if format_minor >= 8 and img_raw_bytes > 0:
        raw_image = bytes(data[raw_image_off:raw_image_off + img_raw_bytes])
    mtp_hidden = None
    if format_minor >= 9 and mtp_hidden_count > 0:
        import numpy as _np
        mtp_hidden = _np.frombuffer(data[mtp_hidden_off:mtp_hidden_off + mtp_hidden_bytes], dtype=_np.float16, count=mtp_hidden_count).reshape(mtp_hidden_n_tokens, mtp_hidden_n_embd).copy()
        sys.stderr.write(f'[mlx-server] parsed MTP hidden: {mtp_hidden_n_tokens}x{mtp_hidden_n_embd} fp16 (KV-inject MTP warmup available)\n')
        sys.stderr.flush()
    return {'format_minor': format_minor, 'token_ids': token_ids, 'prompt_text': prompt_text, 'last_token_id': last_token_id, 'token_count': token_count, 'prompt_token_count': prompt_token_count, 'streamed_prefix_token_count': streamed_prefix_token_count, 'base_prefix_cells': base_prefix_cells, 'requested_n_predict': requested_n_predict, 'sampling_profile': sampling_profile, 'model_fingerprint': model_fingerprint, 'n_layers': n_layers, 'n_ctx': n_ctx, 'kv_state_bytes': kv_state_bytes, 'is_mla': is_mla, 'v_trans': v_trans, 'source_slot_id': source_slot_id, 'has_image_embd': has_image_embd, 'img_cell_count': img_cell_count, 'img_n_embd': img_n_embd, 'img_pos_start': img_pos_start, 'image_embd': image_embd, 'has_raw_image': has_raw_image, 'raw_image': raw_image, 'has_mtp_hidden': has_mtp_hidden, 'mtp_hidden_n_tokens': mtp_hidden_n_tokens, 'mtp_hidden_n_embd': mtp_hidden_n_embd, 'mtp_hidden': mtp_hidden}

def _describe_kva_ref(kva_ref) -> str:
    if isinstance(kva_ref, (bytes, bytearray, memoryview)):
        return f'<memory:{len(kva_ref)} bytes>'
    return str(kva_ref)

def _release_kv_views(*views) -> str:
    n_released = 0
    n_bytes = 0
    for v in views:
        if not isinstance(v, memoryview):
            continue
        try:
            n_bytes = max(n_bytes, len(v))
            v.release()
            n_released += 1
        except ValueError:
            pass
        except BufferError as exc:
            sys.stderr.write(f'[mlx-server] artifact view still exported, not released: {exc}\n')
            sys.stderr.flush()
    return f'{n_released} view(s), {n_bytes} bytes' if n_released else ''

def _keep_warm_enabled() -> bool:
    return os.environ.get('LLAMA_MLX_KEEP_WARM', '1') not in ('', '0')

def _keep_warm_period_s() -> float:
    try:
        return max(0.05, float(os.environ.get('LLAMA_MLX_KEEP_WARM_S', '1.0') or 1.0))
    except ValueError:
        return 1.0

def _start_keep_warm_thread() -> None:
    if not _keep_warm_enabled():
        return
    import threading
    period = _keep_warm_period_s()

    def run() -> None:
        cold_run = 0
        while True:
            time.sleep(period)
            slow_before = _keep_warm_state['slow']
            t0 = time.time()
            _keep_warm_tick()
            dt = time.time() - t0
            if _keep_warm_state['slow'] > slow_before and dt >= 1.0:
                cold_run += 1
                if cold_run >= 3:
                    sys.stderr.write(f'[mlx-server] keep-warm cannot hold the wiring at a {period:g}s period (3 cold ticks of {dt:.1f}s); thread stopped [KEEP_WARM]\n')
                    sys.stderr.flush()
                    return
            else:
                cold_run = 0
    threading.Thread(target=run, name='keep-warm', daemon=True).start()
    sys.stderr.write(f'[mlx-server] keep-warm thread started (period {period:g}s) [KEEP_WARM]\n')
    sys.stderr.flush()
_keep_warm_state = {'last': 0.0, 'n': 0, 'slow': 0}

def _keep_warm_tick() -> None:
    if not _keep_warm_enabled():
        return
    st = _keep_warm_state
    idle_s = time.time() - st['last'] if st['last'] else 0.0
    try:
        import mlx.core as mx
        t0 = time.time()
        mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)
        dt = time.time() - t0
    except Exception as exc:
        sys.stderr.write(f'[mlx-server] keep-warm eval failed: {exc}\n')
        sys.stderr.flush()
        return
    st['n'] += 1
    st['last'] = time.time()
    if dt >= 0.25:
        st['slow'] += 1
        sys.stderr.write(f"[mlx-server] keep-warm eval: {dt:.3f}s after {idle_s:.1f}s idle (tick {st['n']}, slow {st['slow']}) [KEEP_WARM]\n")
        sys.stderr.flush()

def _cache_entry_offset(entry, default: int=0) -> int:
    try:
        children = getattr(entry, 'caches', None)
        if children is not None:
            child_offsets = [_cache_entry_offset(child, -1) for child in children]
            if child_offsets and min(child_offsets) >= 0 and (len(set(child_offsets)) == 1):
                return child_offsets[0]
            return int(default)
        if any((cls.__name__ in ('SlidingWindowCache', 'PoolingCache', 'BatchPoolingCache') for cls in type(entry).__mro__)):
            return int(default)
        offset = getattr(entry, 'offset', None)
        if offset is None and type(entry).__name__ == 'SimpleKVCache':
            offset = entry.cache_length
        if offset is not None and getattr(offset, 'ndim', 0) == 0:
            return int(offset)
    except Exception:
        return int(default)
    return int(default)

def _cache_offset(cache) -> int:
    if not cache:
        return 0
    offsets = [_cache_entry_offset(entry, -1) for entry in cache]
    return offsets[0] if min(offsets) >= 0 and len(set(offsets)) == 1 else -1

def _cache_leaf_offsets(cache) -> List[int]:
    offsets: List[int] = []

    def _visit(entry) -> None:
        children = getattr(entry, 'caches', None)
        if children is not None and children:
            for child in children:
                _visit(child)
            return
        offset = _cache_entry_offset(entry, -1)
        offsets.append(offset)
    for entry in cache or []:
        _visit(entry)
    return offsets

def _can_trim_cache_to_offset(cache, offset: int) -> bool:
    target = int(offset)
    current = _cache_offset(cache)
    if target < 0 or current < target:
        return False
    if current == target:
        return True
    try:

        def _safe(entry):
            children = getattr(entry, 'caches', None)
            if children is not None:
                return bool(children) and all((_safe(child) for child in children))
            floor = getattr(entry, 'trim_floor', None)
            if callable(floor) and target < int(floor()):
                return False
            window = getattr(entry, 'max_size', None)
            if window is not None and current >= int(window):
                start = getattr(entry, 'start_position', None)
                return start is not None and (not getattr(entry, 'keep', 0)) and (int(start) <= max(0, target - int(window) + 1))
            return True
        from mlx_lm.models.cache import can_trim_prompt_cache
        return all((_safe(entry) for entry in cache)) and can_trim_prompt_cache(cache)
    except Exception:
        return False

def _trim_cache_to_offset(cache, offset: int) -> bool:
    target = int(offset)
    if not _can_trim_cache_to_offset(cache, target):
        return False
    current = _cache_offset(cache)
    try:
        if current > target:
            from mlx_lm.models.cache import trim_prompt_cache
            trim_prompt_cache(cache, current - target)

        def _slice_concat(entry):
            for child in getattr(entry, 'caches', ()):
                _slice_concat(child)
            if type(entry).__name__ == 'ConcatenateKVCache':
                state = entry.state
                if target or all((x is not None for x in state)):
                    entry.state = tuple((x[..., :target, :] for x in state))
        for entry in cache:
            _slice_concat(entry)
        return _cache_offset(cache) == target
    except Exception:
        return False

def _validate_kv_injection_cache(cache, offset: int) -> None:
    from mlx_lm.models.cache import KVCache, ConcatenateKVCache

    def _check(entry):
        children = getattr(entry, 'caches', None)
        if children is not None:
            if not children:
                raise RuntimeError('empty KV cache container')
            for child in children:
                _check(child)
            return
        if type(entry) not in (KVCache, ConcatenateKVCache):
            raise RuntimeError(f'KV injection unsupported for {type(entry).__name__}')
        if _cache_entry_offset(entry, -1) != offset:
            raise RuntimeError(f'KV cache offset mismatch: expected={offset}')
        if offset > 0:
            if any((x is None or x.shape[-2] != offset for x in entry.state)):
                raise RuntimeError(f'KV state length does not match offset={offset}')
    if not cache:
        raise RuntimeError('missing KV cache')
    for entry in cache:
        _check(entry)
TBP_MAGIC = 1413632049
TBP_HEADER_SIZE = 52
TBP_MSG_HELLO = 1
TBP_MSG_SESSION_START = 3
TBP_MSG_KV_SEG_BEGIN = 7
TBP_MSG_KV_CHUNK = 8
TBP_MSG_KV_SEG_END = 9
TBP_MSG_KV_DONE = 12
TBP_MSG_QUERY_PREFIX_CACHE = 30
TBP_MSG_PREFIX_CACHE_REPLY = 31

def _tbp1_send_frame(sock: socket.socket, msg_type: int, payload: bytes=b'') -> None:
    import struct as _struct
    import zlib as _zlib
    hdr = bytearray(_struct.pack('<IHHHHII', TBP_MAGIC, 1, 0, msg_type, 0, TBP_HEADER_SIZE, len(payload)))
    hdr += _struct.pack('<QQQ', 0, 0, 0)
    hdr += _struct.pack('<I', 0)
    hdr += _struct.pack('<I', _zlib.crc32(payload) & 4294967295 if payload else 0)
    _struct.pack_into('<I', hdr, 44, _zlib.crc32(bytes(hdr)) & 4294967295)
    sock.sendall(bytes(hdr) + payload)

def _kva_transport_log(sock, stage, event, error=None, got=0, *, peer=None):
    if peer is None:
        try:
            peer = sock.getpeername()
        except OSError:
            pass
    sys.stderr.write(f"[kva-recv] stage={stage} event={event} peer={peer!r} bytes_read={got} errno={getattr(error, 'errno', None)} winerror={getattr(error, 'winerror', None)} exception={(type(error).__name__ if error else '-')}\n")
    sys.stderr.flush()

def _recv_into(sock: socket.socket, view: memoryview, timeout_s: float=30.0, *, stage: str='payload', probe: bool=False) -> bool:
    got = 0
    size = len(view)
    try:
        peer = sock.getpeername()
    except OSError:
        peer = None
    try:
        sock.settimeout(timeout_s)
        while got < size:
            n = sock.recv_into(view[got:], size - got)
            if not n:
                _kva_transport_log(sock, stage, 'probe/eof' if probe and got == 0 else 'eof', got=got, peer=peer)
                return False
            got += n
    except (socket.timeout, TimeoutError) as exc:
        _kva_transport_log(sock, stage, 'timeout', exc, got, peer=peer)
        return False
    except OSError as exc:
        _kva_transport_log(sock, stage, 'reset' if isinstance(exc, ConnectionResetError) else 'error', exc, got, peer=peer)
        return False
    return True

def _recv_all(sock: socket.socket, size: int, timeout_s: float=30.0, *, stage: str='payload', probe: bool=False) -> bytes:
    if size <= 0:
        return b''
    buf = bytearray(size)
    if not _recv_into(sock, memoryview(buf), timeout_s, stage=stage, probe=probe):
        return b''
    return bytes(buf)

def _receive_kva_via_tbp1(client_sock: socket.socket, timeout_s: float=30.0, prefix_query_fn=None) -> Optional[bytes]:
    import struct as _struct
    import zlib as _zlib
    state = 'WAIT_HELLO'
    reassembled = bytearray()
    expected_bytes = 0
    is_prompt_mode = False
    answered_query = False
    kv_buf = None
    kv_off = 0
    try:
        client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    except OSError:
        pass
    first_header = True
    while state != 'COMPLETE':
        hdr = _recv_all(client_sock, TBP_HEADER_SIZE, timeout_s, stage='header', probe=first_header)
        first_header = False
        if len(hdr) < TBP_HEADER_SIZE:
            if answered_query and (not reassembled) and (kv_buf is None):
                return None
            return None
        magic = _struct.unpack_from('<I', hdr, 0)[0]
        msg_type = _struct.unpack_from('<H', hdr, 8)[0]
        payload_sz = _struct.unpack_from('<I', hdr, 16)[0]
        if magic != TBP_MAGIC:
            sys.stderr.write(f'[kva-recv] bad magic 0x{magic:08x}\n')
            return None
        if msg_type == TBP_MSG_KV_CHUNK and state == 'RECV_CHUNKS' and (kv_buf is not None) and (kv_off + payload_sz <= len(kv_buf)):
            if payload_sz > 0 and (not _recv_into(client_sock, memoryview(kv_buf)[kv_off:kv_off + payload_sz], timeout_s)):
                sys.stderr.write('[kva-recv] payload read failed\n')
                return None
            kv_off += payload_sz
            continue
        payload = b''
        if payload_sz > 0:
            payload = _recv_all(client_sock, payload_sz, timeout_s)
            if len(payload) < payload_sz:
                sys.stderr.write('[kva-recv] payload read failed\n')
                return None
        if msg_type == TBP_MSG_HELLO:
            if state == 'WAIT_HELLO':
                state = 'WAIT_SESSION'
        elif msg_type == TBP_MSG_SESSION_START:
            if state == 'WAIT_SESSION':
                info = payload.decode('utf-8', errors='replace')
                for part in info.split():
                    if part.startswith('bytes='):
                        try:
                            expected_bytes = int(part[6:])
                        except ValueError:
                            pass
                is_prompt_mode = 'mode=prompt' in info
                sys.stderr.write(f'[kva-recv] session: {info.strip()}\n')
                state = 'WAIT_SEG_BEGIN'
        elif msg_type == TBP_MSG_KV_SEG_BEGIN:
            if state == 'WAIT_SEG_BEGIN':
                seg_bytes = 0
                for part in payload.decode('utf-8', errors='replace').split(';'):
                    if part.startswith('payload_bytes='):
                        try:
                            seg_bytes = int(part[len('payload_bytes='):])
                        except ValueError:
                            seg_bytes = 0
                if seg_bytes > 0:
                    kv_buf = bytearray(seg_bytes)
                    kv_off = 0
                state = 'RECV_CHUNKS'
        elif msg_type == TBP_MSG_KV_CHUNK:
            if state == 'RECV_CHUNKS':
                reassembled.extend(payload)
        elif msg_type == TBP_MSG_KV_SEG_END:
            if state == 'RECV_CHUNKS':
                state = 'WAIT_DONE'
        elif msg_type == TBP_MSG_KV_DONE:
            state = 'COMPLETE'
        elif msg_type == TBP_MSG_QUERY_PREFIX_CACHE:
            n_tok = payload_sz // 4
            toks = list(_struct.unpack(f'<{n_tok}i', payload[:n_tok * 4])) if n_tok else []
            cached = 0
            try:
                cached = int(prefix_query_fn(toks)) if prefix_query_fn is not None else 0
            except Exception as _qe:
                sys.stderr.write(f'[kva-recv] prefix query failed: {_qe}\n')
                cached = 0
            try:
                _tbp1_send_frame(client_sock, TBP_MSG_PREFIX_CACHE_REPLY, _struct.pack('<BIQ', 1 if cached > 0 else 0, cached, 0))
            except OSError as _se:
                _kva_transport_log(client_sock, 'prefix-send', 'error', _se)
                return None
            answered_query = True
            sys.stderr.write(f'[kva-recv] prefix query: {n_tok} tokens -> resident {cached} [KV_DELTA_QUERY]\n')
            sys.stderr.flush()
    if kv_buf is not None:
        if kv_off != len(kv_buf):
            sys.stderr.write(f'[kva-recv] short artifact: {kv_off} of {len(kv_buf)} bytes\n')
            sys.stderr.flush()
            return None
        if reassembled:
            kv_buf.extend(reassembled)
        reassembled = kv_buf
    sys.stderr.write(f"[kva-recv] received {len(reassembled)} bytes{(' (prompt-direct)' if is_prompt_mode else '')}\n")
    sys.stderr.flush()
    return reassembled

def _tbp1_listener_thread(kv_port: int, kva_path_out: list, event_new_kva, shutdown_event, prefix_query_fn=None, host='127.0.0.1') -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    except OSError:
        pass
    srv.bind((host, kv_port))
    srv.listen(1)
    srv.settimeout(1.0)
    sys.stderr.write(f'[kva-recv] listening on {host}:{kv_port}\n')
    sys.stderr.flush()
    while not shutdown_event.is_set():
        try:
            client, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        sys.stderr.write(f'[kva-recv] connection from {addr}\n')
        sys.stderr.flush()
        try:
            kva_bytes = _receive_kva_via_tbp1(client, timeout_s=30.0, prefix_query_fn=prefix_query_fn)
        finally:
            try:
                client.close()
            except OSError:
                pass
        if kva_bytes is not None and len(kva_bytes) > 0:
            n_stored = len(kva_bytes)
            kva_path_out[0] = kva_bytes
            kva_bytes = None
            event_new_kva.set()
            sys.stderr.write(f'[kva-recv] stored {n_stored} bytes in memory\n')
            sys.stderr.flush()
        else:
            sys.stderr.write('[kva-recv] empty/failed receive\n')
            sys.stderr.flush()
    try:
        srv.close()
    except OSError:
        pass

def prefix_query(self, tokens: List[int]) -> int:
    try:
        cache = self._retained_cache
        ret = self._retained_tokens
        if cache is None or not ret or (not tokens):
            return 0
        cache_n = _cache_offset(cache)
        lim = min(len(tokens), len(ret), cache_n)
        if lim <= 0:
            return 0
        import numpy as _np
        a = _np.asarray(ret[:lim], dtype=_np.int64)
        b = _np.asarray(tokens[:lim], dtype=_np.int64)
        ne = _np.nonzero(a != b)[0]
        lcp = int(ne[0]) if ne.size else lim
        if lcp < self._LCP_REUSE_MIN:
            return 0
        if not _can_trim_cache_to_offset(cache, lcp):
            return 0
        return lcp
    except Exception as _e:
        sys.stderr.write(f'[mlx-server] prefix_query failed: {_e}\n')
        sys.stderr.flush()
        return 0

def inject_mla_rows(cache, layers_mla, mla_meta, safe_cells, _delta_base, _incoming_cells, kv_delta_base):
    from kv_converter import inject_kv_mla, inject_kv_idx
    if _delta_base > 0:
        from kv_converter import append_kv_mla
        _keep_incoming = max(0, safe_cells - _delta_base)
        if safe_cells < _delta_base:
            if not _trim_cache_to_offset(cache, safe_cells):
                raise RuntimeError('delta base did not trim to logit-tail boundary')
            _delta_base = safe_cells
        _validate_kv_injection_cache(cache, _delta_base)
        if _keep_incoming < _incoming_cells:
            layers_mla = [(kv[:, :, :_keep_incoming, :], kp[:, :, :_keep_incoming, :]) for kv, kp in layers_mla]
        append_kv_mla(cache, layers_mla, mla_meta.get('idx_k_mx'), _keep_incoming)
        sys.stderr.write(f'[mlx-server] delta-inject: appended {_keep_incoming} shipped cells behind the {_delta_base}-cell resident base (total {safe_cells}) [KV_DELTA_INJECT]\n')
        sys.stderr.flush()
    else:
        inject_kv_mla(cache, layers_mla)
    if int(kv_delta_base or 0) == 0:
        inject_kv_idx(cache, mla_meta.get('idx_k_mx'), safe_cells)
    _idxl = mla_meta.get('idx_k_mx')
    _c1 = cache[0].caches[1] if hasattr(cache[0], 'caches') and len(cache[0].caches) > 1 else None
    sys.stderr.write(f"[mlx-server] DSA k_idx inject: has_k_idx={mla_meta.get('has_k_idx')} idx_head_size={mla_meta.get('idx_head_size')} n_idx_present={sum((1 for x in _idxl or [] if x is not None))} cache0_is_list={hasattr(cache[0], 'caches')} c1_offset={getattr(_c1, 'offset', 'NA')} idx0_shape={(None if not _idxl or _idxl[0] is None else _idxl[0].shape)}\n")
    sys.stderr.flush()
    if mla_meta.get('has_k_idx'):
        for _layer_i, _layer_cache in enumerate(cache):
            _children = getattr(_layer_cache, 'caches', None)
            if not _children or len(_children) < 2:
                raise RuntimeError(f'DSA k_idx cache missing at layer {_layer_i}')
            _main_offset = _cache_entry_offset(_children[0], -1)
            _indexer_offset = _cache_entry_offset(_children[1], -1)
            if _main_offset != safe_cells or _indexer_offset != safe_cells:
                raise RuntimeError(f'DSA cache offset mismatch at layer {_layer_i}: main={_main_offset} indexer={_indexer_offset} safe_cells={safe_cells}')
        sys.stderr.write(f'[mlx-server] DSA cache offsets aligned: main={safe_cells} indexer={safe_cells} safe_cells={safe_cells}\n')
        sys.stderr.flush()

def select_delta_base(self, prompt_token_ids, delta_base):
    retained_n = len(self._retained_tokens) if self._retained_tokens else 0
    new_n = len(prompt_token_ids)
    cache_n = _cache_offset(self._retained_cache) if self._retained_cache is not None else 0
    ok = 0 < delta_base <= new_n and self._retained_cache is not None and (retained_n >= delta_base) and (cache_n >= delta_base) and (list(self._retained_tokens[:delta_base]) == list(prompt_token_ids[:delta_base]))
    if ok:
        ok = _trim_cache_to_offset(self._retained_cache, delta_base)
    if ok:
        self._retained_tokens = list(self._retained_tokens[:delta_base])
        sys.stderr.write(f"[mlx-server] delta-inject: retained cache trimmed to the {delta_base}-token base (cache {cache_n} -> {delta_base}, tokens {retained_n}); the artifact's cells append there [KV_DELTA_BASE]\n")
        sys.stderr.flush()
        return (self._retained_cache, delta_base)
    sys.stderr.write(f'[mlx-server] delta-inject: the retained cache does NOT cover the {delta_base}-token base (retained={retained_n}, offset={cache_n}); full local prefill required [KV_DELTA_MISS]\n')
    sys.stderr.flush()
    self._retained_cache = None
    self._retained_tokens = []
    return (self._make_cache(), 0)

def _hash_token_ids(token_ids: List[int]) -> int:
    h = 14695981039346656037
    for tok in token_ids:
        v = int(tok) & 4294967295
        for shift in (0, 8, 16, 24):
            h ^= v >> shift & 255
            h = h * 1099511628211 & 18446744073709551615
    return h

def inject_standard_rows(cache, layers, safe_cells, base=0):
    from kv_converter import inject_kv_into_cache
    if base:
        import mlx.core as mx
        _validate_kv_injection_cache(cache, base)
        if len(cache) != len(layers):
            raise ValueError('standard KV layer count mismatch')
        if safe_cells < base:
            if not _trim_cache_to_offset(cache, safe_cells):
                raise ValueError('standard delta could not trim to logit tail')
        else:
            n = safe_cells - base
            for entry, layer in zip(cache, layers):
                keys, values = entry.state
                layer.keys = mx.concatenate([keys, layer.keys[..., :n, :].astype(keys.dtype)], axis=-2)
                layer.values = mx.concatenate([values, layer.values[..., :n, :].astype(values.dtype)], axis=-2)
            inject_kv_into_cache(cache, layers)
        print(f'[KV_DELTA_INJECT] standard KV base={base} total={safe_cells}', file=sys.stderr, flush=True)
    else:
        for layer in layers:
            layer.keys = layer.keys[..., :safe_cells, :]
            layer.values = layer.values[..., :safe_cells, :]
        inject_kv_into_cache(cache, layers)
