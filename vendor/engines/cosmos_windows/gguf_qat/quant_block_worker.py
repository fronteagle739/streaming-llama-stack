# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/quant_block_worker.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from cosmos_windows.gguf_qat.native_quant import QUANT_GEOMETRY
Q4_BLOCK_BYTES = 18

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def post_packet(url: str, packet: dict[str, Any], *, timeout: float=15.0, access_token: str='') -> dict[str, Any]:
    body = json.dumps(packet, separators=(',', ':')).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if access_token:
        headers['X-Studio-Token'] = access_token
    request = urllib.request.Request(url.rstrip('/') + '/training/deltas', data=body, headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode('utf-8'))
    if int(result.get('acked_sequence') or 0) != int(packet['sequence']):
        raise IOError(f"authority ACK mismatch for sequence {packet['sequence']}: {result}")
    return result

def _auth_headers(access_token: str='') -> dict[str, str]:
    return {'X-Studio-Token': access_token} if access_token else {}

def fetch_tensor_overlay(url: str, run_id: str, base_fingerprint: str, tensor_id: str, packed: bytes, *, quant: str, timeout: float=15.0, access_token: str='') -> bytes:
    if quant not in QUANT_GEOMETRY:
        raise ValueError(f'unsupported quant type {quant!r}')
    query = urllib.parse.urlencode({'base_fingerprint': base_fingerprint})
    endpoint = url.rstrip('/') + '/training/deltas/' + urllib.parse.quote(run_id, safe='') + '/tensors/' + urllib.parse.quote(tensor_id, safe='') + '?' + query
    request = urllib.request.Request(endpoint, headers=_auth_headers(access_token))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode('utf-8'))
    block_bytes = QUANT_GEOMETRY[quant][1]
    out = bytearray(packed)
    prefix = tensor_id + ':'
    for row in result.get('blocks') or []:
        block_id = str(row.get('block_id') or '')
        if not block_id.startswith(prefix) or str(row.get('quant') or '') != quant:
            raise IOError(f'authority returned mismatched block {block_id!r}')
        try:
            index = int(block_id[len(prefix):])
            block = base64.b64decode(str(row['packed']), validate=True)
        except (ValueError, KeyError, TypeError) as exc:
            raise IOError(f'authority returned invalid block {block_id!r}') from exc
        start = index * block_bytes
        if len(block) != block_bytes or start < 0 or start + block_bytes > len(out):
            raise IOError(f'authority block {block_id!r} is outside tensor geometry')
        if _sha(block) != str(row.get('new_hash') or ''):
            raise IOError(f'authority block {block_id!r} failed SHA-256')
        out[start:start + block_bytes] = block
    for span in result.get('ranges') or []:
        try:
            start_block = int(span['start_block'])
            count = int(span['count'])
            packed_run = base64.b64decode(str(span['packed']), validate=True)
        except (ValueError, KeyError, TypeError) as exc:
            raise IOError('authority returned invalid compact tensor range') from exc
        start = start_block * block_bytes
        if str(span.get('quant') or '') != quant or count < 1 or len(packed_run) != count * block_bytes or (start < 0) or (start + len(packed_run) > len(out)):
            raise IOError('authority compact tensor range is outside tensor geometry')
        if _sha(packed_run) != str(span.get('new_hash') or ''):
            raise IOError('authority compact tensor range failed SHA-256')
        out[start:start + len(packed_run)] = packed_run
    return bytes(out)

def put_residual_checkpoint(url: str, run_id: str, base_fingerprint: str, tensor_id: str, residual, *, quant: str, sequence: int, timeout: float=60.0, access_token: str='') -> dict[str, Any]:
    raw = residual.detach().to('cpu', dtype=__import__('torch').float16).contiguous().numpy().tobytes()
    query = urllib.parse.urlencode({'base_fingerprint': base_fingerprint, 'quant': quant, 'numel': residual.numel(), 'sequence': int(sequence), 'sha256': _sha(raw)})
    endpoint = url.rstrip('/') + '/training/deltas/' + urllib.parse.quote(run_id, safe='') + '/residuals/' + urllib.parse.quote(tensor_id, safe='') + '?' + query
    headers = {'Content-Type': 'application/octet-stream', **_auth_headers(access_token)}
    request = urllib.request.Request(endpoint, data=raw, headers=headers, method='PUT')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def fetch_residual_checkpoint(url: str, run_id: str, base_fingerprint: str, tensor_id: str, *, device: str, timeout: float=60.0, access_token: str=''):
    import numpy as np
    import torch
    query = urllib.parse.urlencode({'base_fingerprint': base_fingerprint})
    endpoint = url.rstrip('/') + '/training/deltas/' + urllib.parse.quote(run_id, safe='') + '/residuals/' + urllib.parse.quote(tensor_id, safe='') + '?' + query
    request = urllib.request.Request(endpoint, headers=_auth_headers(access_token))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            expected = response.headers.get('X-Residual-Sha256', '')
            numel = int(response.headers.get('X-Residual-Numel', '0'))
            sequence = int(response.headers.get('X-Residual-Sequence', '0'))
            quant = response.headers.get('X-Residual-Quant', '')
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if _sha(raw) != expected or len(raw) != numel * 2:
        raise IOError(f'authority residual failed integrity validation for {tensor_id}')
    residual = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).to(device)
    return {'residual': residual, 'sequence': sequence, 'quant': quant, 'sha256': expected}

@dataclass
class PendingTensorUpdate:
    candidate_packed: bytes
    candidate_residual: Any
    packets: list[dict[str, Any]]
    packet_index: int = 0

@dataclass
class QuantTensorState:
    packed: bytes
    quant: str = 'Q4_0'
    residual: Any = None
    pending: Optional[PendingTensorUpdate] = None

class QuantBlockUpdateEngine:

    def __init__(self, run_id: str, base_fingerprint: str, *, device: str='cuda:0', max_packet_bytes: int=4 * 1024 * 1024, starting_sequence: int=1, stochastic: bool=False, seed: int=0):
        if not run_id or not base_fingerprint:
            raise ValueError('run_id and base_fingerprint are required')
        self.run_id = run_id
        self.base_fingerprint = base_fingerprint
        self.device = device
        self.max_packet_bytes = max(Q4_BLOCK_BYTES, int(max_packet_bytes))
        self.next_sequence = max(1, int(starting_sequence))
        self.stochastic = stochastic
        self.seed = seed
        self.tensors: dict[str, QuantTensorState] = {}

    def register_tensor(self, tensor_id: str, packed: bytes, *, quant: str='Q4_0', residual=None) -> QuantTensorState:
        if quant not in QUANT_GEOMETRY:
            raise NotImplementedError(f'transactional updater does not support {quant!r}')
        _, block_bytes = QUANT_GEOMETRY[quant]
        if not tensor_id or not packed or len(packed) % block_bytes:
            raise ValueError(f'{quant} packed tensor must be non-empty and divisible by {block_bytes} bytes')
        state = QuantTensorState(bytes(packed), quant=quant, residual=residual)
        self.tensors[tensor_id] = state
        return state

    @staticmethod
    def _residual_digest(residual) -> str:
        import torch
        raw = residual.detach().to('cpu', dtype=torch.float16).contiguous().numpy().tobytes()
        return _sha(raw)

    def apply_delta(self, tensor_id: str, delta, *, send: Optional[Callable[[dict[str, Any]], dict]]=None) -> dict:
        import torch
        from cosmos_windows.gguf_qat.sr_requant import merge_q4_0_with_residual
        from cosmos_windows.gguf_qat.native_quant import dequantize_native, quantize_native
        state = self.tensors.get(tensor_id)
        if state is None:
            raise ValueError(f'unregistered quant tensor: {tensor_id}')
        if state.pending is not None:
            raise RuntimeError(f'tensor {tensor_id} has an unacknowledged candidate update')
        block_values, block_bytes = QUANT_GEOMETRY[state.quant]
        delta = delta.reshape(-1).to(self.device, torch.float32)
        expected = len(state.packed) // block_bytes * block_values
        if delta.numel() != expected:
            raise ValueError(f'delta has {delta.numel()} values; {state.quant} tensor needs {expected}')
        residual = state.residual
        if residual is None:
            residual = torch.zeros_like(delta)
        if state.quant == 'Q4_0':
            raw_old = torch.frombuffer(bytearray(state.packed), dtype=torch.uint8).to(self.device)
            generator = torch.Generator(device=self.device).manual_seed(self.seed + self.next_sequence)
            raw_new, residual_new, stats = merge_q4_0_with_residual(raw_old, None, delta, residual=residual, generator=generator, stochastic=self.stochastic, fixed_scale=True)
            new_bytes = raw_new.detach().cpu().numpy().tobytes()
        else:
            old = torch.from_numpy(dequantize_native(state.packed, state.quant).copy()).to(self.device, torch.float32)
            target = old + delta + residual.to(self.device, torch.float32)
            candidate = quantize_native(target.detach().cpu().numpy(), state.quant)
            represented = torch.from_numpy(dequantize_native(candidate, state.quant).copy()).to(self.device, torch.float32)
            old_error = (target - old).square().reshape(-1, block_values).mean(1)
            new_error = (target - represented).square().reshape(-1, block_values).mean(1)
            accept = new_error < old_error
            import numpy as np
            old_blocks = np.frombuffer(state.packed, dtype=np.uint8).reshape(-1, block_bytes)
            candidate_blocks = candidate.reshape(-1, block_bytes)
            selected = candidate_blocks.copy()
            selected[~accept.detach().cpu().numpy()] = old_blocks[~accept.detach().cpu().numpy()]
            new_bytes = selected.tobytes()
            represented = torch.where(accept.repeat_interleave(block_values), represented, old)
            residual_new = (target - represented).to(torch.float16)
            stats = {'accepted_blocks': int(accept.sum().item()), 'mean_abs_residual': float(residual_new.float().abs().mean().item()), 'max_abs_residual': float(residual_new.float().abs().max().item())}
        old_bytes = state.packed
        patches = []
        for offset in range(0, len(old_bytes), block_bytes):
            old_block = old_bytes[offset:offset + block_bytes]
            new_block = new_bytes[offset:offset + block_bytes]
            if old_block == new_block:
                continue
            patches.append({'block_id': f'{tensor_id}:{offset // block_bytes}', 'quant': state.quant, 'old_hash': _sha(old_block), 'new_hash': _sha(new_block), 'packed': base64.b64encode(new_block).decode('ascii')})
        stats['changed_blocks'] = len(patches)
        if not patches:
            state.residual = residual_new.detach()
            return {**stats, 'packets': 0, 'pending': False, 'residual_retained': True}
        per_packet = max(1, self.max_packet_bytes // block_bytes)
        packets = []
        digest = self._residual_digest(residual_new)
        for start in range(0, len(patches), per_packet):
            group = patches[start:start + per_packet]
            ranges = []
            current = []
            prior_index = None
            for row in group:
                index = int(str(row['block_id']).rsplit(':', 1)[1])
                if current and index != prior_index + 1:
                    ranges.append(current)
                    current = []
                current.append(row)
                prior_index = index
            if current:
                ranges.append(current)
            wire_ranges = []
            for rows in ranges:
                first = int(str(rows[0]['block_id']).rsplit(':', 1)[1])
                packed_run = b''.join((base64.b64decode(row['packed']) for row in rows))
                old_hashes = b''.join((bytes.fromhex(row['old_hash']) for row in rows))
                wire_ranges.append({'tensor_id': tensor_id, 'start_block': first, 'count': len(rows), 'quant': state.quant, 'old_hashes': base64.b64encode(old_hashes).decode('ascii'), 'new_hash': _sha(packed_run), 'packed': base64.b64encode(packed_run).decode('ascii')})
            packets.append({'schema': 'studio.micro_delta.v2', 'run_id': self.run_id, 'base_fingerprint': self.base_fingerprint, 'sequence': self.next_sequence + len(packets), 'residual_digest': digest, 'ranges': wire_ranges})
        state.pending = PendingTensorUpdate(new_bytes, residual_new.detach(), packets)
        result = {**stats, 'packets': len(packets), 'pending': True, 'residual_retained': True}
        if send is not None:
            result.update(self.flush_pending(tensor_id, send))
        return result

    def flush_pending(self, tensor_id: str, send: Callable[[dict[str, Any]], dict]) -> dict[str, Any]:
        state = self.tensors.get(tensor_id)
        if state is None or state.pending is None:
            return {'committed': False, 'pending': False}
        pending = state.pending
        while pending.packet_index < len(pending.packets):
            packet = pending.packets[pending.packet_index]
            ack = send(packet)
            if int(ack.get('acked_sequence') or 0) != int(packet['sequence']):
                raise IOError(f"authority did not durably ACK sequence {packet['sequence']}")
            pending.packet_index += 1
            self.next_sequence = int(packet['sequence']) + 1
        state.packed = pending.candidate_packed
        state.residual = pending.candidate_residual
        state.pending = None
        return {'committed': True, 'pending': False, 'next_sequence': self.next_sequence}

    def checkpoint_residual(self, tensor_id: str, url: str, *, access_token: str='', timeout: float=60.0) -> dict[str, Any]:
        state = self.tensors.get(tensor_id)
        if state is None or state.residual is None:
            raise ValueError(f'tensor {tensor_id} has no residual to checkpoint')
        if state.pending is not None:
            raise RuntimeError('cannot checkpoint residual while a tensor update is unacknowledged')
        return put_residual_checkpoint(url, self.run_id, self.base_fingerprint, tensor_id, state.residual, quant=state.quant, sequence=self.next_sequence - 1, timeout=timeout, access_token=access_token)

@dataclass
class FactoredAdafactorState:
    row: Any = None
    col: Any = None
    full: Any = None
    step: int = 0

def factored_adafactor_delta(grad, state: FactoredAdafactorState, *, lr: float, beta2: float=0.999, eps: float=1e-30, clip_threshold: float=1.0):
    import torch
    g = grad.to(torch.float32)
    g2 = g.square().add_(eps)
    if g.ndim != 2:
        if state.full is None:
            state.full = g2
        else:
            state.full.mul_(beta2).add_(g2, alpha=1.0 - beta2)
        state.step += 1
        update = g * state.full.rsqrt()
        rms = update.square().mean().sqrt().clamp_min(eps)
        update = update / torch.maximum(torch.ones((), device=g.device), rms / clip_threshold)
        return -float(lr) * update
    row_now, col_now = (g2.mean(dim=1), g2.mean(dim=0))
    if state.row is None:
        state.row, state.col = (row_now, col_now)
    else:
        state.row.mul_(beta2).add_(row_now, alpha=1.0 - beta2)
        state.col.mul_(beta2).add_(col_now, alpha=1.0 - beta2)
    state.step += 1
    denom = state.row.mean().clamp_min(eps)
    variance = state.row[:, None] * state.col[None, :] / denom
    update = g * variance.rsqrt()
    rms = update.square().mean().sqrt().clamp_min(eps)
    update = update / torch.maximum(torch.ones((), device=g.device), rms / clip_threshold)
    return -float(lr) * update
