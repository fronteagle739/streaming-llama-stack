# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/diloco_sparse_client.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

def select_sparse(values, fraction: float):
    import torch
    flat = values.detach().float().reshape(-1)
    if not 0 < fraction <= 1:
        raise ValueError('sparse DiLoCo top-k fraction must be in (0, 1]')
    count = min(flat.numel(), max(1, int(round(flat.numel() * fraction))))
    chosen = torch.topk(flat.abs(), count, sorted=False).indices
    chosen, _ = torch.sort(chosen)
    return (chosen.detach().cpu().tolist(), flat[chosen].detach().cpu().tolist())

def _request(url: str, *, body=None, token: str='', timeout: float=60.0) -> dict:
    data = json.dumps(body, separators=(',', ':')).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method='POST' if body is not None else 'GET', headers={'Content-Type': 'application/json'})
    if token:
        request.add_header('X-Studio-Token', token)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())

def synchronize(values, *, coordinator_url: str, run_id: str, round_no: int, base_fingerprint: str, schema_hash: str, tensor_id: str, identity_path: str, node_label: str, step_count: int, sync_bits: int=8, topk_fraction: float=0.01, access_token: str='', wait_seconds: float=1800, poll_seconds: float=5.0):
    import torch
    from studio.diloco.codec import Int4SRCodec, QuantizedDelta
    from studio.diloco.coordinator import build_sparse_contribution, load_or_create_identity, unpack_codes, unpack_indices
    indices, selected = select_sparse(values, topk_fraction)
    codec = Int4SRCodec(bits=sync_bits)
    quant = codec.encode(selected, stochastic=True)
    identity = load_or_create_identity(Path(identity_path), label=node_label)
    payload = build_sparse_contribution(identity, run_id=run_id, round_no=round_no, base_fingerprint=base_fingerprint, adapter_schema_hash=schema_hash, tensor_id=tensor_id, vector_length=values.numel(), indices=indices, delta=quant, step_count=step_count)
    endpoint = coordinator_url.rstrip('/') + '/training/diloco/runs/' + urllib.parse.quote(run_id)
    token = access_token or os.environ.get('STUDIO_DILOCO_ACCESS_TOKEN', '')
    result = _request(endpoint + '/contributions', body=payload, token=token)
    deadline = time.time() + wait_seconds
    while result.get('status') == 'waiting' and time.time() < deadline:
        time.sleep(poll_seconds)
        status = _request(endpoint, token=token)
        row = status.get('last_global') or {}
        if int(row.get('completed_round', -1)) == round_no:
            result = {'status': 'aggregated', 'global': row}
            break
    if result.get('status') != 'aggregated':
        raise TimeoutError(f'sparse DiLoCo round {round_no} did not aggregate before timeout')
    row = result['global']
    if not row.get('sparse') or row.get('tensor_id') != tensor_id or int(row['vector_length']) != values.numel():
        raise ValueError('coordinator returned a mismatched sparse global correction')
    out_indices = unpack_indices(row['indices'], int(row['length']))
    decoded = codec.decode(QuantizedDelta(codes=unpack_codes(row['packed'], int(row['bits']), int(row['length'])), scale=float(row['scale']), zero_point=float(row['zero_point']), bits=int(row['bits']), length=int(row['length'])))
    correction = torch.zeros(values.numel(), device=values.device, dtype=torch.float32)
    correction[torch.tensor(out_indices, device=values.device, dtype=torch.long)] = torch.tensor(decoded, device=values.device, dtype=torch.float32)
    return (correction.reshape(values.shape), {'round': round_no, 'selected_coordinates': len(indices), 'global_coordinates': len(out_indices), 'node_fingerprint': identity.fingerprint})
