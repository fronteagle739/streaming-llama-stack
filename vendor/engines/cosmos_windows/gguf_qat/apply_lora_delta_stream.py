# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/apply_lora_delta_stream.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from cosmos_windows.gguf_qat.merge_back import load_lora_deltas
from cosmos_windows.gguf_qat.quant_block_worker import QuantBlockUpdateEngine, post_packet
from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF

def _require_verified_split_manifest(gguf: str, fingerprint: str) -> None:
    if not re.search('-\\d{5}-of-\\d{5}\\.gguf$', gguf, re.IGNORECASE):
        return
    manifest_path = Path(gguf).parent / '.studio_verified_manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'split GGUF needs a verified manifest before authority updates: {manifest_path}') from exc
    if str(manifest.get('base_fingerprint') or '') != fingerprint:
        raise ValueError('verified split-GGUF manifest does not match --base-fingerprint')

def _status(url: str, run_id: str, fingerprint: str, access_token: str='') -> dict:
    query = urllib.parse.urlencode({'base_fingerprint': fingerprint})
    request = urllib.request.Request(url.rstrip('/') + f'/training/deltas/{urllib.parse.quote(run_id)}?{query}')
    if access_token:
        request.add_header('X-Studio-Token', access_token)
    with urllib.request.urlopen(request, timeout=15.0) as response:
        return json.loads(response.read().decode('utf-8'))

def run(args) -> int:
    import torch
    _require_verified_split_manifest(args.gguf, args.base_fingerprint)
    authority_token = args.authority_token or os.environ.get('STUDIO_QUANT_AUTHORITY_TOKEN', '')
    status = _status(args.authority_url, args.run_id, args.base_fingerprint, authority_token)
    engine = QuantBlockUpdateEngine(args.run_id, args.base_fingerprint, device=args.device, max_packet_bytes=args.max_packet_bytes, starting_sequence=int(status.get('acked_sequence') or 0) + 1, stochastic=args.stochastic, seed=args.seed)
    reader = StreamedGGUF(args.gguf, device=args.device)
    deltas, scaling = load_lora_deltas(args.adapter, device=args.device)
    if not deltas:
        raise ValueError('adapter contains no mapped LoRA deltas')
    print(f'[delta4] adapter scaling={scaling:.4f}; {len(deltas)} tensors; authority={args.authority_url}', flush=True)
    changed = packets = 0
    sender = lambda packet: post_packet(args.authority_url, packet, timeout=args.timeout, access_token=authority_token)
    for index, (name, delta) in enumerate(deltas.items(), 1):
        if not reader.has(name):
            raise ValueError(f'base GGUF does not contain adapter target {name}')
        info = reader.info(name)
        if info.quant not in {'Q4_0', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0'}:
            raise ValueError(f'native delta streaming does not support {name} quant {info.quant}')
        raw = reader.raw_bytes_pread(name)
        engine.register_tensor(name, raw.detach().cpu().numpy().tobytes(), quant=info.quant)
        del raw
        flat = delta.reshape(-1).to(torch.float32)
        for _ in range(args.delta_slices):
            result = engine.apply_delta(name, flat / args.delta_slices, send=sender)
            changed += int(result.get('changed_blocks') or 0)
            packets += int(result.get('packets') or 0)
        print(f'[delta4] tensor {index}/{len(deltas)} {name}: changed={changed} packets={packets} next_sequence={engine.next_sequence}', flush=True)
        del flat, delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    reader.close()
    print(f'RESULT DONE changed_blocks={changed} packets={packets} next_sequence={engine.next_sequence}', flush=True)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True, help='first shard or single GGUF')
    parser.add_argument('--adapter', required=True, help='PEFT adapter directory')
    parser.add_argument('--authority-url', required=True)
    parser.add_argument('--authority-token', default='')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--base-fingerprint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-packet-bytes', type=int, default=4 * 1024 * 1024)
    parser.add_argument('--delta-slices', type=int, default=4)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=15.0)
    args = parser.parse_args()
    if args.delta_slices < 1:
        parser.error('--delta-slices must be >= 1')
    return run(args)
if __name__ == '__main__':
    raise SystemExit(main())
