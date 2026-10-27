from __future__ import annotations
import argparse
import struct
import zlib
from pathlib import Path
DEFAULT_PROMPT = 'The sky above the observatory was clear and cold. Explain, in coherent English, how a careful engineer would compare several speculative decode draft depths without confusing speed for quality.'

def build_prompt_direct_kva(prompt: str, *, token_ids: list[int] | None=None, n_layers: int=0, n_ctx: int=0, requested_n_predict: int=-1, model_fingerprint: int=0) -> bytes:
    prompt_bytes = prompt.encode('utf-8')
    ids = list(token_ids or [])
    token_bytes = struct.pack(f'<{len(ids)}i', *ids) if ids else b''
    payload = prompt_bytes + token_bytes
    crc = zlib.crc32(payload) & 4294967295
    last_token_id = ids[-1] if ids else -1
    base = struct.pack('<8s HH III HH I QI', b'KVARTIF1', 1, 5, n_layers, n_ctx, len(ids), 1, 1, 0, len(payload), crc)
    ext = struct.pack('<i BB H Q', last_token_id, 0, 0, 0, model_fingerprint)
    sampling = b'\x00' * 84
    request = struct.pack('<i', int(requested_n_predict))
    prompt_header = struct.pack('<III', len(ids), len(prompt_bytes), 0)
    return base + ext + sampling + request + prompt_header + payload

def write_prompt_direct_kva(output: str | Path, prompt: str, *, requested_n_predict: int=-1) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = build_prompt_direct_kva(prompt, requested_n_predict=requested_n_predict)
    path.write_bytes(blob)
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True, help='output .kva path')
    parser.add_argument('-p', '--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--requested-n-predict', type=int, default=-1, help='optional KVA decode cap; -1 lets decode_config/max-tokens decide')
    args = parser.parse_args()
    out = write_prompt_direct_kva(args.output, args.prompt, requested_n_predict=args.requested_n_predict)
    print(f'wrote prompt-direct KVA: {out} ({out.stat().st_size} bytes)')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
