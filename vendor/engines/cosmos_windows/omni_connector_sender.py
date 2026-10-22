# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/omni_connector_sender.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import socket
import struct
import zlib
from typing import Dict, Mapping, Optional
TBP_MAGIC = 1413632049
TBP_PROTO_MAJOR = 1
TBP_PROTO_MINOR = 0
TBP_HEADER_SIZE = 52
TBP_MSG_HELLO = 1
TBP_MSG_SESSION_START = 3
TBP_MSG_KV_SEG_BEGIN = 7
TBP_MSG_KV_CHUNK = 8
TBP_MSG_KV_SEG_END = 9
TBP_MSG_KV_DONE = 12
_TBP_HEADER = struct.Struct('<IHHHHIIQQQII')

def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 4294967295

def build_tbp_frame(msg_type: int, payload: bytes=b'', *, flags: int=0, session_id: int=0, stream_id: int=0, seq_no: int=0) -> bytes:
    payload_crc32 = _crc32(payload) if payload else 0
    header = bytearray(_TBP_HEADER.pack(TBP_MAGIC, TBP_PROTO_MAJOR, TBP_PROTO_MINOR, msg_type, flags, TBP_HEADER_SIZE, len(payload), session_id, stream_id, seq_no, 0, payload_crc32))
    struct.pack_into('<I', header, 44, _crc32(bytes(header)))
    return bytes(header) + payload

def format_kv_pairs(values: Mapping[str, object]) -> bytes:
    return ';'.join((f'{k}={v}' for k, v in values.items())).encode('utf-8')

def build_request_payload(prompt: str, *, request_id: str='req-1', height: int=256, width: int=256, num_frames: int=1, num_inference_steps: int=35, guidance_scale: float=6.0, flow_shift: float=10.0, seed: Optional[int]=None, negative_prompt: Optional[str]=None, extra: Optional[dict]=None) -> bytes:
    body: Dict[str, object] = {'request_id': request_id, 'prompt': prompt, 'height': height, 'width': width, 'num_frames': num_frames, 'num_inference_steps': num_inference_steps, 'guidance_scale': guidance_scale, 'flow_shift': flow_shift}
    if seed is not None:
        body['seed'] = seed
    if negative_prompt is not None:
        body['negative_prompt'] = negative_prompt
    if extra:
        body.update(extra)
    return json.dumps(body).encode('utf-8')

def build_frame_stream(payload: bytes, *, session_id: int, from_stage: str='rtx_ar', to_stage: str='mac_cosmos', put_key: str='ar_conditioning', chunk_size: int=1 << 16) -> list[bytes]:
    meta = format_kv_pairs({'from_stage': from_stage, 'to_stage': to_stage, 'put_key': put_key, 'payload_bytes': len(payload), 'streams': 1})
    frames = [build_tbp_frame(TBP_MSG_HELLO), build_tbp_frame(TBP_MSG_SESSION_START, session_id=session_id), build_tbp_frame(TBP_MSG_KV_SEG_BEGIN, meta, session_id=session_id, stream_id=1)]
    seq = 0
    for off in range(0, max(1, len(payload)), chunk_size):
        frames.append(build_tbp_frame(TBP_MSG_KV_CHUNK, payload[off:off + chunk_size], session_id=session_id, stream_id=1, seq_no=seq))
        seq += 1
    frames.append(build_tbp_frame(TBP_MSG_KV_SEG_END, session_id=session_id, stream_id=1))
    frames.append(build_tbp_frame(TBP_MSG_KV_DONE, session_id=session_id, stream_id=1))
    return frames

def send_generation_request(host: str, port: int=9100, *, prompt: str, session_id: int=1, timeout_s: float=30.0, **req_kwargs) -> None:
    payload = build_request_payload(prompt, **req_kwargs)
    frames = build_frame_stream(payload, session_id=session_id)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        for fr in frames:
            sock.sendall(fr)
    print(f'sent {len(frames)} TBP1 frames ({len(payload)} byte request) to {host}:{port} -> prompt={prompt!r}')

def main():
    ap = argparse.ArgumentParser(description='RTX->Mac OmniConnector sender (drive Cosmos3 MPS gen from the RTX).')
    ap.add_argument('--host', default='<mac-ip>', help='Mac receiver IP')
    ap.add_argument('--port', type=int, default=9100)
    ap.add_argument('--prompt', default='a red cube on a wooden table')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--num-frames', type=int, default=1)
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--session-id', type=int, default=1)
    args = ap.parse_args()
    send_generation_request(args.host, args.port, prompt=args.prompt, session_id=args.session_id, height=args.height, width=args.width, num_frames=args.num_frames, num_inference_steps=args.steps, seed=args.seed)
if __name__ == '__main__':
    main()
