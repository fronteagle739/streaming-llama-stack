# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_sender.py
# Machine fallback paths cleared; no model data included.
import importlib.util
import os
import socket
import sys
import threading
from cosmos_windows.omni_connector_sender import build_frame_stream, build_request_payload
RECEIVER_PATH = sys.argv[1] if len(sys.argv) > 1 else ''

def _load_receiver(path):
    spec = importlib.util.spec_from_file_location('mac_receiver', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['mac_receiver'] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    if not os.path.exists(RECEIVER_PATH):
        print(f'[SKIP] receiver module not found at {RECEIVER_PATH} (scp it from the Mac to validate).')
        sys.exit(0)
    rx = _load_receiver(RECEIVER_PATH)
    receiver = rx.OmniConnectorReceiver()
    a, b = socket.socketpair()
    t = threading.Thread(target=receiver._handle_client, args=(b,), daemon=True)
    t.start()
    payload = build_request_payload('a red cube on a wooden table', request_id='xmachine-1', height=512, width=512, num_frames=1, num_inference_steps=20, seed=7)
    for fr in build_frame_stream(payload, session_id=1, put_key='ar_conditioning'):
        a.sendall(fr)
    a.shutdown(socket.SHUT_WR)
    t.join(timeout=10.0)
    a.close()
    art = receiver.store.get_artifact('rtx_ar', 'mac_cosmos', 'ar_conditioning')
    assert art is not None, 'receiver did not store the artifact'
    req = rx.decode_generation_request(art)
    print(f'[round-trip OK] receiver reassembled {art.size}B -> request:')
    print(f'  prompt={req.prompt!r}  {req.width}x{req.height}  steps={req.num_inference_steps}  seed={req.seed}')
    ok = req.prompt == 'a red cube on a wooden table' and req.width == 512 and (req.height == 512) and (req.num_inference_steps == 20) and (req.seed == 7)
    print(f"\n=== SENDER PROTOCOL {('VALIDATED' if ok else 'MISMATCH')} (frames round-trip through the real receiver) ===")
    sys.exit(0 if ok else 1)
if __name__ == '__main__':
    main()
