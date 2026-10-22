# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/mac_reasoner_server.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
_MODEL = None
_PROC = None
_DEV = 'mps'
_LOCK = threading.Lock()

def _load(model_path: str, device: str):
    global _MODEL, _PROC, _DEV
    _DEV = device
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    import transformers_cosmos3
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration, Cosmos3OmniConfig
    from transformers import AutoProcessor
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    cfg = Cosmos3OmniConfig.from_pretrained(model_path)
    print(f'[mac-reasoner] config text {cfg.text_config.num_hidden_layers}L kv={cfg.text_config.num_key_value_heads} inter={cfg.text_config.intermediate_size}; loading on {device} ...', flush=True)
    t0 = time.time()
    _MODEL = Cosmos3ForConditionalGeneration.from_pretrained(model_path, config=cfg, torch_dtype=dtype, attn_implementation='sdpa', device_map=device)
    _MODEL.eval()
    _PROC = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    nparams = sum((p.numel() for p in _MODEL.parameters())) / 1000000000.0
    print(f'[mac-reasoner] LOADED {nparams:.2f}B in {time.time() - t0:.0f}s on {device}', flush=True)

def _reason(body: dict) -> dict:
    import torch
    prompt = body.get('prompt', '')
    system = body.get('system')
    max_new = int(body.get('max_new_tokens', 256))
    temperature = float(body.get('temperature', 0.0))
    top_p = float(body.get('top_p', 0.95))
    content = []
    images = []
    if body.get('image_b64'):
        body.setdefault('images_b64', [body['image_b64']])
    if body.get('images_b64'):
        from PIL import Image
        for raw in body.get('images_b64') or []:
            image = Image.open(io.BytesIO(base64.b64decode(raw))).convert('RGB')
            images.append(image)
            content.append({'type': 'image', 'image': image})
    if images and body.get('media_modality') == 'video':
        prompt = 'The attached images are sampled frames from the requested video.\n\n' + prompt
    content.append({'type': 'text', 'text': prompt})
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': [{'type': 'text', 'text': system}]})
    msgs.append({'role': 'user', 'content': content})
    text = _PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = dict(text=[text], return_tensors='pt')
    if images:
        kw['images'] = images
    inputs = _PROC(**kw).to(_DEV)
    gen_kw = dict(max_new_tokens=max_new)
    if temperature and temperature > 0:
        gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kw.update(do_sample=False)
    t0 = time.time()
    with _LOCK, torch.no_grad():
        out = _MODEL.generate(**inputs, **gen_kw)
    gen = out[0][inputs['input_ids'].shape[1]:]
    answer = _PROC.decode(gen, skip_special_tokens=True).strip()
    return {'answer': answer, 'seconds': round(time.time() - t0, 2), 'prompt_tokens': int(inputs['input_ids'].shape[1]), 'gen_tokens': int(gen.shape[0])}

def _decode_kv(raw: bytes, prompt_len: int, next_id: int, max_new: int) -> dict:
    import torch
    from safetensors.torch import load as st_load
    from transformers import DynamicCache
    flat = st_load(raw)
    n = len([k for k in flat if k.startswith('k_')])
    dtype = next(_MODEL.parameters()).dtype
    cache = DynamicCache()
    for i in range(n):
        k = flat[f'k_{i}'].to(_DEV).to(dtype)
        v = flat[f'v_{i}'].to(_DEV).to(dtype)
        cache.update(k, v, i)
    eos = _MODEL.config.text_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    cur = torch.tensor([[int(next_id)]], device=_DEV)
    gen = [cur]
    pos = prompt_len
    t0 = time.time()
    with _LOCK, torch.no_grad():
        for _ in range(max_new - 1):
            if cur.item() in eos_ids:
                break
            o = _MODEL(input_ids=cur, past_key_values=cache, cache_position=torch.tensor([pos], device=_DEV), use_cache=True, return_dict=True)
            cur = o.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(cur)
            pos += 1
    toks = torch.cat(gen, dim=-1)[0]
    return {'answer': _PROC.decode(toks, skip_special_tokens=True).strip(), 'seconds': round(time.time() - t0, 2), 'gen_tokens': int(toks.shape[0]), 'prefill_tokens': prompt_len}

class Handler(BaseHTTPRequestHandler):

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/health':
            import torch
            self._send(200, {'status': 'ok', 'device': _DEV, 'params_b': round(sum((p.numel() for p in _MODEL.parameters())) / 1000000000.0, 2), 'ctx': _MODEL.config.text_config.max_position_embeddings})
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path not in ('/reason', '/decode_kv'):
            return self._send(404, {'error': 'not found'})
        try:
            n = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(n)
            if self.path == '/reason':
                self._send(200, _reason(json.loads(raw or b'{}')))
            else:
                self._send(200, _decode_kv(raw, int(self.headers.get('X-Prompt-Len', 0)), int(self.headers.get('X-Next-Id', 0)), int(self.headers.get('X-Max-New-Tokens', 256))))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {'error': f'{type(e).__name__}: {e}'})

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=os.path.expanduser('~/models/Cosmos3-Nano'))
    ap.add_argument('--device', default='mps', choices=['mps', 'cpu', 'cuda'])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9200)
    args = ap.parse_args()
    _load(args.model, args.device)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'[mac-reasoner] LISTENING on {args.host}:{args.port} — reasoner resident on {args.device}', flush=True)
    srv.serve_forever()
    return 0
if __name__ == '__main__':
    sys.exit(main())
