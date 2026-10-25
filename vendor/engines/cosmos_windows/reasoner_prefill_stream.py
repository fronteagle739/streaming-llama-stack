# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/reasoner_prefill_stream.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

def _require_local_model(model_path):
    s = str(model_path or '').strip()
    looks_local = bool(s) and (s[0] in '/~' or '\\' in s or (len(s) >= 2 and s[1] == ':' and s[0].isalpha()))
    if looks_local and (not os.path.isdir(s)):
        raise RuntimeError(f"reasoner model directory not found: '{model_path}'. cosmos_windows.reasoner_prefill_stream needs a local Cosmos3-Nano checkout on disk; this path does not exist and no model is downloaded automatically. Pass --model <existing Cosmos3-Nano dir> (or set STUDIO_REASONER_MODEL in front-eagle-stack), or use a different H3 enhancer backend (STUDIO_H3_ENHANCER_ENDPOINT / STUDIO_H3_ENHANCER_GGUF).")

def _load(model_path, device):
    _require_local_model(model_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from cosmos_windows import _cosmos_compat
    except Exception:
        pass
    import torch
    import transformers_cosmos3
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration, Cosmos3OmniConfig
    from transformers import AutoProcessor
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    cfg = Cosmos3OmniConfig.from_pretrained(model_path)
    model = Cosmos3ForConditionalGeneration.from_pretrained(model_path, config=cfg, torch_dtype=dtype, attn_implementation='sdpa', device_map=device)
    model.eval()
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return (model, proc)

def cache_to_tensors(cache):
    if hasattr(cache, 'layers'):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))

def tensors_to_cache(kv_list):
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_list):
        cache.update(k, v, i)
    return cache

def prefill_export(model, inputs):
    import torch
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    kv = cache_to_tensors(out.past_key_values)
    seq = kv[0][0].shape[-2]
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)
    meta = {'layers': len(kv), 'seq': int(seq), 'kv_heads': int(kv[0][0].shape[1]), 'head_dim': int(kv[0][0].shape[-1])}
    return (kv, meta, next_id)

def prefill_chunked(model, inputs, chunk=512):
    import torch
    from transformers import DynamicCache
    ids = inputs['input_ids']
    dev = ids.device
    L = ids.shape[1]
    cache = DynamicCache()
    last_logits = None
    extra = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        kw = dict(input_ids=ids[:, s:e], past_key_values=cache, cache_position=torch.arange(s, e, device=dev), use_cache=True, return_dict=True)
        if s == 0 and extra:
            kw.update(extra)
        with torch.no_grad():
            o = model(**kw)
        last_logits = o.logits[:, -1]
    kv = cache_to_tensors(cache)
    next_id = last_logits.argmax(-1, keepdim=True)
    meta = {'layers': len(kv), 'seq': int(kv[0][0].shape[-2]), 'chunk': chunk, 'kv_heads': int(kv[0][0].shape[1]), 'head_dim': int(kv[0][0].shape[-1])}
    return (kv, meta, next_id)

def save_kv(kv_list, path):
    from safetensors.torch import save_file
    flat = {}
    for i, (k, v) in enumerate(kv_list):
        flat[f'k_{i}'] = k.contiguous().cpu()
        flat[f'v_{i}'] = v.contiguous().cpu()
    save_file(flat, str(path))
    return os.path.getsize(path)

def load_kv(path, device, dtype=None):
    from safetensors.torch import load_file
    flat = load_file(str(path))
    n = len([key for key in flat if key.startswith('k_')])
    kv = []
    for i in range(n):
        k = flat[f'k_{i}'].to(device)
        v = flat[f'v_{i}'].to(device)
        if dtype is not None:
            k, v = (k.to(dtype), v.to(dtype))
        kv.append((k, v))
    return kv

def _next_token(logits, temperature, top_p):
    import torch
    if not temperature or temperature <= 0.0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
    if top_p and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        csum = torch.cumsum(sp, dim=-1)
        keep = csum - sp <= top_p
        sp = sp * keep
        sp = sp / sp.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sp, num_samples=1)
        return si.gather(-1, choice)
    return torch.multinomial(probs, num_samples=1)

def decode_from_kv(model, kv_list, next_id, prompt_len, max_new_tokens=40, eos_ids=None, temperature=0.0, top_p=1.0):
    import torch
    dev = next(model.parameters()).device
    cache = tensors_to_cache(kv_list)
    eos_ids = set(eos_ids or [])
    cur = next_id.to(dev)
    generated = [cur]
    pos = prompt_len
    for _ in range(max_new_tokens - 1):
        if cur.item() in eos_ids:
            break
        with torch.no_grad():
            o = model(input_ids=cur, past_key_values=cache, cache_position=torch.tensor([pos], device=dev), use_cache=True, return_dict=True)
        cur = _next_token(o.logits[:, -1], temperature, top_p)
        generated.append(cur)
        pos += 1
    return torch.cat(generated, dim=-1)

def _selftest(model_path, device, max_new=40):
    import torch
    model, proc = _load(model_path, device)
    q = 'List three safety checks a mobile robot should run before lifting a heavy box. Be concise.'
    text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': q}]}], tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors='pt').to(device)
    plen = inputs['input_ids'].shape[1]
    eos = model.config.text_config.eos_token_id
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    t0 = time.time()
    with torch.no_grad():
        full = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    a_tok = full[0][plen:]
    print(f'[A] full generate: {time.time() - t0:.1f}s, {a_tok.shape[0]} tok')
    t0 = time.time()
    kv, meta, next_id = prefill_export(model, inputs)
    print(f'[B] prefill_export: {time.time() - t0:.1f}s, cache {meta}')
    b_tok = decode_from_kv(model, kv, next_id, plen, max_new_tokens=max_new, eos_ids=eos_ids)[0]
    tmp = Path(os.environ.get('TEMP', '/tmp')) / 'reasoner_kv.safetensors'
    nbytes = save_kv(kv, tmp)
    kv2 = load_kv(tmp, device, dtype=next(model.parameters()).dtype)
    c_tok = decode_from_kv(model, kv2, next_id, plen, max_new_tokens=max_new, eos_ids=eos_ids)[0]
    print(f"[C] kv save+load: {nbytes / 1000000.0:.1f} MB on disk for seq={meta['seq']} ({nbytes / 1000000.0 / meta['seq'] * 1000:.1f} MB / 1k-tok)")
    long_q = 'Here is context to reason over.\n' + ''.join((f'Fact {i}: warehouse robot {i} is assigned to aisle {i % 7}. ' for i in range(60))) + '\nQuestion: in one sentence, summarize the robots-to-aisles assignment pattern.'
    ltext = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': long_q}]}], tokenize=False, add_generation_prompt=True)
    linputs = proc(text=[ltext], return_tensors='pt').to(device)
    lplen = linputs['input_ids'].shape[1]
    kv_s, _, nid_s = prefill_export(model, linputs)
    d_single = decode_from_kv(model, kv_s, nid_s, lplen, max_new_tokens=max_new, eos_ids=eos_ids)[0]
    t0 = time.time()
    kv_c, meta_c, nid_c = prefill_chunked(model, linputs, chunk=64)
    d_chunk = decode_from_kv(model, kv_c, nid_c, lplen, max_new_tokens=max_new, eos_ids=eos_ids)[0]
    nd = min(d_single.shape[0], d_chunk.shape[0])
    cd = torch.equal(d_single[:nd].cpu(), d_chunk[:nd].cpu()) and nid_s.item() == nid_c.item()
    n = min(a_tok.shape[0], b_tok.shape[0])
    ab = torch.equal(a_tok[:n].cpu(), b_tok[:n].cpu())
    ac = torch.equal(a_tok[:n].cpu(), c_tok[:n].cpu())
    print(f'\n[A] {proc.decode(a_tok, skip_special_tokens=True)[:120]!r}')
    print(f'[B] {proc.decode(b_tok, skip_special_tokens=True)[:120]!r}')
    print(f'A==B (prefill/decode split): {ab}')
    print(f'A==C (split + serialize round-trip): {ac}')
    print(f"[D] {lplen}-tok prompt, chunked(64) prefill in {time.time() - t0:.1f}s ({meta_c['seq']} cache)")
    print(f'    chunked == single-shot prefill: {cd}')
    ok = ab and ac and cd
    print(f"\n{('PASS' if ok else 'FAIL')} — split exact, cache portable, chunked==single (long-ctx prefill OK).")
    return 0 if ok else 1

def _export(model_path, device, prompt, out_prefix):
    import json
    model, proc = _load(model_path, device)
    text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}], tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors='pt').to(device)
    plen = inputs['input_ids'].shape[1]
    t0 = time.time()
    kv, meta, next_id = prefill_export(model, inputs)
    nbytes = save_kv(kv, out_prefix + '.safetensors')
    with open(out_prefix + '.json', 'w') as f:
        json.dump({'prompt_len': int(plen), 'next_id': int(next_id.item()), 'prompt': prompt, 'meta': meta}, f)
    print(f'[export] prefilled {plen} tok on {device} in {time.time() - t0:.1f}s -> {out_prefix}.safetensors ({nbytes / 1000000.0:.1f} MB) + .json')
    return 0

def _decode(model_path, device, in_prefix, max_new):
    import json
    model, proc = _load(model_path, device)
    with open(in_prefix + '.json') as f:
        m = json.load(f)
    import torch
    kv = load_kv(in_prefix + '.safetensors', device, dtype=next(model.parameters()).dtype)
    next_id = torch.tensor([[m['next_id']]], device=device)
    eos = model.config.text_config.eos_token_id
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    t0 = time.time()
    toks = decode_from_kv(model, kv, next_id, m['prompt_len'], max_new_tokens=max_new, eos_ids=eos_ids)[0]
    print(f"[decode] resumed on {device} from a prefill of {m['prompt_len']} tok ({time.time() - t0:.1f}s)")
    print(f"[decode] prompt: {m['prompt']!r}")
    print(f'[decode] continuation:\n{proc.decode(toks, skip_special_tokens=True)}')
    return 0

def _load_quant(model_path, device, cache_dir):
    import os, sys
    from pathlib import Path
    _require_local_model(model_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from cosmos_windows import _cosmos_compat
    except Exception:
        pass
    import torch, torch.nn as nn
    import transformers_cosmos3
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration, Cosmos3OmniConfig
    from transformers import AutoProcessor, BitsAndBytesConfig
    cfg = Cosmos3OmniConfig.from_pretrained(model_path)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['lm_head'])
    cache_file = os.path.join(cache_dir, 'manual_4bit.safetensors')

    def _fresh():
        m = Cosmos3ForConditionalGeneration.from_pretrained(model_path, config=cfg, attn_implementation='sdpa', device_map=device, quantization_config=qcfg)
        return m.eval()

    def _save_manual(model):
        from safetensors.torch import save_file
        os.makedirs(cache_dir, exist_ok=True)
        sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        save_file(sd, cache_file)

    def _load_manual():
        import bitsandbytes as bnb
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors.torch import load_file
        sd = load_file(cache_file)
        with init_empty_weights():
            model = Cosmos3ForConditionalGeneration(cfg)
        for name, module in model.named_modules():
            for cn, child in list(module.named_children()):
                full = f'{name}.{cn}' if name else cn
                wkey = full + '.weight'
                if isinstance(child, nn.Linear) and wkey in sd:
                    pref = wkey + '.'
                    stats = {k[len(pref):]: sd[k].to(device) for k in sd if k.startswith(pref)}
                    if not stats:
                        continue
                    new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None, compute_dtype=torch.bfloat16, quant_type='nf4')
                    new.weight = bnb.nn.Params4bit.from_prequantized(data=sd[wkey].to(device), quantized_stats=stats, device=device)
                    if child.bias is not None and full + '.bias' in sd:
                        new.bias = nn.Parameter(sd[full + '.bias'].to(device))
                    setattr(module, cn, new)
        miss = []
        for pname, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p.device.type == 'meta':
                if pname in sd:
                    v = sd[pname]
                    set_module_tensor_to_device(model, pname, device, value=(v.to(torch.bfloat16) if v.is_floating_point() else v).to(device))
                else:
                    miss.append(pname)
        if miss:
            raise RuntimeError(f'manual cache missing {len(miss)} params (e.g. {miss[:3]})')
        for module in model.modules():
            for pn, pp in list(module._parameters.items()):
                if pp is not None and (not isinstance(pp, bnb.nn.Params4bit)) and pp.is_floating_point() and (pp.dtype != torch.bfloat16):
                    module._parameters[pn] = nn.Parameter(pp.to(torch.bfloat16), requires_grad=False)
        return model.eval()

    def _probe(m):
        with torch.no_grad():
            m(input_ids=torch.tensor([[1, 2, 3]], device=device), use_cache=False)
    if os.path.exists(cache_file):
        print(f'[quant] reusing 4-bit reasoner from DISK CACHE: {cache_file}', flush=True)
        try:
            m = _load_manual()
            _probe(m)
            print('[quant] disk-cache OK (reused, no re-quantize)', flush=True)
            return (m, proc)
        except Exception as e:
            print(f'[quant] disk-cache unusable ({type(e).__name__}: {str(e)[:140]}); re-quantizing fresh', flush=True)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    print(f'[quant] quantizing 4-bit reasoner from full weights (will cache): {model_path}', flush=True)
    model = _fresh()
    try:
        _save_manual(model)
        print(f'[quant] cached 4-bit reasoner -> {cache_file} (reused on later runs)', flush=True)
    except Exception as e:
        print(f'[quant] (cache save skipped: {type(e).__name__}: {str(e)[:120]})', flush=True)
    return (model, proc)

def _load_reason_images(image_paths):
    if not image_paths:
        return []
    from PIL import Image
    return [Image.open(path).convert('RGB') for path in image_paths]

def _run_resident(model, proc, device, prompt, max_new, tag, image_paths=None, temperature=0.0, top_p=1.0, system=None):
    import time, json, torch
    images = _load_reason_images(image_paths or [])
    content = [{'type': 'image', 'image': img} for img in images]
    if len(images) > 1:
        prompt = 'The attached images are sampled frames from the requested video.\n\n' + prompt
    content.append({'type': 'text', 'text': prompt})
    messages = []
    if system and str(system).strip():
        messages.append({'role': 'system', 'content': [{'type': 'text', 'text': str(system).strip()}]})
    messages.append({'role': 'user', 'content': content})
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kw = dict(text=[text], return_tensors='pt')
    if images:
        kw['images'] = images
    inputs = proc(**kw).to(device)
    plen = inputs['input_ids'].shape[1]
    eos = model.config.text_config.eos_token_id
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    t0 = time.time()
    kv, meta, next_id = prefill_export(model, inputs)
    toks = decode_from_kv(model, kv, next_id, plen, max_new_tokens=max_new, eos_ids=eos_ids, temperature=temperature, top_p=top_p)[0]
    ans = proc.decode(toks, skip_special_tokens=True)
    secs = round(time.time() - t0, 1)
    print(f'[{tag}] prefill+decode on {device}: {plen} prompt tok, {toks.shape[0]} new tok, {secs}s')
    print(f'[{tag}] answer:\n{ans}')
    print('ANSWER_JSON', json.dumps({'answer': ans, 'prefill_tokens': int(plen), 'seconds': secs}))
    return 0

def _local(model_path, device, prompt, max_new, image_paths=None, temperature=0.0, top_p=1.0, system=None):
    model, proc = _load(model_path, device)
    return _run_resident(model, proc, device, prompt, max_new, 'local', image_paths=image_paths, temperature=temperature, top_p=top_p, system=system)

def _quant(model_path, device, prompt, max_new, image_paths=None, temperature=0.0, top_p=1.0, system=None):
    import os
    cache_dir = os.path.join(model_path, '.quant4_manual')
    model, proc = _load_quant(model_path, device, cache_dir)
    return _run_resident(model, proc, device, prompt, max_new, 'quant', image_paths=image_paths, temperature=temperature, top_p=top_p, system=system)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='selftest', choices=['selftest', 'export', 'decode', 'local', 'quant'])
    ap.add_argument('--model', default='')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'])
    ap.add_argument('--prompt', default='Plan, in 4 short numbered steps, how a mobile manipulator clears a table of dishes.')
    ap.add_argument('--out', default='kv_demo', help='export: output prefix')
    ap.add_argument('--in', dest='in_prefix', default='kv_demo', help='decode: input prefix')
    ap.add_argument('--max-new-tokens', type=int, default=40)
    ap.add_argument('--gpu-index', default=os.environ.get('STUDIO_GPU_INDEX', '0'))
    ap.add_argument('--image', action='append', default=[], help='optional image/frame context; repeat for multiple sampled video frames')
    ap.add_argument('--temperature', type=float, default=0.0, help='local/quant: >0 samples (nucleus top-p); 0 (default) = greedy, unchanged')
    ap.add_argument('--top-p', dest='top_p', type=float, default=1.0, help='local/quant: nucleus top-p when sampling')
    ap.add_argument('--system', default=None, help='local/quant: optional system prompt prepended to the reason turn')
    args = ap.parse_args()
    if args.device == 'cuda':
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_index)
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.mode == 'export':
        return _export(args.model, args.device, args.prompt, args.out)
    if args.mode == 'decode':
        return _decode(args.model, args.device, args.in_prefix, args.max_new_tokens)
    if args.mode == 'local':
        return _local(args.model, args.device, args.prompt, args.max_new_tokens, image_paths=args.image, temperature=args.temperature, top_p=args.top_p, system=args.system)
    if args.mode == 'quant':
        return _quant(args.model, args.device, args.prompt, args.max_new_tokens, image_paths=args.image, temperature=args.temperature, top_p=args.top_p, system=args.system)
    return _selftest(args.model, args.device, args.max_new_tokens)
if __name__ == '__main__':
    raise SystemExit(main())
