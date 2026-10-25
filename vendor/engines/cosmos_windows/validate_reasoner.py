# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_reasoner.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import sys
import time

def _make_count_image(path: str, n: int=3):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (448, 448), 'white')
    d = ImageDraw.Draw(img)
    xs = [80, 230, 360, 150, 300][:n]
    ys = [120, 90, 250, 320, 200][:n]
    for x, y in zip(xs, ys):
        d.ellipse([x - 45, y - 45, x + 45, y + 45], fill='red')
    img.save(path)
    return path

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'])
    ap.add_argument('--gpu-index', default=os.environ.get('STUDIO_GPU_INDEX', '0'), help='Explicit CUDA device index in PCI bus order')
    ap.add_argument('--max-new-tokens', type=int, default=160)
    ap.add_argument('--skip-vqa', action='store_true')
    args = ap.parse_args()
    if args.device == 'cuda':
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_index)
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from cosmos_windows import _cosmos_compat
    except Exception as e:
        print('[reasoner] _cosmos_compat shim skipped:', type(e).__name__, str(e)[:80])
    import torch
    import transformers_cosmos3
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration, Cosmos3OmniConfig
    from transformers import AutoProcessor
    dev = args.device
    dtype = torch.bfloat16 if dev != 'cpu' else torch.float32
    cfg = Cosmos3OmniConfig.from_pretrained(args.model)
    print(f'[reasoner] config: text {cfg.text_config.num_hidden_layers}L hidden={cfg.text_config.hidden_size} kv={cfg.text_config.num_key_value_heads} inter={cfg.text_config.intermediate_size}; vision out_hidden={cfg.vision_config.out_hidden_size}', flush=True)
    t0 = time.time()
    print(f'[reasoner] loading on {dev} ...', flush=True)
    model = Cosmos3ForConditionalGeneration.from_pretrained(args.model, config=cfg, torch_dtype=dtype, attn_implementation='sdpa', device_map=dev)
    model.eval()
    nparams = sum((p.numel() for p in model.parameters()))
    wgt = sum((p.numel() * p.element_size() for p in model.parameters())) / 1000000000.0
    print(f'[reasoner] LOADED in {time.time() - t0:.0f}s  params={nparams / 1000000000.0:.2f}B  weights={wgt:.1f}GB', flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ok = True
    q = 'A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Think step by step, then state the final number.'
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors='pt').to(dev)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    ans = proc.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'\n[reasoner] === TEXT REASONING ({time.time() - t1:.1f}s) ===\n{ans.strip()}')
    ok &= '9' in ans
    if not args.skip_vqa:
        from PIL import Image
        img_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'reasoner_vqa_circles.png')
        _make_count_image(img_path, n=3)
        vmsgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img_path}, {'type': 'text', 'text': 'How many red circles are in this image? Answer with just the number.'}]}]
        vtext = proc.apply_chat_template(vmsgs, tokenize=False, add_generation_prompt=True)
        vinputs = proc(text=[vtext], images=[Image.open(img_path)], return_tensors='pt').to(dev)
        t2 = time.time()
        with torch.no_grad():
            vout = model.generate(**vinputs, max_new_tokens=32, do_sample=False)
        vans = proc.decode(vout[0][vinputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f'\n[reasoner] === VQA: count 3 red circles ({time.time() - t2:.1f}s) ===\n{vans.strip()}')
        ok &= '3' in vans or 'three' in vans.lower()
    print(f"\n[reasoner] {('PASS' if ok else 'FAIL')} — AR reasoner functional on {dev}")
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
