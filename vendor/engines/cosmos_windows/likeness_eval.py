# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/likeness_eval.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os, glob, argparse
_IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def _load_clip(device='cuda'):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    return (model, proc, device)

def _embed(paths, clip, device):
    import torch
    from PIL import Image
    model, proc = clip
    imgs = [Image.open(p).convert('RGB') for p in paths]
    pv = proc(images=imgs, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=pv)
    if torch.is_tensor(out):
        f = out
    elif getattr(out, 'image_embeds', None) is not None:
        f = out.image_embeds
    elif getattr(out, 'pooler_output', None) is not None:
        f = out.pooler_output
    else:
        f = out.last_hidden_state.mean(1)
    return torch.nn.functional.normalize(f, dim=-1)

def _paths(spec):
    if os.path.isdir(spec):
        return [p for p in sorted(glob.glob(os.path.join(spec, '*.*'))) if os.path.splitext(p)[1].lower() in _IMG_EXT]
    return [spec] if os.path.splitext(spec)[1].lower() in _IMG_EXT else []

def evaluate(ref_dir, gen, baseline=None, device='cuda'):
    import torch
    model, proc, device = _load_clip(device)
    clip = (model, proc)
    refs = _embed(_paths(ref_dir), clip, device)
    g = _embed(_paths(gen), clip, device).mean(0, keepdim=True)
    gen_vs_ref = (g @ refs.T).mean().item()
    out = {'gen_vs_ref': round(gen_vs_ref, 4), 'n_ref': refs.shape[0]}
    if baseline:
        b = _embed(_paths(baseline), clip, device).mean(0, keepdim=True)
        base_vs_ref = (b @ refs.T).mean().item()
        out['baseline_vs_ref'] = round(base_vs_ref, 4)
        out['delta'] = round(gen_vs_ref - base_vs_ref, 4)
        out['verdict'] = 'PASS' if gen_vs_ref > base_vs_ref else 'FAIL'
    return out

def _selftest():
    import torch, tempfile
    from PIL import Image
    ref = ''
    if not os.path.exists(ref):
        print('[likeness] selftest SKIP (no tifa.jpg)')
        return True
    try:
        model, proc, device = _load_clip('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f'[likeness] selftest SKIP (CLIP unavailable offline: {type(e).__name__}) — tool is ready, needs net once')
        return True
    clip = (model, proc)
    d = tempfile.mkdtemp()
    noise = os.path.join(d, 'noise.png')
    Image.fromarray(torch.rand(256, 256, 3).mul(255).byte().numpy()).save(noise)
    same = (_embed([ref], clip, device) @ _embed([ref], clip, device).T).item()
    diff = (_embed([ref], clip, device) @ _embed([noise], clip, device).T).item()
    ok = same > 0.99 and same > diff
    print(f"[likeness] selftest: sim(tifa,tifa)={same:.3f} > sim(tifa,noise)={diff:.3f} -> {('PASS' if ok else 'FAIL')}")
    return ok
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', help='reference image(s) of the subject (dir or single file)')
    ap.add_argument('--gen', help='generated image(s) from the LoRA model (dir or file)')
    ap.add_argument('--baseline', default=None, help='generated image(s) from the BASE model (for the delta)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest or not (a.ref_dir and a.gen):
        import sys
        sys.exit(0 if _selftest() else 1)
    import json
    print(json.dumps(evaluate(a.ref_dir, a.gen, a.baseline), indent=2))
