# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/merge_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os, json, glob, shutil, argparse
import torch
from safetensors.torch import load_file, save_file

def adapter_deltas(adapter_file: str, scale: float):
    ad = load_file(adapter_file)
    mods = {}
    for k, v in ad.items():
        if '.lora_A.' in k:
            mods.setdefault(k.split('.lora_A.')[0], {})['A'] = v.float()
        elif '.lora_B.' in k:
            mods.setdefault(k.split('.lora_B.')[0], {})['B'] = v.float()
    deltas = {f'{m}.weight': d['B'] @ d['A'] * scale for m, d in mods.items() if 'A' in d and 'B' in d}
    return (deltas, len(ad))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='')
    ap.add_argument('--adapter', default='')
    ap.add_argument('--out', default=None, help='merged transformer dir; omit with --dry-run')
    ap.add_argument('--scale', type=float, default=1.0, help='lora_alpha / r (trainer used alpha==r -> 1.0)')
    ap.add_argument('--dry-run', action='store_true', help='validate mapping + deltas WITHOUT writing 32GB')
    a = ap.parse_args()
    deltas, n_ad = adapter_deltas(a.adapter, a.scale)
    idx = json.load(open(os.path.join(a.base, 'diffusion_pytorch_model.safetensors.index.json')))
    wm = idx['weight_map']
    print(f'[merge] {len(deltas)} weight-deltas from {n_ad} adapter tensors; scale={a.scale}', flush=True)
    missing = [k for k in deltas if k not in wm]
    if missing:
        print(f'[merge] WARNING {len(missing)} delta keys not in base index, e.g. {missing[:3]}', flush=True)
    shards = sorted(set((wm[k] for k in deltas)))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    merged = 0
    sample = None
    all_shards = sorted(set(wm.values()))
    for shard in all_shards if a.out else shards:
        sd = load_file(os.path.join(a.base, shard))
        for wkey in list(sd.keys()):
            if wkey in deltas:
                W = sd[wkey]
                d = deltas[wkey]
                if tuple(d.shape) != tuple(W.shape):
                    print(f'[merge] SHAPE MISMATCH {wkey}: delta{tuple(d.shape)} vs W{tuple(W.shape)}', flush=True)
                    continue
                newW = (W.float() + d).to(W.dtype)
                if sample is None:
                    sample = (wkey, (newW.float() - W.float()).abs().mean().item(), d.abs().mean().item())
                sd[wkey] = newW
                merged += 1
        if a.out:
            save_file(sd, os.path.join(a.out, shard))
    if a.out:
        for f in glob.glob(os.path.join(a.base, '*.json')):
            shutil.copy(f, a.out)
    print(f'[merge] merged {merged}/{len(deltas)} weights' + (f' -> {a.out} ({len(all_shards)} shards)' if a.out else ' (dry-run, not written)'), flush=True)
    if sample:
        k, applied, expected = sample
        print(f'[merge] sample {k}: applied mean|d|={applied:.6f} vs expected {expected:.6f} match={abs(applied - expected) < expected * 0.05 + 1e-06}', flush=True)
    ok = merged == len(deltas) and (not missing) and sample and (abs(sample[1] - sample[2]) < sample[2] * 0.05 + 1e-06)
    print('RESULT', 'PASS' if ok else 'CHECK', flush=True)
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
