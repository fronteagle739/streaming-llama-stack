# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/wan_lora_merge.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import struct
import time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
_ST = {torch.float64: 'F64', torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16', torch.uint8: 'U8', torch.int8: 'I8'}
_ST_INV = {v: k for k, v in _ST.items()}

def _raw(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    return t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()

def _spec(l, k, d=None):
    return (l.get(k) if isinstance(l, dict) else getattr(l, k, d)) if l is not None else d

def _lora_index(loras, expert: str):
    idx: dict = {}
    used = []
    for l in loras or []:
        tgt = _spec(l, 'expert', 'both') or 'both'
        if tgt not in (expert, 'both'):
            continue
        path = _spec(l, 'path')
        w = float(_spec(l, 'weight', 1.0) or 1.0)
        if not path:
            continue
        used.append({'path': str(path), 'strength': w, 'expert': tgt})
        sd = load_file(path)
        alphas, bases = ({}, {})
        for k, v in sd.items():
            bare = k[len('diffusion_model.'):] if k.startswith('diffusion_model.') else k
            for suf in ('.lora_down.weight', '.lora_up.weight', '.diff_b', '.diff', '.alpha'):
                if bare.endswith(suf):
                    base = bare[:-len(suf)]
                    (alphas if suf == '.alpha' else bases.setdefault(base, {}))[base if suf == '.alpha' else suf] = float(v.flatten()[0]) if suf == '.alpha' else v
                    break
        for base, parts in bases.items():
            idx.setdefault(base, []).append((w, parts, alphas.get(base)))
    return (idx, used)

def _delta_for(bare_name: str, lora_idx: dict, device, dtype):
    if bare_name.endswith('.weight'):
        base = bare_name[:-len('.weight')]
        entries = lora_idx.get(base)
        if not entries:
            return None
        acc = None
        for w, parts, alpha in entries:
            if '.lora_down.weight' in parts and '.lora_up.weight' in parts:
                dn = parts['.lora_down.weight'].to(device, torch.float32)
                up = parts['.lora_up.weight'].to(device, torch.float32)
                s = w * (alpha / dn.shape[0] if alpha else 1.0)
                d = s * (up @ dn)
                acc = d if acc is None else acc + d
            if '.diff' in parts:
                d = w * parts['.diff'].to(device, torch.float32)
                acc = d if acc is None else acc + d
        return acc
    if bare_name.endswith('.bias'):
        base = bare_name[:-len('.bias')]
        entries = lora_idx.get(base)
        if not entries:
            return None
        acc = None
        for w, parts, _a in entries:
            if '.diff_b' in parts:
                d = w * parts['.diff_b'].to(device, torch.float32)
                acc = d if acc is None else acc + d
        return acc
    return None

def marker_path(out_file: str) -> Path:
    return Path(str(out_file) + '.lora_state.json')

def lora_state(path: str) -> dict | None:
    mp = marker_path(path)
    if mp.is_file():
        try:
            return json.loads(mp.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            return None
    return None

def _stream_apply(in_file: str, out_file: str, lora_idx: dict, sign: float, device: str):
    with safe_open(in_file, framework='pt', device='cpu') as f:
        keys = [k for k in f.keys() if not (k.endswith('.comfy_quant') or k.endswith('.weight_scale'))]
        header, off, plan = ({}, 0, [])
        for k in keys:
            sl = f.get_slice(k)
            shape = list(sl.get_shape())
            tdt = _ST_INV[sl.get_dtype()]
            ne = 1
            for s in shape:
                ne *= int(s)
            nb = ne * torch.empty(0, dtype=tdt).element_size()
            header[k] = {'dtype': _ST[tdt], 'shape': shape, 'data_offsets': [off, off + nb]}
            plan.append((k, tdt, nb))
            off += nb
        hj = json.dumps(header, separators=(',', ':')).encode('utf-8')
        hj += b' ' * (-len(hj) % 8)
        n = 0
        with open(out_file, 'wb') as out:
            out.write(struct.pack('<Q', len(hj)))
            out.write(hj)
            for k, tdt, nb in plan:
                t = f.get_tensor(k).to(device)
                d = _delta_for(k, lora_idx, device, tdt)
                if d is not None:
                    t = (t.float() + sign * d).to(tdt)
                    n += 1
                raw = _raw(t)
                if len(raw) != nb:
                    raise RuntimeError(f'{k}: {len(raw)} != {nb}')
                out.write(raw)
                del t
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
    return n

def merge_to_disk(base_file: str, out_file: str, loras: list, expert: str, device: str='cuda') -> dict:
    lora_idx, used = _lora_index(loras, expert)
    if not used:
        raise ValueError('no loras for expert=%r' % expert)
    t0 = time.time()
    n = _stream_apply(base_file, out_file, lora_idx, +1.0, device)
    marker = {'base_source': str(base_file), 'expert': expert, 'merged_loras': used, 'tensors_modified': n, 'tool': 'wan_lora_merge.merge_to_disk'}
    marker_path(out_file).write_text(json.dumps(marker, indent=2), encoding='utf-8')
    print(f'[wan-lora-merge] {Path(out_file).name}: baked {len(used)} lora(s) into {n} tensors in {time.time() - t0:.1f}s; marker {marker_path(out_file).name}', flush=True)
    return marker

def unmerge(merged_file: str, out_file: str | None=None, device: str='cuda') -> str:
    st = lora_state(merged_file)
    if not st:
        raise ValueError(f'no .lora_state.json marker beside {merged_file} — nothing to unmerge')
    lora_idx, _ = _lora_index(st.get('merged_loras', []), st.get('expert', 'both'))
    out_file = out_file or str(Path(merged_file).with_suffix('.unmerged.safetensors'))
    n = _stream_apply(merged_file, out_file, lora_idx, -1.0, device)
    marker_path(merged_file).unlink(missing_ok=True)
    print(f'[wan-lora-merge] unmerged {n} tensors -> {out_file} (marker dropped)', flush=True)
    return out_file
