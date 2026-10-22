# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_ideogram_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'cosmos_windows'
from .optim_factory import build_optimizer, MultiPhaseLR
from .streaming_flag import add_streaming_arg, resolve_streaming
QUANT_REPOS = {'nf4': 'ideogram-ai/ideogram-4-nf4', 'fp8': 'ideogram-ai/ideogram-4-fp8'}
_LOCAL_EXPORT_DEFAULTS = {'fp8': [os.environ.get('STUDIO_IDEOGRAM_FP8_DIR'), ''], 'nf4': [os.environ.get('STUDIO_IDEOGRAM_NF4_DIR'), '']}

def _resolve_model(model, quant='nf4'):
    if model:
        return str(model)
    root = os.environ.get('STUDIO_IDEOGRAM_EXPORT_ROOT')
    candidates = list(_LOCAL_EXPORT_DEFAULTS.get(quant, []))
    if root:
        candidates.insert(0, str(Path(root) / f'ideogram-4-{quant}'))
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return cand
    return QUANT_REPOS.get(quant, QUANT_REPOS['nf4'])
DEFAULT_MODEL = None

def _targets(train_mlp: bool):
    t = ['qkv', 'o']
    if train_mlp:
        t += ['w1', 'w2', 'w3']
    return t

def _pick_device():
    import torch
    if not torch.cuda.is_available():
        return 'cpu'
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(best)
    return f'cuda:{best}'

def _patch_load_file_pread(_ideo_pipeline):
    import inspect
    from safetensors.torch import load_file as _orig_load_file
    if getattr(_ideo_pipeline.load_file, '_pread_patched', False):
        return lambda: None
    _fork = 'backend' in inspect.signature(_orig_load_file).parameters
    if not _fork:
        print('[ideo-train] WARNING: stock safetensors (no backend= kwarg) — pread host-RSS fix inactive')

    def _pread_load_file(filename, device='cpu', *, backend='pread', **kw):
        if not _fork:
            return _orig_load_file(filename, device=device, **kw)
        return _orig_load_file(filename, device=device, backend='pread', **kw)
    _pread_load_file._pread_patched = True
    _prev = _ideo_pipeline.load_file
    _ideo_pipeline.load_file = _pread_load_file

    def _restore():
        _ideo_pipeline.load_file = _prev
    return _restore

def build_pipe(model, device='cuda', quant='nf4'):
    import torch
    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
    import ideogram4.pipeline_ideogram4 as _ideo_pipeline
    _patch_load_file_pread(_ideo_pipeline)
    repo = _resolve_model(model, quant=quant)
    if Path(repo).exists():
        _hub = _ideo_pipeline.hf_hub_download

        def _local_aware(*args, **kwargs):
            rid = kwargs.pop('repo_id', args[0] if args else None)
            fn = kwargs.pop('filename', args[1] if len(args) > 1 else None)
            root = Path(str(rid))
            if root.exists() and fn:
                cand = root.joinpath(*[p for p in str(fn).replace('\\', '/').split('/') if p])
                if cand.is_file():
                    return str(cand)
                raise _ideo_pipeline.EntryNotFoundError(f'local Ideogram file not found: {cand}')
            return _hub(*args, **kwargs) if args else _hub(repo_id=rid, filename=fn, **kwargs)
        _ideo_pipeline.hf_hub_download = _local_aware
    if device.startswith('cuda') and ':' in device:
        try:
            torch.cuda.set_device(int(device.split(':', 1)[1]))
        except Exception:
            pass
    cfg = Ideogram4PipelineConfig(weights_repo=repo)
    if Path(repo).exists():

        def _has(rel):
            return Path(repo).joinpath(*rel.split('/')).is_file()
        uncond_idx = cfg.unconditional_index_filename
        uncond_single = uncond_idx.removesuffix('.index.json')
        if not _has(uncond_idx) and (not _has(uncond_single)):
            print('[ideo-train] unconditional_transformer weights absent locally -> reusing conditional weights for the (eval-only) uncond tower (training LoRAs the conditional tower).', flush=True)
            cfg.unconditional_index_filename = cfg.conditional_index_filename
    return Ideogram4Pipeline.from_pretrained(config=cfg, device=device, dtype=torch.bfloat16)

def _make_lora_linear(base, rank, alpha, in_features, out_features, device):
    import torch, torch.nn as nn

    class LoRALinear(nn.Module):

        def __init__(self):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
            nn.init.normal_(self.lora_A, std=1.0 / rank)
            self.scaling = float(alpha) / float(rank)

        def forward(self, x):
            out = self.base(x)
            delta = x.to(torch.float32) @ self.lora_A.t() @ self.lora_B.t()
            return out + self.scaling * delta.to(out.dtype)
    m = LoRALinear()
    for p in m.base.parameters(recurse=True):
        p.requires_grad_(False)
    return m

def _infer_linear_shape(mod):
    inf = getattr(mod, 'in_features', None)
    outf = getattr(mod, 'out_features', None)
    if inf is None or outf is None:
        w = getattr(mod, 'weight', None)
        if w is not None and w.dim() == 2:
            outf, inf = (int(w.shape[0]), int(w.shape[1]))
    return (inf, outf)

def _enable_grad_checkpointing(tf):
    import torch
    from torch.utils.checkpoint import checkpoint
    layers = getattr(tf, 'layers', None)
    if layers is None:
        return 0
    n = 0
    for layer in layers:
        if getattr(layer, '_ckpt_wrapped', False):
            continue
        _orig = layer.forward

        def _ckpt_forward(*args, _orig=_orig, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)
        layer.forward = _ckpt_forward
        layer._ckpt_wrapped = True
        n += 1
    if n:
        print(f'[ideo-train] gradient checkpointing enabled on {n} transformer blocks (recompute fp8 dequant in backward — bounds VRAM).', flush=True)
    return n

def _block_base_tensors(layer):
    out = []
    for mod in layer.modules():
        for pn, p in list(mod._parameters.items()):
            if p is None or p.requires_grad:
                continue
            out.append((mod, pn, True))
        for bn, b in list(mod._buffers.items()):
            if b is None:
                continue
            out.append((mod, bn, False))
    return out

def _enable_layer_streaming(tf, device):
    import torch
    from cosmos_windows.streamed_backward import apply_streamed_multi
    layers = getattr(tf, 'layers', None)
    if layers is None:
        return 0

    def make_bind(tensors):

        def bind():
            for mod, name, is_param in tensors:
                store = mod._parameters if is_param else mod._buffers
                t = store[name]
                if t is not None and t.device.type != device.split(':')[0]:
                    t.data = t.data.to(device)
        return bind

    def make_evict(tensors):

        def evict():
            for mod, name, is_param in tensors:
                store = mod._parameters if is_param else mod._buffers
                t = store[name]
                if t is not None and t.device.type != 'cpu':
                    t.data = t.data.to('cpu')
        return evict
    n = 0
    for layer in layers:
        if getattr(layer, '_sb_wrapped', False):
            continue
        tensors = _block_base_tensors(layer)
        for mod, name, is_param in tensors:
            store = mod._parameters if is_param else mod._buffers
            t = store[name]
            if t is not None and t.device.type != 'cpu':
                t.data = t.data.to('cpu')
        bind, evict = (make_bind(tensors), make_evict(tensors))
        _orig = layer.forward

        def _streamed_forward(x, segment_ids=None, cos=None, sin=None, adaln_input=None, _orig=_orig, _b=bind, _e=evict, _layer=layer):
            call = lambda lyr, h: _orig(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
            return apply_streamed_multi(_layer, call, x, bind=_b, evict=_e)
        layer.forward = _streamed_forward
        layer._sb_wrapped = True
        n += 1
    print(f'[ideo-train] STREAMING=ON: {n} transformer blocks layer-streamed CPU<->{device} (fwd+bwd, ~1 block base resident at a time) — manual LoRA stays resident.', flush=True)
    return n

def build(model, device, rank=16, alpha=None, train_mlp=False, quant='nf4', streaming=False):
    import torch
    pipe = build_pipe(model, device=device, quant=quant)
    tf = pipe.conditional_transformer
    alpha = alpha or rank
    tf.requires_grad_(False)
    dev = torch.device(device)
    targets = set(_targets(train_mlp))
    to_wrap = []
    for name, child in tf.named_modules():
        for cn, sub in list(child.named_children()):
            if cn in targets and hasattr(sub, 'forward'):
                to_wrap.append((child, cn, sub))
    lora_params = []
    wrapped = 0
    for parent, cn, sub in to_wrap:
        inf, outf = _infer_linear_shape(sub)
        if not inf or not outf:
            continue
        w = _make_lora_linear(sub, rank, alpha, inf, outf, dev)
        setattr(parent, cn, w)
        lora_params += [w.lora_A, w.lora_B]
        wrapped += 1
    print(f'[ideo-train] manual LoRA(r={rank}, alpha={alpha}) on {sorted(targets)}: wrapped {wrapped} Linears, {sum((p.numel() for p in lora_params)) / 1000000.0:.2f}M trainable (fp32)', flush=True)
    if not lora_params:
        raise RuntimeError(f'no target Linears wrapped (targets={sorted(targets)}) — check module names')
    if streaming:
        _enable_layer_streaming(tf, device)
    else:
        _enable_grad_checkpointing(tf)
    if getattr(pipe, 'unconditional_transformer', None) is not None:
        pipe.unconditional_transformer = None
        torch.cuda.empty_cache() if device and str(device).startswith('cuda') else None
        print('[ideo-train] freed unconditional_transformer (~9GB VRAM) — not used in training.', flush=True)
    return (pipe, lora_params)

def build_pack(pipe, prompt, height, width):
    import torch
    inputs = pipe._build_inputs([prompt], height=height, width=width)
    with torch.no_grad():
        llm_features = pipe._encode_text(inputs['token_ids'], inputs['text_position_ids'], inputs['indicator'])
    return {'inputs': inputs, 'llm_features': llm_features, 'max_text_tokens': int(inputs['max_text_tokens']), 'num_image_tokens': int(inputs['num_image_tokens']), 'grid_h': int(inputs['grid_h']), 'grid_w': int(inputs['grid_w'])}

def _latent_dim(pipe):
    return int(pipe.conditional_transformer.config.in_channels)

def encode_image(pipe, image_path, height, width, pack):
    import torch
    from PIL import Image
    try:
        patch = pipe.config.patch_size
        ae_scale = pipe.config.ae_scale_factor
        grid_h, grid_w = (pack['grid_h'], pack['grid_w'])
        H, W = (grid_h * patch * ae_scale, grid_w * patch * ae_scale)
        img = Image.open(image_path).convert('RGB').resize((W, H))
        import numpy as np
        arr = np.asarray(img).astype('float32') / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(pipe.device, pipe.dtype)
        with torch.no_grad():
            enc = pipe.autoencoder.encoder(x)
            z_ch = enc.shape[1] // 2
            mean = enc[:, :z_ch]
            b = mean.shape[0]
            mean = mean.view(b, z_ch, grid_h, patch, grid_w, patch)
            mean = mean.permute(0, 2, 4, 3, 5, 1).contiguous()
            z = mean.view(b, grid_h * grid_w, patch * patch * z_ch)
            z = (z.float() - pipe.latent_shift) / pipe.latent_scale
        if z.shape[1] != pack['num_image_tokens'] or z.shape[-1] != _latent_dim(pipe):
            raise ValueError(f"encoded latent shape {tuple(z.shape)} does not match tokens={pack['num_image_tokens']} channels={_latent_dim(pipe)}")
        return z.float()
    except Exception as e:
        raise RuntimeError(f'VAE encode failed for {image_path}: {type(e).__name__}: {e}') from e

def flow_step(pipe, pack, z0, opt, lora_params, t_lo=0.05, t_hi=0.95):
    import torch
    dev = pipe.device
    mtt = pack['max_text_tokens']
    inputs = pack['inputs']
    bsz = z0.shape[0]
    t = torch.rand((), device=dev).clamp(t_lo, t_hi)
    noise = torch.randn_like(z0)
    z_t = (1.0 - t) * z0 + t * noise
    target_v = noise - z0
    text_pad = torch.zeros(bsz, mtt, z0.shape[-1], dtype=z_t.dtype, device=dev)
    x = torch.cat([text_pad, z_t], dim=1)
    t_vec = torch.full((bsz,), float(t), dtype=torch.float32, device=dev)
    opt.zero_grad(set_to_none=True)
    pred = pipe.conditional_transformer(llm_features=pack['llm_features'], x=x, t=t_vec, position_ids=inputs['position_ids'], segment_ids=inputs['segment_ids'], indicator=inputs['indicator'])
    pred_img = pred[:, mtt:].float()
    loss = torch.nn.functional.mse_loss(pred_img, target_v.float())
    loss.backward()
    ng = sum((1 for p in lora_params if p.grad is not None))
    gnorm = math.sqrt(sum((p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None)))
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt.step()
    return (loss.item(), ng, gnorm)

def save_lora(pipe, out_dir):
    import os
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().float().cpu() for n, p in pipe.conditional_transformer.named_parameters() if 'lora_' in n}
    save_file(sd, os.path.join(out_dir, 'ideogram_lora.safetensors'))
    return len(sd)

def _selftest(model, quant='nf4', streaming=False):
    import torch
    device = _pick_device()
    pipe, lora_params = build(model, device, rank=8, quant=quant, streaming=streaming)
    opt = build_optimizer(lora_params, 'adafactor', lr=0.001)
    sched = MultiPhaseLR(opt, '0.001@0,0.0005@2', default_lr=0.001)
    pack = build_pack(pipe, 'a red apple on a wooden table', 512, 512)
    print(f"[ideo-train] pack: max_text={pack['max_text_tokens']} img_tokens={pack['num_image_tokens']} latent_dim={_latent_dim(pipe)} grid=({pack['grid_h']},{pack['grid_w']})", flush=True)
    if getattr(pipe, 'text_encoder', None) is not None:
        pipe.text_encoder = None
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print('[ideo-train] selftest: freed text_encoder (~8GB VRAM) after one-time packing.', flush=True)
    z0 = torch.randn(1, pack['num_image_tokens'], _latent_dim(pipe), device=device, dtype=torch.float32)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    before = [p.detach().clone() for p in lora_params]
    losses = []
    for step in range(3):
        sched.step(step)
        loss, ng, gnorm = flow_step(pipe, pack, z0, opt, lora_params)
        losses.append(loss)
    moved = sum(((p.detach() - b).abs().sum().item() for p, b in zip(lora_params, before)))
    peak = torch.cuda.max_memory_allocated() / 1000000000.0 if device.startswith('cuda') else 0.0
    finite = all((math.isfinite(x) for x in losses))
    ok = ng == len(lora_params) and finite and (gnorm > 0) and (moved > 0)
    mode = 'STREAMING(fwd+bwd layer-streamed)' if streaming else 'RESIDENT(grad-ckpt)'
    print(f'[ideo-train] [{mode}] flow steps loss={[round(x, 4) for x in losses]} grads={ng}/{len(lora_params)} gnorm={gnorm:.4f} moved={moved:.2e} peak={peak:.1f}GB', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1

def train(model, data_path, out_dir, steps, lr, rank, train_mlp, optimizer='adamw', lr_phases='', checkpoint_every=0, quant='nf4', streaming=False):
    import torch
    device = _pick_device()
    print(f"[ideo-train] device={device} streaming={('on' if streaming else 'off')}", flush=True)
    pipe, lora_params = build(model, device, rank=rank, train_mlp=train_mlp, quant=quant, streaming=streaming)
    opt = build_optimizer(lora_params, optimizer, lr=lr)
    sched = MultiPhaseLR(opt, lr_phases, default_lr=lr)
    data = [json.loads(l) for l in open(data_path, encoding='utf-8') if l.strip()]
    if not data:
        raise ValueError('training dataset is empty')
    for index, ex in enumerate(data):
        image = ex.get('image')
        if not image:
            raise ValueError(f'training example {index} has no image')
        if not os.path.isfile(image):
            raise FileNotFoundError(f'training example {index} image does not exist: {image}')
    pack_cache = {}
    t0 = time.time()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        sched.step(step)
        ex = data[step % len(data)]
        h, w = (ex.get('height', 1024), ex.get('width', 1024))
        prompt = ex.get('prompt', '')
        key = (prompt, h, w)
        if key not in pack_cache:
            pack_cache[key] = build_pack(pipe, prompt, h, w)
        pack = pack_cache[key]
        z0 = encode_image(pipe, ex['image'], h, w, pack)
        loss, ng, gnorm = flow_step(pipe, pack, z0, opt, lora_params)
        if step % 10 == 0 or step == steps - 1:
            pk = torch.cuda.max_memory_allocated() / 1000000000.0 if device.startswith('cuda') else 0.0
            print(f'[ideo-train] step {step + 1}/{steps} loss={loss:.4f} lr={sched.current_lr():g} ({(time.time() - t0) / (step + 1):.2f}s/step)' + (f' peak={pk:.1f}GB' if pk else ''), flush=True)
        if checkpoint_every and (step + 1) % checkpoint_every == 0 and (step + 1 < steps):
            ck = os.path.join(out_dir, f'step_{step + 1}')
            save_lora(pipe, ck)
            print(f'[ideo-train] checkpoint @ step {step + 1} -> {ck}', flush=True)
    n = save_lora(pipe, out_dir)
    print(f'[ideo-train] saved {n} LoRA tensors -> {out_dir}/ideogram_lora.safetensors\nRESULT DONE', flush=True)
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL, help='weights repo id or local dir (nf4/fp8/bf16). Local dir uses the local-aware hub shim.')
    ap.add_argument('--quant', default='fp8', choices=['nf4', 'fp8'], help='quant flavour for the default repo lookup (ignored when --model is a local path). fp8 (default) is the RAM-SAFE training base: weight-only fp8 loads meta-init+assign (~3.9GB host RSS) and our MANUAL fp32 LoRA wraps the frozen Fp8Linear directly. nf4 (bnb 4-bit) is also wrappable by the manual LoRA, BUT its loader instantiates the full model on CPU first (host-RAM blowup) — prefer fp8 unless you have a meta-safe nf4.')
    ap.add_argument('--data', default=None, help='jsonl: {image,prompt[,height,width]} per line')
    ap.add_argument('--out', default='ideogram_lora')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=6e-05)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--train-mlp', action='store_true', help='also LoRA the MLP w1/w2/w3 (more capacity)')
    ap.add_argument('--checkpoint-every', type=int, default=0, help='save the adapter every N steps')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adafactor'], help='optimizer for the LoRA params (adamw default, or momentum-less adafactor)')
    ap.add_argument('--lr-phases', default='', help="toolkit-revamped multiphase LR: 'lr@start_step' pairs, e.g. '0.00006@0,0.00003@14000'. Blank = constant at --lr.")
    add_streaming_arg(ap)
    ap.add_argument('--selftest', action='store_true', help='random-latent grad+step capability check')
    a = ap.parse_args()
    streaming = resolve_streaming(a, default=False)
    if a.selftest:
        return _selftest(a.model, quant=a.quant, streaming=streaming)
    if not a.data:
        ap.error('--data required (or --selftest)')
    return train(a.model, a.data, a.out, a.steps, a.lr, a.rank, a.train_mlp, optimizer=a.optimizer, lr_phases=a.lr_phases, checkpoint_every=a.checkpoint_every, quant=a.quant, streaming=streaming)
if __name__ == '__main__':
    raise SystemExit(main())
