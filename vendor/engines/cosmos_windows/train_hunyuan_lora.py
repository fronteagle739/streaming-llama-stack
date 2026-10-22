# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_hunyuan_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'cosmos_windows'
from .optim_factory import build_optimizer, MultiPhaseLR
from .streaming_flag import add_streaming_arg, resolve_streaming
DEFAULT_MODEL = ''

def _targets(train_mlp: bool):
    t = ['qkv_proj', 'o_proj']
    if train_mlp:
        t += ['gate_and_up_proj', 'down_proj']
    return t

def build(model_dir, device, rank=8, alpha=16, train_mlp=False):
    import torch, torch.nn as nn
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cosmos_windows.weight_streaming.hunyuan_backend import build_streamed_backward_hunyuan
    from cosmos_windows.weight_streaming.build_streamed_transformer import materialize_meta_lora
    from peft import LoraConfig
    model, streamer = build_streamed_backward_hunyuan(model_dir, device=device, n_gpu_buffers=1)
    model.load_tokenizer(model_dir)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=_targets(train_mlp), lora_dropout=0.0))
    n = materialize_meta_lora(model, device=device)
    for m in model.modules():
        for pn, p in list(m._parameters.items()):
            if p is not None and p.requires_grad and (p.dtype != torch.float32):
                m._parameters[pn] = nn.Parameter(p.detach().float(), requires_grad=True)
    lora_params = [p for nm, p in model.named_parameters() if 'lora_' in nm and p.requires_grad]
    print(f'[hy-train] {n} LoRA params materialized; {len(lora_params)} trainable', flush=True)
    return (model, streamer, lora_params)

def capture_model_inputs(model, prompt, image_size, device):
    import torch
    captured = {}

    class _Stop(Exception):
        pass
    orig = type(model).forward

    def cap(self, *a, **kw):
        if kw.get('images') is not None:
            captured['kwargs'] = dict(kw)
            raise _Stop()
        return orig(self, *a, **kw)
    model.forward = cap.__get__(model)
    try:
        with torch.no_grad():
            model.generate_image(prompt=prompt, bot_task='auto', image_size=image_size, seed=0, diff_infer_steps=1, use_cache=False)
    except _Stop:
        pass
    finally:
        model.forward = orig.__get__(model)
    if 'kwargs' not in captured:
        raise RuntimeError('did not capture an image forward — generate_image structure changed?')
    return captured['kwargs']

def flow_step(model, streamer, base_inputs, z0, opt, lora_params):
    import torch
    dev = z0.device
    t = torch.rand((), device=dev).clamp(0.05, 0.95)
    noise = torch.randn_like(z0)
    z_t = (1.0 - t) * z0 + t * noise
    target_v = noise - z0
    inputs = dict(base_inputs)
    inputs['images'] = z_t.to(base_inputs['images'].dtype)
    if 'timesteps' in inputs and torch.is_tensor(inputs['timesteps']):
        inputs['timesteps'] = torch.full_like(inputs['timesteps'], float(t))
    inputs['first_step'] = True
    opt.zero_grad(set_to_none=True)
    streamer.new_pass()
    out = model(**inputs)
    pred = out['diffusion_prediction'] if isinstance(out, dict) else out.diffusion_prediction
    pv = pred.float().reshape(-1)
    tv = target_v.float().reshape(-1)
    n = min(pv.numel(), tv.numel())
    loss = torch.nn.functional.mse_loss(pv[:n], tv[:n])
    streamer.new_pass(reverse=True)
    loss.backward()
    ng = sum((1 for p in lora_params if p.grad is not None))
    gnorm = math.sqrt(sum((p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None)))
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt.step()
    return (loss.item(), ng, gnorm)

def save_lora(model, out_dir):
    import os
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n}
    save_file(sd, os.path.join(out_dir, 'hunyuan_lora.safetensors'))
    return len(sd)

def _pick_device():
    import torch
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(best)
    return f'cuda:{best}'

def _selftest(model_dir):
    import torch
    device = _pick_device()
    model, streamer, lora_params = build(model_dir, device)
    opt = torch.optim.AdamW(lora_params, lr=0.001)
    base = capture_model_inputs(model, 'a red apple on a table', (512, 512), device)
    print(f"[hy-train] captured model_inputs keys: {sorted(base.keys())}; images {tuple(base['images'].shape)}", flush=True)
    z0 = torch.randn_like(base['images'].float())
    torch.cuda.reset_peak_memory_stats()
    before = [p.detach().clone() for p in lora_params]
    loss, ng, gnorm = flow_step(model, streamer, base, z0, opt, lora_params)
    moved = sum(((p.detach() - b).abs().sum().item() for p, b in zip(lora_params, before)))
    peak = torch.cuda.max_memory_allocated() / 1000000000.0
    ok = ng == len(lora_params) and math.isfinite(loss) and (gnorm > 0) and (moved > 0) and (peak < 30.0)
    print(f'[hy-train] flow step: loss={loss:.4f} grads={ng}/{len(lora_params)} gnorm={gnorm:.4f} moved={moved:.2e} peak={peak:.1f}GB', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1

def train(model_dir, data_path, out_dir, steps, lr, rank, train_mlp, optimizer='adamw', lr_phases=''):
    import torch
    from PIL import Image
    device = _pick_device()
    model, streamer, lora_params = build(model_dir, device, rank=rank, train_mlp=train_mlp)
    opt = build_optimizer(lora_params, optimizer, lr=lr)
    sched = MultiPhaseLR(opt, lr_phases, default_lr=lr)
    data = [json.loads(l) for l in open(data_path, encoding='utf-8') if l.strip()]
    base_cache = {}
    t0 = time.time()
    for step in range(steps):
        sched.step(step)
        ex = data[step % len(data)]
        size = tuple(ex.get('image_size', (512, 512)))
        key = (ex['prompt'], size)
        if key not in base_cache:
            base_cache[key] = capture_model_inputs(model, ex['prompt'], size, device)
        base = base_cache[key]
        import numpy as np
        img = Image.open(ex['image']).convert('RGB').resize(size)
        arr = np.asarray(img).astype('float32') / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        cfg = base['images'].shape[0]
        with torch.no_grad():
            _, z0 = model.vae_encode(x, cfg_factor=cfg)
        z0 = z0.to(device).float()
        if z0.shape != base['images'].shape:
            z0 = torch.randn_like(base['images'].float())
        loss, ng, gnorm = flow_step(model, streamer, base, z0, opt, lora_params)
        if step % 10 == 0 or step == steps - 1:
            print(f'[hy-train] step {step + 1}/{steps} loss={loss:.4f} ({(time.time() - t0) / (step + 1):.1f}s/step)', flush=True)
    n = save_lora(model, out_dir)
    print(f'[hy-train] saved {n} LoRA tensors -> {out_dir}/hunyuan_lora.safetensors\nRESULT DONE', flush=True)
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--data', default=None, help='jsonl: {image,prompt[,image_size]} per line')
    ap.add_argument('--out', default='hunyuan_lora')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--train-mlp', action='store_true')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adafactor'], help='optimizer for the LoRA params (adamw default, or momentum-less adafactor)')
    ap.add_argument('--lr-phases', default='', help="toolkit-revamped multiphase LR: 'lr@start_step' pairs (blank = constant at --lr)")
    add_streaming_arg(ap)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if not resolve_streaming(a, default=True):
        print('[hy-train] NOTE: --streaming off requested, but HunyuanImage-3 (80B/157GB) has NO resident path on a 32GB card — proceeding with the streamed (fwd+bwd) backward, which is always on here.', flush=True)
    if a.selftest:
        return _selftest(a.model)
    if not a.data:
        ap.error('--data required (or --selftest)')
    return train(a.model, a.data, a.out, a.steps, a.lr, a.rank, a.train_mlp, optimizer=a.optimizer, lr_phases=a.lr_phases)
if __name__ == '__main__':
    raise SystemExit(main())
