# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_reasoner_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'cosmos_windows'
from .optim_factory import build_optimizer, MultiPhaseLR
from .streaming_flag import add_streaming_arg, resolve_streaming

def _find_decoder_layers(model):
    import torch.nn as nn
    cands = ['model.language_model.layers', 'model.model.language_model.layers', 'model.text_model.layers', 'language_model.model.layers', 'model.layers', 'model.model.layers']
    for path in cands:
        obj = model
        ok = True
        for part in path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList) and (len(obj) > 0):
            return obj
    return None

def _wrap_reasoner_streamed(model, device):
    from cosmos_windows.streamed_backward import apply_streamed_multi
    layers = _find_decoder_layers(model)
    if layers is None:
        return 0

    def base_params(layer):
        return [p for p in layer.parameters() if not p.requires_grad]

    def make_bind(layer):

        def bind():
            for p in base_params(layer):
                if p.device.type != device.split(':')[0]:
                    p.data = p.data.to(device)
        return bind

    def make_evict(layer):

        def evict():
            for p in base_params(layer):
                if p.device.type != 'cpu':
                    p.data = p.data.to('cpu')
        return evict
    n = 0
    for layer in layers:
        for p in base_params(layer):
            if p.device.type != 'cpu':
                p.data = p.data.to('cpu')
        cls_fwd = type(layer).forward
        bind, evict = (make_bind(layer), make_evict(layer))

        def wrapped(hidden_states, *args, _layer=layer, _fwd=cls_fwd, _b=bind, _e=evict, **kw):

            def call(lyr, h):
                return _fwd(lyr, h, *args, **{**kw, 'use_cache': False})
            return apply_streamed_multi(_layer, call, hidden_states, bind=_b, evict=_e)
        layer.forward = wrapped
        n += 1
    return n

def _pick_device():
    import torch
    if not torch.cuda.is_available():
        return 'cpu'
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(best)
    return f'cuda:{best}'

def _targets(train_mlp: bool):
    t = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if train_mlp:
        t += ['gate_proj', 'up_proj', 'down_proj']
    return t

def build_reasoner_lora(model_path: str, device: str, rank: int=8, alpha: int=16, train_mlp: bool=False, streaming: bool=False):
    import torch, torch.nn as nn
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cosmos_windows.reasoner_prefill_stream import _load
    from peft import LoraConfig
    model, proc = _load(model_path, device)
    model.train()
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=_targets(train_mlp), lora_dropout=0.0))
    for n, p in model.named_parameters():
        p.requires_grad_('lora_' in n)
    for m in model.modules():
        for pn, p in list(m._parameters.items()):
            if p is not None and p.requires_grad and (p.dtype != torch.float32):
                m._parameters[pn] = nn.Parameter(p.detach().float(), requires_grad=True)
    wrapped = 0
    if streaming:
        wrapped = _wrap_reasoner_streamed(model, device)
        if wrapped:
            model.config.use_cache = False
            print(f'[reason-train] STREAMING=ON: {wrapped} text decoder layers streamed CPU<->{device} (fwd+bwd, ~1 layer base resident) — LoRA resident.', flush=True)
        else:
            print('[reason-train] STREAMING=ON requested but the decoder layer list was not locatable on this model — falling back to RESIDENT + gradient checkpointing (the reasoner fits resident anyway).', flush=True)
    if not wrapped:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    return (model, proc, lora_params)

def _sft_batch(proc, ex, device):
    import torch
    content = []
    img = None
    if ex.get('image'):
        from PIL import Image
        img = Image.open(ex['image']).convert('RGB')
        content.append({'type': 'image', 'image': img})
    content.append({'type': 'text', 'text': ex['prompt']})
    msgs = [{'role': 'user', 'content': content}]
    prompt_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + ex['response']
    kw = dict(text=[full_text], return_tensors='pt')
    pkw = dict(text=[prompt_text], return_tensors='pt')
    if img is not None:
        kw['images'] = [img]
        pkw['images'] = [img]
    inputs = proc(**kw).to(device)
    plen = proc(**pkw)['input_ids'].shape[1]
    labels = inputs['input_ids'].clone()
    labels[:, :plen] = -100
    return (inputs, labels)

def _lm_loss(model, inputs, labels):
    import torch, torch.nn.functional as F
    out = model(**inputs, use_cache=False, return_dict=True)
    logits = out.logits
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1), ignore_index=-100)

def save_lora(model, out_dir):
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n}
    save_file(sd, os.path.join(out_dir, 'reasoner_lora.safetensors'))
    return len(sd)

def train(model_path, data_path, out_dir, steps, lr, rank, train_mlp, optimizer='adamw', lr_phases='', streaming=False):
    import torch
    device = _pick_device()
    print(f"[reason-train] device={device} streaming={('on' if streaming else 'off')}", flush=True)
    model, proc, lora_params = build_reasoner_lora(model_path, device, rank=rank, train_mlp=train_mlp, streaming=streaming)
    print(f'[reason-train] {sum((p.numel() for p in lora_params)) / 1000000.0:.2f}M LoRA params trainable', flush=True)
    data = [json.loads(l) for l in open(data_path, encoding='utf-8') if l.strip()]
    opt = build_optimizer(lora_params, optimizer, lr=lr)
    sched = MultiPhaseLR(opt, lr_phases, default_lr=lr)
    t0 = time.time()
    for step in range(steps):
        sched.step(step)
        ex = data[step % len(data)]
        inputs, labels = _sft_batch(proc, ex, device)
        opt.zero_grad(set_to_none=True)
        loss = _lm_loss(model, inputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f'[reason-train] step {step + 1}/{steps} loss={loss.item():.4f} ({(time.time() - t0) / (step + 1):.2f}s/step)', flush=True)
    n = save_lora(model, out_dir)
    print(f'[reason-train] saved {n} LoRA tensors -> {out_dir}/reasoner_lora.safetensors', flush=True)
    print('RESULT DONE', flush=True)
    return 0

def _selftest(model_path, streaming=False):
    import torch
    device = _pick_device()
    model, proc, lora_params = build_reasoner_lora(model_path, device, streaming=streaming)
    opt = torch.optim.AdamW(lora_params, lr=0.001)
    img = os.environ.get('REASONER_SELFTEST_IMG', '')
    batch = [{'prompt': 'Why slow a robot near people?', 'response': 'To avoid harming them.'}]
    have_img = os.path.exists(img)
    if have_img:
        batch.append({'image': img, 'prompt': 'Describe this.', 'response': 'A red apple.'})
    opt.zero_grad(set_to_none=True)
    losses = {}
    for ex in batch:
        inputs, labels = _sft_batch(proc, ex, device)
        loss = _lm_loss(model, inputs, labels) / len(batch)
        loss.backward()
        losses['image+text' if ex.get('image') else 'text'] = loss.item() * len(batch)
        del inputs, labels, loss
        torch.cuda.empty_cache()
    ng = sum((1 for p in lora_params if p.grad is not None))
    gnorm = math.sqrt(sum((p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None)))
    before = [p.detach().clone() for p in lora_params]
    opt.step()
    moved = sum(((p.detach() - b).abs().sum().item() for p, b in zip(lora_params, before)))
    print(f'[combined] losses={ {k: round(v, 3) for k, v in losses.items()}} grads_nonNone={ng}/{len(lora_params)} gnorm={gnorm:.3f} moved={moved:.2e}', flush=True)
    ok = ng == len(lora_params) and math.isfinite(gnorm) and (gnorm > 0) and (moved > 0) and ('text' in losses) and ('image+text' in losses or not have_img)
    if not have_img:
        print(f'[combined] (note: no test image at {img} -> text-only; set REASONER_SELFTEST_IMG for the VLM path)', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='')
    ap.add_argument('--data', default=None, help='jsonl: {prompt,response[,image]} per line')
    ap.add_argument('--out', default='reasoner_lora')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--train-mlp', action='store_true', help='also LoRA the MLP gate/up/down (more capacity)')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adafactor'], help='optimizer for the LoRA params (adamw default, or momentum-less adafactor)')
    ap.add_argument('--lr-phases', default='', help="toolkit-revamped multiphase LR: 'lr@start_step' pairs (blank = constant at --lr)")
    add_streaming_arg(ap)
    ap.add_argument('--selftest', action='store_true', help='no-data 2-mode grad+step capability check')
    a = ap.parse_args()
    streaming = resolve_streaming(a, default=False)
    if a.selftest:
        return _selftest(a.model, streaming=streaming)
    if not a.data:
        ap.error('--data is required (or use --selftest)')
    return train(a.model, a.data, a.out, a.steps, a.lr, a.rank, a.train_mlp, optimizer=a.optimizer, lr_phases=a.lr_phases, streaming=streaming)
if __name__ == '__main__':
    raise SystemExit(main())
