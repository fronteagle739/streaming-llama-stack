# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, os, sys, time
import torch
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'cosmos_windows'
from .flow_match_loss import sample_sigma_t, rectified_flow_xt, flow_match_loss
from .optim_factory import build_optimizer, MultiPhaseLR
from .streaming_flag import add_streaming_arg, resolve_streaming
GEN_STREAM_TARGETS = ['add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
UND_STREAM_TARGETS = ['to_q', 'to_k', 'to_v', 'to_out']
NANO = ''

def _targets(train_reasoner: bool):
    return GEN_STREAM_TARGETS + (UND_STREAM_TARGETS if train_reasoner else [])

def caption_ce_loss(tf, und_out, pack):
    import torch
    tok = pack.get('input_ids')
    idx = pack.get('text_indexes')
    if tok is None or idx is None or und_out is None:
        return None
    tok = tok.view(-1).long().to(und_out.device)
    idx = idx.view(-1).long().to(und_out.device)
    idx = idx[idx < und_out.shape[0]]
    m = min(tok.numel(), idx.numel())
    if m < 2:
        return None
    h = und_out[idx[:m]].to(tf.lm_head.weight.dtype)
    logits = tf.lm_head(h)
    return torch.nn.functional.cross_entropy(logits[:-1].float(), tok[:m][1:])

def _save_adapter(tf, out_dir):
    import os
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    lora_sd = {n: p.detach().float().cpu() for n, p in tf.named_parameters() if 'lora_' in n}
    save_file(lora_sd, os.path.join(out_dir, 'pytorch_lora_weights.safetensors'))
    return len(lora_sd)

def load_qlora_lora(model_dir: str, rank: int=16, bits: int=4, targets=GEN_STREAM_TARGETS):
    from pathlib import Path
    from diffusers import BitsAndBytesConfig
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    from peft import LoraConfig
    from diffusers.training_utils import cast_training_params
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if bits == 4 else BitsAndBytesConfig(load_in_8bit=True)
    tf = Cosmos3OmniTransformer.from_pretrained(str(Path(model_dir) / 'transformer'), quantization_config=qcfg, torch_dtype=torch.bfloat16)
    tf.requires_grad_(False)
    tf.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=targets))
    cast_training_params(tf, dtype=torch.float32)
    if tf._supports_gradient_checkpointing:
        tf.enable_gradient_checkpointing()
    trainable = [p for p in tf.parameters() if p.requires_grad]
    print(f'[train] QLoRA {bits}-bit base + LoRA(r={rank}) on {targets}: {sum((p.numel() for p in trainable)) / 1000000.0:.1f}M trainable', flush=True)
    return (tf, trainable)

def load_streamed_lora(tf, rank: int=16, targets=GEN_STREAM_TARGETS):
    from peft import LoraConfig
    from diffusers.training_utils import cast_training_params
    tf.requires_grad_(False)
    tf.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=targets))
    cast_training_params(tf, dtype=torch.float32)
    trainable = [p for p in tf.parameters() if p.requires_grad]
    print(f'[train] STREAMED base (DS, reused from pipe) + LoRA(r={rank}): {sum((p.numel() for p in trainable)) / 1000000.0:.1f}M trainable; whole-forward checkpoint (streamer-sequential)', flush=True)
    return (tf, trainable)

def capture_static_pack(cli, modality, prompt, image_path):
    pipe = cli._get_pipe(need_sound=False)
    tf = pipe.transformer
    grabbed = {}

    class _Stop(Exception):
        pass
    orig = tf.forward

    def cap(**kw):
        grabbed.update(kw)
        raise _Stop()
    tf.forward = cap
    try:
        cli.do('/set steps 1')
        cli.generate(modality, prompt, image_path=image_path)
    except _Stop:
        pass
    finally:
        tf.forward = orig
    return (pipe, grabbed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=NANO)
    ap.add_argument('--image', default='')
    ap.add_argument('--prompt', default='a young woman with long brown hair smiling')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--overfit', action='store_true', help='repeat one sample; loss must fall toward 0')
    ap.add_argument('--save-dir', default='', help='where to save the trained adapter')
    ap.add_argument('--target-image', default=None, help='VAE-encode this image (repeated to the clip length) as the REAL clean target (vs the captured-latent stand-in) — the dataset clean-target path')
    ap.add_argument('--target-dir', default=None, help='a tiny CORPUS dir of images — VAE-encode all + cycle per step (the DataLoader iteration)')
    add_streaming_arg(ap, with_aliases=False)
    ap.add_argument('--resident', action='store_true', help='DEPRECATED alias for --streaming off (force QLoRA resident).')
    ap.add_argument('--streamed', action='store_true', help='DEPRECATED alias / §4b: load the base via DirectStorage streaming (Super path) instead of QLoRA (maps onto --streaming on). Prefer --streaming on.')
    ap.add_argument('--streamed-reverse', dest='streamed_reverse', action='store_true', help='DEPRECATED alias / §4c: per-layer streamed backward over single-buffer reverse-ds (build_streamed_backward_transformer). Bounds activations to ~1 layer — the path Super 64B needs. Maps onto --streaming on (the canonical fwd+bwd layer-streamed path).')
    ap.add_argument('--checkpoint-every', type=int, default=0, help='save the adapter every N steps to <save-dir>/step_<N> (0 = only at the end)')
    ap.add_argument('--modality', default='i2v', help='pack-capture modality: i2v (needs --image) or t2i (text-only, e.g. Super-Text2Image)')
    ap.add_argument('--train-reasoner', action='store_true', help='co-train the reasoner (und) tower: add LoRA to to_q/k/v/out + a caption cross-entropy loss on the prompt tokens, joint with the velocity loss. Helps learn new CONCEPTS, not just pixels.')
    ap.add_argument('--reasoner-weight', type=float, default=0.5, help='weight of the caption CE loss relative to the velocity loss (--train-reasoner)')
    ap.add_argument('--backward-buffers', type=int, default=1, help='GPU ring buffers for the --streamed-reverse backward: 1 = single-buffer (lowest VRAM), >=2 = prefetch overlap (+~1 layer/buffer VRAM). Multi-buffer reverse is fixed (the dir-aware recycle) + validated; **use 3 for ~1.7x faster streaming** (same prefetch win measured for batched_generate) if VRAM allows. Default 1 stays conservative for the gradient path.')
    ap.add_argument('--per-caption', action='store_true', help='capture a SEPARATE forward pack per dataset item using its own .txt caption sidecar (curate_dataset format) instead of one fixed --prompt for all. Real varied captions — important for concept learning + --train-reasoner. Captured before the streamed base build (the pipe transformer is freed after).')
    ap.add_argument('--batched', action='store_true', help='LAYER-MAJOR BATCHED streaming (with --streamed-reverse): each optimizer step pushes --batch-size images through each layer in ONE bind/evict, so the model streams twice per step regardless of batch size (~Nx fewer full-model streams = ~Nx faster on a streaming-bound trainer). Gen-only (likeness); reasoner-CE batching is future. Requires per-caption packs (auto-enables --per-caption when --target-dir is set).')
    ap.add_argument('--batch-size', type=int, default=8, help='images per layer-bind for --batched (the minibatch)')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adafactor'], help='optimizer for the LoRA params. adamw (default, unchanged) or adafactor (momentum-less factored 2nd-moment — tiny state, the toolkit-revamped default).')
    ap.add_argument('--lr-phases', default='', help="toolkit-revamped multiphase LR: 'lr@start_step' pairs, e.g. '0.00006@0,0.00003@14000'. Blank = constant at --lr.")
    a = ap.parse_args()
    streaming_on = resolve_streaming(a, default=a.streamed or a.streamed_reverse)
    if streaming_on and (not (a.streamed or a.streamed_reverse)):
        a.streamed_reverse = True
    if not streaming_on:
        a.streamed = a.streamed_reverse = False
    if a.batched:
        a.streamed_reverse = True
        if a.target_dir:
            a.per_caption = True
    targets = _targets(a.train_reasoner)
    from cosmos_windows.cosmos_cli import Cli, DEFAULTS
    st = dict(DEFAULTS)
    st.update(model=a.model, width=256, height=256, num_frames=1, memeff_attn=True)
    cli = Cli(st)
    pipe, pack = capture_static_pack(cli, a.modality, a.prompt, a.image if a.modality != 't2i' else None)
    print(f"[train] captured pack: keys={sorted((k for k in pack))[:6]}... und_len={pack.get('und_len')} vis={(pack['vision_sequence_indexes'].shape[0] if 'vision_sequence_indexes' in pack else '?')}", flush=True)
    import os as _os, glob as _g
    _MX = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}
    packs_by_path = {}
    if a.per_caption and a.target_dir:
        media = [p for p in sorted(_g.glob(_os.path.join(a.target_dir, '*.*'))) if _os.path.splitext(p)[1].lower() in _MX]
        for mp in media:
            capf = _os.path.splitext(mp)[0] + '.txt'
            cap = open(capf, encoding='utf-8').read().strip() if _os.path.exists(capf) else a.prompt
            _, pk = capture_static_pack(cli, a.modality, cap, mp if a.modality != 't2i' else None)
            packs_by_path[mp] = pk
        print(f'[train] per-caption: captured {len(packs_by_path)} packs from dataset .txt captions', flush=True)
    rev_streamer = None
    if a.streamed_reverse:
        from cosmos_windows.weight_streaming.build_streamed_transformer import build_streamed_backward_transformer, materialize_meta_lora
        from peft import LoraConfig
        tf, rev_streamer = build_streamed_backward_transformer(a.model, device='cuda', n_gpu_buffers=a.backward_buffers)
        tf.requires_grad_(False)
        tf.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights='gaussian', target_modules=targets))
        n_lora = materialize_meta_lora(tf, device='cuda')
        tf.gradient_checkpointing = False
        trainable = [p for p in tf.parameters() if p.requires_grad]
        try:
            import gc
            pipe.transformer = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print('[train] (pipe transformer free skipped:', type(e).__name__, e, ')', flush=True)
        print(f'[train] STREAMED-REVERSE (per-layer reverse-ds backward) + LoRA(r={a.rank}): {n_lora} LoRA params materialized, {sum((p.numel() for p in trainable)) / 1000000.0:.1f}M trainable', flush=True)
    elif a.streamed:
        tf, trainable = load_streamed_lora(pipe.transformer, rank=a.rank, targets=targets)
    else:
        tf, trainable = load_qlora_lora(a.model, rank=a.rank, bits=a.bits, targets=targets)
        pipe.transformer = tf
    _captured = {}
    if a.train_reasoner:
        tf.norm.register_forward_hook(lambda m, i, o: _captured.__setitem__('und_out', o))
        print(f'[train] --train-reasoner: LoRA on und+gen towers ({len(targets)} target mods); caption CE weight={a.reasoner_weight}', flush=True)
    import os, glob as _glob
    vis = pack['vision_tokens']
    stand_in = (vis[0] if isinstance(vis, (list, tuple)) else vis).detach().to(torch.bfloat16)
    cond_mask = None

    def _encode(path):
        from PIL import Image
        px = pipe.video_processor.preprocess(Image.open(path).convert('RGB'), height=st['height'], width=st['width'])
        if px.ndim == 4:
            px = px.unsqueeze(2)
        t_pix = (stand_in.shape[2] - 1) * 4 + 1
        px = px.repeat(1, 1, t_pix, 1, 1).to(stand_in.device, torch.bfloat16)
        with torch.no_grad():
            r = pipe._encode_video(px)
        return r.to(stand_in.dtype) if tuple(r.shape) == tuple(stand_in.shape) else None
    _MEDIA_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}
    paths = [p for p in sorted(_glob.glob(os.path.join(a.target_dir, '*.*'))) if os.path.splitext(p)[1].lower() in _MEDIA_EXT] if a.target_dir else [a.target_image] if a.target_image else []
    samples = [(c, packs_by_path.get(p, pack)) for p, c in ((p, _encode(p)) for p in paths) if c is not None]
    if not samples:
        samples = [(stand_in, pack)]
    cleans = [c for c, _ in samples]
    print(f"[train] corpus: {len(samples)} target(s) from {len(paths) or 'stand-in'} shape={tuple(cleans[0].shape)}; packs={('per-caption' if a.per_caption and packs_by_path else 'single')}", flush=True)
    opt = build_optimizer(trainable, a.optimizer, lr=a.lr)
    sched = MultiPhaseLR(opt, a.lr_phases, default_lr=a.lr)
    gen = torch.Generator(device=stand_in.device).manual_seed(0)
    from contextlib import nullcontext
    fwd_ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if a.streamed or a.streamed_reverse else nullcontext
    if a.batched:
        import time as _time
        from cosmos_windows.weight_streaming.batched_backward import batched_train_forward
        bs = min(a.batch_size, len(samples))
        print(f'[train] LAYER-MAJOR BATCHED: {bs} images/step through each layer in one bind/evict (streams the model twice/step for all {bs} images); {len(samples)} corpus -> {(len(samples) + bs - 1) // bs} batches/epoch', flush=True)
        _step_times = []
    for step in range(a.steps):
        sched.step(step)
        if a.batched:
            batch = [samples[(step * bs + j) % len(samples)] for j in range(bs)]
            packs_b, cleans_b, noises_b = ([], [], [])
            for clean_j, spack_j in batch:
                noise_j = torch.randn(clean_j.shape, generator=gen, device=clean_j.device, dtype=clean_j.dtype)
                sigma_j = sample_sigma_t(1, device=clean_j.device, generator=gen)
                xt_j = rectified_flow_xt(clean_j.float(), noise_j.float(), sigma_j).to(clean_j.dtype)
                skw = dict(spack_j)
                skw['vision_tokens'] = [xt_j] if isinstance(vis, (list, tuple)) else xt_j
                skw['vision_timesteps'] = torch.full_like(spack_j['vision_timesteps'], float(sigma_j.item()))
                packs_b.append(skw)
                cleans_b.append(clean_j)
                noises_b.append(noise_j)
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with fwd_ctx():
                preds_list = batched_train_forward(tf, rev_streamer, packs_b)
            vloss = sum((flow_match_loss(preds_list[j][0] if isinstance(preds_list[j], (list, tuple)) else preds_list[j], cleans_b[j].float(), noises_b[j].float(), cond_mask) for j in range(bs))) / bs
            rev_streamer.new_pass(reverse=True)
            vloss.backward()
            opt.step()
            torch.cuda.synchronize()
            _dt = _time.perf_counter() - _t0
            if step > 0:
                _step_times.append(_dt)
            print(f'[train] step {step:3d}  vloss={vloss.item():.4f}  (batch={bs})  {_dt:.1f}s/step = {_dt / bs:.2f}s/img', flush=True)
            if a.checkpoint_every and (step + 1) % a.checkpoint_every == 0 and (step + 1 < a.steps):
                ck = os.path.join(a.save_dir, f'step_{step + 1}')
                _save_adapter(tf, ck)
                print(f'[train] checkpoint @ step {step + 1} -> {ck}', flush=True)
            if step == a.steps - 1 and _step_times:
                avg = sum(_step_times) / len(_step_times)
                print(f'[train] BATCHED THROUGHPUT (steps>0 avg): {avg:.1f}s/step for {bs} imgs = {avg / bs:.2f}s/img, {bs / avg:.2f} img/s. The model streams twice per step regardless of batch — vs per-image streaming twice PER image (~{bs}x fewer full-model streams).', flush=True)
            continue
        clean, spack = samples[step % len(samples)]
        noise = torch.randn(clean.shape, generator=gen, device=clean.device, dtype=clean.dtype)
        sigma = sample_sigma_t(1, device=clean.device, generator=gen)
        xt = rectified_flow_xt(clean.float(), noise.float(), sigma).to(clean.dtype)
        step_kw = dict(spack)
        step_kw['vision_tokens'] = [xt] if isinstance(vis, (list, tuple)) else xt
        step_kw['vision_timesteps'] = torch.full_like(spack['vision_timesteps'], float(sigma.item()))
        with fwd_ctx():
            if a.streamed_reverse:
                preds_vision, _ = tf(**step_kw)
            elif a.streamed:
                from torch.utils.checkpoint import checkpoint
                preds_vision, _ = checkpoint(tf, use_reentrant=False, **step_kw)
            else:
                preds_vision, _ = tf(**step_kw)
        pred = preds_vision[0] if isinstance(preds_vision, (list, tuple)) else preds_vision
        vloss = flow_match_loss(pred, clean.float(), noise.float(), cond_mask)
        closs = caption_ce_loss(tf, _captured.get('und_out'), spack) if a.train_reasoner else None
        loss = vloss + a.reasoner_weight * closs if closs is not None else vloss
        import time as _time
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if a.streamed_reverse:
            rev_streamer.new_pass(reverse=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        _dt = _time.perf_counter() - _t0
        if step % 5 == 0 or step == a.steps - 1:
            ctxt = f' v={vloss.item():.4f} cap={closs.item():.4f}' if closs is not None else ''
            tt = f'  {_dt:.1f}s/img' if a.streamed_reverse else ''
            print(f'[train] step {step:3d}  loss={loss.item():.4f}{ctxt}  sigma={sigma.item():.3f}{tt}', flush=True)
        if a.checkpoint_every and (step + 1) % a.checkpoint_every == 0 and (step + 1 < a.steps):
            ck = os.path.join(a.save_dir, f'step_{step + 1}')
            _save_adapter(tf, ck)
            print(f'[train] checkpoint @ step {step + 1} -> {ck}', flush=True)
    adapter_effect = -1.0
    with torch.no_grad():
        tkw = dict(pack)
        tkw['vision_tokens'] = [cleans[0]] if isinstance(vis, (list, tuple)) else cleans[0]
        tkw['vision_timesteps'] = torch.full_like(pack['vision_timesteps'], 0.5)

        def _fwd():
            with fwd_ctx():
                o = tf(**tkw)[0]
            return (o[0] if isinstance(o, (list, tuple)) else o).float()
        on = _fwd().clone()
        try:
            tf.disable_adapters()
            off = _fwd()
            tf.enable_adapters()
            adapter_effect = (on - off).abs().mean().item()
        except Exception as e:
            print('[train] adapter on/off toggle err:', type(e).__name__, e, flush=True)
    print(f'[train] trained adapter changes forward output by mean|d|={adapter_effect:.4f} (>0 = adapter active)', flush=True)
    import os, glob
    from safetensors.torch import save_file, load_file
    os.makedirs(a.save_dir, exist_ok=True)
    lora_sd = {n: p.detach().float().cpu() for n, p in tf.named_parameters() if 'lora_' in n}
    out = os.path.join(a.save_dir, 'pytorch_lora_weights.safetensors')
    save_file(lora_sd, out)
    rt = load_file(out)
    nz = sum((1 for v in rt.values() if v.abs().sum().item() > 0))
    print(f'[train] saved adapter: {len(lora_sd)} tensors, {nz} non-zero, roundtrip={len(rt) == len(lora_sd)} -> {out}', flush=True)
    finite = all((torch.isfinite(v).all().item() for v in rt.values()))
    ok = bool(lora_sd) and nz > 0 and (len(rt) == len(lora_sd)) and finite
    print('RESULT', 'DONE' if ok else 'SAVE_FAIL', flush=True)
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
