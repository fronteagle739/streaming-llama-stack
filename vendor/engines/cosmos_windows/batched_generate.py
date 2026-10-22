# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/batched_generate.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, os, sys, time, copy
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def capture_packs(cli, modality, prompt, negative, cfg):
    pipe = cli._get_pipe(need_sound=False)
    tf = pipe.transformer
    grabbed = []

    class _Stop(Exception):
        pass
    orig = tf.forward

    def cap(**kw):
        grabbed.append(dict(kw))
        if len(grabbed) >= (2 if cfg != 1.0 else 1):
            raise _Stop()
        return orig(**kw)
    tf.forward = cap
    try:
        cli.do('/set steps 1')
        cli.st['negative_prompt'] = negative
        cli.generate(modality, prompt, image_path=None)
    except _Stop:
        pass
    finally:
        tf.forward = orig
    cond_kw = grabbed[0]
    uncond_kw = grabbed[1] if len(grabbed) > 1 else None
    return (pipe, cond_kw, uncond_kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt', default=None, help='one prompt, generated --num times with different seeds')
    ap.add_argument('--prompts-file', default=None, help='file of prompts (one per line) — batch of different images')
    ap.add_argument('--num', type=int, default=8, help='batch size when --prompt is used')
    ap.add_argument('--seed', type=int, default=0, help='base seed; generated images use seed+i')
    ap.add_argument('--negative', default='', help='CFG negative prompt')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--cfg', type=float, default=6.0)
    ap.add_argument('--modality', default='t2i')
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--backward-buffers', type=int, default=3, help='DirectStorage prefetch ring buffers (forward-only gen). 3 = prefetch overlap keeps the NVMe pipeline saturated (measured ~1.7x over single-buffer). Lower to 1-2 only if VRAM-tight.')
    ap.add_argument('--adapter', default=None, help='optional trained LoRA adapter (pytorch_lora_weights.safetensors) to apply to the streamed base for BATCHED LoRA generation (no-merge; ~Nx faster than per-image gen_with_lora). The LoRA stays resident while the base streams.')
    ap.add_argument('--rank', type=int, default=16, help='LoRA rank (must match the trained adapter)')
    ap.add_argument('--loras', default=None, help='MULTI-LoRA STACK: a JSON list of adapters to stack (no-merge), e.g. \'[{"path":"a.safetensors","weight":0.8},{"path":"b.safetensors","weight":0.5,"rank":32}]\'. Each composes via peft set_adapters with its own weight. Overrides --adapter when given.')
    ap.add_argument('--no-reasoner', action='store_true', help='the adapter was trained WITHOUT --train-reasoner')
    ap.add_argument('--no-cfg-fusion', action='store_false', dest='cfg_fusion', default=True, help='run CFG cond+uncond as SEPARATE streams/step instead of one fused 2N-pack forward. ~2x more streams (slower) but HALVES the per-step held activations — use for BIG video latents that would OOM under fusion. The layer-major streaming still amortizes over the N images.')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = [l.strip() for l in open(a.prompts_file, encoding='utf-8') if l.strip()] if a.prompts_file else [a.prompt] * a.num
    N = len(prompts)
    seeds = [int(a.seed) + i for i in range(N)]
    same_prompt = a.prompts_file is None
    from cosmos_windows.cosmos_cli import Cli, DEFAULTS
    st = dict(DEFAULTS)
    st.update(model=a.model, width=a.width, height=a.height, num_frames=1, memeff_attn=True)
    cli = Cli(st)
    print(f'[bgen] capturing packs for {(1 if same_prompt else N)} prompt(s)...', flush=True)
    pipe, cond0, uncond0 = capture_packs(cli, a.modality, prompts[0], a.negative, a.cfg)
    cond_kws = [cond0] * N if same_prompt else [cond0]
    uncond_kws = [uncond0] * N if same_prompt else [uncond0]
    if not same_prompt:
        for p in prompts[1:]:
            _, ck, uk = capture_packs(cli, a.modality, p, a.negative, a.cfg)
            cond_kws.append(ck)
            uncond_kws.append(uk)
    vt0 = cond0['vision_tokens']
    latent_shape = (vt0[0] if isinstance(vt0, (list, tuple)) else vt0).shape
    dtype = (vt0[0] if isinstance(vt0, (list, tuple)) else vt0).dtype
    scheduler0 = pipe.scheduler
    vae = pipe.vae
    vmean = pipe._vae_latents_mean
    vinv = pipe._vae_latents_inv_std
    video_processor = pipe.video_processor
    import gc
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    from cosmos_windows.weight_streaming.build_streamed_transformer import build_streamed_backward_transformer
    from cosmos_windows.weight_streaming.batched_backward import batched_train_forward
    tf, streamer = build_streamed_backward_transformer(a.model, device='cuda', n_gpu_buffers=a.backward_buffers)
    tf.eval()
    for p in tf.parameters():
        p.requires_grad_(False)
    if a.loras or a.adapter:
        from cosmos_windows.train_lora import _targets
        targets = _targets(not a.no_reasoner)
        if a.loras:
            import json as _json
            from cosmos_windows.gen_with_lora import apply_lora_stack
            apply_lora_stack(tf, _json.loads(a.loras), a.rank, targets)
        else:
            from cosmos_windows.gen_with_lora import apply_lora
            apply_lora(tf, a.adapter, a.rank, targets)
        for p in tf.parameters():
            p.requires_grad_(False)
    g = torch.Generator(device='cuda')
    latents = []
    for s in seeds:
        g.manual_seed(s)
        latents.append(torch.randn(latent_shape, generator=g, device='cuda', dtype=dtype))
    schedulers = [copy.deepcopy(scheduler0) for _ in range(N)]
    for sch in schedulers:
        sch.set_timesteps(a.steps, device='cuda')
    timesteps = schedulers[0].timesteps
    print(f'[bgen] N={N} images, {a.steps} steps, cfg={a.cfg}; latent {tuple(latent_shape)}; 1 fused stream/step over {N * (2 if a.cfg != 1.0 else 1)} forwards (cond+uncond of the whole batch)', flush=True)

    def _pack(base_kw, latent, t):
        kw = dict(base_kw)
        vis = base_kw['vision_tokens']
        kw['vision_tokens'] = [latent] if isinstance(vis, (list, tuple)) else latent
        kw['vision_timesteps'] = torch.full_like(base_kw['vision_timesteps'], float(t))
        return kw
    step_times = []
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tv = t.item()
            cond_packs = [_pack(cond_kws[j], latents[j], tv) for j in range(N)]
            uncond_packs = [_pack(uncond_kws[j], latents[j], tv) for j in range(N)] if a.cfg != 1.0 else []
            if a.cfg_fusion and a.cfg != 1.0:
                preds = batched_train_forward(tf, streamer, cond_packs + uncond_packs)
                cond_preds, uncond_preds = (preds[:N], preds[N:])
            else:
                cond_preds = batched_train_forward(tf, streamer, cond_packs)
                uncond_preds = batched_train_forward(tf, streamer, uncond_packs) if a.cfg != 1.0 else None
            for j in range(N):
                cv = cond_preds[j][0] if isinstance(cond_preds[j], (list, tuple)) else cond_preds[j]
                if a.cfg != 1.0:
                    uv = uncond_preds[j][0] if isinstance(uncond_preds[j], (list, tuple)) else uncond_preds[j]
                    v = uv + a.cfg * (cv - uv)
                else:
                    v = cv
                latents[j] = schedulers[j].step(v.unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False)[0].squeeze(0)
            torch.cuda.synchronize()
            if i > 0:
                step_times.append(time.perf_counter() - t0)
            if i % 5 == 0 or i == len(timesteps) - 1:
                print(f'[bgen] step {i + 1}/{len(timesteps)}  {time.perf_counter() - t0:.1f}s ({N} imgs)', flush=True)
    if step_times:
        avg = sum(step_times) / len(step_times)
        nimg = N * (2 if a.cfg != 1.0 else 1)
        print(f'[bgen] THROUGHPUT: {avg:.1f}s/step for {N} imgs = {avg / N:.2f}s/img/step; full gen ~= {avg * len(timesteps):.0f}s for ALL {N} images (= {avg * len(timesteps) / N:.1f}s/img). 1 fused model-stream/step over {nimg} forwards (cond+uncond of the whole batch).', flush=True)
    print(f'[bgen] decoding {N} latents...', flush=True)
    for j in range(N):
        z = latents[j].to(vae.dtype) / vinv.to(latents[j].device, vae.dtype).view(1, -1, 1, 1, 1) + vmean.to(latents[j].device, vae.dtype).view(1, -1, 1, 1, 1)
        with torch.no_grad():
            dec = vae.decode(z).sample
        frame = video_processor.postprocess_video(dec, output_type='pil')[0][0]
        tag = f'seed{seeds[j]}' if same_prompt else f'p{j:02d}'
        fp = os.path.join(a.out, f'bgen_{tag}.png')
        frame.save(fp)
        print(f'[bgen] saved {fp}', flush=True)
    print('RESULT DONE', flush=True)
if __name__ == '__main__':
    main()
