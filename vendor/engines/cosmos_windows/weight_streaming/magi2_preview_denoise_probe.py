# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_preview_denoise_probe.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

def _setup(model_dir: Path):
    from cosmos_windows.weight_streaming.magi2_attention_adapter import install_dependency_shims, install_magi2_attention_intercept
    from cosmos_windows.weight_streaming.magi2_moe_binding import install_moe_kernel_sm120_patch
    os.environ.setdefault('MAGI2_CKPT_ROOT', str(model_dir))
    os.environ.setdefault('MAGI2_DISABLE_MAGI_COMPILE', '1')
    os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
    os.environ.setdefault('SKIP_LOAD_MODEL', '0')
    install_dependency_shims()
    import inference.model.magi2_preview
    import inference.model.magi2_refiner
    import inference.common.arch
    install_magi2_attention_intercept()
    install_moe_kernel_sm120_patch()

def run_probe(model_dir: Path, resolution: str, steps: int, transport: str, ring_budget_gb: float, txt_tokens: int, seconds: float=10.0, ring_depth=None, mlp_chunk: int=0, dump_latent: Optional[str]=None, compute_dtype: str='bf16', cfg_mode: str='batched') -> dict:
    import torch
    _mf = os.environ.get('MAGI2_MEM_FRACTION')
    if _mf and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_mf), 0)
    from inference.common.magi2_config import load_config
    from inference.model.magi2_preview import Transformer
    from inference.pipeline.preview_data_proxy import CFGConfig, SamplerInput
    from inference.pipeline.sampler import FlowUniPCMultistepScheduler, Magi2PreviewSampler
    from inference.pipeline.preview_data_proxy import Magi2DataProxy
    from cosmos_windows.weight_streaming.magi2_layer_ring import FileLayerSource, build_layer_ring_streamer, choose_ring_regime, plan_layer_ring
    from cosmos_windows.weight_streaming.magi2_moe_binding import build_streamed_magi2_preview
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    preview_dir = model_dir / 'preview'
    presets = {'272p': (448, 256), '540p': (896, 512), '1080p': (896, 512)}
    pw, ph = presets[resolution]
    rep: dict = {'resolution': resolution, 'steps': steps, 'transport_requested': transport, 'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0))}
    repo_root = Path(os.environ.get('MAGI2_REPO_ROOT', ''))
    cfg = load_config(str(repo_root / 'configs' / 'magi2_preview.json'))
    cfg.engine_config.cp_size = 1
    cfg.engine_config.ep_size = 1
    ec = cfg.evaluation_config
    ac = cfg.arch_config
    device = 'cuda'
    dtype = torch.bfloat16
    t_build = time.time()
    with torch.device('meta'):
        model = Transformer(ac, ep_size=1)
    index = build_index(preview_dir)
    vae_stride, patch = (ec.vae_stride, ec.patch_size)
    latent_h = ph // vae_stride[1] // patch[1] * patch[1]
    latent_w = pw // vae_stride[2] // patch[2] * patch[2]
    vll = (int(round(seconds * float(ec.fps) * 2)) - 1) // vae_stride[0] + 1
    all_ = int(round(seconds * float(ec.data_proxy_config.audio_latent_fps)))
    packed_tokens = vll * latent_h * latent_w + all_ + txt_tokens
    base_plan = plan_layer_ring(index, vram_budget_bytes=int(ring_budget_gb * 1000000000.0), max_ring_depth=2)
    if ring_depth in (None, 'auto'):
        depth, reason = choose_ring_regime(base_plan, packed_tokens=packed_tokens)
    else:
        depth = int(ring_depth)
        reason = f'explicit ring_depth={depth}'
    rep['ring_regime'] = reason
    print(f'[MAGI2-RING] {reason}', flush=True)
    plan = plan_layer_ring(index, vram_budget_bytes=int(ring_budget_gb * 1000000000.0), min_ring_depth=depth, max_ring_depth=depth)
    rep['ring_depth'] = plan.ring_depth
    rep['packed_tokens'] = packed_tokens
    rep['bytes_per_pass_gb'] = round(plan.bytes_per_pass() / 1000000000.0, 2)
    rep['num_layers'] = plan.num_layers
    if compute_dtype and compute_dtype != 'bf16':
        from cosmos_windows.weight_streaming.magi2_compute_precision import install_moe_compute_dtype
        rep['compute_precision'] = install_moe_compute_dtype(compute_dtype).get('note')
    rep['compute_dtype'] = compute_dtype
    if mlp_chunk and int(mlp_chunk) > 0:
        from cosmos_windows.weight_streaming.magi2_activation_chunk import install_chunked_shared_mlp
        rep['mlp_token_chunk'] = install_chunked_shared_mlp(chunk_tokens=int(mlp_chunk), min_tokens=0).get('note')
    if transport == 'pinned':
        streamer = build_layer_ring_streamer(plan, device=device, transport='pinned', source=FileLayerSource())
    else:
        streamer = build_layer_ring_streamer(plan, device=device, transport=transport)
    rep['transport'] = getattr(streamer, 'transport', '?')
    rep['ds_mode'] = getattr(streamer, 'mode', None)
    build_streamed_magi2_preview(model, index, streamer, device=device)
    rep['build_seconds'] = round(time.time() - t_build, 1)
    vae_stride, patch, z_dim = (ec.vae_stride, ec.patch_size, ec.z_dim)
    fps = float(ec.fps)
    shift = ec.shift
    latent_h = ph // vae_stride[1] // patch[1] * patch[1]
    latent_w = pw // vae_stride[2] // patch[2] * patch[2]
    video_frame_length = int(round(seconds * fps * 2))
    video_latent_length = (video_frame_length - 1) // vae_stride[0] + 1
    audio_latent_fps = float(ec.data_proxy_config.audio_latent_fps)
    audio_latent_length = int(round(seconds * audio_latent_fps))
    text_ch = ac.text_in_channels
    rep['latent_shape'] = [1, z_dim, video_latent_length, latent_h, latent_w]
    rep['video_tokens'] = video_latent_length * latent_h * latent_w
    rep['audio_tokens'] = audio_latent_length
    rep['text_ch'] = text_ch
    torch.manual_seed(42)
    latent = torch.randn(1, z_dim, video_latent_length, latent_h, latent_w, dtype=torch.float32, device=device)
    audio_latent = torch.randn(1, audio_latent_length, 64, dtype=torch.float32, device=device)
    txt_feat = torch.randn(1, txt_tokens, text_ch, dtype=dtype, device=device)
    null_txt_feat = torch.randn(1, txt_tokens, text_ch, dtype=dtype, device=device)
    ref_audio_feat = torch.zeros(1, 0, 64, dtype=torch.float32, device=device)
    video_scheduler = FlowUniPCMultistepScheduler()
    audio_scheduler = FlowUniPCMultistepScheduler()
    video_scheduler.set_timesteps(steps, device=device, shift=shift)
    audio_scheduler.set_timesteps(steps, device=device, shift=shift)
    cfg_config = CFGConfig(use_cfg_trick=ec.use_cfg_trick, cfg_trick_start_frame=ec.cfg_trick_start_frame, cfg_trick_value=ec.cfg_trick_value, use_dynamic_cfg=ec.use_dynamic_cfg, dynamic_cfg_start_t=ec.dynamic_cfg_start_t, dynamic_cfg_cutoff_value=ec.dynamic_cfg_cutoff_value, video_txt_guidance_scale=ec.video_txt_guidance_scale, audio_txt_guidance_scale=ec.audio_txt_guidance_scale, use_ref_for_uncond=ec.use_ref_for_uncond, use_skimmed_cfg_linear=ec.use_skimmed_cfg_linear, skimmed_cfg_scale=ec.skimmed_cfg_scale, cfg_rescale=ec.cfg_rescale)
    sampler_input = SamplerInput(video_t_list=video_scheduler.timesteps, audio_t_list=audio_scheduler.timesteps, latent=latent, audio_latent=audio_latent, txt_feat=txt_feat, null_txt_feat=null_txt_feat, ref_audio_feat=ref_audio_feat, ref_video_feat=None, video_scheduler=video_scheduler, audio_scheduler=audio_scheduler, cfg_config=cfg_config, ref_image_feat=None, ref_image_feat_len=None, ref_image_special_token_embedding=None)
    data_proxy = Magi2DataProxy(ec.data_proxy_config)
    sampler = Magi2PreviewSampler(model, data_proxy, device, dtype)
    if cfg_mode == 'sequential':
        from cosmos_windows.weight_streaming.magi2_cfg_sequential import install_sequential_cfg
        install_sequential_cfg(sampler)
    rep['cfg_mode'] = cfg_mode
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        latent_out, audio_out = sampler.sample(sampler_input)
    torch.cuda.synchronize()
    total = time.time() - t0
    rep['denoise_seconds'] = round(total, 2)
    rep['s_per_step'] = round(total / max(steps, 1), 2)
    rep['effective_stream_gbps'] = round(plan.bytes_per_pass() * steps / 1000000000.0 / total, 2) if total > 0 else None
    rep['peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1000000000.0, 2)
    rep['latent_out_shape'] = list(latent_out.shape)
    rep['audio_out_shape'] = list(audio_out.shape)
    rep['latent_finite'] = bool(torch.isfinite(latent_out).all())
    rep['audio_finite'] = bool(torch.isfinite(audio_out).all())
    rep['passed'] = rep['latent_finite'] and rep['audio_finite']
    if dump_latent:
        torch.save({'latent': latent_out.detach().cpu(), 'audio': audio_out.detach().cpu()}, dump_latent)
        rep['dumped_latent'] = dump_latent
    streamer.close()
    return rep

def _main(argv: Optional[list]=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, default=Path(''))
    p.add_argument('--resolution', choices=['272p', '540p', '1080p'], default='272p')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--transport', choices=['pinned', 'directstorage', 'auto'], default='pinned')
    p.add_argument('--ring-budget-gb', type=float, default=14.0)
    p.add_argument('--txt-tokens', type=int, default=226)
    p.add_argument('--ring-depth', default='auto', help='auto | 1 | 2 (1=max activation headroom, 2=prefetch overlap)')
    p.add_argument('--mlp-chunk', type=int, default=0, help='token-chunk the shared MLP at N rows (0=off; parity/max-res)')
    p.add_argument('--compute-dtype', choices=['bf16', 'fp8', 'fp4'], default='bf16')
    p.add_argument('--cfg-mode', choices=['batched', 'sequential'], default='batched')
    p.add_argument('--dump-latent', default=None, help='save {latent,audio} to this .pt for a parity compare')
    args = p.parse_args(argv)
    _setup(args.model_dir)
    rd = None if args.ring_depth == 'auto' else int(args.ring_depth)
    rep = run_probe(args.model_dir, args.resolution, args.steps, args.transport, args.ring_budget_gb, args.txt_tokens, ring_depth=rd, mlp_chunk=args.mlp_chunk, dump_latent=args.dump_latent, compute_dtype=args.compute_dtype, cfg_mode=args.cfg_mode)
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep.get('passed') else 1
if __name__ == '__main__':
    sys.path.insert(0, '')
    sys.path.insert(0, os.environ.get('MAGI2_REPO_ROOT', ''))
    raise SystemExit(_main())
