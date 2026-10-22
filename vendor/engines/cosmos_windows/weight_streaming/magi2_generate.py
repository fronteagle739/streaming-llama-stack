# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_generate.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import gc
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
    install_dependency_shims()
    import inference.model.magi2_preview
    import inference.model.magi2_refiner
    import inference.common.arch
    install_magi2_attention_intercept()
    install_moe_kernel_sm120_patch()

def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

def _tensor_stats(t, name: str) -> dict:
    import torch
    f = t.float()
    finite = torch.isfinite(f)
    ff = float(finite.float().mean())
    g = f[finite] if ff < 1.0 else f
    return {'name': name, 'shape': list(t.shape), 'finite_fraction': round(ff, 6), 'mean': round(float(g.mean()), 6) if g.numel() else None, 'std': round(float(g.std()), 6) if g.numel() > 1 else None, 'min': round(float(g.min()), 6) if g.numel() else None, 'max': round(float(g.max()), 6) if g.numel() else None, 'abs_mean': round(float(g.abs().mean()), 6) if g.numel() else None}

def assert_not_degenerate(stats: dict, stage: str) -> None:
    if stats['finite_fraction'] < 1.0:
        raise RuntimeError(f"DEGENERATE OUTPUT at stage '{stage}': only {stats['finite_fraction'] * 100:.4f}% of {stats['name']} is finite (NaN/Inf present) — refusing to continue. stats={stats}")
    if stats.get('abs_mean') is not None and stats['abs_mean'] == 0.0:
        raise RuntimeError(f"DEGENERATE OUTPUT at stage '{stage}': {stats['name']} is ALL ZEROS (would decode to a black clip) — refusing to continue. stats={stats}")
    if stats.get('std') is not None and stats['std'] == 0.0:
        raise RuntimeError(f"DEGENERATE OUTPUT at stage '{stage}': {stats['name']} is FLAT (std=0, constant value {stats.get('mean')}) — refusing to continue. stats={stats}")

def _video_stats(video_np) -> dict:
    import numpy as np
    a = np.asarray(video_np)
    per_frame_max = a.reshape(a.shape[0], -1).max(axis=1) if a.ndim >= 2 else a.max()
    black_frames = int((per_frame_max <= 2).sum())
    return {'shape': list(a.shape), 'mean': round(float(a.mean()), 3), 'std': round(float(a.std()), 3), 'min': int(a.min()), 'max': int(a.max()), 'black_frames': black_frames, 'frames': int(a.shape[0]), 'black_fraction': round(black_frames / max(1, a.shape[0]), 4)}

def assert_video_not_black(vstats: dict) -> None:
    if vstats['max'] <= 2:
        raise RuntimeError(f"BLACK VIDEO: decoded frames are entirely black (max pixel {vstats['max']}). stats={vstats}. Refusing to write an mp4 that looks like a successful render.")
    if vstats['black_fraction'] >= 0.5:
        raise RuntimeError(f"BLACK VIDEO: {vstats['black_fraction'] * 100:.1f}% of frames are black. stats={vstats}")
    if vstats['std'] < 1.0:
        raise RuntimeError(f"FLAT VIDEO: decoded frames have std {vstats['std']:.3f} (near-constant image). stats={vstats}")

def _rss_gb():
    try:
        import os as _os
        import psutil
        return round(psutil.Process(_os.getpid()).memory_info().rss / 1000000000.0, 2)
    except Exception:
        return None

def _lean_get_turbo_vaed(config_path, ckpt_path, device, weight_dtype):
    import json as _json
    import torch
    from inference.model.turbo_vaed import TurboVAED
    with open(config_path, 'r', encoding='utf-8') as f:
        config = _json.load(f)
    with torch.device(device):
        student = TurboVAED.from_config(config)
    ckpt = torch.load(ckpt_path, map_location=device, mmap=True)
    if 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    missing, _unexpected = student.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith('decoder.') and (not sample_key.startswith('encoder.')):
            student.decoder.load_state_dict(state_dict, strict=False)
    student = student.to(device, dtype=weight_dtype)
    student.eval()
    student.requires_grad_(False)
    return student

class _LeanAudioVAE:

    def __init__(self, vae_model, sample_rate, downsampling_ratio, rescale_mean, rescale_std):
        self.vae_model = vae_model
        self.sample_rate = sample_rate
        self.downsampling_ratio = downsampling_ratio
        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std

    @property
    def latent_fps(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    def to(self, *_a, **_k):
        return self

    def decode(self, latents):
        import torch
        latents = latents * self.rescale_std + self.rescale_mean
        with torch.no_grad():
            return self.vae_model.decode(latents)

def _lean_get_audio_vae(audio_model_path, device):
    import json as _json
    import os as _os
    import torch
    from inference.model.sa_audio_vae import create_model_from_config
    from inference.pipeline.audio_decoder import SA_RESCALE_MEAN, SA_RESCALE_STD
    from cosmos_windows.weight_streaming.backends import PortableBackend
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    with open(_os.path.join(audio_model_path, 'model_config.json')) as f:
        full_config = _json.load(f)
    vae_config = full_config['model']['pretransform']['config']
    sample_rate = full_config['sample_rate']
    downsampling_ratio = vae_config['downsampling_ratio']
    autoencoder_config = {'model_type': 'autoencoder', 'sample_rate': sample_rate, 'model': vae_config}
    with torch.device(device):
        vae_model = create_model_from_config(autoencoder_config)
    weights_path = os.path.join(audio_model_path, 'model.safetensors')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'Weight file does not exist: {weights_path}')
    index = build_index(weights_path)
    prefix = 'pretransform.model.'
    from cosmos_windows.weight_streaming.ds_state_load import directstorage_usable, ds_load_state_dict
    if directstorage_usable():
        vae_state_dict = ds_load_state_dict(index, device=device, name_filter=lambda n: n.startswith(prefix), rename=lambda n: n[len(prefix):])
    else:
        backend = PortableBackend(pin_memory=True)
        vae_state_dict = {}
        try:
            for name, ref in index.refs.items():
                if name.startswith(prefix):
                    vae_state_dict[name[len(prefix):]] = backend.read_tensor(ref, device=device, direct=True)
        finally:
            backend.close()
    vae_model.load_state_dict(vae_state_dict)
    vae_model.to(device).eval()
    rescale_mean = torch.tensor(SA_RESCALE_MEAN, dtype=torch.float32, device=device).view(1, -1, 1)
    rescale_std = torch.tensor(SA_RESCALE_STD, dtype=torch.float32, device=device).view(1, -1, 1)
    print(f'[magi2] Stable Audio VAE loaded LEAN direct-read (sr={sample_rate}, ds_ratio={downsampling_ratio})', flush=True)
    return _LeanAudioVAE(vae_model, sample_rate, downsampling_ratio, rescale_mean, rescale_std)

def _decode_video(latent, turbo_vae, dtype, emit=None):
    import numpy as np
    import torch
    z = latent.squeeze(0).to(dtype)
    if z.dim() == 4:
        z = z.unsqueeze(0)
    if emit is not None:
        zf = z.float()
        emit(event='stage_log', stage='vae_decode', message='pre-decode latent (as fed to VAE)', z_stats={'shape': list(z.shape), 'finite_fraction': round(torch.isfinite(zf).float().mean().item(), 6), 'mean': round(zf.mean().item(), 6), 'std': round(zf.std().item(), 6), 'min': round(zf.min().item(), 4), 'max': round(zf.max().item(), 4)})
    with torch.no_grad():
        decoded = turbo_vae.decode(z)
    videos = decoded if isinstance(decoded, list) else [decoded]
    if emit is not None and videos:
        df = (videos[0][0] if isinstance(videos[0], list) else videos[0]).float()
        emit(event='stage_log', stage='vae_decode', message='decoded float frames (pre-uint8)', decoded_stats={'shape': list(df.shape), 'finite_fraction': round(torch.isfinite(df).float().mean().item(), 6), 'mean': round(torch.nan_to_num(df).mean().item(), 6), 'std': round(torch.nan_to_num(df).std().item(), 6), 'min': round(torch.nan_to_num(df).min().item(), 4), 'max': round(torch.nan_to_num(df).max().item(), 4)})
        del df
    out = []
    for video in videos:
        video = video.float().mul(0.5).add(0.5).clamp(0, 1)
        if video.ndim == 5:
            out.extend(list(video.cpu().permute(0, 2, 3, 4, 1).mul(255).numpy().astype(np.uint8)))
        else:
            out.append(video.cpu().permute(1, 2, 3, 0).mul(255).numpy().astype(np.uint8))
    return out[0] if out else None

def _decode_video_tiled(latent, turbo_vae, dtype, budget_gb: float=8.0, emit=None):
    import numpy as np
    import torch
    from cosmos_windows.weight_streaming.magi2_vae_tiling import plan_tiled_decode, tiled_decode
    su = int(getattr(turbo_vae, 'spatial_compression_ratio', 16) or 16)
    z = latent.squeeze(0).to(dtype)
    if z.dim() == 4:
        z = z.unsqueeze(0)
    B, C, T, Hl, Wl = z.shape

    def decode_fn(tile):
        with torch.no_grad():
            out = turbo_vae.decode(tile)
        if isinstance(out, list):
            out = out[0]
        if out.dim() == 4:
            out = out.unsqueeze(0)
        return out
    t_pixels = 24
    plan = plan_tiled_decode(Hl, Wl, T, spatial_upscale=su, hidden_channels=256, dtype_bytes=2, budget_bytes=int(budget_gb * 1000000000.0), t_pixels=t_pixels, overlap_lat=4, min_tile_lat=16)
    if emit is not None:
        emit(event='stage_log', stage='vae_tile_plan', **plan.summary())
    video = tiled_decode(z, decode_fn, plan)
    video = video.float().mul(0.5).add(0.5).clamp(0, 1)
    arr = video.cpu().permute(0, 2, 3, 4, 1).mul(255).numpy().astype(np.uint8)
    del video
    return arr[0]

def _ensure_figure_tokens(prompt: str, figure_ids) -> str:
    import json as _json
    try:
        obj = _json.loads(prompt)
        if not isinstance(obj, dict):
            raise ValueError
        obj['reference_layer'] = ['The first frame refers to <Figure 1>']
        return _json.dumps(obj, ensure_ascii=False)
    except (ValueError, TypeError):
        return prompt + 'reference_layer:The first frame refers to <Figure 1>'

def _encode_ref_image(image_path, vae, video_processor, ref_image_type, height, width, device, dtype):
    import torch
    from diffusers.utils import load_image
    from inference.pipeline.inference_engine import resizepad
    pil = load_image(image_path)
    if ref_image_type == 'square':
        th, tw = (320, 320)
    else:
        m = max(height, width)
        if pil.width > pil.height:
            tw, th = (m, int(pil.height * m / pil.width))
        else:
            th, tw = (m, int(pil.width * m / pil.height))
    img = resizepad(pil, th, tw)
    img = video_processor.preprocess(img, height=th, width=tw)
    img = img.to(device=device, dtype=dtype).unsqueeze(2)[:, :3]
    with torch.no_grad():
        latent = vae.encode(img.float())
    feat = latent.squeeze(0)
    ref_image_feat = feat.unsqueeze(0).unsqueeze(0)
    lh, lw = (int(ref_image_feat.shape[4]), int(ref_image_feat.shape[5]))
    ref_image_feat_len = torch.tensor([[[lh, lw]]], dtype=torch.long)
    ref_image_ids = torch.tensor([[1]], dtype=torch.long)
    return (ref_image_feat, ref_image_feat_len, ref_image_ids)

def _decode_audio(audio_latent, audio_vae, device):
    import torch
    from inference.pipeline.audio_decoder import resample_audio_sinc
    la = audio_latent.squeeze(0).to(device)
    with torch.no_grad():
        waveform = audio_vae.decode(la.T)
    audio_np = waveform.squeeze(0).T.cpu().float().numpy()
    return resample_audio_sinc(audio_np, 441 / 512)

def generate(model_dir: Path, prompt: str, negative_prompt: str, out_prefix: str, resolution: str, steps: int, transport: str, ring_budget_gb: float, seconds: float=10.0, txt_skip_layer: int=2, emit=None, image: Optional[str]=None, compute_dtype: str='bf16', enable_refiner: Optional[bool]=None, refiner_steps: int=5, refiner_width: Optional[int]=None, refiner_height: Optional[int]=None) -> dict:
    import torch
    from inference.common.magi2_config import load_config
    from inference.model.magi2_preview import Transformer
    from inference.pipeline.preview_data_proxy import CFGConfig, Magi2DataProxy, SamplerInput
    from inference.pipeline.sampler import FlowUniPCMultistepScheduler, Magi2PreviewSampler
    from cosmos_windows.weight_streaming.magi2_layer_ring import FileLayerSource, build_layer_ring_streamer, plan_layer_ring
    from cosmos_windows.weight_streaming.magi2_moe_binding import build_streamed_magi2_preview
    from cosmos_windows.weight_streaming.magi2_text_encode import build_streamed_magi2_text_encoder
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    os.environ.setdefault('MAGI2_CKPT_ROOT', str(model_dir))
    device = 'cuda'
    dtype = torch.bfloat16
    rep: dict = {'prompt': prompt, 'resolution': resolution, 'steps': steps, 'compute_dtype': compute_dtype}
    from cosmos_windows.weight_streaming.magi2_compute_precision import install_moe_compute_dtype
    cp = install_moe_compute_dtype(compute_dtype)
    rep['compute_precision'] = cp.get('note')
    from cosmos_windows.weight_streaming.magi2_activation_chunk import install_chunked_shared_mlp
    from cosmos_windows.weight_streaming.magi2_backend import packed_tokens_for
    _res_tokens = packed_tokens_for(resolution, seconds)
    _default_chunk = 1024 if _res_tokens > 20000 else 0
    _chunk_tokens = int(os.environ.get('MAGI2_MLP_CHUNK_TOKENS', str(_default_chunk)))
    mc = install_chunked_shared_mlp(chunk_tokens=_chunk_tokens, min_tokens=20000)
    rep['mlp_token_chunk'] = mc.get('note')
    log = (lambda **k: print(json.dumps({'event': 'magi2_gen', **k}), flush=True)) if emit is None else emit
    repo_root = Path(os.environ.get('MAGI2_REPO_ROOT', ''))
    cfg = load_config(str(repo_root / 'configs' / 'magi2_preview.json'))
    cfg.engine_config.cp_size = 1
    cfg.engine_config.ep_size = 1
    ec = cfg.evaluation_config
    presets = {'272p': (448, 256), '640sq': (640, 640), '800x448': (800, 448), '540p': (896, 512), '1080p': (896, 512)}
    pw, ph = presets[resolution]
    vae_stride, patch, z_dim = (ec.vae_stride, ec.patch_size, ec.z_dim)
    latent_h = ph // vae_stride[1] // patch[1] * patch[1]
    latent_w = pw // vae_stride[2] // patch[2] * patch[2]
    gen_h, gen_w = (latent_h * vae_stride[1], latent_w * vae_stride[2])
    ref_image_feat = ref_image_feat_len = ref_image_ids = None
    if image:
        from diffusers.video_processor import VideoProcessor
        from inference.model.vae2_2 import get_vae2_2
        t = time.time()
        wan_vae = get_vae2_2(os.path.join(ec.vae_model_path, 'Wan2.2_VAE.pth'), device, weight_dtype=torch.float32)
        vproc = VideoProcessor(vae_scale_factor=2 * ec.vae_stride[1])
        ref_image_feat, ref_image_feat_len, ref_image_ids = _encode_ref_image(image, wan_vae, vproc, ec.ref_image_type, gen_h, gen_w, device, dtype)
        del wan_vae
        _free()
        prompt = _ensure_figure_tokens(prompt, ref_image_ids.reshape(-1).tolist())
        rep['ref_image_encode_s'] = round(time.time() - t, 1)
        log(event='ref_image_encode_done', seconds=rep['ref_image_encode_s'], feat=list(ref_image_feat.shape))
    t = time.time()
    enc = build_streamed_magi2_text_encoder(model_dir / 'text_encoder', device=device, skip_layer=txt_skip_layer, dtype=dtype)
    context = enc.encode(prompt).to(dtype)
    null_context = enc.encode(negative_prompt).to(dtype)
    ref_image_special_tokens = None
    if image and ref_image_ids is not None:
        target_strs = [f'<Figure {int(i)}>' for i in ref_image_ids.reshape(-1).tolist()]
        ref_image_special_tokens = enc.get_special_token(prompt, target_strs, context).to(device=device, dtype=dtype).unsqueeze(0)
        ref_image_feat = ref_image_feat.to(device=device, dtype=torch.float32)
        ref_image_feat_len = ref_image_feat_len.to(device)
    rep['text_encode_s'] = round(time.time() - t, 1)
    rep['context_shape'] = list(context.shape)
    _cs = _tensor_stats(context, 'context')
    _ns = _tensor_stats(null_context, 'null_context')
    rep['context_stats'], rep['null_context_stats'] = (_cs, _ns)
    assert_not_degenerate(_cs, 'text_encode/context')
    assert_not_degenerate(_ns, 'text_encode/null_context')
    rep['prompt_chars'], rep['negative_prompt_chars'] = (len(prompt), len(negative_prompt or ''))
    log(event='conditioning_verified', context=_cs, null_context=_ns, prompt_chars=rep['prompt_chars'], negative_prompt_chars=rep['negative_prompt_chars'])
    del enc
    _free()
    rep.setdefault('rss_by_stage', {})['text_encode'] = _rss_gb()
    log(event='text_encode_done', seconds=rep['text_encode_s'], context=list(context.shape), rss_gb=rep['rss_by_stage']['text_encode'])
    if transport != 'pinned':
        _frac = float(os.environ.get('MAGI2_TORCH_MEM_FRACTION', '0.78'))
        if 0 < _frac < 1:
            torch.cuda.set_per_process_memory_fraction(_frac)
            _tot = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
            log(event='stage_log', stage='vram_reserve', fraction=_frac, torch_cap_gb=round(_tot * _frac, 2), total_gb=round(_tot, 2), message='capped torch VRAM so the DirectStorage D3D12 heap cannot be evicted (device-hang guard); OOM here is honest, a hang is not')
    with torch.device('meta'):
        model = Transformer(cfg.arch_config, ep_size=1)
    index = build_index(model_dir / 'preview')
    from cosmos_windows.weight_streaming.magi2_backend import choose_run_config
    _run_cfg = choose_run_config(resolution, mlp_chunk_tokens=_chunk_tokens, seconds=seconds)
    _cfg_env = os.environ.get('MAGI2_CFG_MODE', 'auto').strip().lower()
    _cfg_for_ring = _run_cfg['cfg_mode'] if _cfg_env == 'auto' else _cfg_env
    os.environ['MAGI2_CFG_MODE'] = _cfg_for_ring
    rep['config_matrix'] = _run_cfg['matrix']
    log(event='stage_log', stage='config_matrix', chosen_cfg_mode=_cfg_for_ring, reason=_run_cfg['reason'], matrix=_run_cfg['matrix'])
    _ring = {'ring_depth': _run_cfg['ring_depth'], 'overlap': _run_cfg['overlap'], 'projections': {c['ring_depth']: {'projected_gb': c['projected_gb']} for c in _run_cfg['matrix'] if c['cfg_mode'] == _cfg_for_ring}, 'reason': _run_cfg['reason']}
    _forced = os.environ.get('MAGI2_RING_DEPTH')
    _max_depth = int(_forced) if _forced else _ring['ring_depth']
    if _forced is None and _ring['overlap']:
        os.environ.setdefault('MAGI2_RING_OVERLAP', '1')
    plan = plan_layer_ring(index, vram_budget_bytes=int(ring_budget_gb * 1000000000.0), min_ring_depth=1, max_ring_depth=_max_depth)
    rep['ring_depth'] = plan.ring_depth
    rep['ring_choice'] = _ring['reason']
    rep['ring_projections'] = {str(d): p['projected_gb'] for d, p in _ring['projections'].items()}
    log(event='stage_log', stage='layer_ring', ring_depth=plan.ring_depth, resolution=resolution, overlap=os.environ.get('MAGI2_RING_OVERLAP', '0'), reason=_ring['reason'], projections=rep['ring_projections'])
    if transport == 'pinned':
        streamer = build_layer_ring_streamer(plan, device=device, transport='pinned', source=FileLayerSource())
    else:
        streamer = build_layer_ring_streamer(plan, device=device, transport=transport)
    build_streamed_magi2_preview(model, index, streamer, device=device)
    rep['transport'] = getattr(streamer, 'transport', '?')
    vfl = int(round(seconds * float(ec.fps) * 2))
    video_latent_length = (vfl - 1) // vae_stride[0] + 1
    audio_latent_length = int(round(seconds * float(ec.data_proxy_config.audio_latent_fps)))
    torch.manual_seed(cfg.engine_config.seed)
    latent = torch.randn(1, z_dim, video_latent_length, latent_h, latent_w, dtype=torch.float32, device=device)
    audio_latent = torch.randn(1, audio_latent_length, 64, dtype=torch.float32, device=device)
    ref_audio_feat = torch.zeros(1, 0, 64, dtype=torch.float32, device=device)
    vsched, asched = (FlowUniPCMultistepScheduler(), FlowUniPCMultistepScheduler())
    vsched.set_timesteps(steps, device=device, shift=ec.shift)
    asched.set_timesteps(steps, device=device, shift=ec.shift)
    cfg_config = CFGConfig(use_cfg_trick=ec.use_cfg_trick, cfg_trick_start_frame=ec.cfg_trick_start_frame, cfg_trick_value=ec.cfg_trick_value, use_dynamic_cfg=ec.use_dynamic_cfg, dynamic_cfg_start_t=ec.dynamic_cfg_start_t, dynamic_cfg_cutoff_value=ec.dynamic_cfg_cutoff_value, video_txt_guidance_scale=ec.video_txt_guidance_scale, audio_txt_guidance_scale=ec.audio_txt_guidance_scale, use_ref_for_uncond=ec.use_ref_for_uncond, use_skimmed_cfg_linear=ec.use_skimmed_cfg_linear, skimmed_cfg_scale=ec.skimmed_cfg_scale, cfg_rescale=ec.cfg_rescale)
    sampler_input = SamplerInput(video_t_list=vsched.timesteps, audio_t_list=asched.timesteps, latent=latent, audio_latent=audio_latent, txt_feat=context, null_txt_feat=null_context, ref_audio_feat=ref_audio_feat, ref_video_feat=None, video_scheduler=vsched, audio_scheduler=asched, cfg_config=cfg_config, ref_image_feat=ref_image_feat, ref_image_feat_len=ref_image_feat_len, ref_image_special_token_embedding=ref_image_special_tokens)
    sampler = Magi2PreviewSampler(model, Magi2DataProxy(ec.data_proxy_config), device, dtype)
    torch.cuda.reset_peak_memory_stats()
    _step = {'i': 0, 't': time.time()}
    _orig_forward = sampler.forward
    log(event='denoise_started', steps=steps, stage='preview')

    def _forward_with_progress(model_input):
        out = _orig_forward(model_input)
        _step['i'] += 1
        now = time.time()
        log(event='denoise_step_complete', step=_step['i'], steps=steps, seconds=round(now - _step['t'], 2))
        _step['t'] = now
        return out
    sampler.forward = _forward_with_progress
    _cfg_mode = os.environ.get('MAGI2_CFG_MODE', 'auto').strip().lower()
    if _cfg_mode == 'auto':
        _cfg_mode = 'sequential' if resolution in ('540p', '1080p') else 'batched'
    rep['cfg_mode'] = _cfg_mode
    if _cfg_mode == 'sequential':
        from cosmos_windows.weight_streaming.magi2_cfg_sequential import install_sequential_cfg
        install_sequential_cfg(sampler, emit=log)
    t = time.time()
    try:
        with torch.no_grad():
            latent, audio_latent = sampler.sample(sampler_input)
        torch.cuda.synchronize()
    except BaseException:
        try:
            streamer.close()
        except Exception:
            pass
        try:
            del sampler, sampler_input, model, streamer
        except Exception:
            pass
        _free()
        raise
    rep['denoise_s'] = round(time.time() - t, 1)
    rep['s_per_step'] = round(rep['denoise_s'] / max(steps, 1), 1)
    rep['denoise_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1000000000.0, 2)
    _ls = _tensor_stats(latent, 'video_latent')
    _as = _tensor_stats(audio_latent, 'audio_latent')
    rep['latent_stats'], rep['audio_latent_stats'] = (_ls, _as)
    log(event='latent_stats', stage='preview_denoise', latent=_ls, audio=_as)
    assert_not_degenerate(_ls, 'preview_denoise/video_latent')
    rep['bytes_per_pass_gb'] = round(plan.bytes_per_pass() / 1000000000.0, 1)
    rep['ring_overlap'] = os.environ.get('MAGI2_RING_OVERLAP', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    rep['effective_stream_gbps'] = round(plan.bytes_per_pass() * steps / 1000000000.0 / rep['denoise_s'], 2) if rep['denoise_s'] > 0 else None
    log(event='denoise_done', seconds=rep['denoise_s'], s_per_step=rep['s_per_step'], peak_vram_gb=rep['denoise_peak_vram_gb'], stream_gbps=rep['effective_stream_gbps'], ring_overlap=rep['ring_overlap'])
    streamer.close()
    del model, streamer
    _free()
    _dump = os.environ.get('MAGI2_DUMP_LATENT')
    if _dump:
        import torch as _t
        Path(_dump).parent.mkdir(parents=True, exist_ok=True)
        _t.save({'latent': latent.detach().cpu(), 'audio_latent': audio_latent.detach().cpu(), 'resolution': resolution, 'steps': steps, 'prompt': prompt}, _dump)
        log(event='stage_log', stage='latent_dump', path=_dump, shape=list(latent.shape), message='denoised latent saved for offline decode analysis')
    _ls_post = _tensor_stats(latent, 'video_latent_postfree')
    log(event='latent_stats', stage='post_ring_teardown', latent=_ls_post)
    assert_not_degenerate(_ls_post, 'post_ring_teardown/video_latent')
    rep.setdefault('rss_by_stage', {})['preview_denoise'] = _rss_gb()
    log(event='stage_rss', stage='preview_denoise', rss_gb=rep['rss_by_stage']['preview_denoise'])
    do_refine = enable_refiner if enable_refiner is not None else resolution == '1080p'
    if do_refine and transport != 'pinned':
        _rfrac = float(os.environ.get('MAGI2_REFINER_MEM_FRACTION', '0.97'))
        if 0 < _rfrac <= 1:
            torch.cuda.set_per_process_memory_fraction(_rfrac)
            _tot = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
            log(event='stage_log', stage='vram_reserve', fraction=_rfrac, torch_cap_gb=round(_tot * _rfrac, 2), message='preview ring + DS staging released — cap lifted for the refiner stage')
    rep['refiner'] = bool(do_refine)
    if do_refine:
        from cosmos_windows.weight_streaming.magi2_refiner_stage import load_refiner_config, load_refiner_resident, refine_latent
        rw = int(refiner_width or 1920)
        rh = int(refiner_height or 1088)
        t = time.time()
        rcfg = load_refiner_config(repo_root)
        refiner_model = load_refiner_resident(model_dir / 'refiner', rcfg.magi2_refiner_arch_config, device, emit=log)
        log(event='refiner_started', steps=refiner_steps, target=f'{rw}x{rh}')
        torch.cuda.reset_peak_memory_stats()
        latent, audio_latent = refine_latent(refiner_model, rcfg.evaluation_config, context, null_context, latent, audio_latent, rw, rh, refiner_steps, device=device, dtype=dtype)
        torch.cuda.synchronize()
        rep['refiner_s'] = round(time.time() - t, 1)
        rep['refiner_s_per_step'] = round(rep['refiner_s'] / max(refiner_steps, 1), 1)
        rep['refiner_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1000000000.0, 2)
        rep['refined_latent_shape'] = list(latent.shape)
        rep['refiner_finite'] = bool(torch.isfinite(latent).all())
        log(event='refiner_done', seconds=rep['refiner_s'], s_per_step=rep['refiner_s_per_step'], peak_vram_gb=rep['refiner_peak_vram_gb'], latent_shape=rep['refined_latent_shape'], finite=rep['refiner_finite'])
        del refiner_model
        _free()
        rep.setdefault('rss_by_stage', {})['refiner'] = _rss_gb()
        log(event='stage_rss', stage='refiner', rss_gb=rep['rss_by_stage']['refiner'])
    from inference.pipeline.audio_decoder import SAAudioFeatureExtractor
    t = time.time()
    turbo_vae = _lean_get_turbo_vaed(ec.student_config, ec.student_ckpt, device, weight_dtype=dtype)
    if do_refine:
        try:
            video_np = _decode_video(latent, turbo_vae, dtype, emit=log)
            rep['vae_tiled'] = False
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(event='stage_log', stage='vae_decode', level='warn', message='un-tiled 1080p decode OOM -> spatial-tiled decode (seam-blended)')
            video_np = _decode_video_tiled(latent, turbo_vae, dtype, emit=log)
            rep['vae_tiled'] = True
    else:
        video_np = _decode_video(latent, turbo_vae, dtype, emit=log)
    del turbo_vae
    _free()
    audio_vae = _lean_get_audio_vae(ec.audio_model_path, device)
    audio_np = _decode_audio(audio_latent, audio_vae, device)
    sample_rate = int(getattr(audio_vae, 'sample_rate', 44100))
    rep['vae_decode_s'] = round(time.time() - t, 1)
    rep['video_shape'] = list(getattr(video_np, 'shape', []))
    _vs = _video_stats(video_np)
    rep['video_stats'] = _vs
    log(event='video_stats', **_vs)
    assert_video_not_black(_vs)
    rep.setdefault('rss_by_stage', {})['vae_decode'] = _rss_gb()
    log(event='vae_decode_done', seconds=rep['vae_decode_s'], video_shape=rep['video_shape'], rss_gb=rep['rss_by_stage']['vae_decode'])
    del audio_vae
    _free()
    from inference.pipeline.pipeline import Magi2Evaluator
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    Magi2Evaluator._save_video(video_np, out_prefix, seconds, output_fps=float(ec.fps))
    if audio_np is not None and getattr(audio_np, 'size', 0):
        Magi2Evaluator._mux_audio(out_prefix, audio_np, sample_rate)
    rep['output'] = f'{out_prefix}.mp4'
    rep['output_bytes'] = Path(rep['output']).stat().st_size if Path(rep['output']).is_file() else 0
    rep['passed'] = rep['output_bytes'] > 0
    log(event='completed', output=rep['output'], bytes=rep['output_bytes'])
    return rep

def _main(argv: Optional[list]=None) -> int:
    from cosmos_windows.weight_streaming.magi2_prompt import DEFAULT_NEGATIVE_PROMPT
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, default=Path(''))
    p.add_argument('--prompt', required=True)
    p.add_argument('--image', default=None, help='first-frame still for i2v (omit for t2v)')
    p.add_argument('--negative-prompt', default=None)
    p.add_argument('--out', required=True, help='output path prefix (no extension)')
    p.add_argument('--resolution', choices=['272p', '640sq', '800x448', '540p', '1080p'], default='272p')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--transport', choices=['pinned', 'directstorage', 'auto'], default='pinned')
    p.add_argument('--ring-budget-gb', type=float, default=14.0)
    p.add_argument('--compute-dtype', choices=['bf16', 'fp8', 'fp4'], default='bf16')
    p.add_argument('--refiner', dest='enable_refiner', default=None, action=argparse.BooleanOptionalAction, help='force the 1080p refiner on/off (default: auto at 1080p)')
    p.add_argument('--refiner-steps', type=int, default=5)
    args = p.parse_args(argv)
    _setup(args.model_dir)
    neg = args.negative_prompt if args.negative_prompt is not None else os.environ.get('NEGATIVE_PROMPT') or DEFAULT_NEGATIVE_PROMPT
    rep = generate(args.model_dir, args.prompt, neg, args.out, args.resolution, args.steps, args.transport, args.ring_budget_gb, image=args.image, compute_dtype=args.compute_dtype, enable_refiner=args.enable_refiner, refiner_steps=args.refiner_steps)
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep.get('passed') else 1
if __name__ == '__main__':
    sys.path.insert(0, '')
    sys.path.insert(0, os.environ.get('MAGI2_REPO_ROOT', ''))
    raise SystemExit(_main())
