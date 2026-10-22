# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/bernini_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
import os
from typing import Optional

def _preflight_resident_ram() -> None:
    from . import ram_guard
    need = float(os.environ.get('BERNINI_MIN_FREE_RAM_GB', '10'))
    ram_guard.preflight_host_ram(need, 'Bernini 1.3B resident gen — WanTransformer3DModel (from_pretrained CPU stage) + UMT5 + Wan VAE', skip_env='BERNINI_SKIP_RAM_PREFLIGHT', force_env='BERNINI_FORCE_RESIDENT', advice='Use the streamed 14B Bernini path (per-block mxfp8) on a RAM-constrained box.')

def _rotate_rope_by_source_id(freqs_cos, freqs_sin, source_id: int, dim_per_head: int, theta: float):
    import torch
    if not source_id:
        return (freqs_cos, freqs_sin)
    half = dim_per_head // 2
    inv_freq = 1.0 / theta ** (torch.arange(0, half, device=freqs_cos.device, dtype=torch.float32) / half)
    a = float(source_id) * inv_freq
    cos_a = torch.cos(a).to(freqs_cos.dtype)
    sin_a = torch.sin(a).to(freqs_cos.dtype)
    reps = freqs_cos.shape[-1] // a.shape[-1]
    cos_a = cos_a.repeat(reps)
    sin_a = sin_a.repeat(reps)
    new_cos = freqs_cos * cos_a - freqs_sin * sin_a
    new_sin = freqs_sin * cos_a + freqs_cos * sin_a
    return (new_cos, new_sin)

def patch_wan_in_context(transformer):
    import torch
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    if getattr(transformer, '_bernini_in_context', False):
        return transformer
    orig_forward = transformer.forward
    theta = float(getattr(transformer.rope, 'theta', 10000.0))

    def forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None, return_dict=True, attention_kwargs=None, context_latents=None, **kwargs):
        if context_latents is None:
            context_latents = getattr(transformer, '_bernini_bound_context', None)
        if context_latents is None:
            return orig_forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image, return_dict=return_dict, attention_kwargs=attention_kwargs, **kwargs)
        self = transformer
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf, pph, ppw = (num_frames // p_t, height // p_h, width // p_w)
        dim_per_head = self.config.attention_head_dim if hasattr(self.config, 'attention_head_dim') else hidden_states.shape and self.rope.freqs_cos.shape[-1]
        cos, sin = self.rope(hidden_states)
        dph = cos.shape[-1]
        for i, lat in enumerate(context_latents):
            c_cos, c_sin = self.rope(lat.to(hidden_states.device, hidden_states.dtype))
            c_cos, c_sin = _rotate_rope_by_source_id(c_cos, c_sin, i + 1, dph, theta)
            cos = torch.cat([cos, c_cos], dim=1)
            sin = torch.cat([sin, c_sin], dim=1)
        rotary_emb = (cos, sin)
        x = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        main_len = x.shape[1]
        for lat in context_latents:
            cl = self.patch_embedding(lat.to(hidden_states.device, hidden_states.dtype)).flatten(2).transpose(1, 2)
            x = torch.cat([x, cl], dim=1)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        temb, timestep_proj, enc, enc_img = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len)
        timestep_proj = timestep_proj.unflatten(2, (6, -1)) if ts_seq_len is not None else timestep_proj.unflatten(1, (6, -1))
        if enc_img is not None:
            enc = torch.cat([enc_img, enc], dim=1)
        for block in self.blocks:
            x = block(x, enc, timestep_proj, rotary_emb)
        x = x[:, :main_len]
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = (shift.squeeze(2), scale.squeeze(2))
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = (shift.to(x.device), scale.to(x.device))
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = self.proj_out(x)
        x = x.reshape(batch_size, ppf, pph, ppw, p_t, p_h, p_w, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        out = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return Transformer2DModelOutput(sample=out) if return_dict else (out,)
    transformer.forward = forward
    transformer._bernini_in_context = True
    return transformer

def bind_context_latents(transformer, context_latents):
    prev = getattr(transformer, '_bernini_bound_context', None)
    transformer._bernini_bound_context = context_latents

    def restore():
        transformer._bernini_bound_context = prev
    return restore
TASK_PROMPT_PREFIX = {'default': 'You are a helpful assistant.', 't2i': 'You are a helpful assistant specialized in text-to-image generation.', 't2v': 'You are a helpful assistant specialized in text-to-video generation.', 'i2i': 'You are a helpful assistant specialized in image editing.', 'r2i': 'You are a helpful assistant specialized in subject-to-image generation.', 'i2v': 'You are a helpful assistant specialized in image-to-video generation.', 'v2v': 'You are a helpful assistant specialized in video editing.', 'r2v': 'You are a helpful assistant specialized in subject-to-video generation.', 'vi2v': 'You are a helpful assistant specialized in video editing on content propagation.', 'rv2v': 'You are a helpful assistant specialized in video editing with reference.', 'ads2v': 'You are a helpful assistant specialized in ads insertion.', 'vrc2v': "You are a helpful assistant for editing. You may need to adjust the subject's action or position.", 'mv2v': "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action."}

def task_prefixed_prompt(task: str, prompt: str) -> str:
    pre = TASK_PROMPT_PREFIX.get(task, '')
    return f'{pre} {prompt}'.strip() if pre else prompt
TASK_GUIDANCE_MODE = {'t2i': 't2v_apg', 'i2i': 'v2v', 't2v': 't2v_apg', 'v2v': 'v2v_apg', 'mv2v': 'v2v_apg', 'rv2v': 'rv2v', 'r2v': 'r2v_apg'}

def detect_task(*, want_video: bool, source_video=None, reference_video=None, reference_images=None) -> str:
    has_refs = bool(reference_images)
    if source_video is not None:
        return 'rv2v' if has_refs or reference_video is not None else 'mv2v' if want_video else 'v2v'
    if reference_video is not None:
        return 'rv2v'
    if has_refs:
        return 'r2v' if want_video else 'i2i'
    return 't2v' if want_video else 't2i'

def build_context_latents(vae, *, source_video=None, reference_video=None, reference_images=None, target_hw=None, ref_max_size: int=848, device='cuda', dtype=None):
    import torch

    def _round16(v):
        return max(16, int(round(v / 16) * 16))

    def _to_bcthw(t):
        if t.ndim == 3:
            t = t.unsqueeze(1)
        if t.ndim == 4:
            t = t.unsqueeze(0)
        return t

    def _resize_bcthw(t, H, W):
        b, c, f = t.shape[:3]
        flat = t.permute(0, 2, 1, 3, 4).reshape(b * f, c, t.shape[-2], t.shape[-1])
        flat = torch.nn.functional.interpolate(flat, size=(H, W), mode='bilinear', align_corners=False)
        return flat.reshape(b, f, c, H, W).permute(0, 2, 1, 3, 4)

    def _encode(img_or_vid, resize_to):
        H, W = resize_to
        t = _resize_bcthw(_to_bcthw(img_or_vid), H, W)
        t = t.to(device=device, dtype=dtype or next(vae.parameters()).dtype)
        with torch.no_grad():
            enc = vae.encode(t)
        lat = enc.latent_dist.mode() if hasattr(enc, 'latent_dist') else enc.sample if hasattr(enc, 'sample') else enc
        return lat
    context: list = []
    th, tw = target_hw or (480, 832)
    if source_video is not None:
        context.append(_encode(source_video, (th, tw)))
    for rv in reference_video or [] if not _is_tensor(reference_video) else [reference_video]:
        rh, rw = _ref_hw(rv, ref_max_size, _round16)
        context.append(_encode(rv, (rh, rw)))
    for ri in reference_images or [] if not _is_tensor(reference_images) else [reference_images]:
        rh, rw = _ref_hw(ri, ref_max_size, _round16)
        context.append(_encode(ri, (rh, rw)))
    source_ids = list(range(1, len(context) + 1))
    return (context, source_ids)

def _is_tensor(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False

def _ref_hw(t, ref_max_size, round16):
    h, w = (t.shape[-2], t.shape[-1])
    scale = min(1.0, ref_max_size / float(max(h, w)))
    return (round16(h * scale), round16(w * scale))
_VIDEO_EXTS = ('.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.gif')

def _is_video_file(path) -> bool:
    return str(path).lower().endswith(_VIDEO_EXTS)

def _decode_video_frames(path, max_frames=None):
    import numpy as np
    frames = []
    try:
        import imageio.v3 as iio
        for fr in iio.imiter(path, plugin='pyav'):
            frames.append(np.asarray(fr))
            if max_frames and len(frames) >= max_frames:
                break
    except Exception:
        import imageio
        rdr = imageio.get_reader(path)
        try:
            for fr in rdr:
                frames.append(np.asarray(fr))
                if max_frames and len(frames) >= max_frames:
                    break
        finally:
            rdr.close()
    if not frames:
        raise RuntimeError(f'could not decode any frames from video: {path}')
    return frames

def _frame_to_chw(fr, target_hw, device, dtype):
    import torch
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.asarray(fr)).convert('RGB').resize((target_hw[1], target_hw[0]))
    arr = torch.from_numpy(np.asarray(img)).float().div(127.5).sub(1.0)
    return arr.permute(2, 0, 1).to(device=device, dtype=dtype)

def _load_image_chw(path, target_hw, device, dtype):
    if _is_video_file(path):
        return _frame_to_chw(_decode_video_frames(path, max_frames=1)[0], target_hw, device, dtype)
    import torch
    from PIL import Image
    img = Image.open(path).convert('RGB').resize((target_hw[1], target_hw[0]))
    import numpy as np
    arr = torch.from_numpy(np.asarray(img)).float().div(127.5).sub(1.0)
    return arr.permute(2, 0, 1).to(device=device, dtype=dtype)

def _load_video_cthw(path, target_hw, device, dtype, max_frames=None):
    import torch
    if not _is_video_file(path):
        return _load_image_chw(path, target_hw, device, dtype).unsqueeze(1)
    frames = _decode_video_frames(path, max_frames=max_frames)
    chw = [_frame_to_chw(fr, target_hw, device, dtype) for fr in frames]
    return torch.stack(chw, dim=1)

def bernini_generate_resident(transformer_dir, prompt, *, task=None, reference_image_paths=None, source_video_path=None, reference_video_path=None, height=480, width=832, num_frames=1, steps=40, guidance_scale=5.0, shift=3.0, negative_prompt='', seed=0, umt5_file='', umt5_config_dir=None, umt5_tokenizer_dir=None, vae_path='', device='cuda', dtype=None, progress=None, should_cancel=None):
    import gc
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from diffusers.video_processor import VideoProcessor
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from . import wan_backend
    _preflight_resident_ram()
    DEV, DTYPE = (device, dtype or torch.bfloat16)
    refs = list(reference_image_paths or [])
    want_video = num_frames > 1
    task = task or detect_task(want_video=want_video, source_video=source_video_path, reference_video=reference_video_path, reference_images=refs)
    full_prompt = task_prefixed_prompt(task, prompt)
    print(f'[bernini] task={task} prompt={full_prompt!r} refs={len(refs)} {height}x{width}x{num_frames}f', flush=True)
    tf = WanTransformer3DModel.from_pretrained(transformer_dir, torch_dtype=DTYPE).to(DEV).eval()
    patch_wan_in_context(tf)
    VSS = 8
    vae = (AutoencoderKLWan.from_single_file(vae_path) if __import__('pathlib').Path(vae_path).is_file() else AutoencoderKLWan.from_pretrained(vae_path)).to(DEV, torch.float32).eval()
    vae = wan_backend._configure_wan_vae_memory(vae, label='[bernini]')
    video_processor = VideoProcessor(vae_scale_factor=VSS)
    import glob as _glob, os as _os
    from pathlib import Path as _Path
    snap_root = _Path(transformer_dir).parent
    snap_te, snap_tok = (snap_root / 'text_encoder', snap_root / 'tokenizer')
    from .backends import hf_hub_dirs as _hub_dirs
    _ref = [s for _hub in _hub_dirs() for s in _glob.glob(_os.path.join(_hub, 'models--Wan-AI--Wan2.2-T2V-A14B-Diffusers', 'snapshots', '*'))]
    if umt5_config_dir:
        cfg_dir = umt5_config_dir
    elif (snap_te / 'config.json').is_file():
        cfg_dir = str(snap_te)
    elif _ref:
        cfg_dir = _os.path.join(_ref[0], 'text_encoder')
    else:
        cfg_dir = str(_Path(umt5_file).parent)
    if umt5_tokenizer_dir:
        tok_dir = umt5_tokenizer_dir
    elif (snap_tok / 'tokenizer_config.json').is_file() or (snap_tok / 'spiece.model').is_file():
        tok_dir = str(snap_tok)
    elif _ref:
        tok_dir = _os.path.join(_ref[0], 'tokenizer')
    else:
        tok_dir = cfg_dir
    text_encoder = wan_backend.build_streamed_umt5(umt5_file, cfg_dir, dtype=DTYPE, device=DEV)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    @torch.no_grad()
    def _encode(text):
        text = [prompt_clean(text)]
        ti = tokenizer(text, padding='max_length', max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        seq_lens = ti.attention_mask.gt(0).sum(dim=1).long()
        emb = text_encoder(ti.input_ids.to(DEV), ti.attention_mask.to(DEV)).last_hidden_state.to(DTYPE)
        emb = [u[:v] for u, v in zip(emb, seq_lens)]
        return torch.stack([torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in emb], dim=0)
    do_cfg = guidance_scale > 1.0
    prompt_embeds = _encode(full_prompt).to(DTYPE)
    negative_prompt_embeds = _encode(negative_prompt or '').to(DTYPE) if do_cfg else None
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    restore_ctx = None
    if refs or source_video_path or reference_video_path:

        @torch.no_grad()
        def _encode_ref(chw):
            t = chw.unsqueeze(0).unsqueeze(2).to(DEV, vae.dtype)
            lat = vae.encode(t).latent_dist.mode()
            return lat

        @torch.no_grad()
        def _encode_ref_video(cthw):
            t = cthw.unsqueeze(0).to(DEV, vae.dtype)
            return vae.encode(t).latent_dist.mode()
        ctx = []
        if source_video_path:
            if _is_video_file(source_video_path):
                ctx.append(_encode_ref_video(_load_video_cthw(source_video_path, (height, width), DEV, vae.dtype, max_frames=num_frames if want_video else 1)))
            else:
                ctx.append(_encode_ref(_load_image_chw(source_video_path, (height, width), DEV, vae.dtype)))
        if reference_video_path:
            if _is_video_file(reference_video_path):
                ctx.append(_encode_ref_video(_load_video_cthw(reference_video_path, (height, width), DEV, vae.dtype, max_frames=num_frames if want_video else 1)))
            else:
                ctx.append(_encode_ref(_load_image_chw(reference_video_path, (height, width), DEV, vae.dtype)))
        for rp in refs:
            ctx.append(_encode_ref(_load_image_chw(rp, (height, width), DEV, vae.dtype)))
        restore_ctx = bind_context_latents(tf, [c.to(DTYPE) for c in ctx])
        print(f'[bernini] bound {len(ctx)} context latent(s)', flush=True)
    patch = tf.config.patch_size
    h_mult, w_mult = (VSS * patch[1], VSS * patch[2])
    height = int(height) // h_mult * h_mult
    width = int(width) // w_mult * w_mult
    num_latent_frames = (num_frames - 1) // 4 + 1
    shape = (1, tf.config.in_channels, num_latent_frames, height // VSS, width // VSS)
    g = torch.Generator(device=DEV).manual_seed(int(seed))
    latents = torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
    scheduler, _ = wan_backend._build_wan_scheduler('unipc', shift)
    scheduler.set_timesteps(steps, device=DEV)
    scheduler.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            lmi = latents.to(DTYPE)
            ts = t.to(DEV).expand(1)
            pred = tf(hidden_states=lmi, timestep=ts, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            if do_cfg:
                un = tf(hidden_states=lmi, timestep=ts, encoder_hidden_states=negative_prompt_embeds, return_dict=False)[0]
                pred = un + guidance_scale * (pred - un)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            if progress:
                progress(i + 1, len(scheduler.timesteps), None)
        if restore_ctx:
            restore_ctx()
    del prompt_embeds, negative_prompt_embeds, tf
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        latents = latents.to(vae.dtype)
        z = vae.config.z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / std + mean
        video = vae.decode(latents, return_dict=False)[0]
        frames = video_processor.postprocess_video(video, output_type='pil')
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return frames[0]
