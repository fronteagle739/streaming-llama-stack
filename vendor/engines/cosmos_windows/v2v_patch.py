# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/v2v_patch.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math

def _latent_frames_for(pipe, num_pixel_cond_frames: int) -> int:
    tc = int(getattr(pipe.vae.config, 'scale_factor_temporal', 1) or 1)
    return max(1, (int(num_pixel_cond_frames) - 1) // tc + 1)

def set_v2v_conditioning(pipe, cond_pixels, num_conditional_frames=None) -> int:
    if cond_pixels is None:
        clear_v2v_conditioning(pipe)
        return 0
    P = int(cond_pixels.shape[2])
    nc = P if num_conditional_frames is None else max(1, min(int(num_conditional_frames), P))
    pipe._v2v_cond_pixels = cond_pixels
    pipe._v2v_cond_pixel_frames = nc
    pipe._v2v_num_cond_latent = _latent_frames_for(pipe, nc)
    return pipe._v2v_num_cond_latent

def clear_v2v_conditioning(pipe) -> None:
    for attr in ('_v2v_cond_pixels', '_v2v_cond_pixel_frames', '_v2v_num_cond_latent'):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)

def _v2v_prepare_latents(self, image=None, num_frames=189, height=720, width=1280, fps=24.0, latents=None, sound_latents=None, generator=None, device='cuda', dtype=None, enable_sound=False):
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    cond_pixels = getattr(self, '_v2v_cond_pixels', None)
    if cond_pixels is None or num_frames <= 1:
        return self._v2v_orig_prepare_latents(image=image, num_frames=num_frames, height=height, width=width, fps=fps, latents=latents, sound_latents=sound_latents, generator=generator, device=device, dtype=dtype, enable_sound=enable_sound)
    if dtype is None:
        dtype = torch.bfloat16
    P = int(cond_pixels.shape[2])
    p = min(P, num_frames)
    vision_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
    vision_tensor[:, :, :p] = cond_pixels[:, :, :p].to(device=device, dtype=dtype)
    if p < num_frames:
        vision_tensor[:, :, p:] = cond_pixels[:, :, p - 1:p].to(device=device, dtype=dtype).expand(-1, -1, num_frames - p, -1, -1)
    x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
    vision_shape = tuple(x0_tokens_vision.shape)
    latent_t = x0_tokens_vision.shape[2]
    latent_P = min(_latent_frames_for(self, p), latent_t)
    self._v2v_num_cond_latent = latent_P
    x0_tokens_sound = None
    fps_sound = None
    if enable_sound:
        sound_dim = self.transformer.config.sound_dim
        fps_sound = float(self.transformer.config.sound_latent_fps)
        n_audio_samples = int(num_frames / fps * self.sound_tokenizer.config.sampling_rate)
        hop_size = self.sound_tokenizer._hop_size
        T_sound = (n_audio_samples + hop_size - 1) // hop_size
        x0_tokens_sound = torch.zeros(sound_dim, T_sound, device=device, dtype=dtype)
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_condition_mask[:latent_P, 0, 0] = 1.0
    if latents is None:
        pure_noise = randn_tensor(vision_shape, generator=generator, device=device, dtype=dtype)
        latents = vision_condition_mask * x0_tokens_vision.to(device=device, dtype=dtype) + (1.0 - vision_condition_mask) * pure_noise
    else:
        latents = latents.to(device=device, dtype=dtype)
    sound_condition_mask = None
    if enable_sound and x0_tokens_sound is not None:
        sound_condition_mask = torch.zeros((x0_tokens_sound.shape[1], 1), device=device, dtype=dtype)
        if sound_latents is None:
            pure_noise_sound = randn_tensor(tuple(x0_tokens_sound.shape), generator=generator, device=device, dtype=dtype)
            sound_latents = sound_condition_mask.T * x0_tokens_sound + (1.0 - sound_condition_mask.T) * pure_noise_sound
        else:
            sound_latents = sound_latents.to(device=device, dtype=dtype)
    return (latents, sound_latents, fps, fps_sound, vision_condition_mask, sound_condition_mask)

def _v2v_prepare_vision_segment(self, input_vision_tokens, has_image_condition, mrope_offset, vision_fps, curr, device):
    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_vae_tokens
    latent_P = getattr(self, '_v2v_num_cond_latent', None)
    if not has_image_condition or not latent_P or latent_P <= 1:
        return self._v2v_orig_prepare_vision_segment(input_vision_tokens=input_vision_tokens, has_image_condition=has_image_condition, mrope_offset=mrope_offset, vision_fps=vision_fps, curr=curr, device=device)
    config = self.transformer.config
    latent_patch_size = config.latent_patch_size
    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    num_vision_tokens = latent_t * patch_h * patch_w
    noisy_start = min(int(latent_P), latent_t)
    noisy_frame_indexes = torch.arange(noisy_start, latent_t, device=device, dtype=torch.long)
    frame_token_stride = patch_h * patch_w
    mse_loss_indexes = []
    for frame_idx in range(noisy_start, latent_t):
        frame_start = curr + frame_idx * frame_token_stride
        mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))
    effective_fps = vision_fps if config.enable_fps_modulation else None
    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=latent_t, grid_h=patch_h, grid_w=patch_w, temporal_offset=mrope_offset, reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids, fps=effective_fps, base_fps=float(config.base_fps), temporal_compression_factor=self.vae.config.scale_factor_temporal)
    return {'vision_token_shapes': [(latent_t, patch_h, patch_w)], 'vision_sequence_indexes': torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device), 'vision_mse_loss_indexes': torch.tensor(mse_loss_indexes, dtype=torch.long, device=device), 'vision_noisy_frame_indexes': [noisy_frame_indexes], 'vision_mrope_ids': vision_mrope_ids.to(device), 'num_vision_tokens': num_vision_tokens, 'num_noisy_vision_tokens': (latent_t - noisy_start) * frame_token_stride}

def enable_v2v(pipe) -> None:
    if getattr(pipe, '_v2v_patched', False):
        return
    pipe._v2v_orig_prepare_latents = pipe.prepare_latents
    pipe._v2v_orig_prepare_vision_segment = pipe._prepare_vision_segment
    import types
    pipe.prepare_latents = types.MethodType(_v2v_prepare_latents, pipe)
    pipe._prepare_vision_segment = types.MethodType(_v2v_prepare_vision_segment, pipe)
    pipe._v2v_patched = True

def disable_v2v(pipe) -> None:
    clear_v2v_conditioning(pipe)
    if getattr(pipe, '_v2v_patched', False):
        pipe.prepare_latents = pipe._v2v_orig_prepare_latents
        pipe._prepare_vision_segment = pipe._v2v_orig_prepare_vision_segment
        pipe._v2v_patched = False

def _frames_to_cond_pixels(frames, device, dtype):
    import torch
    import numpy as np
    if isinstance(frames, torch.Tensor):
        t = frames.float()
        if t.dim() == 5:
            pass
        elif t.dim() == 4 and t.shape[1] == 3:
            t = t.permute(1, 0, 2, 3).unsqueeze(0)
        elif t.dim() == 4 and t.shape[-1] == 3:
            t = t.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f'unsupported tensor shape {tuple(frames.shape)}')
        if float(t.max()) > 1.5:
            t = t / 255.0
        if float(t.min()) >= 0.0:
            t = t * 2.0 - 1.0
        return t.to(device=device, dtype=dtype)
    arr = np.stack([np.asarray(f).astype('float32') for f in frames], axis=0)
    if arr.max() > 1.5:
        arr = arr / 255.0
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t.to(device=device, dtype=dtype)

def generate_continuation(pipe, prompt, *, total_frames, chunk_frames=49, overlap=9, first_chunk_kwargs=None, chunk_kwargs=None, on_chunk=None):
    from PIL import Image
    if not getattr(pipe, '_v2v_patched', False):
        enable_v2v(pipe)
    device = getattr(pipe, '_execution_device', None) or getattr(pipe, 'device', 'cuda')
    dtype = getattr(pipe, 'dtype', None)
    overlap = max(1, int(overlap))
    chunk_frames = max(overlap + 1, int(chunk_frames))
    new_per_chunk = chunk_frames - overlap

    def _as_pil(video):
        if isinstance(video, list):
            return video
        import numpy as np, torch
        if isinstance(video, torch.Tensor):
            video = video.detach().float().cpu().numpy()
        if isinstance(video, np.ndarray):
            v = video
            if v.ndim == 4 and v.shape[1] in (1, 3):
                v = v.transpose(0, 2, 3, 1)
            v = (v * 255).clip(0, 255).astype('uint8') if v.max() <= 1.5 else v.astype('uint8')
            return [Image.fromarray(f) for f in v]
        raise ValueError('unrecognised video output type')
    out_frames: list = []
    num_chunks = max(1, math.ceil((int(total_frames) - chunk_frames) / new_per_chunk) + 1)
    for i in range(num_chunks):
        kw = dict(prompt=prompt, num_frames=chunk_frames, output_type='pil')
        if i == 0:
            kw.update(first_chunk_kwargs or {})
            clear_v2v_conditioning(pipe)
        else:
            kw.update(chunk_kwargs or {})
            tail = out_frames[-overlap:]
            cond = _frames_to_cond_pixels(tail, device, dtype)
            set_v2v_conditioning(pipe, cond, num_conditional_frames=overlap)
            kw['image'] = tail[0]
        video = pipe(**kw)
        frames = _as_pil(video.video if hasattr(video, 'video') else video[0] if isinstance(video, tuple) else video)
        clear_v2v_conditioning(pipe)
        contributed = frames if i == 0 else frames[overlap:]
        out_frames.extend(contributed)
        if on_chunk is not None:
            on_chunk(i, contributed)
        if len(out_frames) >= int(total_frames):
            break
    return out_frames[:int(total_frames)]

def supports_diffusion_forcing(pipe) -> bool:
    import os
    if os.environ.get('COSMOS_DIFFUSION_FORCING', '').strip().lower() in ('1', 'true', 'on', 'yes'):
        return True
    cfg = getattr(getattr(pipe, 'transformer', None), 'config', None)
    for flag in ('is_causal', 'causal_temporal', 'diffusion_forcing', 'independent_frame_noise', 'per_frame_timesteps', 'rolling_diffusion'):
        if bool(getattr(cfg, flag, False)):
            return True
    return False

def _selftest() -> bool:
    import torch

    class _Cfg:
        latent_patch_size = 2
        enable_fps_modulation = False
        unified_3d_mrope_reset_spatial_ids = True
        base_fps = 24.0
        sound_dim = 8
        sound_latent_fps = 24.0

    class _VaeCfg:
        scale_factor_temporal = 4
        scale_factor_spatial = 8

    class _Vae:
        config = _VaeCfg()

    class _FakePipe:

        def __init__(self):
            self.transformer = type('T', (), {'config': _Cfg()})()
            self.vae = _Vae()

        def _encode_video(self, x):
            b, c, T, H, W = x.shape
            lt = (T - 1) // 4 + 1
            return torch.zeros(1, 4, lt, H // 8, W // 8)

        def prepare_latents(self, **kw):
            lt = (kw.get('num_frames', 1) - 1) // 4 + 1
            return (torch.zeros(1, 4, lt, 1, 1), None, kw.get('fps'), None, torch.zeros(lt, 1, 1), None)

        def _prepare_vision_segment(self, input_vision_tokens, has_image_condition, mrope_offset, vision_fps, curr, device):
            lt = input_vision_tokens.shape[2]
            ns = 1 if has_image_condition else 0
            return {'vision_noisy_frame_indexes': [torch.arange(ns, lt, dtype=torch.long)], 'num_noisy_vision_tokens': lt - ns}
    pipe = _FakePipe()
    enable_v2v(pipe)
    enable_v2v(pipe)
    P, N, Hh, Ww = (9, 21, 256, 256)
    cond = torch.zeros(1, 3, P, Hh, Ww)
    latent_P = set_v2v_conditioning(pipe, cond, num_conditional_frames=P)
    assert latent_P == 3, f'latent_P expected 3 got {latent_P}'
    latents, sound_latents, fps_v, fps_s, vmask, smask = pipe.prepare_latents(image=None, num_frames=N, height=Hh, width=Ww, fps=24.0, generator=None, device='cpu', dtype=torch.float32, enable_sound=False)
    lt = latents.shape[2]
    assert lt == 6, f'latent_t expected 6 got {lt}'
    assert vmask.shape == (lt, 1, 1)
    assert float(vmask[:3].sum()) == 3.0 and float(vmask[3:].sum()) == 0.0, 'mask must anchor first 3 latent frames'
    seg = pipe._prepare_vision_segment(input_vision_tokens=latents, has_image_condition=True, mrope_offset=0, vision_fps=24.0, curr=10, device='cpu')
    noisy = seg['vision_noisy_frame_indexes'][0]
    assert noisy.tolist() == [3, 4, 5], f'noisy frames expected [3,4,5] got {noisy.tolist()}'
    patch_hw = math.ceil(Hh // 8 / 2) * math.ceil(Ww // 8 / 2)
    assert seg['num_noisy_vision_tokens'] == (lt - 3) * patch_hw, 'noisy token count must skip the 3 cond frames'
    clear_v2v_conditioning(pipe)
    seg0 = pipe._prepare_vision_segment(input_vision_tokens=latents, has_image_condition=True, mrope_offset=0, vision_fps=24.0, curr=10, device='cpu')
    assert seg0['vision_noisy_frame_indexes'][0].tolist() == [1, 2, 3, 4, 5], 'delegation to stock noisy_start=1 failed'
    from PIL import Image
    import numpy as np

    class _GenPipe(_FakePipe):
        _execution_device = 'cpu'
        dtype = torch.float32
        counter = 0

        def __call__(self, prompt=None, num_frames=49, output_type='pil', image=None, **kw):
            frames = []
            for _ in range(num_frames):
                _GenPipe.counter = (_GenPipe.counter + 1) % 256
                frames.append(Image.fromarray(np.full((8, 8, 3), _GenPipe.counter, dtype='uint8')))
            return frames
    gp = _GenPipe()
    enable_v2v(gp)
    chunk_frames, overlap, total = (49, 9, 120)
    cont = generate_continuation(gp, 'a drone shot', total_frames=total, chunk_frames=chunk_frames, overlap=overlap)
    assert len(cont) == total, f'continuation length expected {total} got {len(cont)}'
    assert all((isinstance(f, Image.Image) for f in cont)), 'continuation must yield PIL frames'
    assert not getattr(gp, '_v2v_cond_pixels', None), 'conditioning must be cleared after the run'
    assert supports_diffusion_forcing(gp) is False, 'stock model must report no diffusion-forcing'
    import os
    os.environ['COSMOS_DIFFUSION_FORCING'] = '1'
    assert supports_diffusion_forcing(gp) is True, 'env override must enable diffusion-forcing path'
    os.environ.pop('COSMOS_DIFFUSION_FORCING', None)
    gp.transformer.config.is_causal = True
    assert supports_diffusion_forcing(gp) is True, 'config flag must enable diffusion-forcing path'
    print(f'[v2v_patch] selftest PASS: latent_P=3, mask[0:3], noisy=[3,4,5], delegation OK; continuation {total}f over 3 chunks (49+40+40, seam-dropped); diffusion-forcing gate OK')
    return True
if __name__ == '__main__':
    raise SystemExit(0 if _selftest() else 1)
