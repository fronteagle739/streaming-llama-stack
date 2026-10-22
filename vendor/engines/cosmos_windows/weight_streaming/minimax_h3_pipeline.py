# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_pipeline.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import dataclass
import math
import os
import numpy as np
import torch
H3_FPS = 24
H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32

@dataclass(frozen=True)
class H3LatentGeometry:
    requested_duration: float
    duration: float
    frames: int
    width: int
    height: int
    video_t: int
    video_h: int
    video_w: int
    audio_t: int
    video_rows: int
    audio_rows: int

def align_h3_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    return int(frame_count) + (5 - int(frame_count)) % 17

def h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return (int(frame_count) - 5) // 17 * 5 + 2

def h3_latent_geometry(duration: float, width: int, height: int) -> H3LatentGeometry:
    if duration <= 0:
        raise ValueError(f'H3 duration must be positive (got {duration:.3f}s)')
    if duration > 60:
        raise ValueError(f'H3 duration must be at most 60 seconds (got {duration:.3f}s)')
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError('H3 width and height must be positive multiples of 32')
    frames = align_h3_frame_count(round(duration * H3_FPS))
    aligned_duration = frames / H3_FPS
    video_t = h3_video_latent_t(frames)
    video_h, video_w = (height // 16, width // 16)
    audio_t = round(aligned_duration * 40)
    return H3LatentGeometry(requested_duration=float(duration), duration=aligned_duration, frames=frames, width=width, height=height, video_t=video_t, video_h=video_h, video_w=video_w, audio_t=audio_t, video_rows=video_t * (video_h // 2) * (video_w // 2), audio_rows=audio_t * 2)

def patchify_h3_video(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError('H3 video latent must have shape [1,24,T,H,W]')
    _, channels, frames, height, width = latent.shape
    if height % 2 or width % 2:
        raise ValueError('H3 latent height/width must be divisible by patch size 2')
    return latent.reshape(1, channels, frames, height // 2, 2, width // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, channels * 4)

def unpatchify_h3_video(rows: torch.Tensor, geometry: H3LatentGeometry) -> torch.Tensor:
    expected = (geometry.video_rows, H3_VIDEO_CHANNELS * 4)
    if tuple(rows.shape) != expected:
        raise ValueError(f'H3 video rows must have shape {expected}, got {tuple(rows.shape)}')
    return rows.reshape(1, geometry.video_t, geometry.video_h // 2, geometry.video_w // 2, H3_VIDEO_CHANNELS, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(1, H3_VIDEO_CHANNELS, geometry.video_t, geometry.video_h, geometry.video_w)

def prepare_h3_visual_condition_rows(latent: torch.Tensor, *, latents_mean, latents_std) -> torch.Tensor:
    if latent.ndim != 5 or int(latent.shape[0]) != 1 or int(latent.shape[1]) != 24:
        raise ValueError('H3 visual condition latent must be [1,24,T,H,W]')
    mean = torch.as_tensor(latents_mean, dtype=torch.float32, device='cpu')
    std = torch.as_tensor(latents_std, dtype=torch.float32, device='cpu')
    if tuple(mean.shape) != (24,) or tuple(std.shape) != (24,):
        raise ValueError('H3 visual latent statistics must contain 24 channels')
    if torch.any(std == 0):
        raise ValueError('H3 visual latent standard deviations must be non-zero')
    rounded = latent.detach().to(device='cpu', dtype=torch.float16).float()
    normalized = (rounded - mean.view(1, 24, 1, 1, 1)) / std.view(1, 24, 1, 1, 1)
    return patchify_h3_video(normalized.contiguous())

def prepare_h3_initial_noise(geometry: H3LatentGeometry, *, seed: int=42, device='cpu', dtype=torch.float32):
    video_generator = torch.Generator(device='cpu').manual_seed(int(seed))
    video_raw = torch.randn(1, H3_VIDEO_CHANNELS, geometry.video_t, geometry.video_h, geometry.video_w, generator=video_generator, dtype=torch.float32)
    video = patchify_h3_video(video_raw).to(device=device, dtype=dtype)
    audio = torch.randn(geometry.audio_rows, H3_AUDIO_CHANNELS, generator=torch.Generator(device='cpu').manual_seed(int(seed)), dtype=torch.float32).to(device=device, dtype=dtype)
    return (video, audio)

def noise_h3_keyframe_rows(clean_rows: torch.Tensor, geometry: H3LatentGeometry, *, keyframe_count: int, seed: int, noise_aug: float=0.999):
    frame_rows = geometry.video_h // 2 * (geometry.video_w // 2)
    if tuple(clean_rows.shape) != (keyframe_count * frame_rows, 96):
        raise ValueError('H3 clean keyframe rows have incompatible shape')
    outputs = []
    for index in range(keyframe_count):
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(1, 24, geometry.video_t + keyframe_count, geometry.video_h, geometry.video_w, generator=generator, dtype=torch.float32)[:, :, :1]
        noise_rows = patchify_h3_video(noise).to(clean_rows.device)
        clean = clean_rows[index * frame_rows:(index + 1) * frame_rows].float()
        outputs.append(noise_aug * clean + (1 - noise_aug) * noise_rows)
    return torch.cat(outputs, dim=0).contiguous()

def noise_h3_visual_reference_rows(clean_rows: torch.Tensor, geometry: H3LatentGeometry, *, shapes: list[tuple[int, int, int]], seed: int, noise_aug: float=0.999):
    expected = sum((t * (h // 2) * (w // 2) for t, h, w in shapes))
    if tuple(clean_rows.shape) != (expected, 96):
        raise ValueError('H3 visual reference rows have incompatible shape')
    outputs, offset = ([], 0)
    condition_frames = sum((t for t, _h, _w in shapes))
    for latent_t, latent_h, latent_w in shapes:
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(1, 24, geometry.video_t + condition_frames, latent_h, latent_w, generator=generator, dtype=torch.float32)[:, :, :latent_t]
        noise_rows = patchify_h3_video(noise).to(clean_rows.device)
        count = noise_rows.shape[0]
        clean = clean_rows[offset:offset + count].float()
        offset += count
        outputs.append(noise_aug * clean + (1 - noise_aug) * noise_rows)
    return torch.cat(outputs, dim=0).contiguous()

def build_h3_t2v_packed_layout(geometry: H3LatentGeometry, *, text_len: int) -> dict:
    if text_len < 1:
        raise ValueError('H3 text length must be positive')
    patch_h, patch_w = (geometry.video_h // 2, geometry.video_w // 2)
    frame_rows = patch_h * patch_w
    audio_rows = geometry.audio_t * 2
    video_rows = geometry.video_t * frame_rows
    used = text_len + audio_rows + video_rows
    seq_len = math.ceil(used / 64) * 64
    text_pos = torch.arange(text_len)
    audio_pos = torch.arange(text_len, text_len + audio_rows)
    img_pos = torch.arange(text_len + audio_rows, used)
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    spans = torch.tensor([5 / 3 * (1, 4, 4, 4, 4)[index % 5] for index in range(geometry.video_t)], dtype=torch.float64)
    video_time = text_len + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    sqrt_area = math.sqrt(geometry.video_h * geometry.video_w)

    def axis(dimension, patch):
        ratio = dimension / sqrt_area
        left = (1 - ratio) / 2
        right = left + ratio
        return torch.from_numpy(np.linspace(left, right, dimension // patch, endpoint=False) * 32).double()
    height_axis, width_axis = (axis(geometry.video_h, 2), axis(geometry.video_w, 2))
    hh, ww = torch.meshgrid(height_axis, width_axis, indexing='ij')
    spatial = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
    video_grid = torch.empty(geometry.video_t, frame_rows, 3, dtype=positions.dtype)
    video_grid[:, :, 0] = video_time[:, None]
    video_grid[:, :, 1:] = spatial[None]
    positions[img_pos] = video_grid.reshape(-1, 3)
    audio_time = text_len + torch.arange(geometry.audio_t, dtype=torch.float64)
    positions[audio_pos, 0] = audio_time.repeat(2)
    positions[text_len:text_len + geometry.audio_t, 2] = width_axis[0]
    positions[text_len + geometry.audio_t:text_len + audio_rows, 2] = width_axis[-1]
    tags = torch.full((seq_len,), -1, dtype=torch.long)
    tags[text_pos] = 1
    tags[audio_pos] = 2
    tags[img_pos] = 0
    return {'seq_len': seq_len, 'used_len': used, 'text_pos': text_pos, 'audio_pos': audio_pos, 'img_pos': img_pos, 'token_tags': tags, 'img_position_ids': positions, 'update_mask': torch.ones(video_rows, dtype=torch.bool), 'cu_seqlens': torch.tensor([0, used, seq_len], dtype=torch.int32)}

def build_h3_fl2v_packed_layout(geometry: H3LatentGeometry, *, text_len: int, frame_indices: tuple[int, ...]) -> dict:
    if frame_indices not in ((0,), (-1,), (0, -1)):
        raise ValueError('H3 FL2V frame indices must be (0,), (-1,), or (0,-1)')
    base = build_h3_t2v_packed_layout(geometry, text_len=text_len)
    frame_rows = geometry.video_h // 2 * (geometry.video_w // 2)
    cond_rows = len(frame_indices) * frame_rows
    audio_rows = geometry.audio_rows
    video_rows = geometry.video_rows
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = math.ceil(used / 64) * 64
    text_pos = torch.arange(text_len)
    cond_pos = torch.arange(text_len, text_len + cond_rows)
    audio_pos = torch.arange(cond_pos[-1] + 1 if cond_rows else text_len, text_len + cond_rows + audio_rows)
    target_pos = torch.arange(text_len + cond_rows + audio_rows, used)
    img_input_pos = torch.cat((cond_pos, target_pos))
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[text_pos, 0] = torch.arange(text_len, dtype=torch.float64)
    target_grid = base['img_position_ids'].index_select(0, base['img_pos'])
    positions[target_pos] = target_grid
    positions[audio_pos] = base['img_position_ids'].index_select(0, base['audio_pos'])
    spatial = target_grid[:frame_rows, 1:]
    spans = torch.tensor([5 / 3 * (1, 4, 4, 4, 4)[index % 5] for index in range(geometry.video_t)], dtype=torch.float64)
    temporal_span = float(spans.numpy().sum())
    for block_index, semantic_index in enumerate(frame_indices):
        rows = cond_pos[block_index * frame_rows:(block_index + 1) * frame_rows]
        positions[rows, 0] = float(text_len) if semantic_index == 0 else float(text_len) + temporal_span - 5 / 3
        positions[rows, 1:] = spatial
    tags = torch.full((seq_len,), -1, dtype=torch.long)
    tags[text_pos] = 1
    tags[audio_pos] = 2
    tags[img_input_pos] = 0
    return {'seq_len': seq_len, 'used_len': used, 'text_pos': text_pos, 'audio_pos': audio_pos, 'img_pos': img_input_pos, 'img_input_pos': img_input_pos, 'img_output_pos': target_pos, 'condition_img_pos': cond_pos, 'token_tags': tags, 'img_position_ids': positions, 'update_mask': torch.ones(video_rows, dtype=torch.bool), 'cu_seqlens': torch.tensor([0, used, seq_len], dtype=torch.int32)}

def build_h3_ref2v_packed_layout(geometry: H3LatentGeometry, *, text_len: int, ref_blocks: list[dict], frame_indices: tuple[int, ...]=()) -> dict:
    if frame_indices not in ((), (0,), (-1,), (0, -1)):
        raise ValueError('H3 keyframe indices must be (), (0,), (-1,), or (0,-1)')
    target_area = math.sqrt(geometry.video_h * geometry.video_w)

    def axis(dimension, area):
        ratio = dimension / area
        left = (1 - ratio) / 2
        return torch.from_numpy(np.linspace(left, left + ratio, dimension // 2, endpoint=False) * 32).double()
    target_h, target_w = (axis(geometry.video_h, target_area), axis(geometry.video_w, target_area))
    target_hh, target_ww = torch.meshgrid(target_h, target_w, indexing='ij')
    target_frame = torch.stack((target_hh.flatten(), target_ww.flatten()), dim=-1)
    cursor = text_len
    frame_rows = geometry.video_h // 2 * (geometry.video_w // 2)
    cond_pos = torch.arange(cursor, cursor + len(frame_indices) * frame_rows)
    cursor += int(cond_pos.numel())
    parsed = []
    ref_visual_rows = ref_audio_rows = 0
    for raw in ref_blocks:
        kind = raw.get('kind')
        if kind == 'image':
            h, w = (int(raw['latent_h']), int(raw['latent_w']))
            rows = h // 2 * (w // 2)
            sl = slice(cursor, cursor + rows)
            cursor += rows
            parsed.append({**raw, 'visual_slice': sl})
            ref_visual_rows += rows
        elif kind == 'audio':
            t = int(raw['ref_audio_t'])
            rows = t * 2
            sl = slice(cursor, cursor + rows)
            cursor += rows
            parsed.append({**raw, 'audio_slice': sl})
            ref_audio_rows += rows
        elif kind in ('video', 'video_audio'):
            t, vt = (int(raw.get('ref_audio_t', 0)), int(raw['latent_t']))
            h, w = (int(raw['latent_h']), int(raw['latent_w']))
            ar, vr = (t * 2, vt * (h // 2) * (w // 2))
            asl = slice(cursor, cursor + ar)
            vsl = slice(asl.stop, asl.stop + vr)
            cursor = vsl.stop
            parsed.append({**raw, 'audio_slice': asl, 'visual_slice': vsl})
            ref_audio_rows += ar
            ref_visual_rows += vr
        else:
            raise ValueError(f'unsupported H3 reference kind: {kind!r}')
    target_audio_pos = torch.arange(cursor, cursor + geometry.audio_rows)
    cursor += geometry.audio_rows
    target_img_pos = torch.arange(cursor, cursor + geometry.video_rows)
    cursor += geometry.video_rows
    used = cursor
    seq_len = math.ceil(used / 64) * 64
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    text_pos = torch.arange(text_len)
    positions[text_pos, 0] = text_pos.double()
    ref_imgs, ref_audio = ([], [])
    time_origin = float(text_len)

    def video_times(count, origin):
        spans = torch.tensor([5 / 3 * (1, 4, 4, 4, 4)[i % 5] for i in range(count)], dtype=torch.float64)
        return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))

    def video_span(count):
        return sum((5 / 3 * (1, 4, 4, 4, 4)[i % 5] for i in range(count)))
    for item in parsed:
        kind = item['kind']
        if kind == 'image':
            sl = item['visual_slice']
            ids = torch.arange(sl.start, sl.stop)
            ref_imgs.append(ids)
            h, w = (int(item['latent_h']), int(item['latent_w']))
            area = math.sqrt(h * w)
            hh, ww = torch.meshgrid(axis(h, area), axis(w, area), indexing='ij')
            positions[ids, 0] = time_origin
            positions[ids, 1] = hh.flatten()
            positions[ids, 2] = ww.flatten()
            time_origin += 1
        elif kind == 'audio':
            sl = item['audio_slice']
            ids = torch.arange(sl.start, sl.stop)
            ref_audio.append(ids)
            t = int(item['ref_audio_t'])
            positions[ids, 0] = (time_origin + torch.arange(t).double()).repeat(2)
            if t:
                positions[ids[:t], 2] = target_w[0]
                positions[ids[t:], 2] = target_w[-1]
            time_origin += t
        else:
            asl, vsl = (item['audio_slice'], item['visual_slice'])
            aids, vids = (torch.arange(asl.start, asl.stop), torch.arange(vsl.start, vsl.stop))
            ref_audio.append(aids)
            ref_imgs.append(vids)
            t, vt = (int(item.get('ref_audio_t', 0)), int(item['latent_t']))
            h, w = (int(item['latent_h']), int(item['latent_w']))
            area = math.sqrt(h * w)
            rh, rw = (axis(h, area), axis(w, area))
            hh, ww = torch.meshgrid(rh, rw, indexing='ij')
            if t:
                positions[aids, 0] = (time_origin + torch.arange(t).double()).repeat(2)
                positions[aids[:t], 2] = rw[0]
                positions[aids[t:], 2] = rw[-1]
            frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
            grid = torch.empty(vt, frame.shape[0], 3, dtype=positions.dtype)
            grid[..., 0] = video_times(vt, time_origin)[:, None]
            grid[..., 1:] = frame[None]
            positions[vids] = grid.reshape(-1, 3)
            time_origin += max(t, video_span(vt))
    positions[target_audio_pos, 0] = (time_origin + torch.arange(geometry.audio_t).double()).repeat(2)
    positions[target_audio_pos[:geometry.audio_t], 2] = target_w[0]
    positions[target_audio_pos[geometry.audio_t:], 2] = target_w[-1]
    target_grid = torch.empty(geometry.video_t, target_frame.shape[0], 3, dtype=positions.dtype)
    target_grid[..., 0] = video_times(geometry.video_t, time_origin)[:, None]
    target_grid[..., 1:] = target_frame[None]
    positions[target_img_pos] = target_grid.reshape(-1, 3)
    for block_index, semantic_index in enumerate(frame_indices):
        rows = cond_pos[block_index * frame_rows:(block_index + 1) * frame_rows]
        positions[rows, 0] = time_origin if semantic_index == 0 else time_origin + video_span(geometry.video_t) - 5 / 3
        positions[rows, 1:] = target_frame
    img_input = torch.cat((cond_pos, *ref_imgs, target_img_pos))
    audio_input = torch.cat((*ref_audio, target_audio_pos)) if ref_audio else target_audio_pos
    tags = torch.full((seq_len,), -1, dtype=torch.long)
    tags[text_pos] = 1
    tags[img_input] = 0
    tags[audio_input] = 2
    reference_img_pos = torch.cat(ref_imgs) if ref_imgs else torch.empty(0, dtype=torch.long)
    out = {'seq_len': seq_len, 'used_len': used, 'text_pos': text_pos, 'img_pos': img_input, 'img_input_pos': img_input, 'img_output_pos': target_img_pos, 'audio_pos': audio_input, 'audio_input_pos': audio_input, 'audio_output_pos': target_audio_pos, 'reference_img_pos': reference_img_pos, 'reference_audio_pos': torch.cat(ref_audio) if ref_audio else torch.empty(0, dtype=torch.long), 'token_tags': tags, 'img_position_ids': positions, 'update_mask': torch.ones(geometry.video_rows, dtype=torch.bool), 'audio_update_mask': torch.ones(geometry.audio_rows, dtype=torch.bool), 'cu_seqlens': torch.tensor([0, used, seq_len], dtype=torch.int32)}
    if frame_indices:
        out['keyframe_img_pos'] = cond_pos
        out['condition_img_pos'] = torch.cat((cond_pos, reference_img_pos))
    return out

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, '')
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not a number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive, got {value}')
    return value

def h3_time_shift_sigmas(*, steps: int, shift_scale: float) -> list[float]:
    if steps <= 0 or not math.isfinite(shift_scale) or shift_scale <= 0:
        raise ValueError('H3 steps and shift scale must be positive')
    base = torch.linspace(1.0, 0.0, steps, device='cpu', dtype=torch.float32)
    shifted = shift_scale * base / (1 + (shift_scale - 1) * base)
    shifted = torch.unique_consecutive(shifted)
    if steps > 1 and shifted[-1].item() > 0:
        shifted = torch.cat((shifted, torch.zeros(1, dtype=shifted.dtype)))
    return [float(value) for value in shifted]
H3_SIGMA_SCHEDULES = ('time_shift', 'linear', 'normal', 'simple', 'sgm_uniform', 'ddim_uniform', 'karras', 'exponential', 'linear_quadratic', 'beta', 'beta57', 'kl_optimal')
H3_SIGMA_MIN = 0.001

def h3_sigma_schedule(name: str, *, steps: int, shift_scale: float) -> list[float]:
    if steps <= 0:
        raise ValueError('H3 steps must be positive')
    name = str(name or 'time_shift')
    if name not in H3_SIGMA_SCHEDULES:
        raise ValueError(f"unknown H3 sigma schedule {name!r}; choose from {', '.join(H3_SIGMA_SCHEDULES)}")
    if name == 'time_shift':
        return h3_time_shift_sigmas(steps=steps, shift_scale=shift_scale)
    if name in ('linear', 'normal', 'simple'):
        return h3_time_shift_sigmas(steps=steps, shift_scale=1.0)
    n = max(1, steps - 1)
    if n == 1:
        return [1.0, 0.0]
    if name == 'karras':
        rho = 7.0
        lo, hi = (H3_SIGMA_MIN ** (1 / rho), 1.0)
        values = [(hi + i / (n - 1) * (lo - hi)) ** rho for i in range(n)]
    elif name == 'exponential':
        lo, hi = (math.log(H3_SIGMA_MIN), math.log(1.0))
        values = [math.exp(hi + i / (n - 1) * (lo - hi)) for i in range(n)]
    elif name == 'sgm_uniform':
        values = [1.0 - i / n for i in range(n)]
    elif name == 'ddim_uniform':
        values = [1.0] + [1.0 - (i + 0.5) / n for i in range(1, n)]
    elif name == 'kl_optimal':
        hi, lo = (math.atan(1.0), math.atan(H3_SIGMA_MIN))
        values = [math.tan(hi + i / (n - 1) * (lo - hi)) for i in range(n)]
    elif name == 'linear_quadratic':
        half = max(1, n // 2)
        linear = [1.0 - i / (2 * half) for i in range(half)]
        start, tail = (linear[-1], n - half)
        values = linear + [start * (1 - (i + 1) / (tail + 1)) ** 2 for i in range(tail)]
    else:
        from scipy.stats import beta as _beta
        alpha, betav = (0.5, 0.7) if name == 'beta57' else (0.6, 0.6)
        values = [1.0] + [1.0 - float(_beta.ppf((i + 1) / n, alpha, betav)) for i in range(n - 1)]
    values = [min(1.0, max(H3_SIGMA_MIN, float(v))) for v in values]
    values[0] = 1.0
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            values[i] = values[i - 1] * 0.5
    return values + [0.0]

def h3_rf_v_to_x0(state: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    if state.shape != velocity.shape:
        raise ValueError('H3 state and velocity shapes must match')
    cond_t = timestep.to(device=state.device, dtype=state.dtype)
    while cond_t.ndim < state.ndim:
        cond_t = cond_t.unsqueeze(-1)
    return state + (1 - cond_t) * velocity

def h3_euler_eta0_step(state: torch.Tensor, denoised: torch.Tensor, *, sigma_current: float, sigma_next: float) -> torch.Tensor:
    if state.shape != denoised.shape:
        raise ValueError('H3 state and denoised shapes must match')
    if sigma_current < 0 or sigma_next < 0:
        raise ValueError('H3 sigmas must be non-negative')
    if sigma_current == 0:
        if sigma_next != 0:
            raise ValueError('sigma_next must be zero when sigma_current is zero')
        return state
    compute_dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
    ratio = state.new_tensor(sigma_next / sigma_current, dtype=compute_dtype)
    result = ratio * state.to(compute_dtype) + (1 - ratio) * denoised.to(compute_dtype)
    return result.to(state.dtype)
H3_SAMPLERS = ('euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde')
H3_STOCHASTIC_SAMPLERS = frozenset({'euler_ancestral', 'dpmpp_2m_sde'})

def h3_ancestral_sigmas(sigma_current: float, sigma_next: float, eta: float=1.0):
    if sigma_next <= 0 or sigma_current <= 0 or eta <= 0:
        return (sigma_next, 0.0)
    up = min(sigma_next, eta * math.sqrt(max(0.0, sigma_next ** 2 * (sigma_current ** 2 - sigma_next ** 2) / max(sigma_current ** 2, 1e-12))))
    down = math.sqrt(max(0.0, sigma_next ** 2 - up ** 2))
    return (down, up)

def h3_dpmpp_2m_x0(denoised: torch.Tensor, previous: torch.Tensor | None, *, sigma_prev: float, sigma_current: float, sigma_next: float):
    if previous is None or sigma_next <= 0 or sigma_prev <= 0 or (sigma_current <= 0):
        return denoised
    h = math.log(sigma_current / sigma_next)
    h_last = math.log(sigma_prev / sigma_current)
    if h <= 0 or h_last <= 0:
        return denoised
    r = h_last / h
    compute_dtype = torch.float32 if denoised.dtype in (torch.float16, torch.bfloat16) else denoised.dtype
    scale = 1.0 / (2.0 * r)
    blended = (1 + scale) * denoised.to(compute_dtype) - scale * previous.to(compute_dtype)
    return blended.to(denoised.dtype)

def h3_sampler_step(video: torch.Tensor, audio: torch.Tensor, video_velocity: torch.Tensor, audio_velocity: torch.Tensor, *, video_sigmas, audio_sigmas, step_index: int, sampler: str='euler', previous=None, noise_seed: int=42):
    sampler = str(sampler or 'euler')
    if sampler not in H3_SAMPLERS:
        raise ValueError(f"unknown H3 sampler {sampler!r}; choose from {', '.join(H3_SAMPLERS)}")
    eta = 1.0 if sampler in H3_STOCHASTIC_SAMPLERS else 0.0
    v_cur, v_next = (float(video_sigmas[step_index]), float(video_sigmas[step_index + 1]))
    a_cur, a_next = (float(audio_sigmas[step_index]), float(audio_sigmas[step_index + 1]))
    video_x0 = h3_rf_v_to_x0(video, video_velocity, video.new_tensor(1 - v_cur, dtype=torch.float32))
    audio_x0 = h3_rf_v_to_x0(audio, audio_velocity, audio.new_tensor(1 - a_cur, dtype=torch.float32))
    if sampler in ('euler', 'euler_ancestral'):
        v_use, a_use = (video_x0, audio_x0)
    else:
        prev_v, prev_a, v_prev_sigma, a_prev_sigma = previous or (None, None, 0.0, 0.0)
        v_use = h3_dpmpp_2m_x0(video_x0, prev_v, sigma_prev=v_prev_sigma, sigma_current=v_cur, sigma_next=v_next)
        a_use = h3_dpmpp_2m_x0(audio_x0, prev_a, sigma_prev=a_prev_sigma, sigma_current=a_cur, sigma_next=a_next)
    v_down, v_up = h3_ancestral_sigmas(v_cur, v_next, eta)
    a_down, a_up = h3_ancestral_sigmas(a_cur, a_next, eta)
    video_out = h3_euler_eta0_step(video, v_use, sigma_current=v_cur, sigma_next=v_down)
    audio_out = h3_euler_eta0_step(audio, a_use, sigma_current=a_cur, sigma_next=a_down)
    if v_up > 0 or a_up > 0:
        generator = torch.Generator(device='cpu').manual_seed(int(noise_seed) + step_index)
        if v_up > 0:
            video_out = video_out + torch.randn(video_out.shape, generator=generator, dtype=torch.float32).to(video_out.device, video_out.dtype) * v_up
        if a_up > 0:
            audio_out = audio_out + torch.randn(audio_out.shape, generator=generator, dtype=torch.float32).to(audio_out.device, audio_out.dtype) * a_up
    return (video_out, audio_out, (video_x0, audio_x0, v_cur, a_cur))

def h3_denoise_step(video: torch.Tensor, audio: torch.Tensor, video_velocity: torch.Tensor, audio_velocity: torch.Tensor, *, video_sigma_current: float, video_sigma_next: float, audio_sigma_current: float, audio_sigma_next: float):
    video_t = video.new_tensor(1 - video_sigma_current, dtype=torch.float32)
    audio_t = audio.new_tensor(1 - audio_sigma_current, dtype=torch.float32)
    video_x0 = h3_rf_v_to_x0(video, video_velocity, video_t)
    audio_x0 = h3_rf_v_to_x0(audio, audio_velocity, audio_t)
    return (h3_euler_eta0_step(video, video_x0, sigma_current=video_sigma_current, sigma_next=video_sigma_next), h3_euler_eta0_step(audio, audio_x0, sigma_current=audio_sigma_current, sigma_next=audio_sigma_next))

def _h3_vram_peaks(device) -> dict[str, float]:
    try:
        if not torch.cuda.is_available():
            return {}
        return {'peak_alloc_gb': round(torch.cuda.max_memory_allocated(device) / 1000000000.0, 2), 'peak_reserved_gb': round(torch.cuda.max_memory_reserved(device) / 1000000000.0, 2)}
    except Exception:
        return {}

def _h3_tensor_stats(tensor: torch.Tensor) -> dict[str, float | bool]:
    values = tensor.float()
    return {'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max()), 'rms': float(values.square().mean().sqrt()), 'finite': bool(torch.isfinite(values).all())}

def denoise_h3_t2v(model_dir, prompt_embeddings: torch.Tensor, *, duration: float, width: int, height: int, steps: int, seed: int=42, device='cuda', host_cache_gb: float=0.0, resident: bool=False, loras=(), session: dict | None=None, attention_backend: str='sdpa', condition_video_rows: torch.Tensor | None=None, frame_indices: tuple[int, ...]=(), reference_video_rows: torch.Tensor | None=None, reference_audio_rows: torch.Tensor | None=None, ref_blocks: list[dict] | None=None, event_callback=None, sparse_attention=None, action_text_spans=None, world_head_tags=None, extension_context=None):
    from pathlib import Path
    import time
    from .minimax_h3_backend import build_streamed_h3_transformer
    from .minimax_h3_native import build_meta_h3_native_model
    from .safetensors_index import build_index
    geometry = h3_latent_geometry(duration, width, height)
    video, audio = prepare_h3_initial_noise(geometry, seed=seed, device=device, dtype=torch.float32)
    if ref_blocks is not None:
        if (condition_video_rows is None) != (not frame_indices):
            raise ValueError('frame_indices and condition_video_rows must be given together')
        layout = build_h3_ref2v_packed_layout(geometry, text_len=prompt_embeddings.shape[0], ref_blocks=ref_blocks, frame_indices=tuple(frame_indices))
        if condition_video_rows is not None:
            expected = layout['keyframe_img_pos'].numel()
            if tuple(condition_video_rows.shape) != (expected, 96):
                raise ValueError(f'H3 condition rows must have shape ({expected},96), got {tuple(condition_video_rows.shape)}')
            condition_video_rows = noise_h3_keyframe_rows(condition_video_rows.to(device=device, dtype=video.dtype), geometry, keyframe_count=len(frame_indices), seed=seed).to(device=device, dtype=video.dtype)
        visual_count = layout['reference_img_pos'].numel()
        audio_count = layout['reference_audio_pos'].numel()
        if visual_count:
            if reference_video_rows is None or tuple(reference_video_rows.shape) != (visual_count, 96):
                raise ValueError(f'H3 reference visual rows must have shape ({visual_count},96)')
            shapes = []
            for item in ref_blocks:
                if item['kind'] == 'image':
                    shapes.append((1, int(item['latent_h']), int(item['latent_w'])))
                elif item['kind'] in ('video', 'video_audio'):
                    shapes.append((int(item['latent_t']), int(item['latent_h']), int(item['latent_w'])))
            reference_video_rows = noise_h3_visual_reference_rows(reference_video_rows, geometry, shapes=shapes, seed=seed).to(device=device, dtype=video.dtype)
        if audio_count:
            if reference_audio_rows is None or tuple(reference_audio_rows.shape) != (audio_count, 32):
                raise ValueError(f'H3 reference audio rows must have shape ({audio_count},32)')
            reference_audio_rows = reference_audio_rows.to(device=device, dtype=audio.dtype)
    elif condition_video_rows is None:
        if frame_indices:
            raise ValueError('frame_indices require condition_video_rows')
        layout = build_h3_t2v_packed_layout(geometry, text_len=prompt_embeddings.shape[0])
    else:
        layout = build_h3_fl2v_packed_layout(geometry, text_len=prompt_embeddings.shape[0], frame_indices=frame_indices)
        expected = layout['condition_img_pos'].numel()
        if tuple(condition_video_rows.shape) != (expected, 96):
            raise ValueError(f'H3 condition rows must have shape ({expected},96), got {tuple(condition_video_rows.shape)}')
        condition_video_rows = condition_video_rows.to(device=device, dtype=video.dtype)
        condition_video_rows = noise_h3_keyframe_rows(condition_video_rows, geometry, keyframe_count=len(frame_indices), seed=seed).to(device=device, dtype=video.dtype)
    if action_text_spans is not None:
        from .minimax_h3_world import bind_action_layout
        bind_action_layout(layout, action_text_spans, geometry, device)
        if world_head_tags is not None:
            layout['token_tags'][layout['text_pos'][:len(world_head_tags)]] = torch.tensor(world_head_tags)
        from .minimax_h3_sparse import H3SparseAttentionConfig
        from dataclasses import replace
        sparse_attention = replace(sparse_attention or H3SparseAttentionConfig(), mode='off')
        if event_callback:
            event_callback('world_conditioning', {'latents': geometry.video_t, 'groups': len(layout['action_groups']), 'routing': 'directed', 'sparse': 'disabled: directed routing takes precedence'})
    held = None
    if extension_context is not None:
        from .minimax_h3_extension import install_prefix
        held = install_prefix(video, audio, layout, extension_context)
    _cached = session is not None and 'model' in session
    if _cached:
        index = session['index']
        model, streamer = (session['model'], session['streamer'])
    else:
        transformer_dir = Path(model_dir)
        if (transformer_dir / 'transformer').is_dir():
            transformer_dir = transformer_dir / 'transformer'
        index = build_index(transformer_dir)
        from .minimax_h3_fused import index_looks_fused
        _fused = index_looks_fused(index.refs)
        model = build_meta_h3_native_model(transformer_dir / 'config.json', attention_backend=attention_backend, fused_adaln=_fused)
        if _fused:
            from .minimax_h3_int8 import convert_h3_linears_to_convrot, convert_h3_linears_to_int8, quantised_module_names
            from .minimax_h3_fused import checkpoint_quant_map
            _quantised = quantised_module_names(index.refs, checkpoint_quant_map(transformer_dir))
            _int8_matmul = os.environ.get('STUDIO_H3_INT8_MATMUL', '') == '1'
            _fused_int8 = os.environ.get('STUDIO_H3_FUSED_INT8', '0') == '1'
            if resident or _int8_matmul or _fused_int8:
                _converted = convert_h3_linears_to_int8(model, _quantised, float_matmul=_fused_int8 or not _int8_matmul)
            else:
                _converted = convert_h3_linears_to_convrot(model, _quantised)
            if _fused and (not _quantised):
                raise RuntimeError('H3 checkpoint is fused (adaln_t_table present) but NO quantised modules were detected: it carries neither .comfy_quant sidecars nor a file-level _quantization_metadata. Refusing to bind INT8 weights as dense.')
            if len(_converted) != len(_quantised):
                raise RuntimeError(f'H3 fused conversion covered {len(_converted)} of {len(_quantised)} quantised modules — refusing to run with a mixed block set')
        model, streamer = build_streamed_h3_transformer(model, index, device=device, host_cache_gb=host_cache_gb, resident=resident)
        if loras:
            from .minimax_h3_lora import attach_h3_lora_files
            attached = attach_h3_lora_files(model, loras, device=device)
            if event_callback:
                event_callback('lora_attached', {'paths': list(attached), 'count': len(attached)})
        if session is not None:
            session.update(index=index, model=model, streamer=streamer)
    from .minimax_h3_native import lean_block_configured
    from .minimax_h3_sparse import H3SparseAttentionController
    sparse_controller = H3SparseAttentionController(sparse_attention, device=device, lean_available=lean_block_configured())
    if event_callback:
        event_callback('sparse_attention_selected', sparse_controller.describe())
    used_len = int(layout.get('used_len', layout['seq_len']))
    _shift_v = _env_float('STUDIO_H3_SIGMA_SHIFT_VIDEO', 12.0)
    _shift_a = _env_float('STUDIO_H3_SIGMA_SHIFT_AUDIO', 3.0)
    _schedule = os.environ.get('STUDIO_H3_SIGMA_SCHEDULE', 'time_shift') or 'time_shift'
    _sampler = os.environ.get('STUDIO_H3_SAMPLER', 'euler') or 'euler'
    if _sampler not in H3_SAMPLERS:
        raise ValueError(f"unknown H3 sampler {_sampler!r}; choose from {', '.join(H3_SAMPLERS)}")
    _sampler_state = None
    from .minimax_h3_int8 import set_active_lora_pass
    _prepass = max(0, int(os.environ.get('STUDIO_H3_PREPASS_STEPS', '0') or 0))
    if _prepass >= steps:
        raise ValueError(f'prepass steps ({_prepass}) must leave at least one step for the main pass (steps={steps})')
    set_active_lora_pass(1 if _prepass else 0)
    video_sigmas = h3_sigma_schedule(_schedule, steps=steps + 1, shift_scale=_shift_v)
    audio_sigmas = h3_sigma_schedule(_schedule, steps=steps + 1, shift_scale=_shift_a)
    if event_callback:
        event_callback('sigma_schedule', {'steps': int(steps), 'shift_video': _shift_v, 'shift_audio': _shift_a, 'schedule': _schedule, 'sampler': _sampler, 'prepass_steps': _prepass, 'shift_applies': _schedule == 'time_shift', 'video_sigmas': [round(float(x), 4) for x in video_sigmas], 'video_gaps': [round(float(a) - float(b), 4) for a, b in zip(video_sigmas, video_sigmas[1:])]})
    from .minimax_h3_backend import reset_stream_stats, stream_stats
    reset_stream_stats()
    try:
        with torch.inference_mode():
            for step_index in range(len(video_sigmas) - 1):
                started = time.perf_counter()
                sparse_plan, attention_mode = sparse_controller.plan_for_step(sigma=video_sigmas[step_index], packed_layout=layout, seq_len=used_len, head_dim=model.config.attention_head_dim)
                if ref_blocks is not None:
                    pieces = [rows for rows in (condition_video_rows, reference_video_rows) if rows is not None]
                    model_video = torch.cat((*pieces, video), dim=0) if pieces else video
                    model_audio = audio if reference_audio_rows is None else torch.cat((reference_audio_rows, audio), dim=0)
                else:
                    model_video = video if condition_video_rows is None else torch.cat((condition_video_rows, video), dim=0)
                    model_audio = audio
                has_visual_condition = condition_video_rows is not None or (ref_blocks is not None and reference_video_rows is not None)
                video_velocity, audio_velocity = model(video_rows=model_video, audio_rows=model_audio, prompt_embeds=prompt_embeddings, video_timestep=1 - video_sigmas[step_index], audio_timestep=1 - audio_sigmas[step_index], condition_timestep=max(1 - video_sigmas[step_index], 0.999) if has_visual_condition else None, reference_audio_timestep=None if ref_blocks is None or reference_audio_rows is None else 1.0, packed_layout=layout, sparse_plan=sparse_plan)
                if _prepass and step_index == _prepass:
                    set_active_lora_pass(2)
                    if event_callback:
                        event_callback('lora_pass', {'pass': 2, 'step': step_index, 'prepass_steps': _prepass})
                video, audio, _sampler_state = h3_sampler_step(video, audio, video_velocity, audio_velocity, video_sigmas=video_sigmas, audio_sigmas=audio_sigmas, step_index=step_index, sampler=_sampler, previous=_sampler_state, noise_seed=int(seed))
                if held is not None:
                    from .minimax_h3_extension import restore_prefix
                    restore_prefix(video, audio, layout, held)
                if event_callback:
                    event_callback('denoise_step_complete', {'step': step_index + 1, 'steps': len(video_sigmas) - 1, 'seconds': time.perf_counter() - started, 'video_sigma': video_sigmas[step_index + 1], 'audio_sigma': audio_sigmas[step_index + 1], 'video_latent': _h3_tensor_stats(video), 'audio_latent': _h3_tensor_stats(audio), 'video_velocity': _h3_tensor_stats(video_velocity), 'audio_velocity': _h3_tensor_stats(audio_velocity), 'attention': attention_mode, **_h3_vram_peaks(device)})
        audio_latent = audio.reshape(2, geometry.audio_t, 32).permute(0, 2, 1).contiguous()
        if event_callback:
            _stats = stream_stats()
            event_callback('dit_stream_stats', {**_stats, 'gb': round(_stats['bytes'] / 1000000000.0, 3), 'gb_per_s': round(_stats['bytes'] / 1000000000.0 / _stats['wait_s'], 2) if _stats['wait_s'] > 0 else None})
        return (unpatchify_h3_video(video, geometry), audio_latent, geometry)
    finally:
        if session is None:
            streamer.close()
            del model
            torch.cuda.empty_cache()
