# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/timed_frames.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from contextlib import contextmanager
import hashlib
from pathlib import Path

def timeline(rows, *, frames, temporal_scale):
    if frames < 2:
        raise ValueError('Timed Cosmos references require a video with at least two frames')
    selected = {}
    for row in rows:
        path = str(row.get('image_path') or '').strip()
        if not path:
            raise ValueError('Every Cosmos keyframe needs a source image')
        frame = row.get('frame')
        fraction = row.get('at_fraction')
        if frame is None:
            if fraction is None or not 0 <= float(fraction) <= 1:
                raise ValueError('Every Cosmos keyframe needs a frame index or timeline fraction')
            frame = round(float(fraction) * (frames - 1))
        if type(frame) is not int or not 0 <= frame < frames:
            raise ValueError('Cosmos keyframe is outside the generated video')
        if fraction is not None and abs(float(fraction) * (frames - 1) - frame) > 0.5:
            raise ValueError('Cosmos keyframe frame index and timeline fraction disagree')
        strength = float(row.get('strength', 1))
        if not 0 <= strength <= 1:
            raise ValueError('Cosmos keyframe strength must be between zero and one')
        if not Path(path).is_file():
            raise ValueError('Cosmos keyframe image does not exist')
        latent = 0 if frame == 0 else (frame - 1) // temporal_scale + 1
        data = Path(path).read_bytes()
        receipt = {'frame': frame, 'at_fraction': frame / (frames - 1), 'latent_frame': latent, 'image_path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'strength': strength}
        old = selected.get(latent)
        if old and (old['sha256'], old['strength']) != (receipt['sha256'], strength):
            raise ValueError('Different Cosmos keyframes occupy the same temporal VAE cell; space them farther apart')
        if old and old['frame'] != frame:
            old.setdefault('additional_frames', []).append(frame)
        else:
            selected[latent] = receipt
    return sorted(selected.values(), key=lambda row: row['frame'])

@contextmanager
def timed_frames(pipe, rows, *, frames, height, width):
    import torch
    from PIL import Image
    scale = int(pipe.vae.config.scale_factor_temporal)
    if scale < 1:
        raise ValueError('Cosmos VAE reports an invalid temporal scale')
    references = timeline(rows, frames=frames, temporal_scale=scale)
    if not references:
        raise ValueError('Cosmos timed conditioning requires at least one reference')
    originals = {name: getattr(pipe, name) for name in ('prepare_latents', '_prepare_vision_segment')}
    state = {}

    def prepare_latents(*args, **kwargs):
        kwargs = {**kwargs, 'image': None}
        result = list(originals['prepare_latents'](*args, **kwargs))
        noise = result[0]
        device, dtype = (noise.device, noise.dtype)
        cells = noise.shape[2]
        pixels = torch.empty((1, 3, frames, height, width), device=device, dtype=dtype)
        decoded = []
        for row in references:
            if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Cosmos reference changed after conditioning preflight')
            with Image.open(row['image_path']) as image:
                decoded.append(pipe.video_processor.preprocess(image.convert('RGB'), height=height, width=width).to(device=device, dtype=dtype))
        for index in range(frames):
            cell = 0 if index == 0 else (index - 1) // scale + 1
            nearest = min(range(len(references)), key=lambda i: abs(references[i]['latent_frame'] - cell))
            pixels[:, :, index] = decoded[nearest]
        clean = pipe._encode_video(pixels).contiguous().to(device=device, dtype=dtype)
        if clean.shape != noise.shape:
            raise ValueError('Cosmos timed reference encoding does not match the video latent shape')
        condition = torch.zeros((cells, 1, 1), device=device, dtype=dtype)
        for row in references:
            if row['latent_frame'] >= cells:
                raise ValueError('Cosmos reference maps outside the encoded timeline')
            condition[row['latent_frame']] = row['strength']
        result[0] = condition * clean + (1 - condition) * noise
        from .conditioning_layout import mask_indexes
        vision_mask, _ = mask_indexes(result)
        result[vision_mask] = condition
        state['condition'] = condition
        return tuple(result)

    def segment(*args, **kwargs):
        kwargs = {**kwargs, 'has_image_condition': False}
        result = originals['_prepare_vision_segment'](*args, **kwargs)
        tokens = kwargs['input_vision_tokens']
        _, _, t, h, w = tokens.shape
        patch = int(pipe.transformer.config.latent_patch_size)
        stride = (h + patch - 1) // patch * ((w + patch - 1) // patch)
        noisy = torch.nonzero(state['condition'][:, 0, 0] < 1, as_tuple=False).flatten().long()
        offsets = noisy[:, None] * stride + torch.arange(stride, device=tokens.device)[None]
        result['vision_noisy_frame_indexes'] = [noisy]
        result['vision_mse_loss_indexes'] = offsets.flatten() + int(kwargs['curr'])
        result['num_noisy_vision_tokens'] = len(noisy) * stride
        return result
    try:
        pipe.prepare_latents = prepare_latents
        pipe._prepare_vision_segment = segment
        yield {'references': references, 'temporal_scale': scale, 'num_frames': frames}
    finally:
        for name, method in originals.items():
            setattr(pipe, name, method)
