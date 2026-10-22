# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/image_edit.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from contextlib import contextmanager
import math

def validate_edit(pipe, strength):
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Cosmos edit strength must be finite and between zero and one')
    for name in ('prepare_latents', '_encode_video', '_mask_velocity_predictions'):
        if not callable(getattr(pipe, name, None)):
            raise ValueError('Cosmos image editing requires the Cosmos 3 Omni pipeline')
    if not getattr(pipe.scheduler.config, 'use_flow_sigmas', False):
        raise ValueError('Cosmos image editing requires a flow-sigma scheduler')
    if not callable(getattr(pipe.scheduler, 'set_begin_index', None)):
        raise ValueError('Cosmos edit scheduler must support a starting timestep')
    return strength

@contextmanager
def image_edit(pipe, image, mask, *, strength, steps, height, width, generator):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from diffusers.utils.torch_utils import randn_tensor
    strength = validate_edit(pipe, strength)
    if strength == 0:
        raise ValueError('Zero-strength edits must return the original image before model loading')
    steps = int(steps)
    if steps < 1:
        raise ValueError('Cosmos edit needs at least one inference step')
    source = image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
    pixel_mask = mask.convert('L').resize((width, height), Image.Resampling.NEAREST) if mask is not None else Image.new('L', (width, height), 255)
    scheduler = pipe.scheduler
    original = {'prepare_latents': pipe.prepare_latents, 'set_timesteps': scheduler.set_timesteps}
    start = max(0, steps - max(1, math.ceil(steps * strength)))
    state = {}

    def set_timesteps(*args, **kwargs):
        original['set_timesteps'](*args, **kwargs)
        state['sigma'] = scheduler.sigmas[start].clone()
        scheduler.timesteps = scheduler.timesteps[start:]
        scheduler.sigmas = scheduler.sigmas[start:]
        scheduler.num_inference_steps = len(scheduler.timesteps)
        scheduler.set_begin_index(0)

    def prepare_latents(*args, **kwargs):
        if int(kwargs.get('num_frames', 1)) != 1 or kwargs.get('enable_sound'):
            raise ValueError('Cosmos image editing is a single-frame vision operation')
        device, dtype = (kwargs['device'], kwargs['dtype'])
        pixels = pipe.video_processor.preprocess(source, height=height, width=width).to(device=device, dtype=dtype)
        clean = pipe._encode_video(pixels.unsqueeze(2)).contiguous().float().to(device=device, dtype=dtype)
        noise = randn_tensor(clean.shape, generator=generator, device=device, dtype=dtype)
        mask_tensor = torch.as_tensor(np.asarray(pixel_mask).copy(), device=device, dtype=torch.float32)[None, None] / 255
        editable = F.interpolate(mask_tensor, size=clean.shape[-2:], mode='nearest')[0, 0][None].to(dtype=dtype)
        sigma = state['sigma'].to(device=device, dtype=dtype)
        initial = clean + editable * sigma * (noise - clean)
        state.update(clean=clean, editable=editable)
        from .conditioning_layout import image_latents
        return image_latents(original['prepare_latents'], initial, 1 - editable, kwargs.get('fps', 24))

    def constrain(callback=None):

        def on_step(p, index, timestep, values):
            editable, clean = (state['editable'], state['clean'])
            values['latents'] = editable * values['latents'] + (1 - editable) * clean
            if callback is not None:
                values = callback(p, index, timestep, values)
            return values
        return on_step

    def protect(decoded):
        return Image.composite(decoded.convert('RGB'), source, pixel_mask)
    try:
        scheduler.set_timesteps = set_timesteps
        scheduler.set_timesteps(steps, device=pipe._execution_device)
        pipe.prepare_latents = prepare_latents
        yield {'image': source, 'callback': constrain, 'protect': protect, 'effective_steps': steps - start}
    finally:
        pipe.prepare_latents = original['prepare_latents']
        scheduler.set_timesteps = original['set_timesteps']
        scheduler.set_timesteps(steps, device=pipe._execution_device)
