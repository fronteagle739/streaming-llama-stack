# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_extension.py
# Machine fallback paths cleared; no model data included.
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import torch
OVERLAP_FRAMES = 39
VIDEO_STEPS = 12
AUDIO_TICKS = 65

def extension_prefix(video, audio, geometry):
    if geometry.frames < OVERLAP_FRAMES or video.shape[2] < VIDEO_STEPS:
        raise ValueError('H3 seamless extension needs at least 39 source frames')
    if tuple(video.shape[:2]) != (1, 24) or tuple(audio.shape[:2]) != (2, 32):
        raise ValueError('Invalid H3 continuation latent shape')
    if video.shape[2] != geometry.video_t or audio.shape[-1] != geometry.audio_t:
        raise ValueError('H3 continuation geometry does not match its latents')
    return (video[:, :, -VIDEO_STEPS:].clone(), audio[:, :, -AUDIO_TICKS:].clone())

def install_prefix(video, audio, layout, prefix):
    from .minimax_h3_pipeline import patchify_h3_video
    pv = patchify_h3_video(prefix[0]).to(video)
    pa = prefix[1].permute(0, 2, 1).to(audio)
    if len(pv) >= len(video) or pa.shape[1] >= len(audio) // 2:
        raise ValueError('H3 extension window must exceed its 39-frame context')
    vm = torch.ones(len(video), dtype=torch.bool, device=video.device)
    am = torch.ones((2, len(audio) // 2), dtype=torch.bool, device=audio.device)
    vm[:len(pv)] = False
    am[:, :pa.shape[1]] = False
    video[:len(pv)] = pv
    audio.view(2, -1, 32)[:, :pa.shape[1]] = pa
    layout['update_mask'] = vm
    layout['audio_update_mask'] = am.flatten()
    layout['held_video_pos'] = layout.get('img_output_pos', layout['img_pos'])[:len(pv)]
    layout['held_audio_pos'] = layout.get('audio_output_pos', layout['audio_pos']).reshape(2, -1)[:, :pa.shape[1]].flatten()
    return (video[~vm].clone(), audio[~am.flatten()].clone())

def restore_prefix(video, audio, layout, held):
    video[~layout['update_mask']] = held[0]
    audio[~layout['audio_update_mask']] = held[1]

def context_path(video_path):
    return Path(video_path).with_suffix('.h3-context.safetensors')

def _digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save_context(path, video, audio, geometry):
    from safetensors.torch import save_file
    save_file({'video': video.detach().cpu().contiguous(), 'audio': audio.detach().cpu().contiguous()}, str(context_path(path)), metadata={'version': '1', 'geometry': json.dumps(asdict(geometry)), 'video_sha256': _digest(path)})

def load_context(path, *, width, height, device='cpu'):
    from safetensors import safe_open
    from .minimax_h3_pipeline import H3LatentGeometry
    with safe_open(str(context_path(path)), framework='pt', device=str(device)) as f:
        meta = f.metadata()
        if meta.get('version') != '1' or meta.get('video_sha256') != _digest(path):
            raise ValueError('H3 context is unsupported or its video has changed')
        g = H3LatentGeometry(**json.loads(meta['geometry']))
        if (g.width, g.height) != (width, height):
            raise ValueError('H3 seamless extension requires the source resolution')
        return (f.get_tensor('video'), f.get_tensor('audio'), g)
