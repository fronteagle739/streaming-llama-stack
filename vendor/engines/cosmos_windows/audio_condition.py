# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/audio_condition.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

def require_codec(support):
    path = Path(str((support or {}).get('cosmos_audio_codec') or ''))
    if not (support or {}).get('cosmos_audio_codec') or not path.is_dir():
        raise ValueError('Audio-conditioned Cosmos video needs a local complete Cosmos AVAE codec with its encoder')
    config = path / 'config.json'
    if not config.is_file() or json.loads(config.read_text(encoding='utf-8')).get('_class_name') != 'AutoencoderOobleck':
        raise ValueError('Cosmos audio codec must be a complete compatible AutoencoderOobleck package')
    if not any(path.glob('*.safetensors')):
        raise ValueError('Cosmos audio codec has no local encoder/decoder weights')
    return path

def decoder_fingerprint(module):
    import torch
    digest = hashlib.sha256()
    for value in module.state_dict().values():
        tensor = value.detach().to(device='cpu').contiguous()
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def load_codec(pipe, path):
    import psutil
    from diffusers import AutoencoderOobleck
    size = sum((p.stat().st_size for p in path.glob('*.safetensors')))
    if not size:
        raise ValueError('Complete Cosmos audio codec has no local safetensors weights')
    if psutil.virtual_memory().available < 2 * size + 2 * 1024 ** 3:
        raise ValueError('Not enough free host memory to load the Cosmos audio encoder without paging')
    codec = AutoencoderOobleck.from_pretrained(str(path), local_files_only=True, torch_dtype=pipe.sound_tokenizer.dtype)
    if decoder_fingerprint(codec.decoder) != decoder_fingerprint(pipe.sound_tokenizer.decoder):
        raise ValueError('Audio codec decoder weights do not match the selected Cosmos model')
    expected = int(pipe.transformer.config.sound_dim)
    if int(codec.config.decoder_input_channels) != expected:
        raise ValueError('Audio codec latent channels do not match Cosmos sound tokens')
    if int(codec.config.sampling_rate) != int(pipe.sound_tokenizer.config.sampling_rate):
        raise ValueError('Audio codec sample rate does not match Cosmos')
    if math.prod(codec.config.downsampling_ratios) != pipe.sound_tokenizer._hop_size:
        raise ValueError('Audio codec temporal scale does not match Cosmos')
    codec.decoder = None
    return codec

def encode_reference(pipe, codec, waveform, sample_rate, *, frames, fps, should_cancel=None):
    import torch
    import torch.nn.functional as F
    if should_cancel and should_cancel():
        raise RuntimeError('Cosmos audio conditioning cancelled before encoding')
    if waveform.ndim != 2 or waveform.shape[1] not in (1, 2) or (not torch.isfinite(waveform).all()):
        raise ValueError('Cosmos reference audio must have finite mono or stereo samples')
    if sample_rate <= 0 or waveform.shape[0] < 1:
        raise ValueError('Cosmos reference audio has no valid sample rate or samples')
    target_rate = int(pipe.sound_tokenizer.config.sampling_rate)
    if sample_rate != target_rate:
        import scipy.signal
        from math import gcd
        divisor = gcd(int(sample_rate), target_rate)
        samples = scipy.signal.resample_poly(waveform.cpu().numpy(), target_rate // divisor, sample_rate // divisor, axis=0)
        waveform = torch.from_numpy(samples.copy())
    target_samples = int(frames / fps * target_rate)
    waveform = waveform[:target_samples]
    if waveform.shape[1] == 1:
        waveform = waveform.expand(-1, 2)
    samples = waveform.T[None]
    hop = int(pipe.sound_tokenizer._hop_size)
    length = (target_samples + hop - 1) // hop
    samples = F.pad(samples, (0, length * hop - samples.shape[-1]))
    codec.to(device=pipe._execution_device)
    with torch.no_grad():
        encoded = codec.encode(samples.to(device=pipe._execution_device, dtype=codec.dtype)).latent_dist.mode()[0]
    if encoded.shape != (int(pipe.transformer.config.sound_dim), length) or not torch.isfinite(encoded).all():
        raise ValueError('Cosmos audio encoder returned incompatible or non-finite sound tokens')
    if should_cancel and should_cancel():
        raise RuntimeError('Cosmos audio conditioning cancelled after encoding')
    return (encoded, {'reference_rate': sample_rate, 'model_rate': target_rate, 'samples': target_samples, 'sound_tokens': length, 'num_frames': frames, 'fps': fps, 'mode': 'condition', 'decoder_sha256': decoder_fingerprint(pipe.sound_tokenizer.decoder)})

@contextmanager
def audio_condition(pipe, encoded):
    import torch
    original_prepare = pipe.prepare_latents
    original_forward = pipe.transformer.forward

    def prepare(*args, **kwargs):
        if not kwargs.get('enable_sound'):
            raise ValueError('Cosmos audio conditioning requires joint sound/vision tokens')
        result = list(original_prepare(*args, **kwargs))
        if result[1] is None or result[1].shape != encoded.shape:
            raise ValueError('Cosmos reference sound tokens do not match the generated timeline')
        result[1] = encoded.to(device=result[1].device, dtype=result[1].dtype)
        from .conditioning_layout import mask_indexes
        _, sound_mask = mask_indexes(result)
        result[sound_mask] = torch.ones((encoded.shape[1], 1), device=result[1].device, dtype=result[1].dtype)
        return tuple(result)

    def forward(*args, **kwargs):
        if kwargs.get('sound_tokens') is None:
            raise ValueError('Cosmos audio conditioning lost its sound tokens')
        device = kwargs['sound_tokens'][0].device
        kwargs['sound_timesteps'] = torch.empty(0, device=device)
        kwargs['sound_noisy_frame_indexes'] = [torch.empty(0, device=device, dtype=torch.long)]
        kwargs['sound_mse_loss_indexes'] = torch.empty(0, device=device, dtype=torch.long)
        return original_forward(*args, **kwargs)
    try:
        pipe.prepare_latents = prepare
        pipe.transformer.forward = forward
        yield
    finally:
        pipe.prepare_latents = original_prepare
        pipe.transformer.forward = original_forward
