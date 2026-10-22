# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_text_encode.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from pathlib import Path
from typing import Optional
_TE_PREFIX = 'model.language_model.'
_TE_LAYER_RE = 'model\\.language_model\\.layers\\.(\\d+)\\.'

def _materialize_computed_buffers(model, device: str, torch) -> None:
    for module in model.modules():
        if module.__class__.__name__.endswith('RotaryEmbedding') and hasattr(module, 'config'):
            with torch.device(device):
                fresh = module.__class__(module.config)
            for bname, bval in fresh.named_buffers(recurse=False):
                module.register_buffer(bname, bval.to(device), persistent=False)
            if hasattr(fresh, 'attention_scaling'):
                module.attention_scaling = fresh.attention_scaling
    leftover = [n for n, b in model.named_buffers() if getattr(b, 'is_meta', False)]
    if leftover:
        raise RuntimeError(f'{len(leftover)} computed buffers left on meta after streaming and could not be re-materialised: {leftover[:8]}. Zeroing them would be a silent correctness bug; add an explicit re-materialisation for these before running the encoder.')

def build_streamed_magi2_text_encoder(text_encoder_dir: str | Path, *, device: str='cuda', skip_layer: int=2, max_length: int=7000, dtype=None):
    import torch
    from transformers import AutoConfig, Qwen3_5TextModel
    import inference.model.qwen35 as q35
    from cosmos_windows.weight_streaming.streamed_text_encoder import build_streamed_text_encoder
    dtype = dtype if dtype is not None else torch.bfloat16
    text_encoder_dir = str(text_encoder_dir)

    def _ckpt_to_module(name: str) -> Optional[str]:
        if name.startswith(_TE_PREFIX):
            return name[len(_TE_PREFIX):]
        return None

    def _streamed_from_pretrained(model_path, torch_dtype=None, **kwargs):
        cfg = AutoConfig.from_pretrained(model_path).text_config
        with torch.device('meta'):
            model = Qwen3_5TextModel(cfg)
        build_streamed_text_encoder(model, model_path, layer_regex=_TE_LAYER_RE, decoder_layers=model.layers, ckpt_to_module=_ckpt_to_module, device=device, dtype=torch_dtype if torch_dtype is not None else dtype, log_prefix='[MAGI2-TE-STREAM]')
        _materialize_computed_buffers(model, device, torch)
        model.to = lambda *a, **k: model
        return model
    had_own = 'from_pretrained' in Qwen3_5TextModel.__dict__
    prev = Qwen3_5TextModel.__dict__.get('from_pretrained')
    Qwen3_5TextModel.from_pretrained = staticmethod(_streamed_from_pretrained)
    try:
        encoder = q35.Qwen35TextEncoder(model_path=text_encoder_dir, device=device, precision=dtype, max_length=max_length, skip_layer=skip_layer)
    finally:
        if had_own:
            Qwen3_5TextModel.from_pretrained = prev
        else:
            del Qwen3_5TextModel.from_pretrained
    return encoder
