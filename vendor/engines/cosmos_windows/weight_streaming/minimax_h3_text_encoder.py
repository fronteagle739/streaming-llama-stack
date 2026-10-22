# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_text_encoder.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import time
H3_QWEN_SELECTED_LAYERS = 50
_LAYER_RE = re.compile('^model\\.language_model\\.layers\\.(\\d+)\\.')

@dataclass(frozen=True)
class H3QwenLayout:
    selected_layers: int
    selected_layer_bytes: int
    language_global_bytes: int
    vision_global_bytes: int
    intentionally_skipped_bytes: int
    complete: bool

def group_h3_qwen_layers(index):
    layers: dict[int, list] = {}
    language_globals = []
    vision_globals = []
    skipped = []
    for name in sorted(index.refs):
        ref = index.refs[name]
        match = _LAYER_RE.match(name)
        if match:
            layer = int(match.group(1))
            (layers.setdefault(layer, []) if layer < H3_QWEN_SELECTED_LAYERS else skipped).append(ref)
        elif name == 'lm_head.weight' or name.startswith('model.language_model.norm.') or 'rotary_emb.inv_freq' in name:
            skipped.append(ref)
        elif name.startswith('model.visual.'):
            vision_globals.append(ref)
        else:
            language_globals.append(ref)
    return (layers, language_globals, vision_globals, skipped)

def inspect_h3_qwen_encoder(model_dir: str | Path) -> H3QwenLayout:
    from .safetensors_index import build_index
    index = build_index(Path(model_dir))
    layers, language, vision, skipped = group_h3_qwen_layers(index)
    return H3QwenLayout(selected_layers=len(layers), selected_layer_bytes=sum((ref.nbytes for refs in layers.values() for ref in refs)), language_global_bytes=sum((ref.nbytes for ref in language)), vision_global_bytes=sum((ref.nbytes for ref in vision)), intentionally_skipped_bytes=sum((ref.nbytes for ref in skipped)), complete=tuple(sorted(layers)) == tuple(range(H3_QWEN_SELECTED_LAYERS)))

def build_meta_h3_qwen_encoder(model_dir: str | Path, *, runtime_device='cpu'):
    import torch
    from transformers import AutoConfig, Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    class CheckpointRoot(torch.nn.Module):

        def __init__(self, config):
            super().__init__()
            self.model = Qwen3VLModel(config)

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)
    config = AutoConfig.from_pretrained(Path(model_dir), local_files_only=True)
    config.text_config.num_hidden_layers = H3_QWEN_SELECTED_LAYERS
    with torch.device('meta'):
        model = CheckpointRoot(config)
    model.model.language_model.norm = torch.nn.Identity()
    model.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config, device=runtime_device)
    return model

def build_streamed_h3_qwen_encoder(model_dir: str | Path, *, device='cuda', include_vision: bool=False, n_gpu_buffers: int=3, host_staging_buffers: int=2, event_callback=None):
    from accelerate.utils import set_module_tensor_to_device
    from .ds_backend import DirectStorageStreamer, enforce_ds_mode
    from .safetensors_index import build_index
    root = Path(model_dir)
    index = build_index(root)
    layers, language_globals, vision_globals, _ = group_h3_qwen_layers(index)
    layout = inspect_h3_qwen_encoder(root)
    if not layout.complete:
        raise ValueError(f'H3 Qwen encoder has {layout.selected_layers}/50 selected layers')
    model = build_meta_h3_qwen_encoder(root, runtime_device=device)
    state_keys = set(model.state_dict())

    def emit(event, **payload):
        if event_callback:
            event_callback(event, payload)
    materialized = language_globals + (vision_globals if include_vision else [])
    import torch
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    staging_bytes = n_gpu_buffers * max((sum((ref.nbytes for ref in refs)) for refs in layers.values()))
    need_bytes = sum((ref.nbytes for ref in materialized)) + staging_bytes + 1500000000
    if free_bytes < need_bytes:
        raise RuntimeError(f'H3 text encoder needs {need_bytes / 1000000000.0:.2f} GB free including DirectStorage staging; only {free_bytes / 1000000000.0:.2f} GB available. No upload started.')
    started = time.perf_counter()
    from .minimax_h3_vae_mux import _bind_resident_directstorage
    _bind_resident_directstorage(model, index, materialized, device=device, label='MiniMax H3 Qwen globals')
    emit('text_encoder_globals_loaded', seconds=time.perf_counter() - started, bytes=sum((ref.nbytes for ref in materialized)), include_vision=include_vision)
    for block_id, refs in layers.items():
        missing = [ref.name for ref in refs if ref.name not in state_keys]
        if missing:
            raise ValueError(f'H3 Qwen layer {block_id} checkpoint/model mismatch: {missing[:4]}')
    streamer = DirectStorageStreamer(index, component='text_encoder', n_gpu_buffers=n_gpu_buffers, device=device, layer_groups=layers, host_staging_buffers=host_staging_buffers)
    enforce_ds_mode(streamer, 'MiniMax H3 Qwen layer-50 encoder', 'MINIMAX_H3_TEXT_ENCODER')
    module_layers = model.model.language_model.layers
    for block_id, layer in enumerate(module_layers):
        raw_forward = type(layer).forward

        def wrapped(*args, _id=block_id, _layer=layer, _forward=raw_forward, **kwargs):
            transfer_start = time.perf_counter()
            loaded = streamer.wait_layer(_id)
            for name, tensor in loaded.items():
                set_module_tensor_to_device(model, name, device, value=tensor)
            emit('text_encoder_layer_bound', layer=_id, seconds=time.perf_counter() - transfer_start, bytes=sum((ref.nbytes for ref in layers[_id])))
            try:
                return _forward(_layer, *args, **kwargs)
            finally:
                for ref in layers[_id]:
                    set_module_tensor_to_device(model, ref.name, 'meta')
                streamer.release_layer(_id)
        layer.forward = wrapped
    model.register_forward_pre_hook(lambda _module, _args, _kwargs=None: streamer.new_pass(), with_kwargs=True)
    model.eval()
    return (model, streamer, layout)

def release_h3_qwen_encoder(model, streamer) -> None:
    import torch
    from accelerate.utils import set_module_tensor_to_device
    for name in list(model.state_dict()):
        set_module_tensor_to_device(model, name, 'meta')
    streamer.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def encode_h3_text_prompt(model_dir: str | Path, prompt: str, *, device='cuda', event_callback=None):
    import torch
    from transformers import AutoTokenizer
    if not prompt:
        raise ValueError('prompt must be non-empty')
    root = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    token_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if not token_ids:
        raise ValueError('prompt tokenized to an empty presentation')
    model, streamer, _ = build_streamed_h3_qwen_encoder(root, device=device, include_vision=False, event_callback=event_callback)
    started = time.perf_counter()
    try:
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            hidden = model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
        hidden = hidden.to(torch.bfloat16)
        if list(hidden.shape) != [len(token_ids), 5120]:
            raise ValueError(f'unexpected H3 text embedding shape: {list(hidden.shape)}')
        if not bool(torch.isfinite(hidden).all().item()):
            raise RuntimeError('H3 text encoder produced non-finite embeddings')
        if event_callback:
            event_callback('text_encoding_complete', {'seconds': time.perf_counter() - started, 'tokens': len(token_ids), 'hidden_dim': 5120})
        return hidden
    finally:
        release_h3_qwen_encoder(model, streamer)
