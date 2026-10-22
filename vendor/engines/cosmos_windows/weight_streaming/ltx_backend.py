# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/ltx_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import re
from pathlib import Path
_LTX_BLOCK_RE = re.compile('(?:^|[/.])transformer_blocks\\.(\\d+)\\.')
_LTX_REMAP = [('model.diffusion_model.', ''), ('audio_patchify_proj', 'audio_proj_in'), ('patchify_proj', 'proj_in'), ('av_ca_video_scale_shift_adaln_single', 'av_cross_attn_video_scale_shift'), ('av_ca_a2v_gate_adaln_single', 'av_cross_attn_video_a2v_gate'), ('av_ca_audio_scale_shift_adaln_single', 'av_cross_attn_audio_scale_shift'), ('av_ca_v2a_gate_adaln_single', 'av_cross_attn_audio_v2a_gate'), ('scale_shift_table_a2v_ca_video', 'video_a2v_cross_attn_scale_shift_table'), ('scale_shift_table_a2v_ca_audio', 'audio_a2v_cross_attn_scale_shift_table'), ('audio_prompt_adaln_single', 'audio_prompt_adaln'), ('prompt_adaln_single', 'prompt_adaln'), ('audio_adaln_single', 'audio_time_embed'), ('adaln_single', 'time_embed'), ('q_norm', 'norm_q'), ('k_norm', 'norm_k')]

def _remap_key(name: str) -> str:
    for old, new in _LTX_REMAP:
        if old in name:
            name = name.replace(old, new)
    return name

def _load_meta_ltx(model_file: str, config_dir: str | Path | None=None):
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    p = Path(model_file)
    cfg = None
    candidates = []
    if config_dir is not None:
        c = Path(config_dir)
        candidates.extend((c, c / 'config.json'))
    default_repo = globals().get('LTX23_DEFAULT_COMPONENTS_REPO')
    if default_repo:
        candidates.extend((Path(default_repo) / 'transformer', Path(default_repo) / 'transformer' / 'config.json'))
    candidates.extend((p.parent / 'config.json', p.parent / 'transformer' / 'config.json', p.parent / 'transformer'))
    for cand in candidates:
        try:
            if cand.exists():
                cfg = LTX2VideoTransformer3DModel.load_config(str(cand.parent if cand.name == 'config.json' else cand))
                break
        except Exception:
            cfg = None
    with init_empty_weights():
        if cfg is not None:
            model = LTX2VideoTransformer3DModel.from_config(cfg)
        else:
            model = LTX2VideoTransformer3DModel(num_layers=48, gated_attn=True, audio_gated_attn=True, cross_attn_mod=True, audio_cross_attn_mod=True, use_prompt_embeddings=False)
    return (model, model.config)

def ltx_group_by_block(idx):
    blocks, globs = ({}, [])
    for name, ref in idx.refs.items():
        m = _LTX_BLOCK_RE.search(name)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (blocks, globs)

class _LtxBlockIndex:

    def __init__(self, idx):
        self._idx = idx

    def group_by_layer(self, component=''):
        return ltx_group_by_block(self._idx)

    def __getattr__(self, k):
        return getattr(self._idx, k)

def build_streamed_ltx(model_file: str, device: str='cuda', n_gpu_buffers: int=3, dtype=None, config_dir: str | Path | None=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    from .build_streamed_transformer import StreamingWeightLoader
    dtype = dtype or torch.bfloat16
    model, cfg = _load_meta_ltx(model_file, config_dir=config_dir)
    tf = model
    idx = build_index(model_file)
    sd_keys = set(model.state_dict().keys())
    block_refs, global_refs = ltx_group_by_block(idx)
    pb = PortableBackend(pin_memory=True)
    n_glob_resident = 0
    for ref in global_refs:
        path = _remap_key(StreamingWeightLoader._param_path(ref))
        if path in sd_keys:
            value = pb.read_tensor(ref, device)
            if value.is_floating_point():
                value = value.to(dtype)
            set_module_tensor_to_device(model, path, device, value=value, dtype=dtype if value.is_floating_point() else None)
            n_glob_resident += 1
    streamer = DirectStorageStreamer(_LtxBlockIndex(idx), component='', n_gpu_buffers=n_gpu_buffers, device=device)
    n_blocks = len(tf.transformer_blocks)
    print(f'[LTX-STREAM] {streamer.n} blocks streamed (of {n_blocks} in the module), {n_glob_resident} resident globals (of {len(global_refs)} non-block tensors; rest = vae/vocoder/text-enc, skipped), mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)

    def _pp(name):
        return _remap_key(name.split('/', 1)[1] if '/' in name else name)

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    block_paths = {il: [_pp(r.name) for r in refs if _pp(r.name) in sd_keys] for il, refs in block_refs.items()}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    if t.is_floating_point():
                        t = t.to(dtype)
                    set_module_tensor_to_device(model, _resolve(p), device, value=t, dtype=dtype if t.is_floating_point() else None)
        return bind

    def make_evict(il):

        def evict():
            for p in block_paths[il]:
                set_module_tensor_to_device(model, _resolve(p), 'meta')
            streamer.release_layer(il)
        return evict
    for il, block in enumerate(tf.transformer_blocks):
        cls_forward = type(block).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *a, _block=block, _fwd=cls_forward, _b=bind, _e=evict, **kw):
            _b()
            try:
                out = _fwd(_block, hidden_states, *a, **kw)
            finally:
                _e()
            return out
        block.forward = wrapped
        block._sb_raw_forward = cls_forward
        block._sb_bind = bind
        block._sb_evict = evict
    model.eval()
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[LTX-STREAM] wrapped {n_blocks} LTX2VideoTransformerBlocks (bind/evict per block; linear forward loop, no cross-block skip connections — trivially streamable). Dense model: FETCH_ALL (no MoE).', flush=True)
    return (model, streamer)
LTX23_DEFAULT_MODEL_FILE = ''
LTX23_DEFAULT_COMPONENTS_REPO = ''
LTX23_DEFAULT_VIDEO_VAE_FILE = ''
LTX23_DEFAULT_AUDIO_VOCODER_FILE = ''
LTX23_DEFAULT_TEXT_PROJECTION_FILE = ''
LTX23_DEFAULT_TEXT_ENCODER_DIR = ''
LTX23_LORA_TARGETS = ['to_q', 'to_k', 'to_v', 'to_out.0', 'to_gate_logits', 'ff.net.0.proj', 'ff.net.2', 'audio_ff.net.0.proj', 'audio_ff.net.2']
from .vram_guard import cuda_free_mib
from .vram_guard import preflight_vram as _shared_preflight_vram

def preflight_vram(device: str, need_mib: int, label: str) -> None:
    _shared_preflight_vram(device, need_mib, label, skip_env='LTX_SKIP_VRAM_PREFLIGHT')

def _ltx23_existing_dir(path: str | Path | None, label: str) -> Path:
    p = Path(path) if path else Path()
    if p.exists():
        return p
    raise FileNotFoundError(f'{label} not found: {path}')

def _ltx23_existing_file(path: str | Path | None, label: str) -> Path:
    p = Path(path) if path else Path()
    if p.is_file():
        return p
    raise FileNotFoundError(f'{label} not found: {path}')

def _ltx23_cached_component_dir(component: str, components_repo: str | Path | None=None) -> Path:
    repo = Path(components_repo or LTX23_DEFAULT_COMPONENTS_REPO)
    local = repo / component
    if (local / 'config.json').exists() or (local / 'scheduler_config.json').exists():
        return local
    import glob
    import os
    from .backends import hf_hub_dirs
    matches = [Path(p) for hub in hf_hub_dirs() for p in glob.glob(os.path.join(hub, 'models--diffusers--LTX-2.3-Diffusers', 'snapshots', '*', component)) if (Path(p) / 'config.json').exists()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f'LTX 2.3 {component!r} config not found under {repo} or the HF cache. Run the small config download or point components_repo at a complete LTX-2.3 diffusers repo.')

def _ltx23_component_weight_files(component: str, components_repo: str | Path | None=None) -> list[Path]:
    root = Path(components_repo or LTX23_DEFAULT_COMPONENTS_REPO) / component
    mono = root / 'diffusion_pytorch_model.safetensors'
    if mono.is_file():
        return [mono]
    return sorted(root.glob('diffusion_pytorch_model-*.safetensors'))

def _ltx23_indexed_block_key(name: str, prefix: str, table: dict[int, str]) -> str:
    if not name.startswith(prefix):
        return name
    rest = name[len(prefix):]
    block, dot, tail = rest.partition('.')
    if block.isdigit() and int(block) in table:
        return table[int(block)] + ('.' + tail if dot else '')
    return name

def _map_ltx23_video_vae_key(name: str) -> str | None:
    if 'per_channel_statistics.channel' in name or 'per_channel_statistics.mean-of-stds' in name:
        return None
    if name.startswith('vae.'):
        name = name[4:]
    name = name.replace('res_blocks', 'resnets')
    name = name.replace('per_channel_statistics.mean-of-means', 'latents_mean')
    name = name.replace('per_channel_statistics.std-of-means', 'latents_std')
    encoder = {0: 'encoder.down_blocks.0', 1: 'encoder.down_blocks.0.downsamplers.0', 2: 'encoder.down_blocks.1', 3: 'encoder.down_blocks.1.downsamplers.0', 4: 'encoder.down_blocks.2', 5: 'encoder.down_blocks.2.downsamplers.0', 6: 'encoder.down_blocks.3', 7: 'encoder.down_blocks.3.downsamplers.0', 8: 'encoder.mid_block'}
    decoder = {0: 'decoder.mid_block', 1: 'decoder.up_blocks.0.upsamplers.0', 2: 'decoder.up_blocks.0', 3: 'decoder.up_blocks.1.upsamplers.0', 4: 'decoder.up_blocks.1', 5: 'decoder.up_blocks.2.upsamplers.0', 6: 'decoder.up_blocks.2', 7: 'decoder.up_blocks.3.upsamplers.0', 8: 'decoder.up_blocks.3'}
    name = _ltx23_indexed_block_key(name, 'encoder.down_blocks.', encoder)
    return _ltx23_indexed_block_key(name, 'decoder.up_blocks.', decoder)

def _map_ltx23_audio_vae_key(name: str) -> str | None:
    if name.startswith('audio_vae.'):
        name = name[len('audio_vae.'):]
    elif not name.startswith('per_channel_statistics.'):
        return None
    name = name.replace('per_channel_statistics.mean-of-means', 'latents_mean')
    return name.replace('per_channel_statistics.std-of-means', 'latents_std')

def _map_ltx23_connector_key(name: str) -> str | None:
    if name.startswith('model.diffusion_model.video_embeddings_connector.'):
        name = name.replace('model.diffusion_model.video_embeddings_connector.', 'video_connector.', 1)
    elif name.startswith('model.diffusion_model.audio_embeddings_connector.'):
        name = name.replace('model.diffusion_model.audio_embeddings_connector.', 'audio_connector.', 1)
    else:
        return None
    name = name.replace('transformer_1d_blocks.', 'transformer_blocks.')
    return name.replace('q_norm', 'norm_q').replace('k_norm', 'norm_k')

def _map_ltx23_text_projection_key(name: str) -> str | None:
    if name.startswith('text_embedding_projection.video_aggregate_embed.'):
        return name.replace('text_embedding_projection.video_aggregate_embed.', 'video_text_proj_in.', 1)
    if name.startswith('text_embedding_projection.audio_aggregate_embed.'):
        return name.replace('text_embedding_projection.audio_aggregate_embed.', 'audio_text_proj_in.', 1)
    return None

def _map_ltx23_vocoder_key(name: str) -> str | None:
    if name.startswith('vocoder.vocoder.'):
        name = 'vocoder.' + name[len('vocoder.vocoder.'):]
    elif name.startswith('vocoder.bwe_generator.'):
        name = 'bwe_generator.' + name[len('vocoder.bwe_generator.'):]
    else:
        return None
    name = name.replace('resblocks.', 'resnets.')
    name = name.replace('conv_pre.', 'conv_in.')
    name = name.replace('conv_post.', 'conv_out.')
    name = name.replace('act_post.', 'act_out.')
    name = name.replace('lowpass.filter', 'filter')
    return name.replace('.ups.', '.upsamplers.')

def _load_ltx23_mapped_module(module, sources, map_key, *, device: str, dtype, label: str, strict: bool=True):
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    expected = set(module.state_dict().keys())
    loaded: set[str] = set()
    for source in sources:
        p = Path(source)
        if not p.is_file():
            continue
        with safe_open(str(p), framework='pt', device='cpu') as f:
            for src_key in f.keys():
                dst_key = map_key(src_key)
                if not dst_key or dst_key not in expected or dst_key in loaded:
                    continue
                tensor = f.get_tensor(src_key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                assign_dtype = dtype if dtype is not None and tensor.is_floating_point() else None
                set_module_tensor_to_device(module, dst_key, device, value=tensor, dtype=assign_dtype)
                loaded.add(dst_key)
    state = module.state_dict()
    missing = sorted(expected - loaded)
    meta_missing = [k for k in missing if getattr(state[k], 'is_meta', False)]
    if strict and meta_missing:
        preview = ', '.join(meta_missing[:8])
        raise RuntimeError(f'[LTX-COMP] {label}: {len(meta_missing)} tensors still on meta: {preview}')
    module.eval()
    print(f"[LTX-COMP] {label}: loaded {len(loaded)}/{len(expected)} tensors to {device}{('' if not missing else f' ({len(missing)} constructed buffers left as-is)')}", flush=True)
    return module

def load_local_ltx23_text_encoder(text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, *, dtype=None, device: str | None=None):
    import json
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma3ForConditionalGeneration
    dtype = dtype or torch.bfloat16
    text_encoder_dir = _ltx23_existing_dir(text_encoder_dir, 'LTX 2.3 Gemma3 text encoder')
    print(f'[LTX-COMP] loading Gemma3 text encoder from {text_encoder_dir}', flush=True)
    target_device = str(device or 'cpu')
    config = AutoConfig.from_pretrained(str(text_encoder_dir))
    with init_empty_weights():
        text_encoder = Gemma3ForConditionalGeneration(config)

    def _target_key(src_key: str, state_keys: set[str]) -> str | None:
        if src_key in state_keys:
            return src_key
        if src_key.startswith('language_model.model.'):
            dst = 'model.language_model.' + src_key[len('language_model.model.'):]
            return dst if dst in state_keys else None
        if src_key.startswith('vision_tower.vision_model.'):
            dst = 'model.vision_tower.' + src_key[len('vision_tower.vision_model.'):]
            return dst if dst in state_keys else None
        if src_key.startswith('vision_tower.'):
            dst = 'model.' + src_key
            return dst if dst in state_keys else None
        if src_key.startswith('multi_modal_projector.'):
            dst = 'model.multi_modal_projector.' + src_key[len('multi_modal_projector.'):]
            return dst if dst in state_keys else None
        return None
    state_keys = set(text_encoder.state_dict().keys())
    index_file = Path(text_encoder_dir) / 'model.safetensors.index.json'
    if index_file.exists():
        with open(index_file, encoding='utf-8') as f:
            weight_map = json.load(f).get('weight_map', {})
        files = sorted({Path(text_encoder_dir) / filename for filename in weight_map.values()})
    else:
        files = sorted(Path(text_encoder_dir).glob('model-*.safetensors'))
        if not files:
            files = sorted(Path(text_encoder_dir).glob('*.safetensors'))
    loaded: set[str] = set()
    for path in files:
        with safe_open(str(path), framework='pt', device='cpu') as sf:
            for src_key in sf.keys():
                dst_key = _target_key(src_key, state_keys)
                if dst_key is None:
                    continue
                tensor = sf.get_tensor(src_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                assign_dtype = dtype if tensor.is_floating_point() else None
                set_module_tensor_to_device(text_encoder, dst_key, target_device, value=tensor, dtype=assign_dtype)
                loaded.add(dst_key)
                del tensor
    state = text_encoder.state_dict()
    if 'lm_head.weight' in state_keys and 'lm_head.weight' not in loaded:
        embed = 'model.language_model.embed_tokens.weight'
        if embed not in loaded:
            raise RuntimeError('[LTX-COMP] Gemma3 text encoder missing tied embedding for lm_head.weight')
        set_module_tensor_to_device(text_encoder, 'lm_head.weight', target_device, value=state[embed], dtype=dtype)
        loaded.add('lm_head.weight')
    meta_missing = [name for name, param in text_encoder.named_parameters() if param.device.type == 'meta']
    if meta_missing:
        preview = ', '.join(meta_missing[:8])
        raise RuntimeError(f'[LTX-COMP] Gemma3 text encoder left {len(meta_missing)} tensors on meta: {preview}')
    text_encoder.eval()
    print(f'[LTX-COMP] Gemma3 text encoder: loaded {len(loaded)}/{len(state_keys)} tensors to {target_device}', flush=True)
    return text_encoder
_GEMMA_LM_LAYER_RE = re.compile('language_model\\.(?:model\\.)?layers\\.(\\d+)\\.')

def build_streamed_ltx23_text_encoder(text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, *, device: str='cuda', dtype=None, n_gpu_buffers: int=2):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Gemma3ForConditionalGeneration
    from .streamed_text_encoder import build_streamed_text_encoder
    dtype = dtype or torch.bfloat16
    text_encoder_dir = _ltx23_existing_dir(text_encoder_dir, 'LTX 2.3 Gemma3 text encoder')
    print(f'[LTX-TE-STREAM] meta-init + stream Gemma3 text encoder from {text_encoder_dir}', flush=True)
    config = AutoConfig.from_pretrained(str(text_encoder_dir))
    with init_empty_weights():
        encoder = Gemma3ForConditionalGeneration(config)
    sd_keys = set(encoder.state_dict().keys())

    def _ckpt_to_module(src_key: str) -> str | None:
        if src_key in sd_keys:
            return src_key
        for pre, dst_pre in (('language_model.model.', 'model.language_model.'), ('vision_tower.vision_model.', 'model.vision_tower.'), ('vision_tower.', 'model.vision_tower.'), ('multi_modal_projector.', 'model.multi_modal_projector.')):
            if src_key.startswith(pre):
                dst = dst_pre + src_key[len(pre):]
                return dst if dst in sd_keys else None
        return None
    return build_streamed_text_encoder(encoder, str(text_encoder_dir), layer_regex=_GEMMA_LM_LAYER_RE, decoder_layers=encoder.model.language_model.layers, ckpt_to_module=_ckpt_to_module, device=device, dtype=dtype, tie_lm_head_from='model.language_model.embed_tokens.weight', log_prefix='[LTX-TE-STREAM]')

def load_local_ltx23_tokenizer(text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, tokenizer_dir: str | None=None):
    from transformers import AutoProcessor, AutoTokenizer
    source = Path(tokenizer_dir) if tokenizer_dir else Path(text_encoder_dir)
    if not source.exists():
        source = Path(LTX23_DEFAULT_COMPONENTS_REPO) / 'processor'
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    try:
        processor = AutoProcessor.from_pretrained(str(source))
    except Exception:
        processor = None
    return (tokenizer, processor)

def load_local_ltx23_connectors(model_file: str=LTX23_DEFAULT_MODEL_FILE, components_repo: str=LTX23_DEFAULT_COMPONENTS_REPO, text_projection_file: str=LTX23_DEFAULT_TEXT_PROJECTION_FILE, *, device: str='cpu', dtype=None):
    import torch
    from accelerate import init_empty_weights
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    dtype = dtype or torch.bfloat16
    cfg_dir = _ltx23_cached_component_dir('connectors', components_repo)
    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(LTX2TextConnectors.load_config(str(cfg_dir)))
    official_sources = _ltx23_component_weight_files('connectors', components_repo)
    if official_sources:
        return _load_ltx23_mapped_module(connectors, official_sources, lambda name: name, device=device, dtype=dtype, label='official connectors')
    bundle = _ltx23_existing_file(model_file, 'LTX 2.3 transformer bundle')
    connectors = _load_ltx23_mapped_module(connectors, [bundle], _map_ltx23_connector_key, device=device, dtype=dtype, label='connectors', strict=False)
    projection = Path(text_projection_file)
    if not projection.is_file():
        projection = bundle
    connectors = _load_ltx23_mapped_module(connectors, [projection], _map_ltx23_text_projection_key, device=device, dtype=dtype, label='text projection')
    return connectors

def load_local_ltx23_runtime_components(model_file: str=LTX23_DEFAULT_MODEL_FILE, components_repo: str=LTX23_DEFAULT_COMPONENTS_REPO, *, video_vae_file: str=LTX23_DEFAULT_VIDEO_VAE_FILE, audio_vocoder_file: str=LTX23_DEFAULT_AUDIO_VOCODER_FILE, text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, tokenizer_dir: str | None=None, device: str='cpu', dtype=None, load_vocoder: bool=False):
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    dtype = dtype or torch.bfloat16
    components_repo = components_repo or LTX23_DEFAULT_COMPONENTS_REPO
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(Path(components_repo) / 'scheduler'))
    tokenizer, processor = load_local_ltx23_tokenizer(text_encoder_dir, tokenizer_dir)
    with init_empty_weights():
        vae = AutoencoderKLLTX2Video.from_config(AutoencoderKLLTX2Video.load_config(str(_ltx23_cached_component_dir('vae', components_repo))))
    official_vae_sources = _ltx23_component_weight_files('vae', components_repo)
    vae_sources = official_vae_sources
    vae_map = lambda name: name
    vae_label = 'official video VAE'
    if not vae_sources:
        vae_source = Path(video_vae_file)
        if not vae_source.is_file():
            vae_source = _ltx23_existing_file(model_file, 'LTX 2.3 transformer bundle')
        vae_sources = [vae_source]
        vae_map = _map_ltx23_video_vae_key
        vae_label = 'video VAE'
    vae = _load_ltx23_mapped_module(vae, vae_sources, vae_map, device=device, dtype=dtype, label=vae_label)
    with init_empty_weights():
        audio_vae = AutoencoderKLLTX2Audio.from_config(AutoencoderKLLTX2Audio.load_config(str(_ltx23_cached_component_dir('audio_vae', components_repo))))
    official_audio_sources = _ltx23_component_weight_files('audio_vae', components_repo)
    audio_sources = official_audio_sources
    audio_map = lambda name: name
    audio_label = 'official audio VAE'
    if not audio_sources:
        audio_source = _ltx23_existing_file(audio_vocoder_file, 'LTX 2.3 audio VAE/vocoder bundle')
        audio_sources = [audio_source]
        audio_map = _map_ltx23_audio_vae_key
        audio_label = 'audio VAE'
    audio_vae = _load_ltx23_mapped_module(audio_vae, audio_sources, audio_map, device=device, dtype=dtype, label=audio_label)
    vocoder = None
    if load_vocoder:
        audio_source = _ltx23_existing_file(audio_vocoder_file, 'LTX 2.3 audio VAE/vocoder bundle')
        with init_empty_weights():
            vocoder = LTX2VocoderWithBWE.from_config(LTX2VocoderWithBWE.load_config(str(_ltx23_cached_component_dir('vocoder', components_repo))))
        vocoder = _load_ltx23_mapped_module(vocoder, [audio_source], _map_ltx23_vocoder_key, device=device, dtype=dtype, label='vocoder')
    return dict(scheduler=scheduler, vae=vae, audio_vae=audio_vae, text_encoder=None, tokenizer=tokenizer, connectors=None, transformer=None, vocoder=vocoder, processor=processor)

def assemble_local_ltx23_components(model_file: str=LTX23_DEFAULT_MODEL_FILE, components_repo: str=LTX23_DEFAULT_COMPONENTS_REPO, *, text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, tokenizer_dir: str | None=None, device: str='cpu', dtype=None, load_text_encoder: bool=True, load_connectors: bool=True, load_vocoder: bool=False):
    components = load_local_ltx23_runtime_components(model_file, components_repo, text_encoder_dir=text_encoder_dir, tokenizer_dir=tokenizer_dir, device=device, dtype=dtype, load_vocoder=load_vocoder)
    if load_connectors:
        components['connectors'] = load_local_ltx23_connectors(model_file, components_repo, device=device, dtype=dtype)
    if load_text_encoder:
        components['text_encoder'] = load_local_ltx23_text_encoder(text_encoder_dir, dtype=dtype, device=device)
        if device != 'cpu':
            components['text_encoder'].to(device)
    return components

def streamed_ltx_generate(model_file: str, prompt: str, components_repo: str='', *, negative_prompt: str='worst quality, blurry, jittery, distorted, watermark, text', num_frames: int=97, height: int=704, width: int=1280, num_inference_steps: int=30, guidance_scale: float=4.0, audio_guidance_scale: float=4.0, frame_rate: float=24.0, seed: int=42, device: str='cuda', dtype=None, n_gpu_buffers: int=3, output_type: str='pil', decode_audio: bool=False, use_cross_timestep: bool=True, transformer=None, streamer=None, components: dict | None=None, text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, tokenizer_dir: str | None=None, prompt_max_length: int=1024, release_text_components_after_encode: bool=True, stream_text_encoder: bool=True, loras=None, sampler_name: str | None=None, sigmas_name: str | None=None, image_path: str | None=None, last_image_path: str | None=None, keyframes: list | None=None, video_path: str | None=None, video_condition_strength: float=1.0, decode_video: bool=True, init_image_path: str | None=None, denoise_strength: float | None=None, progress=None):
    import time as _time
    import numpy as np
    import torch
    from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift, retrieve_timesteps
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2ConditionPipeline, LTX2VideoCondition
    t0 = _time.time()
    dtype = dtype or torch.bfloat16
    if int(num_inference_steps) < 2:
        print('[LTX-GEN] num_inference_steps < 2 would produce invalid FlowMatch sigmas; using 2.', flush=True)
        num_inference_steps = 2
    do_cfg = guidance_scale > 1.0
    gen = torch.Generator(device=device).manual_seed(int(seed))
    import gc as _gc
    components = dict(components or {})
    tokenizer = components.get('tokenizer')
    processor = components.get('processor')
    if tokenizer is None:
        tokenizer, processor = load_local_ltx23_tokenizer(text_encoder_dir, tokenizer_dir)
        components['tokenizer'] = tokenizer
        components['processor'] = processor
    text_encoder = components.get('text_encoder')
    te_streamed = bool(getattr(text_encoder, '_sb_streamed', False))
    if text_encoder is None:
        if stream_text_encoder and str(device).startswith('cuda'):
            preflight_vram(device, need_mib=4096, label='LTX Gemma3 text encoder (streamed)')
            text_encoder = build_streamed_ltx23_text_encoder(text_encoder_dir, device=device, dtype=dtype)
            te_streamed = True
        else:
            preflight_vram(device, need_mib=25600, label='LTX Gemma3 text encoder (resident)')
            text_encoder = load_local_ltx23_text_encoder(text_encoder_dir, dtype=dtype, device=device)
    elif not te_streamed and hasattr(text_encoder, 'to'):
        text_encoder.to(device)
    text_encoder.eval()

    def _gemma(p):
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ti = tokenizer([p.strip()], padding='max_length', max_length=int(prompt_max_length), truncation=True, add_special_tokens=True, return_tensors='pt')
        ids = ti.input_ids.to(device)
        am = ti.attention_mask.to(device)
        out = text_encoder(input_ids=ids, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=-1)
        return (hs.flatten(2, 3).to(dtype), am)
    with torch.no_grad():
        prompt_embeds, prompt_am = _gemma(prompt)
        if do_cfg:
            neg_embeds, neg_am = _gemma(negative_prompt)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            prompt_am = torch.cat([neg_am, prompt_am], dim=0)
    if release_text_components_after_encode:
        components['text_encoder'] = None
        del text_encoder
    elif te_streamed:
        components['text_encoder'] = text_encoder
    else:
        components['text_encoder'] = text_encoder.to('cpu') if hasattr(text_encoder, 'to') else text_encoder
    _gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    connectors = components.get('connectors')
    if connectors is None:
        connectors = load_local_ltx23_connectors(model_file, components_repo, device=device, dtype=dtype)
    elif hasattr(connectors, 'to'):
        connectors.to(device)
    connectors.eval()
    prompt_embeds = prompt_embeds.to(device)
    prompt_am = prompt_am.to(device)
    with torch.no_grad():
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(prompt_embeds, prompt_am, padding_side='left')
    del prompt_embeds, prompt_am
    if release_text_components_after_encode:
        components['connectors'] = None
        del connectors
    else:
        components['connectors'] = connectors.to('cpu') if hasattr(connectors, 'to') else connectors
    _gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    if components.get('vae') is None or components.get('audio_vae') is None or components.get('scheduler') is None:
        runtime = load_local_ltx23_runtime_components(model_file, components_repo, text_encoder_dir=text_encoder_dir, tokenizer_dir=tokenizer_dir, device='cpu', dtype=dtype, load_vocoder=decode_audio)
        runtime['tokenizer'] = components.get('tokenizer') or runtime.get('tokenizer')
        runtime['processor'] = components.get('processor') or runtime.get('processor')
        runtime['text_encoder'] = components.get('text_encoder')
        runtime['connectors'] = components.get('connectors')
        components.update(runtime)
    if transformer is None or streamer is None:
        preflight_vram(device, need_mib=8192, label='LTX streamed 22B transformer + denoise')
        transformer, streamer = build_streamed_ltx(model_file, device=device, n_gpu_buffers=n_gpu_buffers, dtype=dtype, config_dir=Path(components_repo) / 'transformer')
        if getattr(streamer, 'mode', None) != 3:
            raise RuntimeError(f"streamed_ltx_generate: streamer mode={getattr(streamer, 'mode', '?')} != DirectStorage(3). A non-streaming load would prestage ~46 GB in RAM on a 32 GB box. Fix DirectStorage init and retry.")
    transformer.eval()
    if loras:
        from cosmos_windows.gen_with_lora import apply_lora_stack
        apply_lora_stack(transformer, loras, rank=None, targets=LTX23_LORA_TARGETS, device=device)
        transformer.eval()
    components['transformer'] = transformer
    pipe = LTX2ConditionPipeline(**{k: components.get(k) for k in ('scheduler', 'vae', 'audio_vae', 'text_encoder', 'tokenizer', 'connectors', 'transformer', 'vocoder')})
    pipe._guidance_scale = guidance_scale
    pipe._audio_guidance_scale = audio_guidance_scale
    pipe._guidance_rescale = 0.0
    pipe._audio_guidance_rescale = 0.0
    pipe._stg_scale = 0.0
    pipe._audio_stg_scale = 0.0
    pipe._modality_scale = 1.0
    pipe._audio_modality_scale = 1.0
    pipe._attention_kwargs = None
    pipe._interrupt = False
    vae = components['vae']
    audio_vae = components['audio_vae']
    scheduler = components['scheduler']
    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    num_channels_latents = transformer.config.in_channels

    def _ltx_condition_index(pixel_frame: int | None, role: str | None=None) -> int:
        if role == 'last':
            return -1
        frame = max(0, int(pixel_frame or 0))
        if frame <= 0:
            return 0
        tcr = int(pipe.vae_temporal_compression_ratio)
        latent_frames = (int(num_frames) - 1) // tcr + 1
        return max(1, min(latent_frames - 1, int(round((frame - 1) / tcr)) + 1))

    def _ltx_conditions() -> list:
        from PIL import Image
        out = []
        seen: set[tuple[str, int]] = set()

        def add(path: str | None, index: int, strength: float=1.0):
            if not path:
                return
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f'LTX condition frame not found: {path}')
            key = (str(p.resolve()), int(index))
            if key in seen:
                return
            seen.add(key)
            out.append(LTX2VideoCondition(frames=Image.open(p).convert('RGB'), index=int(index), strength=float(strength)))
        if video_path:
            vp = Path(video_path)
            if not vp.is_file():
                raise FileNotFoundError(f'LTX foley source video not found: {video_path}')
            from diffusers.utils import load_video
            frames = load_video(str(vp))
            if not frames:
                raise RuntimeError(f'LTX foley source video has no readable frames: {video_path}')
            frames = [f.convert('RGB') for f in frames[:int(num_frames or len(frames))]]
            out.append(LTX2VideoCondition(frames=frames, index=0, strength=float(video_condition_strength)))
        add(image_path, 0, 1.0)
        add(last_image_path, -1, 1.0)
        total_frames = max(1, int(num_frames or 1))
        for item in keyframes or []:
            if hasattr(item, 'model_dump'):
                item = item.model_dump()
            if not isinstance(item, dict):
                continue
            path = item.get('image_path')
            role = item.get('role')
            if role == 'last':
                idx = -1
            else:
                frame = item.get('frame')
                if frame is None and item.get('at_fraction') is not None:
                    frame = round(float(item['at_fraction']) * (total_frames - 1))
                idx = _ltx_condition_index(frame, role)
            add(path, idx, float(item.get('strength') or 1.0))
        return out
    conditions = _ltx_conditions()
    _sdedit = denoise_strength is not None and 0.0 < float(denoise_strength) < 1.0 and bool(init_image_path) and (not conditions)
    _sdedit_clean = None
    if _sdedit:
        from PIL import Image as _Image
        from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import retrieve_latents as _retrieve_latents
        if hasattr(vae, 'to'):
            vae.to(device)
        _px = pipe.video_processor.preprocess_video([_Image.open(init_image_path).convert('RGB')] * num_frames, height, width).to(device, vae.dtype)
        with torch.no_grad():
            _init_lat = _retrieve_latents(vae.encode(_px), generator=gen, sample_mode='argmax')
        latents, conditioning_mask, clean_latents, keyframe_coords = pipe.prepare_latents(conditions=[], batch_size=1, num_channels_latents=num_channels_latents, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate, noise_scale=0.0, dtype=torch.float32, device=device, generator=gen, latents=_init_lat)
        _sdedit_clean = latents
        if hasattr(vae, 'to'):
            vae.to('cpu')
        _gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        print(f'[LTX-GEN] SDEdit i2i: init encoded (clean), strength={denoise_strength}', flush=True)
    elif conditions:
        if hasattr(vae, 'to'):
            vae.to(device)
        latents, conditioning_mask, clean_latents, keyframe_coords = pipe.prepare_latents(conditions=conditions, batch_size=1, num_channels_latents=num_channels_latents, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate, noise_scale=1.0, dtype=torch.float32, device=device, generator=gen, latents=None)
        if hasattr(vae, 'to'):
            vae.to('cpu')
        _gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        print(f'[LTX-GEN] visual conditioning enabled: {len(conditions)} frame condition(s)', flush=True)
    else:
        latents, conditioning_mask, clean_latents, keyframe_coords = pipe.prepare_latents(conditions=[], batch_size=1, num_channels_latents=num_channels_latents, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate, noise_scale=1.0, dtype=torch.float32, device=device, generator=gen, latents=None)
    duration_s = num_frames / frame_rate
    audio_latents_per_second = pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_mel_bins = audio_vae.config.mel_bins if getattr(audio_vae, 'config', None) is not None else 64
    latent_mel_bins = num_mel_bins // pipe.audio_vae_mel_compression_ratio
    num_channels_latents_audio = getattr(getattr(audio_vae, 'config', None), 'latent_channels', 8)
    audio_latents = pipe.prepare_audio_latents(batch_size=1, num_channels_latents=num_channels_latents_audio, audio_latent_length=audio_num_frames, num_mel_bins=num_mel_bins, noise_scale=0.0, dtype=torch.float32, device=device, generator=gen, latents=None)
    try:
        from cosmos_windows.res4lyf_sampling import custom_flow_sigmas, flow_scheduler_for_manual_loop, is_res4lyf_sampler, is_res4lyf_sigma
    except Exception:
        custom_flow_sigmas = None
        flow_scheduler_for_manual_loop = None
        is_res4lyf_sampler = lambda _name: False
        is_res4lyf_sigma = lambda _name: False
    if flow_scheduler_for_manual_loop is not None and (is_res4lyf_sampler(sampler_name) or is_res4lyf_sigma(sigmas_name)):
        scheduler, selected_sigmas = flow_scheduler_for_manual_loop(scheduler, sampler=sampler_name, sigmas=sigmas_name, num_inference_steps=num_inference_steps, log_prefix='[LTX-GEN]')
        components['scheduler'] = scheduler
    else:
        selected_sigmas = custom_flow_sigmas(sigmas_name, num_inference_steps) if custom_flow_sigmas else None
    sigmas = selected_sigmas if selected_sigmas is not None else np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    if _sdedit:
        _full = np.asarray(sigmas, dtype=np.float64)
        _start_k = int(np.searchsorted(-_full, -float(denoise_strength)))
        _start_k = max(0, min(_start_k, len(_full) - 2))
        sigmas = _full[_start_k:]
    mu = calculate_shift(scheduler.config.get('max_image_seq_len', 4096), scheduler.config.get('base_image_seq_len', 1024), scheduler.config.get('max_image_seq_len', 4096), scheduler.config.get('base_shift', 0.95), scheduler.config.get('max_shift', 2.05))
    import copy as _copy
    audio_scheduler = _copy.deepcopy(scheduler)
    audio_timesteps, _ = retrieve_timesteps(audio_scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu)
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    audio_scheduler.set_begin_index(0)
    if _sdedit and _sdedit_clean is not None:
        _sig0 = float(scheduler.sigmas[0])
        _noise = torch.randn(_sdedit_clean.shape, generator=gen, device=_sdedit_clean.device, dtype=_sdedit_clean.dtype)
        latents = ((1.0 - _sig0) * _sdedit_clean + _sig0 * _noise).to(torch.float32)
        print(f'[LTX-GEN] SDEdit i2i: {len(timesteps)} steps, start sigma0={_sig0:.4f}', flush=True)
    video_coords = transformer.rope.prepare_video_coords(latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate)
    audio_coords = transformer.audio_rope.prepare_audio_coords(audio_latents.shape[0], audio_num_frames, audio_latents.device)
    if keyframe_coords is not None:
        video_coords = torch.cat([video_coords, keyframe_coords], dim=2)
    if do_cfg:
        conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])
        video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
        audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
    print(f'[LTX-GEN] denoising {num_frames}f @ {width}x{height}, {num_inference_steps} steps (seq={latents.shape[1]}, audio_frames={audio_num_frames}, cfg={do_cfg})...', flush=True)
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents).to(dtype)
            audio_latent_model_input = (torch.cat([audio_latents] * 2) if do_cfg else audio_latents).to(dtype)
            timestep = t.expand(latent_model_input.shape[0])
            video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask.squeeze(-1))
            t_audio = audio_timesteps[i]
            audio_timestep = t_audio.expand(latent_model_input.shape[0])
            with transformer.cache_context('cond_uncond'):
                noise_pred_video, noise_pred_audio = transformer(hidden_states=latent_model_input, audio_hidden_states=audio_latent_model_input, encoder_hidden_states=connector_prompt_embeds, audio_encoder_hidden_states=connector_audio_prompt_embeds, timestep=video_timestep, audio_timestep=audio_timestep, sigma=timestep, audio_sigma=audio_timestep, encoder_attention_mask=connector_attention_mask, audio_encoder_attention_mask=connector_attention_mask, num_frames=latent_num_frames, height=latent_height, width=latent_width, fps=frame_rate, audio_num_frames=audio_num_frames, video_coords=video_coords, audio_coords=audio_coords, isolate_modalities=False, spatio_temporal_guidance_blocks=None, perturbation_mask=None, use_cross_timestep=use_cross_timestep, attention_kwargs=None, return_dict=False)
            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()
            if do_cfg:
                nv_unc, nv = noise_pred_video.chunk(2)
                nv = pipe.convert_velocity_to_x0(latents, nv, i, scheduler)
                nv_unc = pipe.convert_velocity_to_x0(latents, nv_unc, i, scheduler)
                noise_pred_video = nv + (guidance_scale - 1.0) * (nv - nv_unc)
                na_unc, na = noise_pred_audio.chunk(2)
                na = pipe.convert_velocity_to_x0(audio_latents, na, i, audio_scheduler)
                na_unc = pipe.convert_velocity_to_x0(audio_latents, na_unc, i, audio_scheduler)
                noise_pred_audio = na + (audio_guidance_scale - 1.0) * (na - na_unc)
            else:
                noise_pred_video = pipe.convert_velocity_to_x0(latents, noise_pred_video, i, scheduler)
                noise_pred_audio = pipe.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)
            bsz = noise_pred_video.size(0)
            noise_pred_video = (noise_pred_video * (1 - conditioning_mask[:bsz]) + clean_latents * conditioning_mask[:bsz]).to(noise_pred_video.dtype)
            noise_pred_video = pipe.convert_x0_to_velocity(latents, noise_pred_video, i, scheduler)
            noise_pred_audio = pipe.convert_x0_to_velocity(audio_latents, noise_pred_audio, i, audio_scheduler)
            latents = scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]
            if not torch.isfinite(latents).all() or not torch.isfinite(audio_latents).all():
                raise RuntimeError(f'LTX denoise produced non-finite latents at step {i + 1}/{len(timesteps)}; check scheduler sigmas/step count instead of saving a black clip.')
        if progress is not None:
            try:
                progress(i + 1, len(timesteps), latents)
            except Exception as e:
                print(f'[LTX-GEN] progress hook note (ignored): {e}', flush=True)
        else:
            print(f'[LTX-GEN] step {i + 1}/{len(timesteps)}', flush=True)
    base_token_count = latent_num_frames * latent_height * latent_width
    latents = latents[:, :base_token_count]
    latents = pipe._unpack_latents(latents, latent_num_frames, latent_height, latent_width, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
    if not torch.isfinite(latents).all():
        raise RuntimeError('LTX final video latents are non-finite; refusing to decode a black clip.')
    audio = None
    video = None
    if output_type == 'latent':
        video = pipe._denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
    else:
        if decode_video and hasattr(vae, 'to'):
            vae.to(device)
        if decode_video:
            with torch.no_grad():
                latents = pipe._denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor).to(vae.dtype)
                dec_t = None
                video = vae.decode(latents, dec_t, return_dict=False)[0]
            video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        if decode_audio:
            if components.get('vocoder') is None:
                raise RuntimeError('decode_audio=True requires the local LTX 2.3 vocoder component')
            if hasattr(audio_vae, 'to'):
                audio_vae.to(device=device, dtype=dtype)
            if hasattr(components['vocoder'], 'to'):
                components['vocoder'].to(device=device, dtype=dtype)
            with torch.no_grad():
                audio_latents = pipe._denormalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
                audio_latents = pipe._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins).to(device=device, dtype=dtype)
                mel = audio_vae.decode(audio_latents, return_dict=False)[0]
                audio = components['vocoder'](mel.to(device=device, dtype=dtype))
    seconds = _time.time() - t0
    print(f'[LTX-GEN] done in {seconds:.0f}s', flush=True)
    return dict(video=video, audio=audio, latents=latents, seconds=seconds, audio_sample_rate=int(getattr(pipe, 'audio_sampling_rate', 24000) or 24000), transformer=transformer, streamer=streamer, components=components)

def _wrap_backward_ltx_block(block, bind, evict):
    from ..streamed_backward import apply_streamed_multi
    cls_forward = type(block).forward

    def wrapped(hidden_states, audio_hidden_states, *args, **kwargs):

        def call(blk, video, audio):
            return cls_forward(blk, video, audio, *args, **kwargs)
        return apply_streamed_multi(block, call, hidden_states, audio_hidden_states, bind=bind, evict=evict)
    block.forward = wrapped
    block._sb_raw_forward = cls_forward
    block._sb_bind = bind
    block._sb_evict = evict

def build_streamed_backward_ltx(model_file: str, device: str='cuda', n_gpu_buffers: int=1, config_dir: str | Path | None=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .build_streamed_transformer import StreamingWeightLoader
    from .safetensors_index import build_index
    from ..streamed_backward import apply_streamed_multi
    model, cfg = _load_meta_ltx(model_file, config_dir=config_dir)
    tf = model
    idx = build_index(model_file)
    sd_keys = set(model.state_dict().keys())
    block_refs, global_refs = ltx_group_by_block(idx)
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = _remap_key(StreamingWeightLoader._param_path(ref))
        if path in sd_keys:
            set_module_tensor_to_device(model, path, device, value=pb.read_tensor(ref, device))
    streamer = DirectStorageStreamer(_LtxBlockIndex(idx), component='', n_gpu_buffers=n_gpu_buffers, device=device)
    n_blocks = len(tf.transformer_blocks)
    print(f"[LTX-BWD] mode={streamer.mode} {streamer.n} blocks, {len(streamer.tbufs)} GPU buf(s), {len([r for r in global_refs if _remap_key(StreamingWeightLoader._param_path(r)) in sd_keys])} resident globals (reverse-capable; {('prefetch ON' if n_gpu_buffers > 1 else 'single-buffer=validated reverse')})", flush=True)

    def _pp(name):
        return _remap_key(name.split('/', 1)[1] if '/' in name else name)

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    block_paths = {il: [_pp(r.name) for r, _ in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in streamer.plan}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    set_module_tensor_to_device(model, _resolve(p), device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for p in block_paths[il]:
                set_module_tensor_to_device(model, _resolve(p), 'meta')
            streamer.release_layer(il)
        return evict
    for il, block in enumerate(tf.transformer_blocks):
        bind, evict = (make_bind(il), make_evict(il))
        _wrap_backward_ltx_block(block, bind, evict)
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[LTX-BWD] wrapped {n_blocks} LTX2VideoTransformerBlocks in StreamedLayerMulti — attach LoRA, materialize_meta_lora, forward, then streamer.new_pass(reverse=True) before loss.backward()', flush=True)
    return (model, streamer)
if __name__ == '__main__':
    import sys
    mf = sys.argv[1] if len(sys.argv) > 1 else ''
    from .safetensors_index import build_index
    from .build_streamed_transformer import StreamingWeightLoader
    model, cfg = _load_meta_ltx(mf)
    n_blocks = len(model.transformer_blocks)
    print(f"meta model: {type(model).__name__}, {n_blocks} transformer_blocks, num_layers(cfg)={getattr(cfg, 'num_layers', '?')}, cross_attn_mod={getattr(cfg, 'cross_attn_mod', '?')}")
    sd_keys = set(model.state_dict().keys())
    idx = build_index(mf)
    blocks, globs = ltx_group_by_block(idx)
    print(f'index: {len(idx)} tensors total; {len(blocks)} block-groups, {len(globs)} non-block tensors')
    if 0 in blocks:
        b0 = blocks[0]
        b0_mapped = [_remap_key(StreamingWeightLoader._param_path(r)) for r in b0]
        b0_homed = [p for p in b0_mapped if p in sd_keys]
        print(f'block 0: {len(b0)} checkpoint tensors -> {len(b0_homed)} land in the diffusers tree ({len(b0) - len(b0_homed)} skipped)')
    block_paths = set()
    for refs in blocks.values():
        for r in refs:
            block_paths.add(_remap_key(StreamingWeightLoader._param_path(r)))
    block_missing = sorted(block_paths - sd_keys)
    glob_paths = {_remap_key(StreamingWeightLoader._param_path(r)) for r in globs}
    glob_homed = sorted(glob_paths & sd_keys)
    glob_skipped = len(glob_paths) - len(glob_homed)
    block_re = _LTX_BLOCK_RE
    all_mapped = block_paths | glob_paths
    tree_unfilled = sorted((k for k in sd_keys if k not in all_mapped))
    print(f'\nKEYSET REPORT:')
    print(f'  block tensors with NO module home (FATAL if >0): {len(block_missing)}')
    for k in block_missing[:20]:
        print(f'    BLOCK-MISSING: {k}')
    print(f'  globals homed: {len(glob_homed)} ; globals skipped (vae/vocoder/text-enc/connectors): {glob_skipped}')
    print(f'  module params with NO checkpoint weight (caption_projection sourced separately / stays meta): {len(tree_unfilled)}')
    for k in tree_unfilled[:20]:
        print(f'    UNFILLED: {k}')
    block_ok = len(block_missing) == 0
    count_ok = len(blocks) == n_blocks
    print(f"\nRESULT: {('OK' if block_ok and count_ok else 'CHECK')} (blocks={('PASS' if count_ok else f'MISMATCH {len(blocks)}!={n_blocks}')}, block-keyset={('PASS' if block_ok else 'FAIL')})")
    sys.exit(0 if block_ok and count_ok else 1)

def streamed_ltx_generate_legacy_pipeline(model_file: str, prompt: str, components_repo: str='', *, negative_prompt: str='worst quality, blurry, jittery, distorted, watermark, text', num_frames: int=97, height: int=704, width: int=1280, num_inference_steps: int=30, guidance_scale: float=4.0, audio_guidance_scale: float=4.0, frame_rate: float=24.0, seed: int=42, device: str='cuda', dtype=None, n_gpu_buffers: int=3, output_type: str='pil', decode_audio: bool=False, use_cross_timestep: bool=True, transformer=None, streamer=None, components: dict | None=None, text_encoder_dir: str=LTX23_DEFAULT_TEXT_ENCODER_DIR, tokenizer_dir: str | None=None, prompt_max_length: int=1024, release_text_components_after_encode: bool=True, progress=None):
    import time as _time
    import numpy as np
    import torch
    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, calculate_shift, retrieve_timesteps
    t0 = _time.time()
    dtype = dtype or torch.bfloat16
    if int(num_inference_steps) < 2:
        print('[LTX-GEN] num_inference_steps < 2 would produce invalid FlowMatch sigmas; using 2.', flush=True)
        num_inference_steps = 2
    if transformer is None or streamer is None:
        transformer, streamer = build_streamed_ltx(model_file, device=device, n_gpu_buffers=n_gpu_buffers, dtype=dtype, config_dir=Path(components_repo) / 'transformer')
        if getattr(streamer, 'mode', None) != 3:
            raise RuntimeError(f"streamed_ltx_generate: streamer mode={getattr(streamer, 'mode', '?')} != DirectStorage(3). A non-streaming load would prestage ~46 GB in RAM on a 32 GB box. Fix DirectStorage init and retry.")
    transformer.eval()
    if components is None:
        pipe = LTX2Pipeline.from_pretrained(components_repo, transformer=transformer, torch_dtype=dtype)
        for name, mod in pipe.components.items():
            if name == 'transformer' or not hasattr(mod, 'to'):
                continue
            try:
                mod.to(device)
            except Exception as e:
                print(f'[LTX-GEN] note: could not move {name} to {device}: {e}', flush=True)
        components = dict(pipe.components)
    else:
        components = dict(components)
        components['transformer'] = transformer
        pipe = LTX2Pipeline(**{k: v for k, v in components.items() if k in ('scheduler', 'vae', 'audio_vae', 'text_encoder', 'tokenizer', 'connectors', 'transformer', 'vocoder')})
    pipe._guidance_scale = guidance_scale
    pipe._audio_guidance_scale = audio_guidance_scale
    pipe._guidance_rescale = 0.0
    pipe._audio_guidance_rescale = 0.0
    pipe._stg_scale = 0.0
    pipe._audio_stg_scale = 0.0
    pipe._modality_scale = 1.0
    pipe._audio_modality_scale = 1.0
    vae = components['vae']
    audio_vae = components['audio_vae']
    connectors = components['connectors']
    text_encoder = components['text_encoder']
    tokenizer = components['tokenizer']
    scheduler = components['scheduler']
    do_cfg = guidance_scale > 1.0
    gen = torch.Generator(device=device).manual_seed(int(seed))

    def _gemma(p):
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ti = tokenizer([p.strip()], padding='max_length', max_length=1024, truncation=True, add_special_tokens=True, return_tensors='pt')
        ids = ti.input_ids.to(device)
        am = ti.attention_mask.to(device)
        out = text_encoder(input_ids=ids, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=-1)
        return (hs.flatten(2, 3).to(dtype), am)
    with torch.no_grad():
        prompt_embeds, prompt_am = _gemma(prompt)
        if do_cfg:
            neg_embeds, neg_am = _gemma(negative_prompt)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            prompt_am = torch.cat([neg_am, prompt_am], dim=0)
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(prompt_embeds, prompt_am, padding_side='left')
    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    num_channels_latents = transformer.config.in_channels
    latents = pipe.prepare_latents(batch_size=1, num_channels_latents=num_channels_latents, height=height, width=width, num_frames=num_frames, noise_scale=0.0, dtype=torch.float32, device=device, generator=gen, latents=None)
    duration_s = num_frames / frame_rate
    audio_latents_per_second = pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    audio_num_frames = round(duration_s * audio_latents_per_second)
    num_mel_bins = audio_vae.config.mel_bins if getattr(audio_vae, 'config', None) is not None else 64
    latent_mel_bins = num_mel_bins // pipe.audio_vae_mel_compression_ratio
    num_channels_latents_audio = getattr(getattr(audio_vae, 'config', None), 'latent_channels', 8)
    audio_latents = pipe.prepare_audio_latents(batch_size=1, num_channels_latents=num_channels_latents_audio, audio_latent_length=audio_num_frames, num_mel_bins=num_mel_bins, noise_scale=0.0, dtype=torch.float32, device=device, generator=gen, latents=None)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(scheduler.config.get('max_image_seq_len', 4096), scheduler.config.get('base_image_seq_len', 1024), scheduler.config.get('max_image_seq_len', 4096), scheduler.config.get('base_shift', 0.95), scheduler.config.get('max_shift', 2.05))
    import copy as _copy
    audio_scheduler = _copy.deepcopy(scheduler)
    retrieve_timesteps(audio_scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu)
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    audio_scheduler.set_begin_index(0)
    video_coords = transformer.rope.prepare_video_coords(latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate)
    audio_coords = transformer.audio_rope.prepare_audio_coords(audio_latents.shape[0], audio_num_frames, audio_latents.device)
    if do_cfg:
        video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
        audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
    print(f'[LTX-GEN] denoising {num_frames}f @ {width}x{height}, {num_inference_steps} steps (seq={latents.shape[1]}, audio_frames={audio_num_frames}, cfg={do_cfg})...', flush=True)
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents).to(dtype)
            audio_latent_model_input = (torch.cat([audio_latents] * 2) if do_cfg else audio_latents).to(dtype)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred_video, noise_pred_audio = transformer(hidden_states=latent_model_input, audio_hidden_states=audio_latent_model_input, encoder_hidden_states=connector_prompt_embeds, audio_encoder_hidden_states=connector_audio_prompt_embeds, timestep=timestep, sigma=timestep, encoder_attention_mask=connector_attention_mask, audio_encoder_attention_mask=connector_attention_mask, num_frames=latent_num_frames, height=latent_height, width=latent_width, fps=frame_rate, audio_num_frames=audio_num_frames, video_coords=video_coords, audio_coords=audio_coords, isolate_modalities=False, spatio_temporal_guidance_blocks=None, perturbation_mask=None, use_cross_timestep=use_cross_timestep, attention_kwargs=None, return_dict=False)
            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()
            if do_cfg:
                nv_unc, nv = noise_pred_video.chunk(2)
                nv = pipe.convert_velocity_to_x0(latents, nv, i, scheduler)
                nv_unc = pipe.convert_velocity_to_x0(latents, nv_unc, i, scheduler)
                noise_pred_video_g = nv + (guidance_scale - 1.0) * (nv - nv_unc)
                noise_pred_video = pipe.convert_x0_to_velocity(latents, noise_pred_video_g, i, scheduler)
                na_unc, na = noise_pred_audio.chunk(2)
                na = pipe.convert_velocity_to_x0(audio_latents, na, i, audio_scheduler)
                na_unc = pipe.convert_velocity_to_x0(audio_latents, na_unc, i, audio_scheduler)
                noise_pred_audio_g = na + (audio_guidance_scale - 1.0) * (na - na_unc)
                noise_pred_audio = pipe.convert_x0_to_velocity(audio_latents, noise_pred_audio_g, i, audio_scheduler)
            latents = scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]
            if not torch.isfinite(latents).all() or not torch.isfinite(audio_latents).all():
                raise RuntimeError(f'LTX denoise produced non-finite latents at step {i + 1}/{len(timesteps)}; check scheduler sigmas/step count instead of saving a black clip.')
        if progress is not None:
            try:
                progress(i + 1, len(timesteps), latents)
            except Exception as e:
                print(f'[LTX-GEN] progress hook note (ignored): {e}', flush=True)
        else:
            print(f'[LTX-GEN] step {i + 1}/{len(timesteps)}', flush=True)
    latents = pipe._unpack_latents(latents, latent_num_frames, latent_height, latent_width, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
    if not torch.isfinite(latents).all():
        raise RuntimeError('LTX final video latents are non-finite; refusing to decode a black clip.')
    audio = None
    if output_type == 'latent':
        video = pipe._denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
    else:
        with torch.no_grad():
            latents = pipe._denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor).to(vae.dtype)
            dec_t = None
            video = vae.decode(latents, dec_t, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        if decode_audio:
            with torch.no_grad():
                audio_latents = pipe._denormalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
                audio_latents = pipe._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins).to(audio_vae.dtype)
                mel = audio_vae.decode(audio_latents, return_dict=False)[0]
                audio = components['vocoder'](mel)
    seconds = _time.time() - t0
    print(f'[LTX-GEN] done in {seconds:.0f}s', flush=True)
    return dict(video=video, audio=audio, latents=latents, seconds=seconds, audio_sample_rate=int(getattr(pipe, 'audio_sampling_rate', 24000) or 24000), transformer=transformer, streamer=streamer, components=components)
