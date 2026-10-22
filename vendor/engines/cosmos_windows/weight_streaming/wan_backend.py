# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/wan_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import re
from pathlib import Path
_WAN_BLOCK_RE = re.compile('(?:^|[/.])blocks\\.(\\d+)\\.')
_DM_PREFIX = 'model.diffusion_model.'
_WAN_LEGACY_FP8_SCALE = 448.0
_WAN_OFFICIAL_T2V_ROOT = Path('')

def _env_enabled(name: str, default: bool=True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}

def _configure_wan_vae_memory(vae, *, label: str='[WAN-VAE]'):
    enabled: list[str] = []
    if _env_enabled('WAN_VAE_TILING', True):
        try:
            if hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
                enabled.append('tiling')
            if hasattr(vae, 'enable_temporal_tiling'):
                vae.enable_temporal_tiling()
                enabled.append('temporal_tiling')
        except Exception as e:
            print(f'{label} VAE tiling enable skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)
    if _env_enabled('WAN_VAE_SLICING', True):
        try:
            if hasattr(vae, 'enable_slicing'):
                vae.enable_slicing()
                enabled.append('slicing')
        except Exception as e:
            print(f'{label} VAE slicing enable skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)
    if enabled:
        print(f"{label} VAE memory controls enabled: {', '.join(enabled)}", flush=True)
    return vae

def _close_streamer_for_headroom(streamer, label: str) -> None:
    if streamer is None:
        return
    close = getattr(streamer, 'close', None)
    if not callable(close):
        return
    try:
        close()
        print(f'[wan-gen] closed {label} DirectStorage streamer', flush=True)
    except Exception as e:
        print(f'[wan-gen] {label} DirectStorage streamer close skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)

def _release_bernini_restore(restore):
    if restore is None:
        return None
    try:
        restore()
    except Exception as e:
        print(f'[bernini] context restore skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)
    return None

def _find_cached_wan_scheduler_dir() -> Path | None:
    root = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--Wan-AI--Wan2.2-T2V-A14B-Diffusers' / 'snapshots'
    if not root.exists():
        return None
    for config_file in root.glob('*/scheduler/scheduler_config.json'):
        return config_file.parent
    return None

def _comfy_simple_sigmas(steps: int, shift: float):
    sh = 8.0 if shift is None else float(shift)
    out = []
    for x in range(int(steps)):
        tn = (int(steps) - x) / float(steps)
        out.append(sh * tn / (1.0 + (sh - 1.0) * tn))
    return out

def _build_wan_scheduler(scheduler_name: str, shift: float | None, *, sigmas_name: str | None=None, steps: int=30):
    name = (scheduler_name or 'flowmatch').lower()
    if (sigmas_name or '').lower() == 'simple':
        from diffusers import FlowMatchEulerDiscreteScheduler
        try:
            from diffusers import FlowMatchLCMScheduler
            sched = FlowMatchLCMScheduler(shift=1.0) if name == 'lcm' else FlowMatchEulerDiscreteScheduler(shift=1.0)
        except Exception:
            sched = FlowMatchEulerDiscreteScheduler(shift=1.0)
        return (sched, _comfy_simple_sigmas(steps, shift))
    try:
        from cosmos_windows.res4lyf_sampling import flow_scheduler_for_manual_loop, is_res4lyf_sampler, is_res4lyf_sigma
    except Exception:
        flow_scheduler_for_manual_loop = None
        is_res4lyf_sampler = lambda _name: False
        is_res4lyf_sigma = lambda _name: False
    if flow_scheduler_for_manual_loop is not None and (is_res4lyf_sampler(name) or is_res4lyf_sigma(sigmas_name)):
        from diffusers import FlowMatchEulerDiscreteScheduler
        base = FlowMatchEulerDiscreteScheduler(shift=5.0 if shift is None else float(shift))
        return flow_scheduler_for_manual_loop(base, sampler=scheduler_name, sigmas=sigmas_name, num_inference_steps=int(steps or 30), shift=5.0 if shift is None else float(shift), log_prefix='[wan-gen]')
    if name == 'unipc':
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        scheduler_dir = _find_cached_wan_scheduler_dir()
        if scheduler_dir is not None:
            kwargs = {} if shift is None else {'flow_shift': float(shift)}
            scheduler = UniPCMultistepScheduler.from_pretrained(str(scheduler_dir), **kwargs)
            print(f'[wan-gen] UniPC scheduler from {scheduler_dir}', flush=True)
            return (scheduler, None)
        kwargs = dict(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.0 if shift is None else float(shift), predict_x0=True, solver_order=2, solver_type='bh2', timestep_spacing='linspace', final_sigmas_type='zero', lower_order_final=True)
        return (UniPCMultistepScheduler(**kwargs), None)
    if name in {'flowmatch', 'flow', 'euler', 'euler_flow', 'simple'}:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return (FlowMatchEulerDiscreteScheduler(shift=5.0 if shift is None else float(shift)), None)
    if name == 'lcm':
        from diffusers import FlowMatchEulerDiscreteScheduler
        try:
            from diffusers import FlowMatchLCMScheduler
            return (FlowMatchLCMScheduler(shift=5.0 if shift is None else float(shift)), None)
        except Exception:
            return (FlowMatchEulerDiscreteScheduler(shift=5.0 if shift is None else float(shift)), None)
    raise ValueError(f'Unsupported Wan sampler={scheduler_name!r}; use flowmatch, unipc, lcm, simple, or er_sde')
_WAN22_14B_BASE = dict(patch_size=(1, 2, 2), num_attention_heads=40, attention_head_dim=128, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=13824, num_layers=40, cross_attn_norm=True, qk_norm='rms_norm_across_heads', eps=1e-06, added_kv_proj_dim=None)
WAN22_14B_T2V_CONFIG = dict(_WAN22_14B_BASE, in_channels=16)
WAN22_14B_I2V_CONFIG = dict(_WAN22_14B_BASE, in_channels=36)

def _infer_in_channels(idx, component: str) -> int:
    for name, ref in idx.refs.items():
        if name.endswith('patch_embedding.weight') and len(ref.shape) >= 2:
            return int(ref.shape[1])
    return 16

def _load_meta_wan(idx=None, component: str='', in_channels: int | None=None):
    import torch
    from accelerate import init_empty_weights
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    if in_channels is None:
        in_channels = _infer_in_channels(idx, component) if idx is not None else 16
    cfg = dict(_WAN22_14B_BASE, in_channels=in_channels)
    with init_empty_weights():
        model = WanTransformer3DModel.from_config(cfg)
    return (model, cfg, in_channels)
_WAN_RENAME_ORDERED = [('time_embedding.0', 'condition_embedder.time_embedder.linear_1'), ('time_embedding.2', 'condition_embedder.time_embedder.linear_2'), ('text_embedding.0', 'condition_embedder.text_embedder.linear_1'), ('text_embedding.2', 'condition_embedder.text_embedder.linear_2'), ('time_projection.1', 'condition_embedder.time_proj'), ('cross_attn', 'attn2'), ('self_attn', 'attn1'), ('.o.', '.to_out.0.'), ('.q.', '.to_q.'), ('.k.', '.to_k.'), ('.v.', '.to_v.'), ('.k_img.', '.add_k_proj.'), ('.v_img.', '.add_v_proj.'), ('.norm_k_img.', '.norm_added_k.'), ('head.modulation', 'scale_shift_table'), ('head.head', 'proj_out'), ('modulation', 'scale_shift_table'), ('ffn.0', 'ffn.net.0.proj'), ('ffn.2', 'ffn.net.2'), ('norm2', 'norm__placeholder'), ('norm3', 'norm2'), ('norm__placeholder', 'norm3'), ('img_emb.proj.0', 'condition_embedder.image_embedder.norm1'), ('img_emb.proj.1', 'condition_embedder.image_embedder.ff.net.0.proj'), ('img_emb.proj.3', 'condition_embedder.image_embedder.ff.net.2'), ('img_emb.proj.4', 'condition_embedder.image_embedder.norm2')]

def _rename_one(name: str) -> str:
    new = name
    for old, repl in _WAN_RENAME_ORDERED:
        new = new.replace(old, repl)
    return new

def _key_rename_only(diffusion_model_names):
    return {n: _rename_one(n) for n in diffusion_model_names}

def _has_legacy_scaled_fp8_marker(idx, component: str='') -> bool:
    pre = component + '/' if component else ''
    marker = f'{pre}{_DM_PREFIX}scaled_fp8'
    return marker in idx.refs

def _legacy_scaled_fp8_matrix_needs_rescale(t, *, name: str) -> bool:
    if not name.endswith('.weight') or getattr(t, 'ndim', 0) < 2:
        return False
    try:
        import torch
        if t.dtype != torch.float16:
            return False
        return torch.amax(torch.abs(t.detach().float())).item() > 2.0
    except Exception:
        return False

def _wan_legacy_fp8_role(model_file: str) -> str:
    name = Path(model_file).name.lower()
    return 'low' if 'low_noise' in name or 'lowv' in name or 'low' in name else 'high'

def _find_legacy_fp8_reference_file(model_file: str) -> Path | None:
    role = _wan_legacy_fp8_role(model_file)
    env_role = os.environ.get(f'WAN_FP8_REFERENCE_{role.upper()}')
    env_any = os.environ.get('WAN_FP8_REFERENCE_FILE')
    for raw in (env_role, env_any):
        if raw:
            p = Path(raw)
            if p.is_file():
                return p
    filename = f'wan2.2_t2v_{role}_noise_14B_fp16.safetensors'
    p = Path(model_file)
    candidates = [p.parent / 'split_files' / 'diffusion_models' / filename, p.parent.parent / 'split_files' / 'diffusion_models' / filename, _WAN_OFFICIAL_T2V_ROOT / filename]
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            pass
    return None

class _WanLegacyFp8Calibrator:

    def __init__(self, reference_file: Path, names: list[str]):
        from safetensors import safe_open
        self.reference_file = Path(reference_file)
        self._target_std: dict[str, float] = {}
        self._scale: dict[str, float] = {}
        wanted = {n for n in names if n.endswith('.weight')}
        with safe_open(str(self.reference_file), framework='pt', device='cpu') as sf:
            present = set(sf.keys())
            for name in sorted(wanted & present):
                t = sf.get_tensor(name)
                if getattr(t, 'ndim', 0) >= 2 and t.is_floating_point():
                    std = float(t.detach().float().std().item())
                    if std > 0.0:
                        self._target_std[name] = std
        if not self._target_std:
            raise RuntimeError(f'Wan legacy scaled-fp8 reference has no matching matrix tensors: {self.reference_file}')

    def scale_tensor_(self, t, *, name: str):
        if not _legacy_scaled_fp8_matrix_needs_rescale(t, name=name):
            return t
        target_std = self._target_std.get(name)
        if not target_std:
            raise RuntimeError(f'Wan legacy scaled-fp8 tensor {name!r} has no fp16 reference in {self.reference_file}. Set WAN_LEGACY_FP8_MODE=global to use the old 1/448 fallback for diagnostics.')
        scale = self._scale.get(name)
        if scale is None:
            raw_std = float(t.detach().float().std().item())
            if raw_std <= 0.0:
                return t
            scale = target_std / raw_std
            self._scale[name] = scale
        return t.mul_(scale)

def _build_legacy_fp8_calibrator(model_file: str, names: list[str]):
    mode = os.environ.get('WAN_LEGACY_FP8_MODE', 'reference').strip().lower()
    if mode in ('off', 'none', 'false', '0'):
        print('[WAN-STREAM] legacy scaled_fp8 marker detected; rescale disabled by WAN_LEGACY_FP8_MODE=off.', flush=True)
        return None
    if mode in ('global', 'legacy', '448'):
        print(f'[WAN-STREAM] legacy scaled_fp8 marker detected; using diagnostic global 1/{_WAN_LEGACY_FP8_SCALE:g} rescale (WAN_LEGACY_FP8_MODE={mode}).', flush=True)
        return None
    reference = _find_legacy_fp8_reference_file(model_file)
    if reference is None:
        raise RuntimeError('Wan legacy scaled_fp8 checkpoint has no per-tensor scales and no official fp16 reference was found. Install the official Wan2.2 fp16 high/low shards under split_files/diffusion_models or set WAN_FP8_REFERENCE_HIGH / WAN_FP8_REFERENCE_LOW. Set WAN_LEGACY_FP8_MODE=global only for diagnostics.')
    cal = _WanLegacyFp8Calibrator(reference, names)
    print(f'[WAN-STREAM] legacy scaled_fp8 marker detected; calibrated {len(cal._target_std)} tensors from {reference}', flush=True)
    return cal

def _maybe_unscale_legacy_fp8_tensor(t, *, name: str, enabled: bool, calibrator=None):
    if not enabled:
        return t
    if calibrator is not None:
        return calibrator.scale_tensor_(t, name=name)
    if not _legacy_scaled_fp8_matrix_needs_rescale(t, name=name):
        return t
    return t.div_(_WAN_LEGACY_FP8_SCALE)

def _unswizzle_cublas_block_scales(scale_u8, num_rows: int, num_cols: int):
    import torch

    def _cdiv(a, b):
        return -(-a // b)
    n_row_blocks = _cdiv(num_rows, 128)
    n_col_blocks = _cdiv(num_cols, 4)
    padded_rows, padded_cols = (n_row_blocks * 128, n_col_blocks * 4)
    s = scale_u8.reshape(-1, 32, 16)
    s = s.reshape(-1, 32, 4, 4).transpose(1, 2)
    s = s.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    s = s.reshape(n_row_blocks, n_col_blocks, 128, 4)
    s = s.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return s[:num_rows, :num_cols]

def dequant_mxfp8(weight_fp8, scale_u8, block: int=32, dtype=None):
    import torch
    dtype = dtype or torch.bfloat16
    w = weight_fp8.to(torch.float32)
    n_out, n_in = (w.shape[0], w.shape[-1])
    n_blk = -(-n_in // block)
    s = _unswizzle_cublas_block_scales(scale_u8, n_out, n_blk)
    factor = s.to(torch.int32) << 23
    factor = torch.where(s == 0, torch.zeros_like(factor), factor).view(torch.float32)
    factor = factor.repeat_interleave(block, dim=-1)[..., :n_in]
    return (w * factor).to(dtype)

def _load_wan_lora_diffusers(path: str) -> dict:
    from safetensors.torch import load_file
    sd = load_file(path)
    keys = list(sd.keys())
    non_diffusers = any((k.startswith('diffusion_model.') or '.lora_down.' in k or '.lora_up.' in k for k in keys))
    if not non_diffusers:
        return sd
    from diffusers.loaders import lora_conversion_utils as _conv
    is_musubi = any((k.startswith('lora_unet_') for k in keys))
    converter = _conv._convert_musubi_wan_lora_to_diffusers if is_musubi else _conv._convert_non_diffusers_wan_lora_to_diffusers
    out = converter(sd)
    print(f'[WAN-STREAM] converted LoRA {Path(path).name}: {len(keys)} comfy keys -> {len(out)} diffusers keys', flush=True)
    return out

def compute_wan_lora_deltas(loras, expert: str, device, dtype, *, alpha_default=None):
    import torch as _torch
    from safetensors.torch import load_file as _load
    out: dict = {}

    def _spec(l, k, d=None):
        return (l.get(k) if isinstance(l, dict) else getattr(l, k, d)) if l is not None else d
    n_lr = n_add = 0
    for l in loras or []:
        tgt = _spec(l, 'expert', 'both') or 'both'
        if tgt not in (expert, 'both'):
            continue
        path = _spec(l, 'path')
        w = float(_spec(l, 'weight', 1.0) or 1.0)
        if not path:
            continue
        sd = _load(path)
        bases: dict = {}
        alphas: dict = {}
        for k, v in sd.items():
            bare = k[len('diffusion_model.'):] if k.startswith('diffusion_model.') else k
            for suf in ('.lora_down.weight', '.lora_up.weight', '.diff_b', '.diff', '.alpha'):
                if bare.endswith(suf):
                    base = bare[:-len(suf)]
                    if suf == '.alpha':
                        alphas[base] = float(v.flatten()[0])
                    else:
                        bases.setdefault(base, {})[suf] = v
                    break
        for base, parts in bases.items():
            if '.lora_down.weight' in parts and '.lora_up.weight' in parts:
                dn = parts['.lora_down.weight'].to(device, dtype)
                up = parts['.lora_up.weight'].to(device, dtype)
                scale = w * (alphas[base] / dn.shape[0] if base in alphas else 1.0)
                key = base + '.weight'
                out.setdefault(key, {'lr': [], 'add': None})['lr'].append((float(scale), up, dn))
                n_lr += 1
            if '.diff' in parts:
                key = base + '.weight'
                d = parts['.diff'].to(device, dtype) * w
                e = out.setdefault(key, {'lr': [], 'add': None})
                e['add'] = d if e['add'] is None else e['add'] + d
                n_add += 1
            if '.diff_b' in parts:
                key = base + '.bias'
                d = parts['.diff_b'].to(device, dtype) * w
                e = out.setdefault(key, {'lr': [], 'add': None})
                e['add'] = d if e['add'] is None else e['add'] + d
                n_add += 1
    if out:
        print(f'[WAN-STREAM] {expert} LoRA resident (low-rank, ~factors): {len(out)} targets ({n_lr} low-rank, {n_add} diff/diff_b) — reconstructed per-layer at bind', flush=True)
    return out

def _mxfp8_scale_name(weight_name: str):
    if not weight_name.endswith('.weight'):
        return None
    return weight_name[:-len('.weight')] + '.weight_scale'

def _has_mxfp8_marker(idx, component: str='') -> bool:
    pre = component + '/' if component else ''
    for name in idx.names():
        bare = name[len(pre):] if pre and name.startswith(pre) else name
        if bare.endswith('.weight_scale'):
            ref = idx.refs.get(name)
            dt = str(getattr(ref, 'dtype', '')).lower()
            if 'u8' in dt or 'uint8' in dt:
                return True
    return False

def _umt5_config_from_dir(config_dir):
    import json as _json, os as _os
    from transformers import UMT5Config
    p = _os.path.join(config_dir, 'config.json') if config_dir else ''
    if p and _os.path.isfile(p):
        return UMT5Config(**{k: v for k, v in _json.load(open(p, encoding='utf-8')).items() if not k.startswith('_')})
    return UMT5Config(d_model=4096, d_ff=10240, d_kv=64, num_heads=64, num_layers=24, relative_attention_num_buckets=32, relative_attention_max_distance=128, vocab_size=256384, feed_forward_proj='gated-gelu', is_gated_act=True, layer_norm_epsilon=1e-06, dropout_rate=0.0, tie_word_embeddings=False)

def load_local_umt5(fp8_file: str, config_dir: str, dtype=None, device: str='cpu'):
    import json, os, torch
    from transformers import UMT5EncoderModel, UMT5Config
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    dtype = dtype or torch.bfloat16
    cfg = _umt5_config_from_dir(config_dir)
    with init_empty_weights():
        model = UMT5EncoderModel(cfg)
    sd_keys = set(model.state_dict().keys())
    with safe_open(fp8_file, 'pt', device='cpu') as sf:
        present = set(sf.keys())
        for k in present:
            if k.endswith('.scale_weight') or k not in sd_keys:
                continue
            w = sf.get_tensor(k)
            sk = k[:-len('.weight')] + '.scale_weight' if k.endswith('.weight') else None
            w = w.to(dtype) * sf.get_tensor(sk).to(dtype) if sk and sk in present else w.to(dtype)
            set_module_tensor_to_device(model, k, device, value=w)
            del w
    for n, p in list(model.named_parameters()):
        if p.device.type == 'meta':
            src = 'shared.weight' if 'embed_tokens' in n else 'encoder.embed_tokens.weight' if 'shared' in n else None
            if src and src in sd_keys:
                set_module_tensor_to_device(model, n, device, value=model.state_dict()[src])
            else:
                raise RuntimeError(f'umt5 load left {n} on meta (not in file)')
    return model
_UMT5_BLOCK_RE = re.compile('(?:^|[/.])(?:encoder\\.)?block\\.(\\d+)\\.')

def build_streamed_umt5(fp8_file: str, config_dir: str, *, dtype=None, device: str='cuda'):
    import json
    import os
    import torch
    from accelerate import init_empty_weights
    from transformers import UMT5Config, UMT5EncoderModel
    from .streamed_text_encoder import build_streamed_text_encoder
    dtype = dtype or torch.bfloat16
    cfg = _umt5_config_from_dir(config_dir)
    with init_empty_weights():
        encoder = UMT5EncoderModel(cfg)
    sd_keys = set(encoder.state_dict().keys())

    def _ckpt_to_module(src_key: str):
        if src_key.endswith('.scale_weight') or src_key.endswith('scaled_fp8') or src_key.endswith('spiece_model'):
            return None
        return src_key if src_key in sd_keys else None
    from .backends import PortableBackend
    _scale_pb = PortableBackend(pin_memory=False)

    def _fp8_dequant(idx, ref, gpu_tensor):
        if not ref.name.endswith('.weight'):
            return gpu_tensor
        scale_name = ref.name[:-len('.weight')] + '.scale_weight'
        scale_ref = idx.refs.get(scale_name)
        if scale_ref is None:
            return gpu_tensor
        scale = _scale_pb.read_tensor(scale_ref, str(gpu_tensor.device))
        return gpu_tensor.to(dtype) * scale.to(dtype)
    print(f'[WAN-TE-STREAM] meta-init + stream UMT5-xxl from {fp8_file}', flush=True)
    return build_streamed_text_encoder(encoder, fp8_file, layer_regex=_UMT5_BLOCK_RE, decoder_layers=encoder.encoder.block, ckpt_to_module=_ckpt_to_module, device=device, dtype=dtype, tie_map={'encoder.embed_tokens.weight': 'shared.weight'}, tensor_transform=_fp8_dequant, log_prefix='[WAN-TE-STREAM]')

def load_bundled_umt5(checkpoint_file: str, config_dir: str, dtype=None, device: str='cpu', prefix: str='text_encoders.umt5xxl.transformer.'):
    import json, os, torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import UMT5Config, UMT5EncoderModel
    dtype = dtype or torch.bfloat16
    cfg = _umt5_config_from_dir(config_dir)
    with init_empty_weights():
        model = UMT5EncoderModel(cfg)
    sd_keys = set(model.state_dict().keys())
    marker = prefix + 'scaled_fp8'
    with safe_open(checkpoint_file, 'pt', device='cpu') as sf:
        if prefix + 'shared.weight' not in sf.keys():
            raise FileNotFoundError(f'{checkpoint_file} does not contain bundled UMT5 weights under {prefix!r}')
        legacy_scaled_fp8 = marker in sf.keys()
        if legacy_scaled_fp8:
            print(f'[WAN-TEXT] bundled UMT5 legacy scaled_fp8 marker detected; unscaling large fp16 matrix weights by 1/{_WAN_LEGACY_FP8_SCALE:g}.', flush=True)
        for src in sf.keys():
            if not src.startswith(prefix) or src == marker:
                continue
            dst = src[len(prefix):]
            if dst not in sd_keys:
                continue
            w = sf.get_tensor(src)
            w = _maybe_unscale_legacy_fp8_tensor(w, name=dst, enabled=legacy_scaled_fp8)
            if w.is_floating_point():
                w = w.to(dtype)
            set_module_tensor_to_device(model, dst, device, value=w)
            del w
    state = model.state_dict()
    for n, p in list(model.named_parameters()):
        if p.device.type == 'meta':
            if n == 'encoder.embed_tokens.weight' and 'shared.weight' in sd_keys:
                set_module_tensor_to_device(model, n, device, value=state['shared.weight'])
            else:
                raise RuntimeError(f'bundled umt5 load left {n} on meta (not in checkpoint)')
    return model

def wan_group_by_layer(idx, component: str=''):
    pre = component + '/' if component else ''
    layers, globs = ({}, [])
    for name, ref in idx.refs.items():
        if pre and (not name.startswith(pre)):
            continue
        bare = name[len(pre):] if pre else name
        if bare.startswith('text_encoders.') or bare.startswith('vae.'):
            continue
        if bare.endswith('scaled_fp8'):
            continue
        if bare.startswith(_DM_PREFIX):
            bare = bare[len(_DM_PREFIX):]
        elif 'diffusion_model' in bare:
            continue
        m = _WAN_BLOCK_RE.search(bare)
        if m:
            layers.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (layers, globs)

def build_streamed_wan(model_file: str, device: str='cuda', n_gpu_buffers: int=3, in_channels: int | None=None, lora_deltas: dict | None=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    from .build_streamed_transformer import StreamingWeightLoader
    p = Path(model_file)
    component = '' if p.is_file() and p.suffix == '.safetensors' else 'transformer'
    idx = build_index(model_file)
    if in_channels is None:
        in_channels = _infer_in_channels(idx, component)
    model, cfg, in_channels = _load_meta_wan(idx, component, in_channels)
    _wan_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(os.environ.get('WAN_STREAM_DTYPE', 'fp16'), torch.float16)
    model = model.to(_wan_dtype)
    tf = model
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = wan_group_by_layer(idx, component)
    n_blocks = len(layer_refs)
    if n_blocks == 0:
        raise RuntimeError(f'wan_group_by_layer found 0 blocks in {model_file} — the `blocks.N.` regex matched nothing. Check the tensor names: {list(idx.names())[:4]}')
    pre = component + '/' if component else ''

    def _bare(name):
        b = name[len(pre):] if pre and name.startswith(pre) else name
        return b[len(_DM_PREFIX):] if b.startswith(_DM_PREFIX) else b
    dit_names = [_bare(r.name) for refs in layer_refs.values() for r in refs] + [_bare(r.name) for r in global_refs]
    remap = _key_rename_only(dit_names)

    def _pp(name):
        return remap.get(_bare(name), _bare(name))
    legacy_scaled_fp8 = _has_legacy_scaled_fp8_marker(idx, component)
    fp8_calibrator = _build_legacy_fp8_calibrator(model_file, dit_names) if legacy_scaled_fp8 else None
    if legacy_scaled_fp8:
        detail = 'reference-calibrated' if fp8_calibrator is not None else f'global 1/{_WAN_LEGACY_FP8_SCALE:g}'
        print(f'[WAN-STREAM] legacy scaled_fp8 DiT matrix rescale active: {detail}.', flush=True)
    _mxfp8 = _has_mxfp8_marker(idx, component)
    if _mxfp8:
        print('[WAN-STREAM] MXFP8 (microscaling fp8) DiT detected — per-block E8M0 dequant active.', flush=True)
    _mxfp8_pb = PortableBackend(pin_memory=False) if _mxfp8 else None

    def _maybe_dequant_mxfp8(t, ref_name, scale_lookup):
        if not _mxfp8:
            return t
        sname = _mxfp8_scale_name(ref_name)
        if sname is None:
            return t
        scale = scale_lookup(sname)
        if scale is None:
            return t
        return dequant_mxfp8(t, scale.to(t.device))
    pb = PortableBackend(pin_memory=True)

    def _apply_lora_delta(t, bare_name):
        if not lora_deltas:
            return t
        spec = lora_deltas.get(bare_name)
        if spec is None:
            return t
        acc = None
        for scale, up, dn in spec['lr']:
            dl = up @ dn * scale
            acc = dl if acc is None else acc.add_(dl)
        if spec['add'] is not None:
            acc = spec['add'] if acc is None else acc.add_(spec['add'])
        if acc is not None:
            if acc.shape != t.shape:
                acc = acc.reshape(t.shape) if acc.numel() == t.numel() else acc
            t = t + (acc if acc.dtype == t.dtype else acc.to(t.dtype))
        return t

    def _global_scale_lookup(scale_name):
        for _r in global_refs:
            if _bare(_r.name) == scale_name or _r.name == scale_name:
                return _mxfp8_pb.read_tensor(_r, device)
        sref = idx.refs.get(scale_name) or idx.refs.get(pre + scale_name if pre else scale_name)
        return _mxfp8_pb.read_tensor(sref, device) if sref is not None else None
    for ref in global_refs:
        path = _pp(ref.name)
        if path in sd_keys:
            value = pb.read_tensor(ref, device)
            value = _maybe_dequant_mxfp8(value, _bare(ref.name), _global_scale_lookup)
            value = _maybe_unscale_legacy_fp8_tensor(value, name=_bare(ref.name), enabled=legacy_scaled_fp8, calibrator=fp8_calibrator)
            value = _apply_lora_delta(value, _bare(ref.name))
            set_module_tensor_to_device(model, path, device, value=value)
    for name, buf in list(model.named_buffers()):
        if buf.device.type != 'meta' and str(buf.device) != device:
            set_module_tensor_to_device(model, name, device, value=buf.to(device))
    wan_idx = _WanGroupedIndex(idx, component)
    streamer = DirectStorageStreamer(wan_idx, component='', n_gpu_buffers=n_gpu_buffers, device=device)
    from .ds_backend import enforce_ds_mode
    enforce_ds_mode(streamer, 'Wan', 'WAN')
    print(f'[WAN-STREAM] {n_blocks} blocks streamed, {len(global_refs)} resident globals, in_channels={in_channels}, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    layer_paths = {il: [_pp(r.name) for r in refs if _pp(r.name) in sd_keys] for il, refs in layer_refs.items()}

    def _lora_aware_path(pp):
        if '.' not in pp:
            return pp
        mod_path, leaf = pp.rsplit('.', 1)
        if leaf not in ('weight', 'bias'):
            return pp
        try:
            mod = model.get_submodule(mod_path)
        except AttributeError:
            return pp
        return f'{mod_path}.base_layer.{leaf}' if hasattr(mod, 'base_layer') else pp

    def make_bind(il):

        def bind():
            streamed = streamer.wait_layer(il)
            scales = {_bare(n): v for n, v in streamed.items() if _bare(n).endswith('.weight_scale')} if _mxfp8 else {}
            for name, t in streamed.items():
                pp = _pp(name)
                if pp in sd_keys:
                    t = _maybe_dequant_mxfp8(t, _bare(name), lambda s: scales.get(s))
                    t = _maybe_unscale_legacy_fp8_tensor(t, name=_bare(name), enabled=legacy_scaled_fp8, calibrator=fp8_calibrator)
                    t = _apply_lora_delta(t, _bare(name))
                    set_module_tensor_to_device(model, _lora_aware_path(pp), device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for pp in layer_paths[il]:
                set_module_tensor_to_device(model, _lora_aware_path(pp), 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(tf.blocks):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *a, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kw):
            _b()
            try:
                out = _fwd(_layer, hidden_states, *a, **kw)
            finally:
                _e()
            return out
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.eval()
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f"[WAN-STREAM] wrapped {n_blocks} WanTransformerBlocks (bind/evict per block; single residual stream). Dense DiT — FETCH_ALL (no per-token MoE; Wan 2.2 'MoE' is the High/Low denoiser split, handled by two streamers).", flush=True)
    return (model, streamer)

def build_streamed_backward_wan(model_file: str, device: str='cuda', n_gpu_buffers: int=1, in_channels: int | None=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    from .build_streamed_transformer import StreamingWeightLoader
    from ..streamed_backward import apply_streamed_multi
    p = Path(model_file)
    component = '' if p.is_file() and p.suffix == '.safetensors' else 'transformer'
    idx = build_index(model_file)
    if in_channels is None:
        in_channels = _infer_in_channels(idx, component)
    model, cfg, in_channels = _load_meta_wan(idx, component, in_channels)
    tf = model
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = wan_group_by_layer(idx, component)
    n_blocks = len(layer_refs)
    pre = component + '/' if component else ''

    def _bare(name):
        b = name[len(pre):] if pre and name.startswith(pre) else name
        return b[len(_DM_PREFIX):] if b.startswith(_DM_PREFIX) else b
    dit_names = [_bare(r.name) for refs in layer_refs.values() for r in refs] + [_bare(r.name) for r in global_refs]
    remap = _key_rename_only(dit_names)

    def _pp(name):
        return remap.get(_bare(name), _bare(name))
    legacy_scaled_fp8 = _has_legacy_scaled_fp8_marker(idx, component)
    fp8_calibrator = _build_legacy_fp8_calibrator(model_file, dit_names) if legacy_scaled_fp8 else None
    if legacy_scaled_fp8:
        detail = 'reference-calibrated' if fp8_calibrator is not None else f'global 1/{_WAN_LEGACY_FP8_SCALE:g}'
        print(f'[WAN-BWD] legacy scaled_fp8 DiT matrix rescale active: {detail}.', flush=True)

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = _pp(ref.name)
        if path in sd_keys:
            value = pb.read_tensor(ref, device)
            value = _maybe_unscale_legacy_fp8_tensor(value, name=_bare(ref.name), enabled=legacy_scaled_fp8, calibrator=fp8_calibrator)
            set_module_tensor_to_device(model, path, device, value=value)
    for name, buf in list(model.named_buffers()):
        if buf.device.type != 'meta' and str(buf.device) != device:
            set_module_tensor_to_device(model, name, device, value=buf.to(device))
    wan_idx = _WanGroupedIndex(idx, component)
    streamer = DirectStorageStreamer(wan_idx, component='', n_gpu_buffers=n_gpu_buffers, device=device)
    print(f"[WAN-BWD] mode={streamer.mode} {n_blocks} blocks, {len(streamer.tbufs)} GPU buf(s), {len(global_refs)} resident globals (reverse-capable; {('prefetch ON' if n_gpu_buffers > 1 else 'single-buffer=validated reverse')})", flush=True)
    layer_paths = {il: [_pp(r.name) for r, _ in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in streamer.plan}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                pp = _pp(name)
                if pp in sd_keys:
                    t = _maybe_unscale_legacy_fp8_tensor(t, name=_bare(name), enabled=legacy_scaled_fp8, calibrator=fp8_calibrator)
                    set_module_tensor_to_device(model, _resolve(pp), device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for pp in layer_paths[il]:
                set_module_tensor_to_device(model, _resolve(pp), 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(tf.blocks):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *args, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kwargs):
            call = lambda lyr, h: _fwd(lyr, h, *args, **kwargs)
            return apply_streamed_multi(_layer, call, hidden_states, bind=_b, evict=_e)
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[WAN-BWD] wrapped {n_blocks} WanTransformerBlocks in StreamedLayerMulti — attach LoRA + materialize_meta_lora, forward, then streamer.new_pass(reverse=True) before loss.backward()', flush=True)
    return (model, streamer)

class _WanGroupedIndex:

    def __init__(self, base_idx, component: str=''):
        layers, _globs = wan_group_by_layer(base_idx, component)
        self._layers = layers
        self.refs = {r.name: r for refs in layers.values() for r in refs}

    def group_by_layer(self, component: str=''):
        return (self._layers, [])

    def names(self):
        return list(self.refs.keys())

def build_streamed_wan_pipeline(high_file: str, low_file: str | None, vae_path: str, text_encoder_path: str, tokenizer_path: str | None=None, boundary_ratio: float | None=0.875, device: str='cuda', n_gpu_buffers: int=3, i2v: bool=False, in_channels: int | None=None):
    from diffusers import WanPipeline, WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
    if in_channels is None:
        in_channels = 36 if i2v else 16
    transformer, streamer_high = build_streamed_wan(high_file, device=device, n_gpu_buffers=n_gpu_buffers, in_channels=in_channels)
    transformer_2, streamer_low = (None, None)
    if low_file is not None:
        transformer_2, streamer_low = build_streamed_wan(low_file, device=device, n_gpu_buffers=n_gpu_buffers, in_channels=in_channels)
    vae = _load_wan_vae(vae_path, device)
    text_encoder = _load_umt5(text_encoder_path, device)
    tokenizer = _load_umt5_tokenizer(tokenizer_path or text_encoder_path)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    PipeCls = WanImageToVideoPipeline if i2v else WanPipeline
    pipe = PipeCls(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, transformer=transformer, transformer_2=transformer_2, boundary_ratio=boundary_ratio)
    print(f'[WAN-PIPE] {PipeCls.__name__} assembled: High+Low streamed (boundary_ratio={boundary_ratio}), VAE+UMT5 resident.', flush=True)
    return (pipe, streamer_high, streamer_low)

def _load_wan_vae(vae_path: str, device: str):
    from diffusers import AutoencoderKLWan
    if Path(vae_path).is_file():
        vae = AutoencoderKLWan.from_single_file(vae_path)
    else:
        vae = AutoencoderKLWan.from_pretrained(vae_path)
    return _configure_wan_vae_memory(vae.to(device), label='[WAN-PIPE]')

def _load_umt5(text_encoder_path: str, device: str):
    from transformers import UMT5EncoderModel
    return UMT5EncoderModel.from_pretrained(text_encoder_path, device_map={'': device})

def _load_umt5_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)

def resolve_wan_crossover_step(num_steps: int, crossover_step: int | None=None, boundary_ratio: float | None=None, timesteps=None, num_train_timesteps: int=1000) -> int:
    if crossover_step is not None:
        out = int(crossover_step)
    elif boundary_ratio is not None:
        if timesteps is None:
            raise ValueError('timesteps are required when resolving boundary_ratio')
        boundary_timestep = float(boundary_ratio) * int(num_train_timesteps)
        out = sum((1 for tt in timesteps if float(tt) >= boundary_timestep))
    else:
        out = max(1, int(num_steps) // 2)
    if out < 0 or out > int(num_steps):
        raise ValueError(f'crossover_step must be in [0, {int(num_steps)}], got {out}')
    return out
if __name__ == '__main__':
    import sys
    mf = sys.argv[1] if len(sys.argv) > 1 else ''
    from .safetensors_index import build_index
    p = Path(mf)
    component = '' if p.is_file() and p.suffix == '.safetensors' else 'transformer'
    print(f'file: {mf}  component={component!r}')
    idx = build_index(mf)
    in_ch = _infer_in_channels(idx, component)
    model, cfg, in_ch = _load_meta_wan(idx, component, in_ch)
    sd_keys = set(model.state_dict().keys())
    print(f"meta model: {type(model).__name__}, {len(model.blocks)} blocks, dim={cfg['attention_head_dim'] * cfg['num_attention_heads']}, in_channels={in_ch}")
    layer_refs, global_refs = wan_group_by_layer(idx, component)
    print(f'index: {len(layer_refs)} block-groups, {len(global_refs)} globals (dropped bundled text_encoders/vae)')
    pre = component + '/' if component else ''

    def _bare(name):
        b = name[len(pre):] if pre and name.startswith(pre) else name
        return b[len(_DM_PREFIX):] if b.startswith(_DM_PREFIX) else b
    dit_names = [_bare(r.name) for refs in layer_refs.values() for r in refs] + [_bare(r.name) for r in global_refs]
    remap = _key_rename_only(dit_names)
    idx_paths = {remap.get(_bare(r.name), _bare(r.name)) for refs in layer_refs.values() for r in refs}
    idx_paths |= {remap.get(_bare(r.name), _bare(r.name)) for r in global_refs}
    missing_in_tree = sorted(idx_paths - sd_keys)
    extra_in_tree = sorted(sd_keys - idx_paths)
    nexp_per_block = len(layer_refs.get(0, []))
    print(f'block 0: {nexp_per_block} tensors (the streamed unit)')
    print(f'idx param paths (remapped): {len(idx_paths)} | meta state_dict keys: {len(sd_keys)}')
    print(f'MISSING in module tree (FATAL if >0): {len(missing_in_tree)}')
    for k in missing_in_tree[:20]:
        print(f'  MISSING: {k}')
    print(f'extra in module tree (buffers/tied, usually OK): {len(extra_in_tree)}')
    for k in extra_in_tree[:10]:
        print(f'  extra: {k}')
    ok = len(layer_refs) == cfg['num_layers'] and len(missing_in_tree) == 0
    print('RESULT', 'OK' if ok else 'BLOCK_COUNT_MISMATCH' if len(layer_refs) != cfg['num_layers'] else 'KEYSET_FAIL')
    sys.exit(0 if ok else 1)

def streamed_wan_generate(high_file: str, prompt: str, height: int=480, width: int=832, num_frames: int=81, steps: int=30, *, low_file: str | None=None, vae_path: str='', umt5_file: str='', umt5_config_dir: str=None, umt5_tokenizer_dir: str=None, negative_prompt: str='', guidance_scale: float=5.0, guidance_scale_2: float | None=None, boundary_ratio: float | None=None, crossover_step: int | None=None, low_schedule_steps: int | None=None, shift: float | None=None, seed: int=0, in_channels: int | None=None, n_gpu_buffers: int=3, device: str='cuda', dtype=None, max_sequence_length: int=512, scheduler_name: str='flowmatch', sigmas_name: str | None=None, return_latents: bool=False, lazy_denoisers: bool=True, use_bundled_text_encoder: bool=False, image_path: str | None=None, last_image_path: str | None=None, keyframes: list | None=None, stream_text_encoder: bool=True, bernini_reference_images: list | None=None, bernini_source_video: str | None=None, bernini_reference_video: str | None=None, bernini_task: str | None=None, loras: list | None=None, progress=None, should_cancel=None, init_image_path: str | None=None, denoise_strength: float=1.0):
    import gc
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    DEV = device
    DTYPE = dtype or torch.float16
    if image_path and in_channels is None:
        from .safetensors_index import build_index
        inferred = _infer_in_channels(build_index(high_file), '')
        if inferred != 36:
            raise RuntimeError(f'Wan image conditioning requires a 36-channel I2V checkpoint, but {high_file!r} has in_channels={inferred}. Register a Wan I2V high-noise denoiser for i2v/fflf.')
        in_channels = inferred
    _ld_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(os.environ.get('WAN_STREAM_DTYPE', 'fp16'), torch.float16)
    _high_deltas = compute_wan_lora_deltas(loras, 'high', DEV, _ld_dtype) if loras else None
    _low_deltas = compute_wan_lora_deltas(loras, 'low', DEV, _ld_dtype) if loras else None
    transformer, streamer_high = build_streamed_wan(high_file, device=DEV, n_gpu_buffers=n_gpu_buffers, in_channels=in_channels, lora_deltas=_high_deltas)
    _bernini = bool(bernini_reference_images or bernini_source_video or bernini_reference_video or bernini_task)
    if _bernini:
        from . import bernini_backend as _bb
        _bb.patch_wan_in_context(transformer)

    def _loras_for(expert: str):
        out = []
        for l in loras or []:
            tgt = (l.get('expert') if isinstance(l, dict) else getattr(l, 'expert', None)) or 'both'
            if tgt in (expert, 'both'):
                out.append(l)
        return out

    def _apply_loras(tf, expert: str):
        specs = _loras_for(expert)
        if not specs:
            return
        import torch as _torch
        from accelerate.utils import set_module_tensor_to_device
        names, weights, converted = ([], [], {})
        for i, l in enumerate(specs):
            path = l.get('path') if isinstance(l, dict) else getattr(l, 'path', None)
            w = float((l.get('weight') if isinstance(l, dict) else getattr(l, 'weight', 1.0)) or 1.0)
            name = (l.get('name') if isinstance(l, dict) else getattr(l, 'name', None)) or f'lora{i}'
            sd = _load_wan_lora_diffusers(path)
            try:
                tf.load_lora_adapter(sd, adapter_name=name)
            except Exception as e:
                raise RuntimeError(f'failed to load LoRA {path!r} onto the {expert} Wan/Bernini expert: {e}') from e
            names.append(name)
            weights.append(w)
            converted[name] = {k.split('transformer.', 1)[-1]: v for k, v in sd.items() if '.lora_' in k}
        attached = any((getattr(m, 'lora_A', None) for _, m in tf.named_modules()))
        if not attached:
            raise RuntimeError(f"LoRA(s) for the {expert} expert attached to NO modules (0 target Linears matched). Keys don't map to the diffusers WanTransformer3DModel even after conversion — the gen would be unconditioned. Check the LoRA is a Wan/Bernini adapter.")
        placed = 0
        for pname, p in list(tf.named_parameters()):
            if '.lora_' not in pname:
                continue
            for name in names:
                tok = f'.{name}.'
                if tok not in pname:
                    continue
                src_key = pname.replace(tok, '.')
                val = converted.get(name, {}).get(src_key)
                if val is not None:
                    set_module_tensor_to_device(tf, pname, device, value=val.to(device=device, dtype=p.dtype))
                    placed += 1
                elif p.device.type == 'meta':
                    set_module_tensor_to_device(tf, pname, device, value=_torch.zeros(p.shape, dtype=p.dtype, device=device))
                break
        if hasattr(tf, 'set_adapters'):
            tf.set_adapters(names, weights)
        print(f'[WAN-STREAM] applied {len(names)} LoRA(s) to the {expert} expert ({placed} A/B tensors placed resident): {list(zip(names, weights))}', flush=True)
        print(f'[WAN-STREAM] applied {len(names)} LoRA(s) to the {expert} expert: {list(zip(names, weights))}', flush=True)
    transformer_2, streamer_low = (None, None)
    lazy = bool(lazy_denoisers) and low_file is not None
    if low_file is not None and (not lazy):
        transformer_2, streamer_low = build_streamed_wan(low_file, device=DEV, n_gpu_buffers=n_gpu_buffers, in_channels=in_channels, lora_deltas=_low_deltas)
        if _bernini:
            from . import bernini_backend as _bb
            _bb.patch_wan_in_context(transformer_2)
    transformer_dtype = transformer.dtype
    vae = AutoencoderKLWan.from_single_file(vae_path) if Path(vae_path).is_file() else AutoencoderKLWan.from_pretrained(vae_path)
    vae = _configure_wan_vae_memory(vae.to(DEV, torch.float32).eval(), label='[wan-gen]')
    VST = vae.config.scale_factor_temporal
    VSS = vae.config.scale_factor_spatial
    video_processor = VideoProcessor(vae_scale_factor=VSS)
    import glob as _glob, os as _os

    def _wan_ref_snapshot():
        hubs = [_os.environ.get('HF_HUB_CACHE') or '', _os.path.join(_os.environ.get('HF_HOME') or '', 'hub') if _os.environ.get('HF_HOME') else '', '', _os.path.expanduser('~/.cache/huggingface/hub')]
        for hub in hubs:
            if not hub or not _os.path.isdir(hub):
                continue
            snaps = _glob.glob(_os.path.join(hub, 'models--Wan-AI--Wan2.2-T2V-A14B-Diffusers', 'snapshots', '*'))
            for s in snaps:
                if _os.path.isfile(_os.path.join(s, 'tokenizer', 'spiece.model')):
                    return s
            if snaps:
                return snaps[0]
        return None
    _ref_snap = _wan_ref_snapshot()
    cfg_dir = umt5_config_dir or (_os.path.join(_ref_snap, 'text_encoder') if _ref_snap else str(Path(umt5_file).parent))
    tok_dir = umt5_tokenizer_dir or (_os.path.join(_ref_snap, 'tokenizer') if _ref_snap else cfg_dir)
    text_encoder = None
    if use_bundled_text_encoder:
        try:
            text_encoder = load_bundled_umt5(high_file, cfg_dir, dtype=DTYPE, device=DEV).eval()
        except Exception as e:
            print(f'[WAN-TEXT] bundled UMT5 unavailable, falling back to external text encoder: {e}', flush=True)
    if text_encoder is None:
        if stream_text_encoder and str(DEV).startswith('cuda') and (not use_bundled_text_encoder):
            from .vram_guard import preflight_vram as _pf
            _pf(DEV, need_mib=6144, label='Wan UMT5-xxl text encoder (streamed)')
            text_encoder = build_streamed_umt5(umt5_file, cfg_dir, dtype=DTYPE, device=DEV)
        else:
            text_encoder = load_local_umt5(umt5_file, cfg_dir, dtype=DTYPE, device=DEV).eval()
    te_streamed = bool(getattr(text_encoder, '_sb_streamed', False))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    @torch.no_grad()
    def _encode_one(text):
        text = [prompt_clean(text)]
        ti = tokenizer(text, padding='max_length', max_length=max_sequence_length, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        ids, mask = (ti.input_ids, ti.attention_mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = text_encoder(ids.to(DEV), mask.to(DEV)).last_hidden_state
        emb = emb.to(dtype=DTYPE, device=DEV)
        emb = [u[:v] for u, v in zip(emb, seq_lens)]
        emb = torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in emb], dim=0)
        return emb
    do_cfg = guidance_scale > 1.0
    _prompt = prompt
    if _bernini and bernini_task:
        from . import bernini_backend as _bb
        _prompt = _bb.task_prefixed_prompt(bernini_task, prompt)
    prompt_embeds = _encode_one(_prompt).to(transformer_dtype)
    _WAN_CANON_NEG = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    negative_prompt_embeds = _encode_one(negative_prompt or _WAN_CANON_NEG).to(transformer_dtype) if do_cfg else None
    _cfg_zero_star = os.environ.get('WAN_CFG_ZERO_STAR', '1') == '1'
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    patch = transformer.config.patch_size
    h_mult, w_mult = (VSS * patch[1], VSS * patch[2])
    height = int(height) // h_mult * h_mult
    width = int(width) // w_mult * w_mult
    if num_frames % VST != 1:
        num_frames = num_frames // VST * VST + 1
    num_frames = max(num_frames, 1)

    def _normalize_keyframes() -> list[tuple[int, str, float]]:
        total = max(1, int(num_frames or 1))
        end_frame = max(0, total - 1)
        out: dict[int, tuple[str, float]] = {}

        def add(path: str | None, frame: int | None, strength: float=1.0):
            if not path:
                return
            f = max(0, min(end_frame, int(frame or 0)))
            out[f] = (str(path), float(strength))
        add(image_path, 0, 1.0)
        add(last_image_path, end_frame, 1.0)
        for item in keyframes or []:
            if hasattr(item, 'model_dump'):
                item = item.model_dump()
            if not isinstance(item, dict):
                continue
            role = item.get('role')
            if role == 'first':
                frame = 0
            elif role == 'last':
                frame = end_frame
            else:
                frame = item.get('frame')
                if frame is None and item.get('at_fraction') is not None:
                    frame = round(float(item['at_fraction']) * end_frame)
            add(item.get('image_path'), frame, float(item.get('strength') or 1.0))
        return [(frame, path, strength) for frame, (path, strength) in sorted(out.items())]
    conditioned_frames = _normalize_keyframes()
    is_i2v = bool(conditioned_frames)
    if is_i2v and int(transformer.config.in_channels) != 36:
        raise RuntimeError(f'Wan image conditioning requires a 36-channel I2V checkpoint, got in_channels={transformer.config.in_channels}.')
    if last_image_path and num_frames < 2:
        raise RuntimeError('Wan first/last-frame conditioning needs num_frames >= 2')
    num_channels_latents = int(vae.config.z_dim if is_i2v else transformer.config.in_channels)
    num_latent_frames = (num_frames - 1) // VST + 1
    shape = (1, num_channels_latents, num_latent_frames, height // VSS, width // VSS)
    g = torch.Generator(device=DEV).manual_seed(int(seed))
    latents = torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
    i2v_condition = None
    if is_i2v:
        from PIL import Image
        from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

        def _load_condition_image(path: str, label: str):
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f'Wan {label} frame not found: {path}')
            img = Image.open(p).convert('RGB')
            return video_processor.preprocess(img, height=height, width=width).to(DEV, dtype=torch.float32)
        video_condition = torch.zeros(1, 3, num_frames, height, width, device=DEV, dtype=torch.float32)
        latent_height, latent_width = (height // VSS, width // VSS)
        mask_lat_size = torch.zeros(1, 1, num_frames, latent_height, latent_width, dtype=latents.dtype, device=DEV)
        for frame, path, strength in conditioned_frames:
            frame_tensor = _load_condition_image(path, f'keyframe@{frame}').unsqueeze(2)
            video_condition[:, :, frame:frame + 1] = frame_tensor
            mask_lat_size[:, :, frame:frame + 1] = float(strength)
        video_condition = video_condition.to(device=DEV, dtype=vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latent_condition = retrieve_latents(vae.encode(video_condition), sample_mode='argmax').to(latents.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=VST)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(1, -1, VST, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device)
        i2v_condition = torch.concat([mask_lat_size, latent_condition], dim=1)
        cond_desc = ', '.join((f'{frame}:{Path(path).name}@{strength:g}' for frame, path, strength in conditioned_frames))
        print(f'[wan-gen] image conditioning enabled: {cond_desc} (latent input channels {latents.shape[1]}+{i2v_condition.shape[1]}=36)', flush=True)
    scheduler, custom_sigmas = _build_wan_scheduler(scheduler_name, shift, sigmas_name=sigmas_name, steps=steps)
    if custom_sigmas is not None:
        scheduler.set_timesteps(sigmas=custom_sigmas, device=DEV)
    else:
        scheduler.set_timesteps(steps, device=DEV)
    timesteps = scheduler.timesteps
    scheduler.set_begin_index(0)
    sdedit_start = 0
    _sdedit = bool(init_image_path) and (not is_i2v) and (0.0 < float(denoise_strength) < 1.0)
    if _sdedit:
        if low_schedule_steps is not None and low_file is not None:
            raise RuntimeError('Wan SDEdit i2i is not supported with the two-denoiser high/low schedule (low_schedule_steps + low_file). Use a single-denoiser model, or strength=1.0.')
        from PIL import Image as _Image
        from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents as _retrieve_latents
        from .sdedit import flow_start_index as _flow_start_index, flow_noise_latent as _flow_noise_latent
        with torch.no_grad():
            _img = _Image.open(init_image_path).convert('RGB')
            _vid = video_processor.preprocess(_img, height=height, width=width).to(DEV, vae.dtype)
            _vid = _vid.unsqueeze(2)
            if num_frames > 1:
                _vid = _vid.repeat(1, 1, num_frames, 1, 1)
            _lmean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            _lstd = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            _x0 = _retrieve_latents(vae.encode(_vid), sample_mode='argmax').to(latents.dtype)
            _x0 = ((_x0 - _lmean) * _lstd).to(torch.float32)
            sdedit_start = _flow_start_index(len(timesteps), denoise_strength)
            _sigma0 = float(scheduler.sigmas[sdedit_start])
            _noise = torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
            latents = _flow_noise_latent(_x0, _noise, _sigma0).to(torch.float32)
        scheduler.set_begin_index(sdedit_start)
        print(f'[wan-gen] SDEdit i2i: strength={denoise_strength} -> start step {sdedit_start}/{len(timesteps)} sigma0={_sigma0:.4f}', flush=True)
        timesteps = timesteps[sdedit_start:]
    scheduler_low = None
    timesteps_low = None
    if low_schedule_steps is not None and low_file is not None:
        scheduler_low, custom_sigmas_low = _build_wan_scheduler(scheduler_name, shift, sigmas_name=sigmas_name, steps=int(low_schedule_steps))
        if custom_sigmas_low is not None:
            scheduler_low.set_timesteps(sigmas=custom_sigmas_low, device=DEV)
        else:
            scheduler_low.set_timesteps(int(low_schedule_steps), device=DEV)
        timesteps_low = scheduler_low.timesteps
    if low_file is None:
        effective_crossover_step = None
    else:
        effective_crossover_step = resolve_wan_crossover_step(len(timesteps), crossover_step=crossover_step, boundary_ratio=boundary_ratio, timesteps=timesteps, num_train_timesteps=scheduler.config.num_train_timesteps)
    if guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale
    if scheduler_low is not None and effective_crossover_step is not None:
        xo = int(effective_crossover_step)
        scheduler_low.set_begin_index(xo)
        _step_plan = [(scheduler, t) for t in timesteps[:xo]] + [(scheduler_low, t) for t in timesteps_low[xo:]]
        timesteps = [t for _, t in _step_plan]
        print(f'[wan-gen] two-schedule mode: HIGH {xo} steps of {len(scheduler.timesteps)}, LOW {len(_step_plan) - xo} steps of {len(timesteps_low)} (comfy pair semantics)', flush=True)
    else:
        _step_plan = [(scheduler, t) for t in timesteps]
    import time as _time
    attention_kwargs = None
    split_desc = '' if low_file is None else f'/{guidance_scale_2}@step{effective_crossover_step}'
    if low_file is not None and boundary_ratio is not None and (crossover_step is None):
        split_desc += f'(ratio={boundary_ratio:g})'
    sigma_desc = f'/{sigmas_name}' if sigmas_name else ''
    print(f'[wan-gen] {height}x{width}x{num_frames}f ({num_latent_frames} lat-frames), {steps} steps, cfg={guidance_scale}{split_desc}, {scheduler_name}{sigma_desc}, seed={seed}', flush=True)
    _bernini_restore = None
    if _bernini:
        from . import bernini_backend as _bb
        from PIL import Image
        import numpy as _np

        def _img_latent(path):
            chw = _bb._load_image_chw(path, (height, width), DEV, vae.dtype)
            t5 = chw.unsqueeze(0).unsqueeze(2).to(DEV, vae.dtype)
            return vae.encode(t5).latent_dist.mode().to(transformer_dtype)

        def _video_latent(path):
            cthw = _bb._load_video_cthw(path, (height, width), DEV, vae.dtype, max_frames=num_frames)
            t5 = cthw.unsqueeze(0).to(DEV, vae.dtype)
            return vae.encode(t5).latent_dist.mode().to(transformer_dtype)
        _ctx = []
        if bernini_source_video:
            _ctx.append(_video_latent(bernini_source_video) if _bb._is_video_file(bernini_source_video) else _img_latent(bernini_source_video))
        if bernini_reference_video:
            _ctx.append(_video_latent(bernini_reference_video) if _bb._is_video_file(bernini_reference_video) else _img_latent(bernini_reference_video))
        for _rp in bernini_reference_images or []:
            _ctx.append(_img_latent(_rp))
        if _ctx:
            _bernini_restore = _bb.bind_context_latents(transformer, _ctx)
            print(f'[bernini] bound {len(_ctx)} context latent(s) to the streamed transformer', flush=True)
    high_active = True
    _step_sync = _os.environ.get('WAN_STEP_SYNC_TIMING', '0') == '1'
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if should_cancel is not None and should_cancel():
                raise RuntimeError('generation cancelled')
            if _step_sync:
                torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            if effective_crossover_step is None or i < effective_crossover_step:
                current_model, cur_g = (transformer, guidance_scale)
            else:
                if high_active:
                    if hasattr(scheduler, 'model_outputs'):
                        _order = int(getattr(getattr(scheduler, 'config', None), 'solver_order', 2) or 2)
                        scheduler.model_outputs = [None] * _order
                    if hasattr(scheduler, 'lower_order_nums'):
                        scheduler.lower_order_nums = 0
                    if hasattr(scheduler, 'last_sample'):
                        scheduler.last_sample = None
                if lazy and high_active:
                    _bernini_restore = _release_bernini_restore(_bernini_restore)
                    _close_streamer_for_headroom(streamer_high, 'HIGH before LOW build')
                    transformer = None
                    streamer_high = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    transformer_2, streamer_low = build_streamed_wan(low_file, device=DEV, n_gpu_buffers=n_gpu_buffers, in_channels=in_channels, lora_deltas=_low_deltas)
                    if _bernini:
                        from . import bernini_backend as _bb
                        _bb.patch_wan_in_context(transformer_2)
                        if _ctx:
                            _bernini_restore = _bb.bind_context_latents(transformer_2, _ctx)
                    print(f'[wan-gen] crossover@step{i + 1}: evicted HIGH, streaming LOW (lazy)', flush=True)
                high_active = False
                current_model, cur_g = (transformer_2, guidance_scale_2)
            latent_model_input = latents.to(device=DEV, dtype=transformer_dtype)
            if i2v_condition is not None:
                latent_model_input = torch.cat([latent_model_input, i2v_condition.to(device=DEV, dtype=transformer_dtype)], dim=1)
            timestep = t.to(DEV).expand(latents.shape[0])
            noise_pred = current_model(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds, attention_kwargs=attention_kwargs, return_dict=False)[0]
            if do_cfg:
                noise_uncond = current_model(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=negative_prompt_embeds, attention_kwargs=attention_kwargs, return_dict=False)[0]
                if _cfg_zero_star:
                    _flat_c = noise_pred.flatten(1).float()
                    _flat_u = noise_uncond.flatten(1).float()
                    _alpha = (_flat_c * _flat_u).sum(1) / (_flat_u.square().sum(1) + 1e-08)
                    _alpha = _alpha.view(-1, *[1] * (noise_pred.ndim - 1)).to(noise_pred.dtype)
                    noise_pred = noise_uncond * _alpha + cur_g * (noise_pred - noise_uncond * _alpha)
                else:
                    noise_pred = noise_uncond + cur_g * (noise_pred - noise_uncond)
                del noise_uncond
            if _cfg_zero_star and i == 0:
                noise_pred = torch.zeros_like(noise_pred)
            if os.environ.get('WAN_STEP_STATS') == '1':
                _sg = float(t) / 1000.0
                _x0 = latents.float() - _sg * noise_pred.float()
                print(f'[wan-stats] step {i + 1} t={float(t):.1f} v_std={noise_pred.float().std().item():.4f} v_mean={noise_pred.float().mean().item():.4f} x_std={latents.float().std().item():.4f} x0_std={_x0.std().item():.4f} x0_mean={_x0.mean().item():.4f}', flush=True)
            latents = _step_plan[i][0].step(noise_pred, t, latents, return_dict=False)[0]
            del noise_pred, latent_model_input
            if _step_sync:
                torch.cuda.synchronize()
            if i % 5 == 0 or i == len(timesteps) - 1:
                print(f"[wan-gen] step {i + 1}/{len(timesteps)}  {_time.perf_counter() - _t0:.1f}s ({('high' if current_model is transformer else 'low')})", flush=True)
            if progress is not None:
                try:
                    progress(i + 1, len(timesteps), latents)
                except Exception as e:
                    if should_cancel is not None and should_cancel():
                        raise
                    print(f'[wan-gen] progress hook note (ignored): {e}', flush=True)
    _bernini_restore = _release_bernini_restore(_bernini_restore)
    _close_streamer_for_headroom(streamer_high, 'HIGH before VAE decode')
    _close_streamer_for_headroom(streamer_low, 'LOW before VAE decode')
    del prompt_embeds, negative_prompt_embeds, i2v_condition, transformer, transformer_2
    del streamer_high, streamer_low
    gc.collect()
    torch.cuda.empty_cache()
    if return_latents:
        return latents
    with torch.no_grad():
        latents = latents.to(vae.dtype)
        z_dim = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = vae.decode(latents, return_dict=False)[0]
        frames = video_processor.postprocess_video(video, output_type='pil')
    del vae, latents, video
    gc.collect()
    torch.cuda.empty_cache()
    return frames[0]
