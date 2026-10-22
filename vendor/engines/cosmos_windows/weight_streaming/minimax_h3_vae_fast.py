# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_vae_fast.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
from pathlib import Path
import types
import torch
import torch.nn.functional as F
DECODER_MODES = ('released', 'int8_convrot', 'auto')
ENCODER_MODES = ('released', 'fused', 'fused_acc16', 'auto')
INT8_CONVROT_FILENAME = 'minimax_h3_video_vae_int8_convrot.safetensors'
_HEAVY = ('attn.to_qkv', 'attn.to_out', 'ff.w1', 'ff.w2')
_DECODER_KERNELS = ('int8_linear', 'int8_attention', 'rms_rope_split_half_')
_ENCODER_KERNELS = ('group_norm_silu_pad3d', 'fp16_conv3d')

def kernels_available(names) -> bool:
    try:
        import comfy_kitchen as ck
    except Exception:
        return False
    if not all((hasattr(ck, n) for n in names)):
        return False
    if 'int8_linear' in names:
        import inspect
        try:
            params = inspect.signature(ck.int8_linear).parameters
        except (TypeError, ValueError):
            return False
        if not {'residual', 'residual_scale', 'input_act_weight'} <= set(params):
            return False
    return True

def _env_mode(name: str, allowed) -> str:
    value = os.environ.get(name, 'released').strip().lower() or 'released'
    if value not in allowed:
        raise ValueError(f'{name} must be one of {allowed}, got {value!r}')
    return value

def decoder_mode(component_dir=None) -> str:
    value = _env_mode('MINIMAX_H3_VAE_DECODER', DECODER_MODES)
    if value != 'auto':
        return value
    if not kernels_available(_DECODER_KERNELS):
        return 'released'
    if component_dir is not None and (not default_int8_convrot_file(component_dir).is_file()):
        return 'released'
    return 'int8_convrot'

def encoder_mode() -> str:
    value = _env_mode('MINIMAX_H3_VAE_ENCODER', ENCODER_MODES)
    if value != 'auto':
        return value
    return 'fused_acc16' if kernels_available(_ENCODER_KERNELS) else 'released'

def describe_modes(component_dir=None) -> dict:
    ck_version = None
    try:
        import importlib.metadata
        ck_version = importlib.metadata.version('comfy-kitchen')
    except Exception:
        pass
    file = default_int8_convrot_file(component_dir) if component_dir is not None else None
    return dict(decoder=decoder_mode(component_dir), encoder=encoder_mode(), decoder_requested=os.environ.get('MINIMAX_H3_VAE_DECODER', 'released'), encoder_requested=os.environ.get('MINIMAX_H3_VAE_ENCODER', 'released'), comfy_kitchen=ck_version, int8_file=str(file) if file else None, int8_file_present=bool(file and file.is_file()), decoder_kernels=kernels_available(_DECODER_KERNELS), encoder_kernels=kernels_available(_ENCODER_KERNELS))

def int8_compute_dtype():
    value = os.environ.get('MINIMAX_H3_VAE_INT8_DTYPE', 'float16').strip().lower()
    table = {'float16': torch.float16, 'fp16': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
    if value not in table:
        raise ValueError('MINIMAX_H3_VAE_INT8_DTYPE must be float16 or bfloat16')
    return table[value]

def _kitchen():
    try:
        import comfy_kitchen as ck
    except Exception as exc:
        raise RuntimeError(f'the fast H3 VAE kernels need comfy_kitchen: {exc}') from exc
    return ck

def require_kernels(*names: str):
    ck = _kitchen()
    missing = [n for n in names if not hasattr(ck, n)]
    if missing:
        version = getattr(ck, '__version__', '?')
        raise RuntimeError(f'comfy_kitchen {version} lacks {missing}; the fast H3 VAE needs comfy-kitchen>=0.2.35')
    return ck

def default_int8_convrot_file(component_dir) -> Path:
    override = os.environ.get('MINIMAX_H3_VAE_INT8_CONVROT_FILE', '').strip()
    if override:
        return Path(override)
    resolved = Path(component_dir).resolve()
    roots = [resolved, (Path(component_dir).parent / 'video_vae').resolve()]
    for root in roots:
        for parent in root.parents:
            candidate = parent / 'comfy_vae' / INT8_CONVROT_FILENAME
            if candidate.is_file():
                return candidate
    return resolved.parent.parent / 'comfy_vae' / INT8_CONVROT_FILENAME

def _conv_parts(conv):
    pad_t, pad_h, pad_w = (int(p) for p in conv.padding)
    if pad_h or pad_w:
        if conv.pad_mode != 'reflect':
            raise ValueError(f'fused H3 encoder expects reflect spatial padding, got {conv.pad_mode}')
    if pad_t and (not conv.causal or conv.pad_mode_t != 'constant'):
        raise ValueError('fused H3 encoder expects causal zero temporal padding')
    return (pad_t, (pad_w, pad_w, pad_h, pad_h))

class _FastEncoder:

    def __init__(self, model, *, accumulate_fp16: bool):
        self.ck = require_kernels(*_ENCODER_KERNELS)
        self.accumulate_fp16 = accumulate_fp16
        self.encoder = model.encoder
        self.quant_conv = model.quant_conv
        self.weights = {}
        for module in list(self.encoder.modules()) + [self.quant_conv]:
            if isinstance(module, torch.nn.Conv3d):
                w = module.weight.detach().to(torch.float16).contiguous(memory_format=torch.channels_last_3d)
                b = None if module.bias is None else module.bias.detach().to(torch.float16)
                self.weights[id(module)] = (w, b)
            elif isinstance(module, torch.nn.GroupNorm):
                self.weights[id(module)] = (module.weight.detach().to(torch.float16), module.bias.detach().to(torch.float16))
        self.resident_bytes = sum((t.numel() * t.element_size() for pair in self.weights.values() for t in pair if t is not None))

    def _conv(self, conv, x, *, norm=None, spatial_pad=None, residual=None):
        pad_t, default_pad = _conv_parts(conv)
        spatial_pad = default_pad if spatial_pad is None else spatial_pad
        front = 0 if x.shape[2] == 1 else pad_t * 2
        weight, bias = self.weights[id(conv)]
        if x.shape[2] == 1 and pad_t:
            if norm is not None:
                nw, nb = self.weights[id(norm)]
                x = self.ck.group_norm_silu_pad3d(x, nw, nb, norm.num_groups, norm.eps, (*spatial_pad, 0), silu=True)
            elif any(spatial_pad):
                x = self.ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (*spatial_pad, 0), silu=False)
            out = F.conv3d(x, weight[:, :, -1:].contiguous(memory_format=torch.channels_last_3d), bias, conv.stride)
            return out if residual is None else out.add_(residual)
        if norm is not None:
            nw, nb = self.weights[id(norm)]
            x = self.ck.group_norm_silu_pad3d(x, nw, nb, norm.num_groups, norm.eps, (*spatial_pad, front), silu=True)
        elif front or any(spatial_pad):
            x = self.ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (*spatial_pad, front), silu=False)
        elif not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        if self.accumulate_fp16:
            return self.ck.fp16_conv3d(x, weight, bias, residual, conv.stride)
        out = F.conv3d(x, weight, bias, conv.stride)
        return out if residual is None else out.add_(residual)

    def _resblock(self, block, x):
        h = self._conv(block.conv1, x, norm=block.norm1)
        if block.in_channels != block.out_channels:
            x = self._conv(block.nin_shortcut, x)
        return self._conv(block.conv2, h, norm=block.norm2, residual=x)

    def forward(self, x, zq=None):
        if zq is not None:
            raise NotImplementedError('fused H3 encoder does not support spatial-norm conditioning')
        enc = self.encoder
        h = self._conv(enc.conv_in, x.to(torch.float16).contiguous(memory_format=torch.channels_last_3d))
        for level in enc.down:
            for block in level.block:
                h = self._resblock(block, h)
            down = getattr(level, 'downsample', None)
            if down is not None:
                pad = (0, 1, 0, 1) if down.space_stride == 2 else None
                h = self._conv(down.conv, h, spatial_pad=pad)
        return self._conv(enc.conv_out, h, norm=enc.norm_out)

    def encode(self, x):
        h = self.forward(x)
        moments = self._conv(self.quant_conv, h)
        return moments.float().contiguous()

def install_fast_encoder(model, mode: str | None=None):
    mode = encoder_mode() if mode is None else mode
    if mode == 'released':
        return None
    if getattr(model, 'encoder_parallel', False):
        raise ValueError('fused H3 encoder does not support encoder_parallel')
    fast = _FastEncoder(model, accumulate_fp16=mode == 'fused_acc16')
    model._h3_fast_encoder = fast
    model._h3_fast_encoder_mode = mode
    model.encode = types.MethodType(lambda self, x: self._h3_fast_encoder.encode(x), model)
    return fast

class H3Int8ConvRotLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, *, bias=True, dtype=torch.float16, groupsize=256):
        super().__init__()
        self.in_features, self.out_features, self.groupsize = (in_features, out_features, groupsize)
        self.register_buffer('weight', torch.empty(out_features, in_features, dtype=torch.int8, device='meta'))
        self.register_buffer('weight_scale', torch.empty(out_features, 1, dtype=torch.float32, device='meta'))
        self.register_buffer('bias', torch.empty(out_features, dtype=dtype, device='meta') if bias else None)

    def forward(self, x, **kw):
        return int8_call(self, x, x.dtype, **kw)

def int8_call(lin, x, dtype, **kw):
    ck = _kitchen()
    bias = lin.bias if lin.bias is None or lin.bias.dtype == dtype else lin.bias.to(dtype)
    return ck.int8_linear(x, lin.weight, lin.weight_scale, bias, dtype, convrot=True, convrot_groupsize=lin.groupsize, **kw)

def _parse_quant(blob) -> dict:
    import json
    return json.loads(bytes(blob.tolist()).decode('utf-8'))

def replace_heavy_linears(model, *, dtype=None) -> tuple:
    dtype = int8_compute_dtype() if dtype is None else dtype
    names = []
    for block_id, block in enumerate(model.decoder.transformer_blocks):
        for name in _HEAVY:
            parent_name, _, leaf = name.rpartition('.')
            parent = block.get_submodule(parent_name)
            old = getattr(parent, leaf)
            if not isinstance(old, H3Int8ConvRotLinear):
                setattr(parent, leaf, H3Int8ConvRotLinear(old.in_features, old.out_features, bias=old.bias is not None, dtype=dtype))
            names.append(f'decoder.transformer_blocks.{block_id}.{name}')
    model._h3_int8_convrot_names = frozenset(names)
    return tuple(names)

def bind_int8_convrot_blocks(model, path, *, device, dtype=None):
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    dtype = int8_compute_dtype() if dtype is None else dtype
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'INT8 ConvRot H3 VAE not found: {path}')
    names = getattr(model, '_h3_int8_convrot_names', None) or replace_heavy_linears(model, dtype=dtype)
    state = model.state_dict()
    bound, total = ([], 0)
    with safe_open(str(path), framework='pt', device=str(device)) as handle:
        keys = set(handle.keys())
        for name in names:
            conf = _parse_quant(handle.get_tensor(name + '.comfy_quant'))
            if conf.get('format') != 'int8_tensorwise' or not conf.get('convrot'):
                raise ValueError(f'{name}: expected int8_tensorwise+convrot, got {conf}')
            model.get_submodule(name).groupsize = int(conf.get('convrot_groupsize', 256))
        for key in sorted(keys):
            if not key.startswith('decoder.transformer_blocks.') or key.endswith('.comfy_quant'):
                continue
            if key not in state:
                raise ValueError(f'{path.name}: {key} has no place in the decoder')
            value = handle.get_tensor(key)
            target = state[key]
            if value.dtype != torch.int8 and target.dtype != value.dtype:
                value = value.to(target.dtype)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f'{key}: file {tuple(value.shape)} vs model {tuple(target.shape)}')
            set_module_tensor_to_device(model, key, device, value=value, dtype=value.dtype)
            bound.append(key)
            total += value.numel() * value.element_size()
    expected = {k for k in state if k.startswith('decoder.transformer_blocks.')}
    missing = sorted(expected - set(bound))
    if missing:
        raise ValueError(f'{path.name} does not cover decoder blocks: {missing[:8]}')
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return (bound, total)

def _rope_table(rotary_pos_emb, dtype, cache):
    cos, sin = rotary_pos_emb
    key = (cos.data_ptr(), tuple(cos.shape), dtype)
    if cache.get('key') == key:
        return cache['table']
    half = cos.shape[-1] // 2
    c, s = (cos[..., :half].float(), sin[..., :half].float())
    table = torch.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2).to(dtype).contiguous()
    cache.update(key=key, table=table, ref=cos)
    return table

def _int8_block_forward(block, hidden_states, rotary_pos_emb=None, pack_info=None, *, ck, dtype, rope_cache, last):
    if pack_info:
        raise NotImplementedError('int8 H3 decoder blocks do not support masked/packed attention')
    x = hidden_states.to(dtype)
    attn, ff = (block.attn, block.ff)
    batch, seq, _ = x.shape
    cast = lambda t: t if t.dtype == dtype else t.to(dtype)
    qkv = int8_call(attn.to_qkv, x, dtype, input_act='rms_norm', input_act_weight=cast(block.norm1.weight), input_act_eps=block.norm1.eps)
    qkv = qkv.view(batch, seq, -1, 3 * attn.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    ones_key = ('ones', attn.dim_head, x.device, dtype)
    ones = rope_cache.get(ones_key)
    if ones is None:
        ones = rope_cache[ones_key] = torch.ones(attn.dim_head, device=x.device, dtype=dtype)
    if rotary_pos_emb is not None:
        table = _rope_table(rotary_pos_emb, dtype, rope_cache)
        query, key = ck.rms_rope_split_half_(query, key, table, ones, epsilon=attn.norm_q.eps, rot_dim=table.shape[-3] * 2)
    else:
        query = F.rms_norm(query, (attn.dim_head,), eps=attn.norm_q.eps)
        key = F.rms_norm(key, (attn.dim_head,), eps=attn.norm_k.eps)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    out = ck.int8_attention(query, key, value)
    out = torch.nan_to_num(out.transpose(1, 2).reshape(batch, seq, -1))
    x = int8_call(attn.to_out, out, dtype, residual=x, residual_scale=cast(block.scale1))
    h = int8_call(ff.w1, x, dtype, input_act='rms_norm', input_act_weight=cast(block.norm2.weight), input_act_eps=block.norm2.eps)
    x = int8_call(ff.w2, h, dtype, input_act='swiglu', residual=x, residual_scale=cast(block.scale2))
    return x.float() if last else x

def install_int8_decoder_forward(model, *, dtype=None):
    ck = require_kernels(*_DECODER_KERNELS)
    dtype = int8_compute_dtype() if dtype is None else dtype
    rope_cache: dict = {}
    transformer_blocks = model.decoder.transformer_blocks
    for block_id, block in enumerate(transformer_blocks):
        if block.attn.norm_q is None or getattr(block.attn.norm_q, 'weight', None) is not None:
            raise ValueError('int8 H3 decoder expects non-affine RMS qk norms')
        if not isinstance(block.norm1, torch.nn.RMSNorm) or not block.use_scale:
            raise ValueError('int8 H3 decoder expects RMSNorm pre-norms with residual scales')
        last = block_id == len(transformer_blocks) - 1
        block.forward = types.MethodType(lambda self, hs, rope=None, pack_info=None, _last=last: _int8_block_forward(self, hs, rope, pack_info, ck=ck, dtype=dtype, rope_cache=rope_cache, last=_last), block)
    return len(transformer_blocks)
__all__ = ['DECODER_MODES', 'ENCODER_MODES', 'decoder_mode', 'encoder_mode', 'default_int8_convrot_file', 'describe_modes', 'kernels_available', 'install_fast_encoder', 'replace_heavy_linears', 'bind_int8_convrot_blocks', 'install_int8_decoder_forward', 'H3Int8ConvRotLinear', 'int8_call', 'require_kernels']
