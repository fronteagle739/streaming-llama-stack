# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/anima_lllite.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import logging
import os
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)
LLLITE_ARCH_VERSION = '2'
ATOMIC_SPECIFIERS: Tuple[str, ...] = ('self_attn_q_pre', 'self_attn_kv_pre', 'cross_attn_q_pre', 'mlp_fc1_pre')
PRESETS: dict = {'self_attn_q': ('self_attn_q_pre',), 'self_attn_qkv': ('self_attn_q_pre', 'self_attn_kv_pre'), 'self_attn_qkv_cross_q': ('self_attn_q_pre', 'self_attn_kv_pre', 'cross_attn_q_pre')}
_ATOMIC_TO_CHILDREN: dict = {'self_attn_q_pre': [('self_attn_q_proj', ('attn1', 'to_q'))], 'self_attn_kv_pre': [('self_attn_k_proj', ('attn1', 'to_k')), ('self_attn_v_proj', ('attn1', 'to_v'))], 'cross_attn_q_pre': [('cross_attn_q_proj', ('attn2', 'to_q'))], 'mlp_fc1_pre': [('mlp_layer1', ('ff', 'net', 0, 'proj'))]}

def parse_target_layers(spec: str) -> Tuple[str, ...]:
    if not isinstance(spec, str):
        raise TypeError(f'target_layers must be str, got {type(spec).__name__}')
    spec = spec.strip()
    if not spec:
        raise ValueError('target_layers spec is empty')
    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(',') if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(f'unknown target_layers atomic specifier(s): {bad}. valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}')
    return tuple((a for a in ATOMIC_SPECIFIERS if a in parts))

def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)

class _ResBlock(nn.Module):

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h
ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)

class _ASPP(nn.Module):

    def __init__(self, ch: int, dilations: Tuple[int, ...]=ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, 'ASPP needs at least one dilation'
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())
        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))

class _Conditioning1(nn.Module):

    def __init__(self, cond_dim: int, cond_emb_dim: int, n_resblocks: int, use_aspp: bool=False, aspp_dilations: Tuple[int, ...]=ASPP_DEFAULT_DILATIONS, cond_in_channels: int=3):
        super().__init__()
        assert cond_dim % 2 == 0, f'cond_dim must be even, got {cond_dim}'
        assert cond_in_channels >= 1, f'cond_in_channels must be >= 1, got {cond_in_channels}'
        ch_half = cond_dim // 2
        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)
        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None
        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h

class LLLiteModuleDiT(nn.Module):

    def __init__(self, name: str, org_module: nn.Linear, cond_emb_dim: int, mlp_dim: int, dropout: Optional[float]=None, multiplier: float=1.0):
        super().__init__()
        self.lllite_name = name
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier
        in_dim = org_module.in_features
        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)
        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.cond_emb: Optional[torch.Tensor] = None
        self.org_forward = None
        self.layer_idx: int = -1
        self._depth_embeds_ref: List[nn.Parameter] = []

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)
        cx = self.cond_emb
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)
        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x)
        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)
        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx
        h = F.silu(self.down(x_proc))
        gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)
        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)
        out = self.up(m) * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return self.org_forward(x + out)

class ControlNetLLLiteDiT(nn.Module):

    def __init__(self, transformer: nn.Module, cond_emb_dim: int=32, mlp_dim: int=64, target_layers: str='self_attn_q', dropout: Optional[float]=None, multiplier: float=1.0, cond_dim: int=64, cond_resblocks: int=1, use_aspp: bool=False, aspp_dilations: Tuple[int, ...]=ASPP_DEFAULT_DILATIONS, cond_in_channels: int=3, inpaint_masked_input: bool=False):
        super().__init__()
        atomics = parse_target_layers(target_layers)
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input
        self.conditioning1 = _Conditioning1(cond_dim, cond_emb_dim, cond_resblocks, use_aspp=use_aspp, aspp_dilations=aspp_dilations, cond_in_channels=cond_in_channels)
        modules = self._create_modules(transformer, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        if not modules:
            raise RuntimeError(f'ControlNet-LLLite created 0 modules for target={target_layers!r}. The transformer has no transformer_blocks, or the atomics map missed every Linear — check the diffusers module map.')
        self.lllite_modules = nn.ModuleList(modules)
        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]
        logger.info('ControlNet-LLLite (Anima v%s): %d modules for target=%r (atomics=%s), cond_in_channels=%d, cond_dim=%d, cond_resblocks=%d, aspp=%s, cond_emb_dim=%d, mlp_dim=%d%s', LLLITE_ARCH_VERSION, n, target_layers, list(atomics), cond_in_channels, cond_dim, cond_resblocks, 'on' + str(list(self.aspp_dilations)) if use_aspp else 'off', cond_emb_dim, mlp_dim, f', inpaint=on(masked_input={inpaint_masked_input})' if cond_in_channels != 3 else '')

    @staticmethod
    def _resolve(block: nn.Module, path):
        obj = block
        for step in path:
            obj = obj[step] if isinstance(step, int) else getattr(obj, step)
        return obj

    def _create_modules(self, transformer, cond_emb_dim, mlp_dim, atomics, dropout, multiplier):
        blocks = getattr(transformer, 'transformer_blocks', None)
        if blocks is None:
            raise RuntimeError('transformer has no .transformer_blocks (not a CosmosTransformer3DModel?)')
        mods: List[LLLiteModuleDiT] = []
        for i, blk in enumerate(blocks):
            for atomic in atomics:
                for child_name, attr_path in _ATOMIC_TO_CHILDREN[atomic]:
                    lin = self._resolve(blk, attr_path)
                    if not isinstance(lin, nn.Linear):
                        raise RuntimeError(f'block {i} atomic {atomic}: resolved {attr_path} to {type(lin).__name__}, expected nn.Linear (diffusers layout changed?)')
                    name = f'lllite_dit_blocks_{i}_{child_name}'
                    mods.append(LLLiteModuleDiT(name, lin, cond_emb_dim, mlp_dim, dropout, multiplier))
        return mods

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()
_INTERNAL_MODULES_PREFIX = 'lllite_modules.'
_INTERNAL_COND_PREFIX = 'conditioning1.'
_INTERNAL_DEPTH_KEY = 'depth_embeds'
_SAVED_COND_PREFIX = 'lllite_conditioning1.'
_SAVED_DEPTH_SUFFIX = '.depth_embed'

def _from_saved_state_dict(lllite: 'ControlNetLLLiteDiT', weights_sd: dict) -> dict:
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}
    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[:-len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition('.')
        if dot and head in name_to_idx:
            out[f'{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}'] = v
            continue
        out[k] = v
    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(f'depth_embed slices missing for module idx(es) {missing}')
        out[_INTERNAL_DEPTH_KEY] = torch.stack([depth_slices[i] for i in range(n_modules)], dim=0)
    return out

def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool=False):
    if os.path.splitext(file)[1] == '.safetensors':
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location='cpu')
    if any((k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd)):
        raise RuntimeError(f"weights at {file} look like a legacy LLLite format (keys '{_INTERNAL_MODULES_PREFIX}*'). This port reads the v2 named-key format (e.g. 'lllite_dit_blocks_0_self_attn_q_proj.down.weight').")
    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info('loaded LLLite weights from %s: %s', file, info)
    return info

def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != '.safetensors':
        return {}
    from safetensors import safe_open
    with safe_open(file, framework='pt') as f:
        meta = f.metadata()
    return meta or {}

def _to_bchw(image, device, dtype) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        try:
            from PIL import Image as _PIL
        except Exception:
            _PIL = None
        if _PIL is not None and isinstance(image, _PIL.Image):
            import numpy as np
            arr = np.asarray(image.convert('RGB'), dtype='float32') / 255.0
            image = torch.from_numpy(arr)
        else:
            image = torch.as_tensor(image, dtype=torch.float32)
    img = image.to(dtype=torch.float32)
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1).unsqueeze(0)
    elif img.ndim == 4 and img.shape[-1] in (1, 3):
        img = img.permute(0, 3, 1, 2)
    elif img.ndim == 4 and img.shape[1] in (1, 3, 4):
        pass
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.unsqueeze(0)
    else:
        raise ValueError(f'Unexpected cond image shape {tuple(img.shape)} — give HWC, BHWC, CHW or BCHW.')
    img = img[:1]
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    return img.to(device=device, dtype=dtype)

def prepare_cond_image(image, height: int, width: int, device, dtype) -> torch.Tensor:
    img = _to_bchw(image, device, torch.float32)
    if img.shape[-2] != height or img.shape[-1] != width:
        img = F.interpolate(img.float(), size=(height, width), mode='bicubic', align_corners=False).clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)

def _prepare_mask(mask, height: int, width: int, device, dtype) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        try:
            from PIL import Image as _PIL
        except Exception:
            _PIL = None
        if _PIL is not None and isinstance(mask, _PIL.Image):
            import numpy as np
            mask = torch.from_numpy(np.asarray(mask.convert('L'), dtype='float32') / 255.0)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32)
    m = mask.to(dtype=torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(1)
    elif m.ndim == 4 and m.shape[1] == 1:
        pass
    else:
        raise ValueError(f'Unexpected mask shape {tuple(m.shape)} — give HxW, BxHxW or Bx1xHxW.')
    m = m[:1]
    if m.shape[-2] != height or m.shape[-1] != width:
        m = F.interpolate(m, size=(height, width), mode='nearest')
    m = (m >= 0.5).to(dtype=dtype)
    return m.to(device=device)

def prepare_inpaint_cond_image(image, mask, height: int, width: int, device, dtype, masked_input: bool=False) -> torch.Tensor:
    rgb = prepare_cond_image(image, height, width, device, dtype)
    mk = _prepare_mask(mask, height, width, device, dtype)
    if masked_input:
        rgb = rgb * (mk < 0.5).to(rgb.dtype)
    mask_pm1 = mk.to(rgb.dtype) * 2.0 - 1.0
    return torch.cat([rgb, mask_pm1], dim=1)

def attach_lllite(transformer, weights_path: str, strength: float=1.0, device='cuda', dtype=None) -> ControlNetLLLiteDiT:
    if dtype is None:
        dtype = torch.bfloat16
    meta = read_lllite_metadata(weights_path)
    cond_in_channels = int(meta.get('lllite.cond_in_channels', 3))
    use_aspp = str(meta.get('lllite.use_aspp', 'false')).lower() == 'true'
    aspp_dilations_meta = meta.get('lllite.aspp_dilations')
    if use_aspp and aspp_dilations_meta:
        aspp_dilations = tuple((int(d) for d in aspp_dilations_meta.split(',') if d.strip()))
    else:
        aspp_dilations = ASPP_DEFAULT_DILATIONS
    lllite = ControlNetLLLiteDiT(transformer, cond_emb_dim=int(meta.get('lllite.cond_emb_dim', 32)), mlp_dim=int(meta.get('lllite.mlp_dim', 64)), target_layers=meta.get('lllite.target_atomics', meta.get('lllite.target_layers', 'self_attn_q')), multiplier=strength, cond_dim=int(meta.get('lllite.cond_dim', 64)), cond_resblocks=int(meta.get('lllite.cond_resblocks', 1)), use_aspp=use_aspp, aspp_dilations=aspp_dilations, cond_in_channels=cond_in_channels, inpaint_masked_input=str(meta.get('lllite.inpaint_masked_input', 'false')).lower() == 'true')
    load_lllite_weights(lllite, weights_path, strict=False)
    lllite.eval().requires_grad_(False).to(device=device, dtype=dtype)
    lllite.apply_to()
    return lllite
