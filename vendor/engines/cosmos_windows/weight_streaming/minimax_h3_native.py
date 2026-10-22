# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_native.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
LEAN_BLOCK_DEFAULT = '1'

def lean_block_configured() -> bool:
    return os.environ.get('STUDIO_H3_LEAN_BLOCK', LEAN_BLOCK_DEFAULT) != '0'

def lean_block_enabled() -> bool:
    return lean_block_configured() and (not torch.is_grad_enabled())
_PROLOGUE_CALLS = {'fused': 0, 'chunked': 0}

def prologue_stats() -> dict:
    return dict(_PROLOGUE_CALLS)

def reset_prologue_stats() -> None:
    _PROLOGUE_CALLS.update(fused=0, chunked=0)

def fused_rope_enabled() -> bool:
    return os.environ.get('STUDIO_H3_FUSED_ROPE', '0') not in ('', '0', 'false', 'off')
_PROLOGUE_ROWS = 4096

def attention_query_chunk_rows() -> int:
    return int(os.environ.get('STUDIO_H3_ATTN_QUERY_CHUNK', '32768'))

def mlp_chunk_rows() -> int:
    return int(os.environ.get('STUDIO_H3_MLP_CHUNK', '0'))
_CK = None

def comfy_kitchen():
    global _CK
    if _CK is None:
        try:
            import comfy_kitchen as ck
            _CK = ck
        except Exception:
            _CK = False
    return _CK or None

def rope_rotation_table(angles: torch.Tensor, dtype) -> torch.Tensor:
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = (torch.cos(ang), torch.sin(ang))
    return torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2).to(dtype)

def modulation_segments(indices: torch.Tensor) -> list[tuple[int, int, int]]:
    if indices.numel() == 0:
        return []
    values, counts = torch.unique_consecutive(indices, return_counts=True)
    values, counts = (values.tolist(), counts.tolist())
    out, start = ([], 0)
    for value, count in zip(values, counts):
        out.append((start, start + count, int(value)))
        start += count
    return out
_SEGMENT_ROWS = 8192

def modulate_segments_(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, segments: list[tuple[int, int, int]]) -> torch.Tensor:
    for a, b, row in segments:
        h[a:b].mul_(1 + scale[row]).add_(shift[row])
    return h

def rms_modulate_segments(x, norm, shift, scale, segments):
    ck = comfy_kitchen() if os.environ.get('STUDIO_H3_FUSED_ADALN', '0') == '1' else None
    if ck is None or not x.is_cuda or torch.is_grad_enabled():
        return modulate_segments_(norm(x), shift, scale, segments)
    return _rms_adaln_segments(ck, x, norm, shift, scale, segments)

def _rms_adaln_segments(ck, x, norm, shift, scale, segments):
    effective_scale = scale.float()
    if norm.weight is not None:
        effective_scale = (1 + effective_scale) * norm.weight.float() - 1
    out = torch.empty_like(x)
    for a, b, row in segments:
        for start in range(a, b, _SEGMENT_ROWS):
            stop = min(start + _SEGMENT_ROWS, b)
            out[start:stop] = ck.rms_adaln(x[start:stop].float(), effective_scale[row], shift[row].float(), norm.eps)
    return out

def gate_segments_(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, segments: list[tuple[int, int, int]]) -> torch.Tensor:
    for a, b, row in segments:
        g = gate[row]
        for start in range(a, b, _SEGMENT_ROWS):
            stop = min(start + _SEGMENT_ROWS, b)
            x[start:stop].add_(other[start:stop] * g)
    return x

@dataclass(frozen=True)
class H3NativeConfig:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-05
    qk_norm_eps: float = 1e-05
    final_norm_eps: float = 1e-05
    fused_adaln: bool = False
    fused_adaln_rank: int = 8

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def adaln_input_dim(self) -> int:
        return self.fused_adaln_rank if self.fused_adaln else self.time_embed_dim

def config_from_checkpoint(path: str | Path) -> H3NativeConfig:
    path = Path(path)
    if path.is_dir():
        path = path / 'config.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    return H3NativeConfig(hidden_size=int(raw['hidden_size']), num_layers=int(raw['num_layers']), token_refiner_num_layers=int(raw['token_refiner_num_layers']), num_attention_heads=int(raw['num_attention_heads']), attention_head_dim=int(raw['attention_head_dim']), ffn_hidden_size=int(raw['ffn_hidden_size']), latents_dim=int(raw['latents_dim']), audio_latents_dim=int(raw['audio_latents_dim']), patch_size=tuple((int(x) for x in raw['patch_size'])), text_dim=int(raw['text_dim']), timestep_input_dim=int(raw['timestep_input_dim']), time_embed_hidden_size=int(raw['time_embed_hidden_size']), time_embed_dim=int(raw['time_embed_dim']), rope_inv_freq_len=int(raw['rope_inv_freq_len']), norm_eps=float(raw['norm_eps']), qk_norm_eps=float(raw['qk_norm_eps']), final_norm_eps=float(raw['final_norm_eps']))

def _linear(in_features: int, out_features: int, *, bias: bool, dtype, device):
    return nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)

def _rms(size: int, eps: float, *, device):
    return nn.RMSNorm(size, eps=eps, dtype=torch.bfloat16, device=device)

class MiniMaxH3NativeAttention(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta', attention_backend: str='sdpa', sageattn=None) -> None:
        super().__init__()
        self.config = config
        from .minimax_h3_attention import resolve_h3_attention_backend
        attention_backend = resolve_h3_attention_backend(attention_backend)
        self.attention_backend = attention_backend
        if attention_backend == 'sageattention' and sageattn is None:
            from .minimax_h3_attention import load_sageattn
            sageattn = load_sageattn()
        self._flash_varlen = None
        if attention_backend == 'flash_attn2':
            from .minimax_h3_attention import load_flash_attn_varlen
            self._flash_varlen = load_flash_attn_varlen()
        elif attention_backend == 'flash_attn2_int8':
            from .minimax_h3_attention import load_int8_attn_varlen
            self._flash_varlen = load_int8_attn_varlen()
        if attention_backend not in ('sdpa', 'sageattention', 'flash_attn2', 'flash_attn2_int8'):
            raise ValueError(f'unsupported H3 attention backend: {attention_backend}')
        self._sageattn = sageattn
        self.qkv_proj = _linear(config.hidden_size, 3 * config.inner_dim, bias=False, dtype=torch.bfloat16, device=device)
        self.q_norm = _rms(config.attention_head_dim, config.qk_norm_eps, device=device)
        self.k_norm = _rms(config.attention_head_dim, config.qk_norm_eps, device=device)
        self.out_proj = _linear(config.inner_dim, config.hidden_size, bias=False, dtype=torch.bfloat16, device=device)

    @staticmethod
    def _apply_rope(x, cos, sin):
        rotary_dim = cos.shape[-1]
        rotary, passthrough = (x[..., :rotary_dim], x[..., rotary_dim:])
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)

    @staticmethod
    def _segment_mask(cu_seqlens, tokens: int, device):
        if cu_seqlens is None:
            return None
        bounds = [int(x) for x in cu_seqlens]
        if not bounds or bounds[0] != 0 or bounds[-1] != tokens:
            raise ValueError('cu_seqlens must start at zero and end at token count')
        if len(bounds) == 2:
            return None
        segment = torch.empty(tokens, dtype=torch.int32, device=device)
        for index, (start, stop) in enumerate(zip(bounds, bounds[1:])):
            segment[start:stop] = index
        return segment[:, None] == segment[None, :]

    def _lean_prologue_ready(self, x, rope_cache) -> bool:
        return lean_block_enabled() and self.attention_backend in ('flash_attn2', 'flash_attn2_int8') and x.is_cuda and (rope_cache is not None) and (self._flash_varlen is not None)

    def _prologue_in_place_(self, q, k, cos, sin, table) -> None:
        tokens = q.shape[0]
        rot = int(cos.shape[-1])
        if fused_rope_enabled() and table is not None and (comfy_kitchen() is not None):
            _PROLOGUE_CALLS['fused'] += 1
            comfy_kitchen().rms_rope_split_half_(q.unsqueeze(0), k.unsqueeze(0), table.detach(), self.q_norm.weight.detach(), self.k_norm.weight.detach(), epsilon=float(self.q_norm.eps), rot_dim=rot)
            return
        _PROLOGUE_CALLS['chunked'] += 1
        cos = cos.to(q.dtype).view(tokens, 1, rot)
        sin = sin.to(q.dtype).view(tokens, 1, rot)
        for start in range(0, tokens, _PROLOGUE_ROWS):
            stop = min(start + _PROLOGUE_ROWS, tokens)
            for view, norm in ((q, self.q_norm), (k, self.k_norm)):
                piece = view[start:stop]
                piece.copy_(self._apply_rope(norm(piece), cos[start:stop], sin[start:stop]))

    def _sparse_forward(self, x, *, table, plan):
        ck = comfy_kitchen()
        tokens = x.shape[0]
        heads, head_dim = (self.config.num_attention_heads, self.config.attention_head_dim)
        step = int(plan.producer_chunk)

        def chunks():
            for start in range(0, tokens, step):
                rows = self.qkv_proj(x[start:start + step])
                yield rows.view(-1, heads, 3, head_dim).permute(0, 2, 1, 3).reshape(-1, 3 * heads * head_dim)
        stats = plan.stats_for(self)
        out, kmean, vscale = ck.sol_attn_chunked(chunks, tokens, heads, table.detach(), (self.q_norm.weight.detach(), self.k_norm.weight.detach()), kmean=None if stats is None else stats[0], vscale=None if stats is None else stats[1], tau=float(plan.tau), topk_ratio=float(plan.topk_ratio), sink_blocks=list(plan.sink_blocks), sink_q=list(plan.sink_q), rope_eps=float(self.q_norm.eps), token_aug=int(plan.extra_tokens))
        plan.remember(self, kmean, vscale)
        return self.out_proj(out.reshape(tokens, self.config.inner_dim))

    def forward(self, x, *, rope_cache=None, cu_seqlens=None, sparse=None):
        tokens = x.shape[0]
        if sparse is not None and getattr(self, 'action_groups', None) is None:
            if rope_cache is None or len(rope_cache) < 3 or rope_cache[2] is None:
                raise ValueError('sparse attention needs the rope rotation table in rope_cache')
            return self._sparse_forward(x, table=rope_cache[2], plan=sparse)
        if self._lean_prologue_ready(x, rope_cache):
            cos, sin = (rope_cache[0], rope_cache[1])
            table = rope_cache[2] if len(rope_cache) >= 3 else None
            qkv = self.qkv_proj(x).view(tokens, self.config.num_attention_heads, 3, self.config.attention_head_dim)
            q, k, v = (qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :])
            self._prologue_in_place_(q, k, cos, sin, table)
            if getattr(self, 'action_groups', None) is not None:
                from .minimax_h3_world import directed_attention
                attended = directed_attention(q, k, v, self.action_groups, self._flash_varlen)
                return self.out_proj(attended.reshape(tokens, self.config.inner_dim))
            bounds = torch.tensor([0, tokens], dtype=torch.int32, device=q.device) if cu_seqlens is None else cu_seqlens.to(device=q.device, dtype=torch.int32)
            seg = bounds[1:] - bounds[:-1]
            max_seqlen = int(seg.max().item()) if seg.numel() else int(tokens)
            chunk = attention_query_chunk_rows()
            if chunk > 0 and tokens > chunk and (int(bounds.numel()) == 2):
                attended = torch.empty_like(q)
                for start in range(0, tokens, chunk):
                    stop = min(start + chunk, tokens)
                    q_bounds = torch.tensor([0, stop - start], dtype=torch.int32, device=q.device)
                    piece = self._flash_varlen(q[start:stop], k, v, q_bounds, bounds, stop - start, tokens, dropout_p=0.0, causal=False)
                    attended[start:stop] = piece[0] if isinstance(piece, tuple) else piece
                    del piece
            else:
                attended = self._flash_varlen(q, k, v, bounds, bounds, max_seqlen, max_seqlen, dropout_p=0.0, causal=False)
                if isinstance(attended, tuple):
                    attended = attended[0]
            return self.out_proj(attended.reshape(tokens, self.config.inner_dim))
        qkv = self.qkv_proj(x).view(tokens, self.config.num_attention_heads, 3, self.config.attention_head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = (self.q_norm(q), self.k_norm(k))
        if rope_cache is not None:
            cos, sin = (rope_cache[0], rope_cache[1])
            cos = cos.to(q.dtype).view(tokens, 1, -1)
            sin = sin.to(q.dtype).view(tokens, 1, -1)
            q, k = (self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin))
        if getattr(self, 'action_groups', None) is not None:
            from .minimax_h3_world import directed_attention
            attended = directed_attention(q, k, v, self.action_groups, getattr(self, '_flash_varlen', None))
            return self.out_proj(attended.reshape(tokens, self.config.inner_dim))
        if self.attention_backend in ('flash_attn2', 'flash_attn2_int8'):
            bounds = torch.tensor([0, tokens], dtype=torch.int32, device=q.device) if cu_seqlens is None else cu_seqlens.to(device=q.device, dtype=torch.int32)
            seg = bounds[1:] - bounds[:-1]
            max_seqlen = int(seg.max().item()) if seg.numel() else int(tokens)
            attended = self._flash_varlen(q, k, v, bounds, bounds, max_seqlen, max_seqlen, dropout_p=0.0, causal=False)
            if isinstance(attended, tuple):
                attended = attended[0]
            attended = attended.reshape(tokens, self.config.inner_dim)
        elif self.attention_backend == 'sageattention':
            bounds = [0, tokens] if cu_seqlens is None else [int(x) for x in cu_seqlens]
            if not bounds or bounds[0] != 0 or bounds[-1] != tokens:
                raise ValueError('cu_seqlens must start at zero and end at token count')
            chunks = [self._sageattn(q[start:stop].unsqueeze(0), k[start:stop].unsqueeze(0), v[start:stop].unsqueeze(0), tensor_layout='NHD', is_causal=False) for start, stop in zip(bounds, bounds[1:]) if stop > start]
            attended = torch.cat(chunks, dim=1).squeeze(0).reshape(tokens, self.config.inner_dim)
        else:
            mask = self._segment_mask(cu_seqlens, tokens, x.device)
            q, k, v = (item.transpose(0, 1).unsqueeze(0) for item in (q, k, v))
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[None, None], dropout_p=0.0, is_causal=False)
            attended = attended.squeeze(0).transpose(0, 1).reshape(tokens, self.config.inner_dim)
        return self.out_proj(attended)

class MiniMaxH3NativeMLP(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta') -> None:
        super().__init__()
        self.fc1 = _linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False, dtype=torch.bfloat16, device=device)
        self.fc2 = _linear(config.ffn_hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16, device=device)

    def forward(self, x):
        rows = int(x.shape[0])
        chunk = mlp_chunk_rows()
        if not lean_block_enabled():
            gate, up = self.fc1(x).chunk(2, dim=-1)
            return self.fc2(F.silu(gate) * up)
        if chunk <= 0 or rows <= chunk:
            return self.fc2(self._swiglu_(self.fc1(x)))
        out = None
        for start in range(0, rows, chunk):
            piece = self.fc2(self._swiglu_(self.fc1(x[start:start + chunk])))
            if out is None:
                out = torch.empty(rows, piece.shape[-1], dtype=piece.dtype, device=piece.device)
            out[start:start + piece.shape[0]] = piece
            del piece
        return out

    @staticmethod
    def _swiglu_(projected: torch.Tensor) -> torch.Tensor:
        gate, up = projected.chunk(2, dim=-1)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        return gate

class MiniMaxH3NativeAdaLN(nn.Module):

    def __init__(self, config: H3NativeConfig, expand_ratio: int, modality_num: int, *, device='meta') -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.linear = _linear(config.adaln_input_dim, expand_ratio * modality_num * config.hidden_size, bias=True, dtype=torch.bfloat16, device=device)

    def forward(self, conditioning):
        projected = self.linear(conditioning.to(torch.bfloat16))
        projected = projected.view(conditioning.shape[0] * self.modality_num, self.expand_ratio * self.hidden_size)
        return projected.chunk(self.expand_ratio, dim=-1)

class MiniMaxH3NativeBlock(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta', attention_backend: str='sdpa', sageattn=None) -> None:
        super().__init__()
        self.norm1 = _rms(config.hidden_size, config.norm_eps, device=device)
        self.norm2 = _rms(config.hidden_size, config.norm_eps, device=device)
        self.attn = MiniMaxH3NativeAttention(config, device=device, attention_backend=attention_backend, sageattn=sageattn)
        self.mlp = MiniMaxH3NativeMLP(config, device=device)
        self.adaln_proj = MiniMaxH3NativeAdaLN(config, 6, 3, device=device)

    @staticmethod
    def _modulate(x, shift, scale, indices):
        return x * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    def forward(self, x, *, adaln_input, combined_indices, rope_cache=None, cu_seqlens=None, mod_segments=None, sparse=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_proj(adaln_input)
        if mod_segments is not None and lean_block_enabled():
            hidden = rms_modulate_segments(x, self.norm1, shift_a, scale_a, mod_segments)
            hidden = self.attn(hidden, rope_cache=rope_cache, cu_seqlens=cu_seqlens, sparse=sparse)
            gate_segments_(x, gate_a, hidden, mod_segments)
            del hidden
            hidden = rms_modulate_segments(x, self.norm2, shift_m, scale_m, mod_segments)
            hidden = self.mlp(hidden)
            gate_segments_(x, gate_m, hidden, mod_segments)
            return x
        if sparse is not None:
            raise ValueError('sparse attention requires the lean block path (mod_segments)')
        residual = x
        hidden = self._modulate(self.norm1(x), shift_a, scale_a, combined_indices)
        hidden = self.attn(hidden, rope_cache=rope_cache, cu_seqlens=cu_seqlens)
        x = residual + gate_a.index_select(0, combined_indices) * hidden
        residual = x
        hidden = self._modulate(self.norm2(x), shift_m, scale_m, combined_indices)
        hidden = self.mlp(hidden)
        return residual + gate_m.index_select(0, combined_indices) * hidden

class MiniMaxH3NativeTokenRefinerBlock(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta', attention_backend: str='sdpa', sageattn=None) -> None:
        super().__init__()
        self.norm1 = _rms(config.hidden_size, config.norm_eps, device=device)
        self.norm2 = _rms(config.hidden_size, config.norm_eps, device=device)
        self.attn = MiniMaxH3NativeAttention(config, device=device, attention_backend=attention_backend, sageattn=sageattn)
        self.mlp = MiniMaxH3NativeMLP(config, device=device)

    def forward(self, x, *, cu_seqlens=None):
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens)
        return x + self.mlp(self.norm2(x))

class MiniMaxH3NativeTimeEmbedder(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta') -> None:
        super().__init__()
        self.proj_in = _linear(config.timestep_input_dim, config.time_embed_hidden_size, bias=True, dtype=torch.float32, device=device)
        self.proj_out = _linear(config.time_embed_hidden_size, config.time_embed_dim, bias=True, dtype=torch.float32, device=device)
        self.frequency_embedding_size = config.timestep_input_dim
        self._frequency_cache = None

    def forward(self, timestep):
        half = self.frequency_embedding_size // 2
        frequencies = self._frequency_cache
        if frequencies is None or frequencies.device != timestep.device:
            frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
            self._frequency_cache = frequencies
        args = timestep.float()[:, None] * frequencies[None]
        embedded = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedded)))

class MiniMaxH3NativeTokenRefiner(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta', attention_backend: str='sdpa', sageattn=None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3NativeTokenRefinerBlock(config, device=device, attention_backend=attention_backend, sageattn=sageattn) for _ in range(config.token_refiner_num_layers)])
        self.final_norm = _rms(config.hidden_size, config.final_norm_eps, device=device)

    def forward(self, x, *, cu_seqlens=None):
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens)
        return self.final_norm(x)

class MiniMaxH3NativeFinalLayer(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta') -> None:
        super().__init__()
        self.norm = _rms(config.hidden_size, config.final_norm_eps, device=device)
        self.adaln_proj = MiniMaxH3NativeAdaLN(config, 2, 1, device=device)
        video_dim = config.latents_dim
        for size in config.patch_size:
            video_dim *= size
        self.video_out = _linear(config.hidden_size, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = _linear(config.hidden_size, config.audio_latents_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, *, adaln_input, inverse_indices, mod_segments=None):
        shift, scale = self.adaln_proj(adaln_input)
        hidden = self.norm(x)
        if mod_segments is not None and lean_block_enabled():
            for a, b, row in mod_segments:
                hidden[a:b].mul_(1 + scale[row])
                hidden[a:b].add_(shift[row])
            rows = int(hidden.shape[0])
            chunk = mlp_chunk_rows() or rows
            video = torch.empty(rows, self.video_out.out_features, dtype=torch.float32, device=hidden.device)
            audio = torch.empty(rows, self.audio_out.out_features, dtype=torch.float32, device=hidden.device)
            for start in range(0, rows, chunk):
                piece = hidden[start:start + chunk].float()
                video[start:start + chunk] = self.video_out(piece)
                audio[start:start + chunk] = self.audio_out(piece)
                del piece
            return (video, audio)
        hidden = hidden * (1 + scale.index_select(0, inverse_indices))
        hidden = hidden + shift.index_select(0, inverse_indices)
        hidden = hidden.float()
        return (self.video_out(hidden), self.audio_out(hidden))

class MiniMaxH3NativeDiT(nn.Module):

    def __init__(self, config: H3NativeConfig, *, device='meta', attention_backend: str='sdpa', sageattn=None) -> None:
        super().__init__()
        self.config = config
        video_dim = config.latents_dim
        for size in config.patch_size:
            video_dim *= size
        self.video_patch_proj = _linear(video_dim, config.hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = _linear(config.audio_latents_dim, config.hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = _linear(config.text_dim, config.hidden_size, bias=True, dtype=torch.bfloat16, device=device)
        if config.fused_adaln:
            self.register_buffer('adaln_t_table', torch.empty(1025, config.fused_adaln_rank, dtype=torch.float32, device=device), persistent=True)
            self.register_buffer('adaln_basis', torch.empty(config.fused_adaln_rank, config.time_embed_dim, dtype=torch.float32, device=device), persistent=True)
            self.register_buffer('adaln_mean', torch.empty(config.time_embed_dim, dtype=torch.float32, device=device), persistent=True)
        else:
            self.time_embedder = MiniMaxH3NativeTimeEmbedder(config, device=device)
        self.rope = nn.Module()
        self.rope.register_buffer('inv_freq', torch.empty(config.rope_inv_freq_len, dtype=torch.float32, device=device), persistent=True)
        self.attention_backend = attention_backend
        self.token_refiner = MiniMaxH3NativeTokenRefiner(config, device=device, attention_backend=attention_backend, sageattn=sageattn)
        self.blocks = nn.ModuleList([MiniMaxH3NativeBlock(config, device=device, attention_backend=attention_backend, sageattn=sageattn) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3NativeFinalLayer(config, device=device)

    def _rope_cache(self, position_ids, *, dtype):
        if position_ids.ndim != 2 or position_ids.shape[1] != 3:
            raise ValueError('H3 position_ids must have shape [tokens,3]')
        per_axis = position_ids.float().unsqueeze(-1) * self.rope.inv_freq.view(1, 1, -1)
        temporal, height, width = per_axis.unbind(dim=1)
        half = torch.cat((temporal, height, width), dim=-1)
        frequencies = torch.cat((half, half), dim=-1)
        table = rope_rotation_table(frequencies, dtype) if lean_block_enabled() and comfy_kitchen() is not None else None
        return (torch.cos(frequencies).to(dtype), torch.sin(frequencies).to(dtype), table)

    def forward(self, *, video_rows, audio_rows, prompt_embeds, video_timestep, audio_timestep, packed_layout, condition_timestep=None, reference_audio_timestep=None, sparse_plan=None):
        device = video_rows.device
        text_len = int(prompt_embeds.shape[0])
        if prompt_embeds.ndim != 2 or prompt_embeds.shape[1] != self.config.text_dim:
            raise ValueError('H3 prompt embeddings must have shape [text,5120]')
        seq_len = int(packed_layout.get('used_len', packed_layout['seq_len']))
        img_pos = packed_layout.get('img_input_pos', packed_layout['img_pos']).to(device=device, dtype=torch.long)
        img_output_pos = packed_layout.get('img_output_pos', img_pos).to(device=device, dtype=torch.long)
        audio_pos = packed_layout.get('audio_input_pos', packed_layout['audio_pos']).to(device=device, dtype=torch.long)
        audio_output_pos = packed_layout.get('audio_output_pos', audio_pos).to(device=device, dtype=torch.long)
        text_pos = packed_layout['text_pos'].to(device=device, dtype=torch.long)
        token_tags = packed_layout['token_tags'][:seq_len].to(device=device, dtype=torch.long)
        cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
        position_ids = packed_layout['img_position_ids'][:seq_len].to(device=device)
        text = self.condition_proj(prompt_embeds.to(device=device, dtype=torch.bfloat16))
        text = self.token_refiner(text, cu_seqlens=torch.tensor(packed_layout.get('action_refiner_bounds', [0, text_len]), dtype=torch.int32, device=device))
        hidden = torch.zeros(seq_len, self.config.hidden_size, dtype=torch.bfloat16, device=device)
        hidden.index_copy_(0, text_pos, text)
        hidden.index_copy_(0, img_pos, self.video_patch_proj(video_rows.float()).to(torch.bfloat16))
        hidden.index_copy_(0, audio_pos, self.audio_patch_proj(audio_rows.float()).to(torch.bfloat16))
        raw_candidates = [float(video_timestep), float(audio_timestep)]
        if condition_timestep is not None:
            raw_candidates.append(float(condition_timestep))
        if reference_audio_timestep is not None:
            raw_candidates.append(float(reference_audio_timestep))
        held_positions = [packed_layout.get('held_video_pos'), packed_layout.get('held_audio_pos')]
        if any((pos is not None for pos in held_positions)):
            raw_candidates.append(1.0)
        candidates = torch.tensor(raw_candidates, dtype=torch.float32, device=device)
        unique_timesteps, mapping = torch.unique(candidates, sorted=True, return_inverse=True)
        inverse_indices = torch.full((seq_len,), int(mapping[0]), dtype=torch.long, device=device)
        inverse_indices.index_fill_(0, audio_pos, int(mapping[1]))
        condition_pos = packed_layout.get('condition_img_pos', packed_layout.get('reference_img_pos'))
        if condition_pos is not None and condition_pos.numel():
            if condition_timestep is None:
                raise ValueError('condition_img_pos requires condition_timestep')
            inverse_indices.index_fill_(0, condition_pos.to(device=device, dtype=torch.long), int(mapping[2]))
        reference_audio_pos = packed_layout.get('reference_audio_pos')
        if reference_audio_pos is not None and reference_audio_pos.numel():
            if reference_audio_timestep is None:
                raise ValueError('reference_audio_pos requires reference_audio_timestep')
            inverse_indices.index_fill_(0, reference_audio_pos.to(device=device, dtype=torch.long), int(mapping[3 if condition_timestep is not None else 2]))
        for pos in held_positions:
            if pos is not None:
                inverse_indices.index_fill_(0, pos.to(device=device, dtype=torch.long), int(mapping[-1]))
        if self.config.fused_adaln:
            from .minimax_h3_fused import adaln_coefficients
            adaln_input = adaln_coefficients(self.adaln_t_table, unique_timesteps).to(torch.bfloat16)
        else:
            adaln_input = F.silu(self.time_embedder(unique_timesteps)).to(torch.bfloat16)
        combined_indices = token_tags.clamp(min=0) + inverse_indices * 3
        rope_cache = self._rope_cache(position_ids, dtype=torch.bfloat16)
        lean = lean_block_enabled()
        mod_segments = modulation_segments(combined_indices) if lean else None
        final_segments = modulation_segments(inverse_indices) if lean else None
        if sparse_plan is not None and (not lean):
            raise ValueError('sparse attention requires the lean block path (STUDIO_H3_LEAN_BLOCK)')
        for index, block in enumerate(self.blocks):
            block.attn.action_groups = packed_layout.get('action_groups')
            plan = sparse_plan if sparse_plan is not None and sparse_plan.applies_to_block(index) else None
            hidden = block(hidden, adaln_input=adaln_input, combined_indices=combined_indices, rope_cache=rope_cache, cu_seqlens=cu_seqlens, mod_segments=mod_segments, sparse=plan)
        video_logits, audio_logits = self.final_layer(hidden, adaln_input=adaln_input, inverse_indices=inverse_indices, mod_segments=final_segments)
        video_logits = video_logits.index_select(0, img_output_pos)
        audio_logits = audio_logits.index_select(0, audio_output_pos)
        update_mask = packed_layout.get('update_mask')
        if update_mask is not None:
            video_logits = video_logits * update_mask.to(device).view(-1, 1)
        audio_update_mask = packed_layout.get('audio_update_mask')
        if audio_update_mask is not None:
            audio_logits = audio_logits * audio_update_mask.to(device).view(-1, 1)
        return (video_logits, audio_logits)

def build_meta_h3_native_model(config_path: str | Path, *, attention_backend: str='sdpa', fused_adaln: bool=False) -> MiniMaxH3NativeDiT:
    config = config_from_checkpoint(config_path)
    if fused_adaln:
        config = replace(config, fused_adaln=True)
    if config.num_layers != 50:
        raise ValueError(f'released MiniMax H3 requires 50 DiT blocks, got {config.num_layers}')
    sageattn = None
    if attention_backend == 'sageattention':
        from .minimax_h3_attention import load_sageattn
        sageattn = load_sageattn()
    return MiniMaxH3NativeDiT(config, device='meta', attention_backend=attention_backend, sageattn=sageattn)
