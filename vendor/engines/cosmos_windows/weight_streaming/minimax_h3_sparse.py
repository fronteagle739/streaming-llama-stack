# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_sparse.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import math
from typing import Any, Callable
import torch
SPARSE_ATTENTION_MODES = ('auto', 'on', 'off')
SPARSE_ATTENTION_METHODS = ('sol-attn', 'sla')
SPARSE_SINK_CONDITIONING = ('exact_kv', 'exact_kv_and_rows', 'off')
BLOCK_SIZE = 64
HEAD_DIM = 128

@dataclass(frozen=True)
class H3SparseAttentionConfig:
    mode: str = 'auto'
    method: str = 'sol-attn'
    tau: float = 1.3
    keep_percent: float = 10.0
    start_percent: float = 0.2
    end_percent: float = 1.0
    min_tokens: int = 12288
    extra_tokens: int = 256
    sink_conditioning: str = 'exact_kv_and_rows'
    producer_chunk: int = 4096
    dense_blocks: tuple[int, ...] = ()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.mode not in SPARSE_ATTENTION_MODES:
            errors.append('sparse attention mode must be one of ' + ', '.join(SPARSE_ATTENTION_MODES))
        if self.method not in SPARSE_ATTENTION_METHODS:
            errors.append('sparse attention method must be one of ' + ', '.join(SPARSE_ATTENTION_METHODS))
        if not 0.0 <= float(self.tau) <= 4.0:
            errors.append('sparse tau must be within [0, 4]')
        if not 0.5 <= float(self.keep_percent) <= 95.0:
            errors.append('sparse keep_percent must be within [0.5, 95]')
        if not (0.0 <= float(self.start_percent) <= 1.0 and 0.0 <= float(self.end_percent) <= 1.0):
            errors.append('sparse start/end percent must be within [0, 1]')
        if float(self.end_percent) < float(self.start_percent):
            errors.append('sparse end_percent must not be before start_percent')
        if int(self.min_tokens) < 0:
            errors.append('sparse min_tokens must be non-negative')
        if not 0 <= int(self.extra_tokens) <= 256:
            errors.append('sparse extra_tokens must be within [0, 256]')
        if self.sink_conditioning not in SPARSE_SINK_CONDITIONING:
            errors.append('sparse sink_conditioning must be one of ' + ', '.join(SPARSE_SINK_CONDITIONING))
        if int(self.producer_chunk) <= 0 or int(self.producer_chunk) % BLOCK_SIZE:
            errors.append(f'sparse producer_chunk must be a positive multiple of {BLOCK_SIZE}')
        return errors

    @property
    def topk_ratio(self) -> float:
        return 0.0 if self.method == 'sol-attn' else float(self.keep_percent) / 100.0

    @property
    def sigma_start(self) -> float:
        return 1.0 - float(self.start_percent)

    @property
    def sigma_end(self) -> float:
        return 1.0 - float(self.end_percent)

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class H3SparseAttentionCapability:
    available: bool
    reason: str | None = None
    kitchen_version: str = ''
    device: str = ''

    def as_dict(self) -> dict:
        return asdict(self)

class H3SparseAttentionUnavailable(RuntimeError):

    def __init__(self, capability: H3SparseAttentionCapability) -> None:
        self.capability = capability
        super().__init__(f"H3 sparse attention was required (mode=on) but is unavailable: {capability.reason or 'unknown reason'}")

    def as_dict(self) -> dict:
        return {'code': 'h3_sparse_attention_unavailable', 'capability': self.capability.as_dict(), 'requirements': {'package': 'comfy-kitchen>=0.2.33 (sol_attn_chunked)', 'gpu': 'CUDA sm_80+', 'head_dim': HEAD_DIM}}

def detect_h3_sparse_attention(device: str | torch.device | int | None='cuda', *, import_module: Callable[[str], Any]=importlib.import_module) -> H3SparseAttentionCapability:
    try:
        ck = import_module('comfy_kitchen')
    except Exception as exc:
        return H3SparseAttentionCapability(False, f'comfy_kitchen is not importable: {exc}', '', str(device))
    try:
        version = importlib.metadata.version('comfy-kitchen')
    except Exception:
        version = str(getattr(ck, '__version__', '') or 'unknown')
    for name in ('sol_attn_chunked', 'sol_attn_is_available', 'rms_rope_split_half_'):
        if not callable(getattr(ck, name, None)):
            return H3SparseAttentionCapability(False, f'comfy_kitchen {version} has no {name}; need >= 0.2.33', version, str(device))
    try:
        if not torch.cuda.is_available():
            return H3SparseAttentionCapability(False, 'CUDA is not available', version, str(device))
        target = torch.device(device) if not isinstance(device, torch.device) else device
        if target.type != 'cuda':
            return H3SparseAttentionCapability(False, f'{target} is not a CUDA device', version, str(device))
        ok = bool(ck.sol_attn_is_available(target))
    except Exception as exc:
        return H3SparseAttentionCapability(False, f'sol_attn_is_available failed: {exc}', version, str(device))
    if not ok:
        return H3SparseAttentionCapability(False, 'no compiled Sol-Attn kernel for this GPU (needs CUDA sm_80+)', version, str(device))
    return H3SparseAttentionCapability(True, None, version, str(device))

def sparse_sinks(packed_layout: dict, seq_len: int, sink_conditioning: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if sink_conditioning == 'off':
        return ((0, 0), (0, 0))
    img_pos = packed_layout.get('img_output_pos')
    if img_pos is None or int(img_pos.numel()) == 0:
        img_pos = packed_layout.get('img_input_pos', packed_layout.get('img_pos'))
    if img_pos is None or int(img_pos.numel()) == 0:
        return ((0, 0), (0, 0))
    video_start = int(img_pos.min())
    if video_start <= 0:
        return ((0, 0), (0, 0))
    blocks = (0, math.ceil(video_start / BLOCK_SIZE))
    if sink_conditioning != 'exact_kv_and_rows':
        return (blocks, (0, 0))
    audio_pos = packed_layout.get('audio_output_pos')
    if audio_pos is None or int(audio_pos.numel()) == 0:
        audio_pos = packed_layout.get('audio_input_pos', packed_layout.get('audio_pos'))
    if audio_pos is None or int(audio_pos.numel()) == 0:
        return (blocks, blocks)
    audio_start = int(audio_pos.min())
    if audio_start >= video_start:
        raise ValueError(f'sparse sinks expect the target audio rows before the target video rows (audio rows {audio_start}..{int(audio_pos.max())}, video rows {video_start}..{int(img_pos.max())}, seq_len {seq_len}). Reference images produce this layout; see the caller.')
    return (blocks, (audio_start // BLOCK_SIZE, blocks[1]))

class H3SparseAttentionPlan:

    def __init__(self, config: H3SparseAttentionConfig, *, sink_blocks: tuple[int, int], sink_q: tuple[int, int], tokens: int, stats: dict) -> None:
        self.tau = float(config.tau)
        self.topk_ratio = float(config.topk_ratio)
        self.extra_tokens = int(config.extra_tokens)
        self.producer_chunk = int(config.producer_chunk)
        self.sink_blocks = tuple((int(v) for v in sink_blocks))
        self.sink_q = tuple((int(v) for v in sink_q))
        self.tokens = int(tokens)
        self._dense_blocks = frozenset((int(b) for b in config.dense_blocks))
        self._stats = stats

    def applies_to_block(self, index: int) -> bool:
        return int(index) not in self._dense_blocks

    def stats_for(self, module) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._stats.get((id(module), self.tokens))

    def remember(self, module, kmean: torch.Tensor, vscale: torch.Tensor) -> None:
        self._stats[id(module), self.tokens] = (kmean, vscale)

    def as_dict(self) -> dict:
        return {'tau': self.tau, 'topk_ratio': self.topk_ratio, 'extra_tokens': self.extra_tokens, 'producer_chunk': self.producer_chunk, 'sink_blocks': list(self.sink_blocks), 'sink_q': list(self.sink_q), 'tokens': self.tokens, 'dense_blocks': sorted(self._dense_blocks)}

class H3SparseAttentionController:

    def __init__(self, config: H3SparseAttentionConfig | None, *, device='cuda', capability: H3SparseAttentionCapability | None=None, lean_available: bool=True) -> None:
        self.config = config or H3SparseAttentionConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError('; '.join(errors))
        self.capability = capability if capability is not None else detect_h3_sparse_attention(device) if self.config.mode != 'off' else H3SparseAttentionCapability(False, 'mode=off', '', str(device))
        self.lean_available = bool(lean_available)
        self._stats: dict = {}
        if self.config.mode == 'on':
            if not self.capability.available:
                raise H3SparseAttentionUnavailable(self.capability)
            if not self.lean_available:
                raise H3SparseAttentionUnavailable(H3SparseAttentionCapability(False, 'the lean block path is disabled (STUDIO_H3_LEAN_BLOCK=0)', self.capability.kitchen_version, self.capability.device))
        self.active = self.config.mode != 'off' and self.capability.available and self.lean_available

    @property
    def inactive_reason(self) -> str | None:
        if self.config.mode == 'off':
            return 'mode=off'
        if not self.capability.available:
            return self.capability.reason or 'unavailable'
        if not self.lean_available:
            return 'lean block path disabled'
        return None

    def describe(self) -> dict:
        return {'active': self.active, 'reason': self.inactive_reason, 'config': self.config.as_dict(), 'capability': self.capability.as_dict()}

    def plan_for_step(self, *, sigma: float, packed_layout: dict, seq_len: int, head_dim: int) -> tuple[H3SparseAttentionPlan | None, str]:
        if not self.active:
            return (None, f'dense: {self.inactive_reason}')
        if int(head_dim) != HEAD_DIM:
            return (None, f'dense: head_dim {head_dim} != {HEAD_DIM}')
        if int(seq_len) < int(self.config.min_tokens):
            return (None, f'dense: {seq_len} tokens < min_tokens {self.config.min_tokens}')
        sigma = float(sigma)
        if sigma > self.config.sigma_start + 1e-06 or sigma < self.config.sigma_end - 1e-06:
            return (None, f'dense: sigma {sigma:.3f} outside [{self.config.sigma_end:.3f}, {self.config.sigma_start:.3f}]')
        sink_blocks, sink_q = sparse_sinks(packed_layout, int(seq_len), self.config.sink_conditioning)
        plan = H3SparseAttentionPlan(self.config, sink_blocks=sink_blocks, sink_q=sink_q, tokens=int(seq_len), stats=self._stats)
        return (plan, f'sparse: {self.config.method} sinks {sink_blocks}/{sink_q}')

    def reset(self) -> None:
        self._stats.clear()
__all__ = ['BLOCK_SIZE', 'HEAD_DIM', 'H3SparseAttentionCapability', 'H3SparseAttentionConfig', 'H3SparseAttentionController', 'H3SparseAttentionPlan', 'H3SparseAttentionUnavailable', 'SPARSE_ATTENTION_METHODS', 'SPARSE_ATTENTION_MODES', 'SPARSE_SINK_CONDITIONING', 'detect_h3_sparse_attention', 'sparse_sinks']
