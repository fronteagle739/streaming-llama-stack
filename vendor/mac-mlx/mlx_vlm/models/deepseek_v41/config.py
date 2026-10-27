"""Original DeepSeek-V4.1 Flash configuration; also importable by file on CPU hosts."""

import os
from dataclasses import dataclass, field, fields
from typing import Optional

if __package__:
    from ..base import BaseModelConfig
else:
    # The package base imports MLX. Mirror its two config helpers for file imports.
    class BaseModelConfig:
        @classmethod
        def from_dict(cls, params):
            names = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in (params or {}).items() if k in names})

        def to_dict(self):
            return {k: v for k, v in vars(self).items() if v is not None}



def _kv_override(name: str, current: int) -> int:
    """Environment override for a KV-selection width, or the checkpoint's own value.

    V4.1 does not keep a dense KV cache. Older keys/values are POOLED by `Compressor` into groups of
    `compress_ratios[layer]`, and at attention time `Indexer` selects only `index_topk` of those
    pooled entries; `sliding_window` is the recent span kept dense. So the cache is an approximation
    of everything except the last `sliding_window` tokens, which is where the fourfold saving comes
    from and also where its inconsistency comes from -- a fact recalled from outside the selected set
    is reconstructed from a pooled group rather than read back exactly.

    These two widths are the SAFE way to trade that back, and what they cost is COMPUTE rather than
    memory in the shipped configuration. Both bound what attention READS, not what is stored:
    `_LayerCache` grows every field by doubling, and `DeepseekV41Cache.finish` only trims `win_kv`
    back to `window_size` when `DSV41_BOUNDED_CACHE=1`, which is NOT the default ("0"). So with the
    default unbounded cache, raising `sliding_window` adds attention work over an already-stored
    window and adds no storage; under `DSV41_BOUNDED_CACHE=1` it does also raise storage,
    proportionally. `index_topk` is compute either way -- the pooled entries are stored regardless
    and it only selects how many are attended. Neither changes any weights.

    ⚠️ THE POOLING ITSELF IS NOT SAFELY SWITCHABLE, so do not add an override for `compress_ratios`.
    `checkpoint_layout` lays out tensors per ratio (compressor/indexer weights exist only where the
    ratio is non-zero), and a zero ratio also drops the indexer AND uses plain `rope_theta` with YaRN
    off -- which is precisely wrong for long context. Turning the approximation "off" that way would
    not load the checkpoint it was trained as.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return current
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = 'deepseek_v41_text'
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    hidden_act: str = 'silu'
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 1048576
    rope_theta: int = 10000
    rope_scaling: dict = field(default_factory=lambda: {'rope_type': 'yarn', 'factor': 16, 'beta_fast': 32, 'beta_slow': 1, 'original_max_position_embeddings': 65536})
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = 'sqrtsoftplus'
    topk_method: str = 'noaux_tc'
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    sliding_window: int = 128
    compress_ratios: list[int] = field(default_factory=lambda: [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    compress_rope_theta: int = 160000
    kv_source_layer_ids: list[int] = field(default_factory=lambda: [2, 8, 14, 20])
    index_source_layer_ids: list[int] = field(default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36])
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-06
    engram_layer_ids: list[int] = field(default_factory=lambda: [1, 14])
    engram_num_embeddings: list[int] = field(default_factory=lambda: [384006168, 384016682])
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    engram_source: str = "hybrid"
    engram_ram_chunks: int = 8
    model_dir: Optional[str] = None
    gate_temp: float = 1.0

    def __post_init__(self):
        # Widen how much context is kept EXACTLY, without touching the pooling the checkpoint was
        # trained with. See `_kv_override`: defaults are the release's own 512 / 128.
        self.index_topk = _kv_override("DSV41_INDEX_TOPK", self.index_topk)
        self.sliding_window = _kv_override("DSV41_SLIDING_WINDOW", self.sliding_window)
        # The release includes three MTP ratios after the 40 backbone ratios.
        n = self.num_hidden_layers
        if len(self.compress_ratios) not in (n, n + self.num_nextn_predict_layers):
            raise ValueError("compress_ratios must cover each backbone layer, optionally including MTP")
        if any(type(r) is not int or r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError("compress_ratios values must be integers in {0, 1, 2}")
        if any(i < 0 or i >= n or self.compress_ratios[i] <= 0 for i in self.kv_source_layer_ids):
            raise ValueError("kv_source_layer_ids must be backbone layers with ratio > 0")
        if self.engram_source not in ("ram", "nvme", "hybrid"):
            raise ValueError("engram_source must be ram, nvme or hybrid")
        if self.engram_ram_chunks < 1:
            raise ValueError("engram_ram_chunks must be positive")


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = 'deepseek_v41_vision'
    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: int = 10000
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: Optional[float] = None


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "deepseek_v41"
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None
    image_token_id: int = 129264
    quantization_config: Optional[dict] = None
    model_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        raw_text = params.get("text_config") or {}
        text = dict(vars(raw_text)) if isinstance(raw_text, TextConfig) else dict(raw_text)
        for key in ("model_dir", "engram_source"):
            if params.get(key) is not None:
                text[key] = params[key]
        params["text_config"] = TextConfig.from_dict(text)
        vision = params.get("vision_config")
        if vision is not None and not isinstance(vision, VisionConfig):
            params["vision_config"] = VisionConfig.from_dict(vision)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})
