from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """MiMo V2.5/V2.6 configuration for the pinned mlx-lm architecture.

    Defaults retain V2.5 compatibility; checkpoint fields supply V2.6 value
    scaling, projection layout and the model-specific attention/experts sizes.
    """

    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 6144
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 70
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    n_shared_experts: Optional[int] = None
    n_routed_experts: int = 384
    routed_scaling_factor: Optional[float] = None
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    # per-layer pattern lists (len == num_hidden_layers). hybrid_layer_pattern: 0 = full attention layer,
    # 1 = sliding-window. moe_layer_freq: 0 = dense MLP layer, 1 = MoE. Defaults filled at from_dict time
    # (a 70-long list literal would bloat this; the real config.json always supplies them).
    hybrid_layer_pattern: List[int] = field(default_factory=list)
    moe_layer_freq: List[int] = field(default_factory=list)
    layernorm_epsilon: float = 1e-05
    rope_theta: float = 10000000.0
    head_dim: int = 192
    v_head_dim: int = 128
    partial_rotary_factor: float = 0.334
    sliding_window_size: int = 128
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_num_attention_heads: int = 128
    swa_num_key_value_heads: int = 8
    swa_rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576
    # carried through for completeness (ignored by mimo_v2_flash if unknown there)
    attention_value_scale: Optional[float] = None
    attention_projection_layout: Optional[str] = None
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    # The inferencerlabs sanitize-tool version (config key "mlx-sanitized"); MiMo INF ships "0.30.7".
    # Unlike GLM there is no lightning-indexer whose RoPE drifted, so this is informational only.
    mlx_sanitized: Optional[str] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        if "mlx-sanitized" in params:
            params.setdefault("mlx_sanitized", params["mlx-sanitized"])
        return super().from_dict(params)
