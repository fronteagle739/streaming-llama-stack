from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """GLM-5.1 (`glm_moe_dsa`) config. The arch is mlx-lm's `deepseek_v32` (MLA + lightning Indexer + MoE)
    reused verbatim; this just carries the fields mlx-lm's `glm_moe_dsa.ModelArgs` needs so the mlx-vlm
    loader (which has the INF mixed-quant path) can instantiate it. Defaults are GLM-5.1-4.8bit-INF's real
    config.json values, but `from_dict` overrides them per checkpoint."""

    model_type: str = "glm_moe_dsa"
    vocab_size: int = 154880
    hidden_size: int = 6144
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 3
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-05
    rope_parameters: Optional[Dict] = None
    attention_bias: bool = False
    # carried through for completeness / mlx-lm from_dict (ignored if unknown there)
    head_dim: int = 64
    qk_head_dim: int = 256
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict] = None
    rope_theta: Optional[float] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    attention_dropout: float = 0.0
    index_skip_topk_offset: Optional[int] = None
    index_topk_freq: Optional[int] = None
    index_topk_pattern: Optional[Union[str, List[str]]] = None
    index_share_for_mtp_iteration: Optional[bool] = None
    indexer_rope_interleave: Optional[bool] = None
    indexer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    rope_interleave: Optional[bool] = None
    # The inferencerlabs sanitize-tool version (config key "mlx-sanitized"). Selects the indexer forward:
    # 0.30.x checkpoints need the legacy partial-rope lightning indexer; 0.31+ use the current mlx-lm one.
    mlx_sanitized: Optional[str] = None

    @classmethod
    def from_dict(cls, params):
        # carry the hyphenated config key "mlx-sanitized" into the mlx_sanitized field (BaseModelConfig
        # filters by exact field name, so the hyphen form would otherwise be dropped).
        params = dict(params or {})
        if "mlx-sanitized" in params:
            params.setdefault("mlx_sanitized", params["mlx-sanitized"])
        return super().from_dict(params)
