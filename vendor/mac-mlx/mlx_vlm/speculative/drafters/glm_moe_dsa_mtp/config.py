import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.glm_moe_dsa.config import ModelConfig as GlmConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return GlmConfig.from_dict(params)


@dataclass
class GlmMoeDsaMTPConfig(BaseModelConfig):
    model_type: str = "glm_moe_dsa_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False
    source_layer: Optional[int] = None
    recurrent_state: str = "raw"
    compute_dtype: str = "float16"
    share_topk_source: str = "internal"
    dsa_mode: str = "sparse"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.source_layer is None and self.text_config is not None:
            self.source_layer = int(getattr(self.text_config, "num_hidden_layers", 78))
        if self.runtime_block_size is None and self.text_config is not None:
            nextn_depth = getattr(self.text_config, "num_nextn_predict_layers", 1)
            self.runtime_block_size = min(self.block_size, int(nextn_depth) + 1)
        self.recurrent_state = str(self.recurrent_state or "shared_head_norm").lower()
        self.recurrent_state = self.recurrent_state.replace("-", "_")
        if self.recurrent_state not in ("shared_head_norm", "raw"):
            raise ValueError(
                "GlmMoeDsaMTPConfig.recurrent_state must be shared_head_norm or raw"
            )
        self.compute_dtype = str(self.compute_dtype or "float16").lower()
        self.compute_dtype = self.compute_dtype.replace("-", "_")
        dtype_aliases = {
            "fp16": "float16",
            "f16": "float16",
            "bf16": "bfloat16",
            "native": "native",
            "auto": "native",
            "none": "native",
        }
        self.compute_dtype = dtype_aliases.get(self.compute_dtype, self.compute_dtype)
        if self.compute_dtype not in ("float16", "bfloat16", "native"):
            raise ValueError(
                "GlmMoeDsaMTPConfig.compute_dtype must be float16, bfloat16, or native"
            )
        self.share_topk_source = str(self.share_topk_source or "internal").lower()
        self.share_topk_source = self.share_topk_source.replace("-", "_")
        source_aliases = {
            "default": "internal",
            "mtp": "internal",
            "mtp_internal": "internal",
            "target_prev": "target",
            "target_previous": "target",
            "none": "off",
            "disabled": "off",
            "false": "off",
            "0": "off",
        }
        self.share_topk_source = source_aliases.get(
            self.share_topk_source, self.share_topk_source
        )
        if self.share_topk_source not in ("target", "internal", "off"):
            raise ValueError(
                "GlmMoeDsaMTPConfig.share_topk_source must be target, internal, or off"
            )
        self.dsa_mode = str(self.dsa_mode or "sparse").lower()
        self.dsa_mode = self.dsa_mode.replace("-", "_")
        dsa_aliases = {
            "default": "sparse",
            "enabled": "sparse",
            "true": "sparse",
            "1": "sparse",
            "off": "dense",
            "disabled": "dense",
            "false": "dense",
            "0": "dense",
        }
        self.dsa_mode = dsa_aliases.get(self.dsa_mode, self.dsa_mode)
        if self.dsa_mode not in ("sparse", "dense"):
            raise ValueError(
                "GlmMoeDsaMTPConfig.dsa_mode must be sparse or dense"
            )

    @classmethod
    def from_dict(cls, params: dict) -> "GlmMoeDsaMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        nextn_depth = text_config.get("num_nextn_predict_layers", 1)
        flat.setdefault("block_size", int(nextn_depth) + 1)
        if "source_layer" not in flat and "num_hidden_layers" in text_config:
            flat["source_layer"] = int(text_config["num_hidden_layers"])
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
