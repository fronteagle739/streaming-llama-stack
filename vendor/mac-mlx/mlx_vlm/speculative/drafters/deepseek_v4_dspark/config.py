"""Config for the DeepSeek-V4-Flash-DSpark speculative draft head.

DSpark is DeepSeek's own speculative decoder for the V4-Flash release. Unlike a
NextN/MTP head (one recurrent layer that autoregressively drafts one token at a
time), DSpark drafts a whole block of ``dspark_block_size`` tokens in a single
backbone pass, then sharpens that parallel block into a coherent sequence with a
first-order **Markov** transition head and gates it with a per-position
**confidence** head. See ``deepseek_v4_dspark.py`` for the exact reference the
forward mirrors (``inference/model.py`` ``DSparkBlock`` /
``Transformer.forward_spec`` of ``deepseek-ai/DeepSeek-V4-Flash-DSpark``).

The dspark_* hyper-parameters below come verbatim from the release
``inference/config.json`` (n_mtp_layers=3, dspark_block_size=5,
dspark_noise_token_id=128799, dspark_target_layer_ids=[40,41,42],
dspark_markov_rank=256) and are pinned weight-free by
``mlx_vlm/models/deepseek_v4/test_checkpoint_layout.py``.
"""

import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return DeepseekV4Config.from_dict(params)


def _default_target_layer_ids() -> List[int]:
    return [40, 41, 42]


@dataclass
class DeepseekV4DSparkConfig(BaseModelConfig):
    model_type: str = "deepseek_v4_dspark"
    text_config: Optional[TextConfig] = None
    # DSpark draft-head hyper-parameters (release inference/config.json).
    n_draft_layers: int = 3
    block_size: int = 5
    noise_token_id: int = 128799
    target_layer_ids: List[int] = field(default_factory=_default_target_layer_ids)
    markov_rank: int = 256
    # Runtime knob (paper Sec 3.2): keep the drafted prefix up to (excluding) the
    # first position whose sigmoid(confidence) < confidence_threshold. 0.0 keeps
    # the full block (no confidence truncation).
    confidence_threshold: float = 0.0
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None:
            self.runtime_block_size = int(self.block_size)

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV4DSparkConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        # dspark_* may live either at the drafter-config top level or (as in the
        # source release) inside the target text_config — accept both.
        alias = {
            "dspark_block_size": "block_size",
            "dspark_noise_token_id": "noise_token_id",
            "dspark_target_layer_ids": "target_layer_ids",
            "dspark_markov_rank": "markov_rank",
            "n_mtp_layers": "n_draft_layers",
        }
        for src, dst in alias.items():
            if dst not in flat:
                if src in flat:
                    flat[dst] = flat[src]
                elif isinstance(text_config, dict) and src in text_config:
                    flat[dst] = text_config[src]
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
