"""Config for the DeepSeek-V4.1-Flash DSpark speculative draft head.

DSpark is DeepSeek's own block drafter: rather than a NextN/MTP head that
extends one token at a time, it drafts a whole block of ``dspark_block_size``
(5) tokens in ONE pass of ``num_nextn_predict_layers`` (3) full V4.1 blocks,
sharpens that parallel block with a first-order Markov transition head, and
gates it with a per-position confidence head.

The ``dspark_*`` hyper-parameters are those of the release
``config.json`` ``text_config`` (``num_nextn_predict_layers`` 3,
``dspark_block_size`` 5, ``dspark_noise_token_id`` 128799,
``dspark_target_layer_ids`` [37, 38, 39], ``dspark_markov_rank`` 256,
``dspark_n_routed_experts`` 128, ``dspark_num_experts_per_tok`` 3) and are pinned
weight-free by ``mlx_vlm/models/deepseek_v41/test_checkpoint_layout_v41.py``
(the ``mtp.*`` keys) and ``test_dspark_dsv41.py``.

Field names deliberately match our V4 DSpark drafter (``block_size``,
``noise_token_id``, ``target_layer_ids``, ``markov_rank``, ``n_draft_layers``,
``confidence_threshold``) so ``speculative/dspark.py``'s round loop, which reads
``draft_model.config.target_layer_ids`` / ``block_size`` /
``runtime_block_size``, drives this drafter unchanged.
"""

import inspect
from dataclasses import MISSING, dataclass, field, fields
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.deepseek_v41.config import TextConfig as DeepseekV41TextConfig


class TextConfig:
    """Factory shim: the drafter's ``text_config`` is the target's text config."""

    @classmethod
    def from_dict(cls, params: dict):
        return DeepseekV41TextConfig.from_dict(params)


def _default_target_layer_ids() -> List[int]:
    return [37, 38, 39]


@dataclass
class DeepseekV41DSparkConfig(BaseModelConfig):
    model_type: str = "deepseek_v41_dspark"
    text_config: Optional[TextConfig] = None
    # DSpark draft-head hyper-parameters (release text_config).
    n_draft_layers: int = 3
    block_size: int = 5
    noise_token_id: int = 128799
    target_layer_ids: List[int] = field(default_factory=_default_target_layer_ids)
    markov_rank: int = 256
    # Runtime knob: keep the drafted prefix up to (excluding) the first position
    # whose sigmoid(confidence) < confidence_threshold. 0.0 keeps the whole block.
    confidence_threshold: float = 0.0
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        self.target_layer_ids = list(self.target_layer_ids or [])
        self._sync_from_text_config()
        if self.runtime_block_size is None:
            self.runtime_block_size = int(self.block_size)

    def _default(self, name):
        for f in fields(self):
            if f.name == name:
                if f.default is not MISSING:
                    return f.default
                if f.default_factory is not MISSING:  # type: ignore[misc]
                    return f.default_factory()  # type: ignore[misc]
        return None

    def _sync_from_text_config(self):
        """Take the DSpark knobs from the target text config.

        The class defaults ARE the release's values, so "unset" cannot be
        detected by truthiness (a 5 is both the default and a legitimate
        explicit value). A field is refreshed from the text config only while it
        still equals its dataclass default -- an explicitly different drafter
        config always wins, and a target with different knobs (a tiny test
        config, a future release) is still followed.
        """
        text = self.text_config
        if text is None:
            return
        for src, dst in (("dspark_block_size", "block_size"),
                         ("dspark_noise_token_id", "noise_token_id"),
                         ("dspark_markov_rank", "markov_rank"),
                         ("num_nextn_predict_layers", "n_draft_layers")):
            value = getattr(text, src, 0)
            if value and (not getattr(self, dst, 0) or getattr(self, dst) == self._default(dst)):
                setattr(self, dst, int(value))
        target = list(getattr(text, "dspark_target_layer_ids", []) or [])
        if target and (not self.target_layer_ids
                       or self.target_layer_ids == self._default("target_layer_ids")):
            self.target_layer_ids = target

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV41DSparkConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        # dspark_* may live at the drafter-config top level or (as in the release)
        # inside the target text_config -- accept both spellings of each knob.
        alias = {
            "dspark_block_size": "block_size",
            "dspark_noise_token_id": "noise_token_id",
            "dspark_target_layer_ids": "target_layer_ids",
            "dspark_markov_rank": "markov_rank",
            "num_nextn_predict_layers": "n_draft_layers",
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
