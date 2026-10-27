"""VLM interface for DeepSeek-V4.1 Flash."""

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig


class Model(nn.Module):
    # Conservative until long-context chunk parity is validated with QAT enabled.
    # Keep the wrapper and direct-LM generation contracts identical.
    no_chunked_prefill = True
    from .native_guard import NATIVE_STATUS as native_kernel_status

    def __init__(self, config: ModelConfig):
        from .language import LanguageModel

        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        if config.vision_config is not None and config.vision_config.num_hidden_layers:
            from .vision import Aligner, VisionModel

            self.vision = VisionModel(config.vision_config)
            self.aligner = Aligner(config.vision_config, config.text_config.hidden_size)
            self.language_model.image_token_id = config.image_token_id
            for name in ("image_start", "image_end", "image_newline"):
                setattr(self, name, mx.zeros((config.text_config.hidden_size,), dtype=mx.bfloat16))

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values:
            from .vision import merge_image_embeddings

            if not hasattr(self, "vision"):
                raise ValueError("Images require a configured vision tower")
            embeds = merge_image_embeddings(self, input_ids, embeds, pixel_values)
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        if pixel_values is not None:
            kwargs["inputs_embeds"] = self.get_input_embeddings(input_ids, pixel_values).inputs_embeds
        if hasattr(self, "vision"):
            kwargs["image_mask"] = input_ids == self.config.image_token_id
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        from .vision_layout import is_vision_key, sanitize_vision_weights

        vision_weights = {k: v for k, v in weights.items() if is_vision_key(k)}
        if hasattr(self, "vision"):
            sanitize_vision_weights(vision_weights, self.config.vision_config,
                                    self.config.text_config.hidden_size)
        elif vision_weights:
            raise ValueError("Checkpoint has vision tensors but config has no vision tower")
        text_weights = {k.removeprefix("language_model."): v for k, v in weights.items()
                        if not is_vision_key(k)}
        weights = self.language_model.sanitize(text_weights)
        return {**vision_weights, **{
            (key if key.startswith("language_model.") else "language_model." + key): value
            for key, value in weights.items()
        }}

    @property
    def quant_predicate(self):
        from .vision_layout import is_vision_key

        text_predicate = self.language_model.quant_predicate
        return lambda path, module: False if is_vision_key(path) else text_predicate(path, module)

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def attach_engram_storage(self, model_path):
        return self.language_model.attach_engram_storage(model_path)

    def pipeline(self, group):
        self.language_model.pipeline(group)
        return self

    def shard(self, group=None):
        self.language_model.shard(group)
        return self
