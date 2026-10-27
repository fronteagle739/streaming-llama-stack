"""DeepSeek-V4.1; defer the runtime import for CPU-only layout inspection."""

__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig"]


def __getattr__(name):
    if name == "Model":
        from .deepseek_v41 import Model

        return Model
    if name in ("ModelConfig", "TextConfig", "VisionConfig"):
        from . import config

        return getattr(config, name)
    raise AttributeError(name)
