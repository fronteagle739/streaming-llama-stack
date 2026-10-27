from .config import DeepseekV41DSparkConfig as ModelConfig
from .config import TextConfig
from .deepseek_v41_dspark import DeepseekV41DSparkDraftModel
from .deepseek_v41_dspark import DeepseekV41DSparkDraftModel as Model

__all__ = [
    "DeepseekV41DSparkDraftModel",
    "Model",
    "ModelConfig",
    "TextConfig",
]
