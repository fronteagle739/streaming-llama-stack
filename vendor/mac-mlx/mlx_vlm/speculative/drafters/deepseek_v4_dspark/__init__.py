from .config import DeepseekV4DSparkConfig as ModelConfig
from .config import TextConfig
from .deepseek_v4_dspark import DeepseekV4DSparkDraftModel
from .deepseek_v4_dspark import DeepseekV4DSparkDraftModel as Model

__all__ = [
    "DeepseekV4DSparkDraftModel",
    "Model",
    "ModelConfig",
    "TextConfig",
]
