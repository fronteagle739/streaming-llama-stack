import glob
import importlib
import inspect
import json
import logging
import math
import os
import struct
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
import safetensors
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map
from PIL import Image, ImageOps
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from .models.base import BaseImageProcessor
from .tokenizer_utils import load_tokenizer
from .trainer.utils import apply_lora_layers
from . import inf_decode

# Modes that support activation quantization
ACTIVATION_QUANTIZATION_MODES = {"nvfp4", "mxfp8"}

# Constants
MODEL_REMAPPING = {
    "llava_qwen2": "fastvlm",  # Apple's FastVLM, note it's different to the one below
    "llava-qwen2": "llava_bunny",
    "bunny-llama": "llava_bunny",
    "lfm2-vl": "lfm2_vl",
    "cohere2_vision": "aya_vision",
    "jvlm": "jina_vlm",
    "phi4-siglip": "phi4_siglip",
    "sam3_video": "sam3",
    "sam3.1_video": "sam3_1",
    "granite-vision": "granite_vision",
    "granite4-vision": "granite4_vision",
    "granite4_vision": "granite4_vision",
    "rf-detr": "rfdetr",
    "falcon-perception": "falcon_perception",
    "nemotronh_nano_omni_reasoning_v3": "nemotron_h_nano_omni",
    "cohere2moe": "cohere2_moe",
}

MAX_FILE_SIZE_GB = 5

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]

SAFETENSORS_DTYPE_FALLBACKS = {"F8_E8M0": "U8"}


def _e4m3_decode_table() -> mx.array:
    """Return a 256-entry ``float32`` LUT mapping every E4M3FN byte to its value.

    OCP ``E4M3FN``: 1 sign / 4 exponent (bias 7) / 3 mantissa bits,
    no infinities, and ``(exp=15, mant=7)`` reserved for NaN (max finite
    magnitude is therefore 448). This matches the byte convention MLX uses for
    ``nvfp4`` scales (verified empirically).
    """
    table = []
    for byte in range(256):
        sign = (byte >> 7) & 1
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            value = (mantissa / 8.0) * 2.0**-6  # subnormal
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table.append(-value if sign else value)
    return mx.array(table, dtype=mx.float32)


# Built once; reused by every NVFP4 fold.
_E4M3_DECODE_LUT = _e4m3_decode_table()


def _f32_to_e4m3(x: mx.array) -> mx.array:
    """Encode non-negative ``float32`` values to ``E4M3FN`` bytes.

    Pure-MLX bit manipulation (MLX exposes no float8 dtype). Saturates to 448
    on overflow and flushes to the subnormal grid / zero on underflow. Inputs
    are assumed ``>= 0`` (NVFP4 group scales are magnitudes), so the sign bit is
    always 0.
    """
    x = mx.maximum(x.astype(mx.float32), 0.0)
    bits = x.view(mx.uint32)
    fexp = (bits >> 23) & 0xFF  # fp32 exponent, bias 127
    fman = bits & 0x7FFFFF  # fp32 mantissa, 23 bits

    # Normal path: target E4M3 biased exponent e = (fexp - 127) + 7.
    exponent = fexp.astype(mx.int32) - 120
    drop = 20  # 23 -> 3 mantissa bits
    round_bit = (fman >> (drop - 1)) & 1
    sticky = (fman & ((1 << (drop - 1)) - 1)) != 0
    mantissa = fman >> drop
    roundup = round_bit & (sticky.astype(mx.uint32) | (mantissa & 1))
    mantissa = mantissa + roundup
    carry = mantissa >> 3  # mantissa overflowed past 7 -> bump exponent
    mantissa = mantissa & 0x7
    exponent = exponent + carry.astype(mx.int32)

    # Saturate: e > 15, or the NaN slot (e == 15, mant == 7), clamps to 448.
    over = (exponent > 15) | ((exponent == 15) & (mantissa == 7))
    exponent = mx.where(over, mx.array(15, mx.int32), exponent)
    mantissa = mx.where(over, mx.array(6, mx.uint32), mantissa)
    normal_byte = (exponent.astype(mx.uint32) << 3) | mantissa
    normal_valid = exponent >= 1

    # Subnormal path: value = m * 2^-9, so m = round(x * 512) (RNE).
    # m == 8 lands exactly on the smallest normal (0x08 = e1 m0 = 2^-6).
    sub = x * 512.0
    sub_floor = mx.floor(sub)
    frac = sub - sub_floor
    sub_floor_u32 = sub_floor.astype(mx.uint32)
    up = (frac > 0.5) | ((frac == 0.5) & ((sub_floor_u32 & 1) == 1))
    sub_byte = sub_floor_u32 + up.astype(mx.uint32)

    byte = mx.where(normal_valid, normal_byte, sub_byte)
    return byte.astype(mx.uint8)


def _transform_compressed_tensors_nvfp4_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Dict[str, mx.array]:
    """Fold compressed-tensors NVFP4 weights into MLX-native ``nvfp4`` weights.

    A ``nvfp4-pack-quantized`` checkpoint stores, per quantized Linear:

    - ``<p>.weight_packed``       ``uint8``  ``[out, in // 2]``
      (2x E2M1 per byte)
    - ``<p>.weight_scale``        ``uint8``  ``[out, in // 16]``
      (E4M3 per group of 16, loaded by ``mx.load`` as raw bytes -- the same
      byte layout MLX uses for nvfp4 scales)
    - ``<p>.weight_global_scale`` ``float32`` ``[1]`` (per-tensor; the real
      weight is ``fp4 * weight_scale / weight_global_scale``)

    MLX ``nvfp4`` ``QuantizedLinear`` expects ``<p>.weight`` (``uint32``) plus
    ``<p>.scales`` (``uint8`` E4M3) and is single-level: the per-tensor global
    scale is not representable (and is rejected on the Metal backend). Both
    decodes are linear in the FP4 codes, so the global scale can be folded
    directly into the per-group E4M3 scales:
    ``scale_mlx = E4M3(weight_scale / global_scale)``. We keep the original
    packed E2M1 codes bit-exact, avoiding the weight dequantize/re-quantize
    round-trip entirely.
    """
    packed_suffix = ".weight_packed"

    new_weights = {}
    for key in list(weights.keys()):
        if key.endswith(packed_suffix):
            prefix = key[: -len(packed_suffix)]
            packed = weights[key]
            scale = weights[f"{prefix}.weight_scale"]
            global_scale = weights[f"{prefix}.weight_global_scale"].astype(mx.float32)

            # weight_packed is uint8 [out, in//2]; reinterpret as uint32
            # [out, in//8] to match MLX's nvfp4 layout (bit-identical).
            new_weights[f"{prefix}.weight"] = packed.view(mx.uint32)

            # Fold the per-tensor global scale into the per-group E4M3 scales:
            # decode E4M3 -> divide by global scale -> re-encode E4M3.
            # The FP4 codes are untouched; only the much smaller scale tensor
            # is re-rounded once.
            decoded = _E4M3_DECODE_LUT[scale.astype(mx.uint32)]
            new_weights[f"{prefix}.scales"] = _f32_to_e4m3(decoded / global_scale)
        elif key.endswith(".weight_scale") or key.endswith(".weight_global_scale"):
            # Consumed alongside their ``.weight_packed``.
            continue
        else:
            new_weights[key] = weights[key]

    return new_weights


def _transform_compressed_tensors_int4_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
    """Remap compressed-tensors INT4 ``pack-quantized`` weights to MLX affine.

    A symmetric int4 ``pack-quantized`` checkpoint stores, per quantized Linear:

    - ``<p>.weight_packed`` ``int32`` ``[out, in // 8]``
      (8x int4 per word, LSB-first)
    - ``<p>.weight_scale``  (``bf16``/``float``)
      ``[out, in // group_size]``
    - ``<p>.weight_shape``  ``int64`` ``[2]`` (unused by MLX)

    MLX affine ``QuantizedLinear`` uses the same int4 packing and dequantizes as
    ``w * scale + bias``. Symmetric int4 stores values in ``[0, 15]``
    representing ``[-8, 7]``, i.e. ``value = packed - 8``, so we set
    ``bias = -8 * scale``. The packed ``int32`` is bit-identical to MLX's
    ``uint32`` layout (reinterpreted via ``view``).
    """
    weights_cfg = (
        quantization_config.get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
    )
    group_size = weights_cfg.get("group_size", 32)
    bits = weights_cfg.get("num_bits", 4)
    packed_suffix = ".weight_packed"

    new_weights = {}
    for key in list(weights.keys()):
        if key.endswith(packed_suffix):
            prefix = key[: -len(packed_suffix)]
            scale = weights[f"{prefix}.weight_scale"]
            new_weights[f"{prefix}.weight"] = weights[key].view(mx.uint32)
            new_weights[f"{prefix}.scales"] = scale
            new_weights[f"{prefix}.biases"] = -(2 ** (bits - 1)) * scale
        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            # Consumed alongside their ``.weight_packed`` (shape is unused).
            continue
        else:
            new_weights[key] = weights[key]

    return new_weights, {"group_size": group_size, "bits": bits, "mode": "affine"}


def _transform_compressed_tensors_weights(
    weights: Dict[str, mx.array],
    quantization_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, mx.array], Optional[Dict[str, Any]]]:
    """Transform compressed-tensors weights before model-specific sanitization.

    This runs before MoE expert stacking so per-expert tensors have already been
    renamed to MLX-native ``.weight`` / ``.scales`` / ``.biases`` keys.
    """
    if quantization_config is None:
        return weights, None
    if quantization_config.get("quant_method") != "compressed-tensors":
        return weights, None
    if not any(key.endswith(".weight_packed") for key in weights):
        return weights, None

    weights_cfg = (
        quantization_config.get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
    )
    ct_format = quantization_config.get("format") or quantization_config.get(
        "config_groups", {}
    ).get("group_0", {}).get("format")
    quant_type = weights_cfg.get("type")

    if ct_format == "nvfp4-pack-quantized":
        return _transform_compressed_tensors_nvfp4_weights(
            weights, quantization_config
        ), {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    if ct_format == "pack-quantized" and quant_type == "int":
        return _transform_compressed_tensors_int4_weights(weights, quantization_config)

    return weights, None


def _quant_scales_key(weight_key: str) -> str:
    """Return the scales key for a terminal ``.weight`` tensor key."""
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected a .weight tensor key, got {weight_key!r}")
    return weight_key.rsplit(".weight", 1)[0] + ".scales"


def quantize_activations(model: nn.Module) -> nn.Module:

    def _maybe_qq(m: nn.Module) -> nn.Module:
        """Convert a QuantizedLinear layer to QQLinear if compatible."""
        if isinstance(m, nn.QuantizedLinear):
            if m.mode not in ACTIVATION_QUANTIZATION_MODES:
                raise ValueError(
                    f"Mode ({m.mode}) does not support activation quantization. "
                    f"Supported modes: {', '.join(ACTIVATION_QUANTIZATION_MODES)}"
                )
            if m.get("bias", False):
                raise ValueError(
                    "Linear layer with bias does not support activation quantization"
                )
            # Get dimensions from quantized weight
            out_dims, in_dims = m.weight.shape
            in_dims *= 32 // m.bits
            qq = nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            qq.quantize()
            return qq
        return m

    leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)

    return model


def skip_multimodal_module(path: str) -> bool:
    """
    Check if a multimodal module (vision/audio) should skip quantization.

    Args:
        path: The module path to check

    Returns:
        bool: True if the module is multimodal and should skip quantization, False otherwise
    """
    multimodal_modules = (
        "vision_model",
        "vision_tower",
        "vl_connector",
        "sam_model",
        "audio_model",
        "audio_tower",
        "code_predictor",
        "img_projector",
        "multi_modal_projector",
    )
    return any(module in path for module in multimodal_modules)


def _has_quantized_weights(path: str, weights: Optional[Dict[str, mx.array]]) -> bool:
    return weights is not None and f"{path}.scales" in weights


def get_class_predicate(skip_vision=False, weights=None, quantization_config=None):
    def predicate(p, m):
        if (
            skip_multimodal_module(p)
            and skip_vision
            and not _has_quantized_weights(p, weights)
        ):
            return False
        if quantization_config is not None and p in quantization_config:
            return quantization_config[p]
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False
        if weights is not None:
            return f"{p}.scales" in weights
        return True

    return predicate


def get_model_and_args(config: dict):
    """
    Retrieve the model object based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    raw_model_type = config.get("model_type") or config.get("speculators_model_type")
    if raw_model_type is None:
        raise KeyError("model_type")
    model_type = raw_model_type.lower()

    model_type = MODEL_REMAPPING.get(model_type, model_type)

    is_dflash = config.get("dflash_config", None) is not None
    if is_dflash:
        model_type += "_dflash"

    last_err: Optional[ImportError] = None
    for pkg in ("mlx_vlm.models", "mlx_vlm.speculative.drafters"):
        try:
            arch = importlib.import_module(f"{pkg}.{model_type}")
            return arch, model_type
        except ImportError as e:
            if model_type not in str(e):
                raise
            last_err = e
            continue

    if _is_text_only_config(config):
        arch = importlib.import_module("mlx_vlm.models.text_only")
        return arch, "text_only"

    msg = f"Model type {model_type} not supported. Error: {last_err}"
    logging.error(msg)
    raise ValueError(msg)


def _has_config(config: dict, key: str) -> bool:
    value = config.get(key)
    return value is not None and value != {}


def _is_text_only_config(config: dict) -> bool:
    return not any(
        _has_config(config, key)
        for key in ("vision_config", "audio_config", "dflash_config")
    )


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    force_download: bool = False,
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=allow_patterns
                or [
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jinja",
                ],
                force_download=force_download,
            )
        )
    return model_path


def load_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
        quantize_activations (bool, optional): If True, convert QuantizedLinear layers
            to QQLinear layers for activation quantization. Only supported for models
            quantized with 'nvfp4' or 'mxfp8' modes. Default: ``False``.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    requested_weight_files = kwargs.pop("weight_files", None)
    load_weights = kwargs.pop("load_weights", True)
    config = load_config(model_path, **kwargs)
    if load_weights and config.get("model_type") == "mimo_v2":
        from .models.mimo_v2.raw_loader import is_native, load_native

        if is_native(config):
            if requested_weight_files is not None:
                raise ValueError("Native MiMo needs tensor-level rank loading, not shard-file selection")
            return load_native(model_path)
    if config.get("model_type") == "deepseek_v41":
        config["model_dir"] = str(model_path)
        # update_module_configs rebuilds TextConfig below; preserve runtime fields.
        config["text_config"] = dict(config.get("text_config") or {})
        config["text_config"]["model_dir"] = config["model_dir"]
        if config.get("engram_source") is not None:
            config["text_config"]["engram_source"] = config["engram_source"]

    # Find all .safetensors files in the model_path, excluding consolidated model weights
    all_weight_files = sorted(
        wf
        for wf in glob.glob(str(model_path / "*.safetensors"))
        if not wf.endswith("consolidated.safetensors")
    )

    if load_weights and not all_weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weight_files = all_weight_files
    if requested_weight_files is not None:
        requested_names = {Path(wf).name for wf in requested_weight_files}
        weight_files = [
            wf for wf in all_weight_files if Path(wf).name in requested_names
        ]
        missing_weight_files = requested_names - {Path(wf).name for wf in weight_files}
        if missing_weight_files:
            raise FileNotFoundError(
                "Requested safetensors shards were not found in "
                f"{model_path}: {sorted(missing_weight_files)}"
            )

    weights = {}
    if load_weights:
        for wf in weight_files:
            weights.update(_load_safetensors(wf))

    # inferencerlabs -INF checkpoints store OBFUSCATED tensor keys (base64 of the real name XOR a fixed
    # per-checkpoint keystream) with no native safetensors index. Decode the keys back to real weight
    # names so the standard sanitize/quantize/load path sees the expected namespace, and flag the
    # per-tensor mixed quantizer (INF carries bits AND group_size per tensor; config.json has none of it).
    # Without this, a rank-local INF load gets thousands of unmatched obfuscated keys (extra=…/missing=…)
    # -> a broken graph. (The ring SHIP path already decodes via _InfShardReader; this is its local-load
    # analogue, needed for rank 0's OWN layers in the 2-Mac pipeline.)
    if load_weights and weights and inf_decode.is_inf_checkpoint(model_path):
        keystream = inf_decode.index_keystream(model_path)
        if keystream is None:
            keystream = inf_decode.decode_keystream(model_path, config)
        name_map = inf_decode.build_name_map(model_path, keystream)
        weights = {name_map.get(k, k): v for k, v in weights.items()}
        config["_inf_mixed_quant"] = True

    if all_weight_files:
        with safetensors.safe_open(all_weight_files[0], framework="np") as f:
            is_mlx_format = f.metadata() and f.metadata().get("format") == "mlx"
    else:
        # Weightless graph (load_weights=False, e.g. a pipeline rank that RECEIVES
        # shipped weights over the ring): there is no local shard to probe, so infer
        # the MLX format from the config (MLX exports carry a quantization block).
        is_mlx_format = ("quantization" in config) or ("quantization_config" in config)

    model_class, _ = get_model_and_args(config=config)

    # Initialize text and vision configs if not present
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    has_quantization = "quantization" in config

    # Initialize model config and update it with module configs
    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)

    model = model_class.Model(model_config)

    quantization_config = config.get("quantization_config", None)
    if quantization_config is None:
        quantization_config = config.get("text_config", {}).get(
            "quantization_config", None
        )
        if quantization_config is not None:
            config["quantization_config"] = quantization_config

    weights, transformed_quantization = _transform_compressed_tensors_weights(
        weights, quantization_config
    )
    if transformed_quantization is not None:
        config["quantization"] = transformed_quantization
        config["quantization_config"] = transformed_quantization

    # MLX-format checkpoints normally SKIP sanitize (their keys are already in MLX
    # layout). But a VLM-wrapper bridge (deepseek_v4, glm_moe_dsa, mimo_v2, minimax_*)
    # owns a `language_model.` parameter prefix that the upstream text-model checkpoint
    # keys lack — its `sanitize` is a remap that adds that prefix (and is idempotent /
    # no-op for already-sanitized weights). If we don't run it, `load_weights(strict=False)`
    # silently drops EVERY `model.*` / `lm_head.*` key (none match the `language_model.model.*`
    # params) -> a fully random-init model -> garbage generation (looks like an "arch bug").
    # Detect the prefix mismatch directly so this covers every bridged arch, not a hardcoded
    # list. (True multimodal MLX exports already ship the `language_model.` prefix -> skipped;
    # plain text-only MLX models aren't wrapper-prefixed -> skipped.)
    if is_mlx_format and hasattr(model, "sanitize"):
        _sample_pname = next((k for k, _ in tree_flatten(model.parameters())), "")
        _sample_wkey = next(iter(weights), "") if weights else ""
        _needs_wrapper_sanitize = _sample_pname.startswith(
            "language_model."
        ) and not _sample_wkey.startswith("language_model.")
        # GLM-DSA MTP sidecars are MLX-format but their checkpoint keys (mtp_block.*,
        # shared_head_norm.*) do not match the runtime module tree either, so the prefix test
        # above misses them and they load as random init. Detect them explicitly.
        _needs_glm_mtp_runtime_sanitize = config.get("model_type") == "glm_moe_dsa_mtp" and any(
            key.startswith("mtp_block.") or key.startswith("shared_head_norm.")
            for key in weights
        )
        if _needs_wrapper_sanitize or _needs_glm_mtp_runtime_sanitize:
            logging.info(
                "MLX-format checkpoint (model_type=%s): running sanitize so "
                "checkpoint keys match the runtime module tree.",
                config.get("model_type"),
            )
            weights = sanitize_weights(model, weights)

    if not is_mlx_format:
        # Sanitize weights
        weights = sanitize_weights(model, weights)

        if hasattr(model, "thinker") and hasattr(model.thinker, "sanitize"):
            weights = sanitize_weights(model.thinker, weights)
            weights = sanitize_weights(model.thinker.vision_tower, weights)
            weights = sanitize_weights(model.thinker.audio_tower, weights)
            weights = sanitize_weights(model.thinker.language_model, weights)
            weights = sanitize_weights(model.code2wav, weights)
            weights = sanitize_weights(model.talker, weights)
        else:
            if hasattr(model_class, "VisionModel"):
                weights = sanitize_weights(
                    model_class.VisionModel, weights, model_config.vision_config
                )
            if hasattr(model_class, "LanguageModel"):
                weights = sanitize_weights(
                    model_class.LanguageModel, weights, model_config.text_config
                )
            if hasattr(model_class, "AudioModel"):
                weights = sanitize_weights(
                    model_class.AudioModel, weights, model_config.audio_config
                )

    if not has_quantization:
        quantization_config = config.get("quantization_config", None)
        if quantization_config is None:
            text_config = config.get("text_config", {})
            quantization_config = text_config.get("quantization_config", None)
            if quantization_config is not None:
                config["quantization_config"] = quantization_config

        if quantization_config is not None:
            quant_method = quantization_config.get("quant_method")
            quantization = None
            if quant_method == "compressed-tensors":
                quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            elif quant_method == "mxfp4":
                quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            elif quant_method == "fp8" and config.get("model_type") == "deepseek_v4":
                from .models.deepseek_v4.language import make_quantization_config

                quantization = make_quantization_config(model)
            elif quant_method == "fp8" and config.get("model_type") == "deepseek_v41":
                from .models.deepseek_v41.checkpoint_layout import (
                    make_quantization_config_v41,
                )

                quantization = make_quantization_config_v41(model)
            elif quant_method in ("awq", "gptq", "bitnet"):
                logging.warning(
                    "Quantization method %s is not supported in mlx_vlm.load_model()",
                    quant_method,
                )

            if quantization is not None:
                config["quantization"] = quantization
                config["quantization_config"] = quantization

    if has_quantization:
        for quantization_key in ("quantization", "quantization_config"):
            quantization_value = getattr(model_config, quantization_key, None)
            if quantization_value is not None:
                config[quantization_key] = quantization_value

    quantization = config.get("quantization", None)
    # Treat empty dict as None (some models set quantization: {})
    if quantization is not None and not quantization:
        quantization = None
    if quantization is not None:
        # Handle legacy models which may or may not have vision quantized
        # TODO: Re-upload the models with the new quantization config and remove this
        skip_vision = config.get("vision_config", {}).get("skip_vision", False)
        quantized_model = (
            model.language_model._model
            if getattr(model, "_is_text_model", False)
            else model
        )

        def get_class_predicate(p, m):
            # Skip legacy multimodal layers unless the checkpoint has quantized
            # tensors for this exact module.
            if (
                skip_multimodal_module(p)
                and skip_vision
                and not _has_quantized_weights(p, weights)
            ):
                return False
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            quantized_model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    # Handle per-tensor mixed quantization (e.g. inferencerlabs format).
    # Detect packed weights from matching .weight/.scales tensors and quantize
    # those modules exactly once per compatible shape group.
    quantization_config = config.get("quantization_config", {})
    group_size = quantization_config.get("group_size", 64)
    has_quant_weights = any(k.endswith(".scales") for k in weights)
    inf_mixed = config.get("_inf_mixed_quant", False)
    # Plain-key SANITIZED INF (inferencerlabs *-INF re-exported with de-obfuscated keys, e.g. GLM-5.1 /
    # MiMo-V2.5-Pro: config carries "mlx-sanitized" + an empty quantization_config, and the keys are real
    # HF names so is_inf_checkpoint() is False and _inf_mixed_quant was never set). The weights are STILL
    # per-tensor MIXED quant (bits AND group_size vary per tensor; config.json has none of it). The generic
    # per-layer inference below mis-derives bits for 3D SwitchLinear MoE experts (e.g. 12 bits -> ValueError);
    # quant_groups_from_model handles 2D Linear AND 3D SwitchLinear correctly. A model.inf sidecar is the
    # inferencerlabs fingerprint (no plain MLX export ships one), so it safely gates this route.
    if not inf_mixed and has_quant_weights and (model_path / "model.inf").exists():
        inf_mixed = True
        config["_inf_mixed_quant"] = True
    if has_quant_weights and inf_mixed:
        # INF per-tensor mixed quant: bits AND group_size vary per tensor (recovered from the model
        # graph; config.json carries none of it). One nn.quantize per distinct (bits, group_size) combo.
        # Local-load analogue of the ring SHIP path's _ship_quant_groups_from_disk + _apply_ship_quant.
        _inf_groups = inf_decode.quant_groups_from_model(model, weights)
        logging.info(
            "INF per-tensor mixed quantization: %d layers across %d (bits,group_size) combos",
            sum(len(v) for v in _inf_groups.values()), len(_inf_groups))
        for (_bits, _gsize, _mode), _layer_set in sorted(_inf_groups.items()):
            def _inf_predicate(p, m, _ls=_layer_set):
                names = (p, f"language_model.{p}", p.removeprefix("language_model."))
                return any(name in _ls for name in names) and hasattr(m, "to_quantized")
            nn.quantize(model, group_size=_gsize, bits=_bits, mode=_mode, class_predicate=_inf_predicate)
        # Stash the EXACT applied groups (as JSON, so it's never mistaken for a param-tree dict) so the
        # weight-ship holder can send a manifest that quantizes the weightless receiver BYTE-IDENTICALLY.
        model._ship_quant_manifest_json = json.dumps(
            {"groups": [[list(k), sorted(v)] for k, v in _inf_groups.items()]})
    elif has_quant_weights and group_size:
        # Infer bits per layer from (packed_dim, n_groups). This works for
        # both 2D Linear weights and 3D SwitchLinear MoE expert weights.
        quant_groups = {}
        # A FLAT quantization_config (e.g. GLM-5.2-mxfp4: {group_size:32, bits:4, mode:"mxfp4"})
        # carries the real quant mode at the TOP LEVEL, NOT under per-layer-path keys. Falling back
        # to a hardcoded "affine" below silently re-dequantizes mxfp4 tensors with the affine codebook
        # -> garbage decode. Use the global mode (quantization_config, else config["quantization"]) as
        # the fallback; absent any top-level mode this stays "affine" (prior behavior preserved).
        _global_quant_mode = (
            quantization_config.get("mode")
            or (config.get("quantization") or {}).get("mode")
            or "affine"
        )
        _module_in_dim = {}
        for _mp, _ma in tree_flatten(model.parameters()):
            if _mp.endswith(".weight") and hasattr(_ma, "shape") and len(_ma.shape) >= 2:
                _module_in_dim[_mp.rsplit(".weight", 1)[0]] = int(_ma.shape[-1])
        for wk in weights:
            if not wk.endswith(".weight"):
                continue
            sk = _quant_scales_key(wk)
            if sk not in weights:
                continue
            w_arr = weights[wk]
            s_arr = weights[sk]
            if w_arr.dtype != mx.uint32 or w_arr.ndim not in (2, 3):
                continue
            # For affine quant: packed_last = input_dim * bits / 32,
            # and scales_last = input_dim / group_size.
            # So bits = packed_last * 32 / (scales_last * group_size).
            # 2026-09-10: the group size is PER MODULE. DeepSeek-V4-Flash-0731-MLX carries a global
            # group_size 64 plus 494 per-module overrides {group_size: 32, mode: mxfp4|mxfp8}; using
            # the global 64 halved every derived bit width (4-bit mxfp4 -> "2-bit mxfp4 @ 64") and
            # mx.quantize refused ("mxfp4 quantization requires group size 32 but got 64"). Take the
            # override's group_size, and when the module is in the graph, prefer the exact value
            # input_dim / scales_last (the same derivation inf_decode.quant_groups_from_model uses).
            layer_path = wk.rsplit(".weight", 1)[0]
            cfg = quantization_config.get(
                layer_path.removeprefix("language_model."), {}
            )
            if not isinstance(cfg, dict):
                cfg = {}
            gs = int(cfg.get("group_size", group_size) or group_size)
            _in_dim = _module_in_dim.get(layer_path) or _module_in_dim.get(
                "language_model." + layer_path) or _module_in_dim.get(layer_path.removeprefix("language_model."))
            if _in_dim and _in_dim % int(s_arr.shape[-1]) == 0 and (_in_dim // int(s_arr.shape[-1])) in (32, 64, 128):
                gs = _in_dim // int(s_arr.shape[-1])
            input_dim = s_arr.shape[-1] * gs
            bits = round(w_arr.shape[-1] * 32 / input_dim)
            mode = cfg.get("mode", _global_quant_mode)
            quant_groups.setdefault((bits, gs, mode), set()).add(layer_path)

        logging.info(
            "Detected per-tensor mixed quantization (config group_size=%d): "
            "%d quantized layers across %d (bits, group_size, mode) combos",
            group_size,
            sum(len(v) for v in quant_groups.values()),
            len(quant_groups),
        )

        for (bits, gs, mode), layer_set in sorted(quant_groups.items()):
            logging.info(
                "  Converting %d modules to %d-bit %s (group_size=%d)",
                len(layer_set),
                bits,
                mode,
                gs,
            )

            def _predicate(p, m, _layer_set=layer_set):
                names = (p, f"language_model.{p}", p.removeprefix("language_model."))
                return any(name in _layer_set for name in names) and hasattr(
                    m, "to_quantized"
                )

            nn.quantize(
                model,
                group_size=gs,
                bits=bits,
                mode=mode,
                class_predicate=_predicate,
            )
        # Stash the EXACT applied groups (per-tensor bits/group_size/mode) so the weight-ship
        # holder sends the real manifest (all quantized modules) -> receiver quantizes byte-identically.
        model._ship_quant_manifest_json = json.dumps({"groups": [
            [[bits, gs, mode], sorted(p.removeprefix("language_model.") for p in layer_set)]
            for (bits, gs, mode), layer_set in quant_groups.items()]})

    if kwargs.get("quantize_activations", False):
        if quantization is None:
            raise ValueError(
                "Activation quantization requires the model to be quantized first. "
                "Please use a quantized model with mode 'nvfp4' or 'mxfp8'."
            )
        model = quantize_activations(model)

    weight_list = list(weights.items())

    # Detect upfront if strict=False is needed to avoid double-loading
    # (which doubles peak memory — fatal for 400GB+ models).
    def _collect_param_names(module, prefix=""):
        names = set()
        for k, v in module.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                names |= _collect_param_names(v, full)
            else:
                names.add(full)
        return names

    try:
        if load_weights:
            param_names = _collect_param_names(model.parameters())
            weight_names = set(n for n, _ in weight_list)
            extra_weights = weight_names - param_names
            missing_weights = param_names - weight_names
            needs_non_strict = len(extra_weights) > 10 or len(missing_weights) > 10
        else:
            needs_non_strict = False
            extra_weights = set()
            missing_weights = set()
    except Exception:
        needs_non_strict = False
        extra_weights = set()
        missing_weights = set()

    if needs_non_strict:
        logging.info(
            "Using strict=False for weight loading (extra=%d missing=%d)",
            len(extra_weights), len(missing_weights),
        )
        if extra_weights:
            sample = list(extra_weights)[:5]
            logging.info("  Sample extra keys: %s", sample)
        if missing_weights:
            sample = list(missing_weights)[:5]
            logging.info("  Sample missing keys: %s", sample)

    if load_weights:
        model.load_weights(weight_list, strict=not needs_non_strict)

    if load_weights and not lazy:
        mx.eval(model.parameters())

    model.model_path = model_path
    model.eval()
    if load_weights and hasattr(model, "attach_engram_storage"):
        try:
            model.attach_engram_storage(model_path)
        except Exception:
            logging.exception("Failed to attach Engram storage from %s", model_path)
            raise
    return model


def _load_safetensors(path: str) -> dict:
    try:
        return mx.load(path)
    except RuntimeError as e:
        # A Q4i container XOR-masks the safetensors HEADER, so mx.load cannot parse it and reports a
        # JSON error at byte 0 ("[json.exception.parse_error.101] ... last read: '<U+0012>'"). The
        # tensor DATA is plain, so inf_decode reads the header and maps the bytes directly. Tried
        # only after mx.load has failed, so an ordinary checkpoint keeps the native path untouched.
        if "parse_error" in str(e) or "json.exception" in str(e):
            try:
                return inf_decode.load_masked_safetensors(path)
            except Exception:
                raise e
        if not any(dtype in str(e) for dtype in SAFETENSORS_DTYPE_FALLBACKS):
            raise
        load_error = e

    with open(path, "r+b") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        original_header = f.read(header_len)
        header = json.loads(original_header)
        changed = False

        for tensor_info in header.values():
            if not isinstance(tensor_info, dict):
                continue
            dtype = tensor_info.get("dtype")
            if dtype in SAFETENSORS_DTYPE_FALLBACKS:
                tensor_info["dtype"] = SAFETENSORS_DTYPE_FALLBACKS[dtype]
                changed = True

        if not changed:
            raise load_error

        patched_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(patched_header) > header_len:
            raise RuntimeError(
                f"Cannot reinterpret unsupported safetensors dtype in {path}: "
                "patched header is larger than the original header."
            )

        try:
            f.seek(8)
            f.write(patched_header)
            f.write(b" " * (header_len - len(patched_header)))
            f.flush()
            return mx.load(path)
        finally:
            f.seek(8)
            f.write(original_header)
            f.flush()


def _lookup_sharded_weight_file(weight_index: Dict[str, str], key: str) -> Optional[str]:
    for candidate in (key, key.removeprefix("language_model.")):
        file_name = weight_index.get(candidate)
        if file_name is not None:
            return file_name
    return None


def sanitize_weights(model_obj, weights, config=None):
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        weights = model_obj.sanitize(weights)
    return weights


def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr) and config.get(config_attr) is not None:
            config_class = getattr(model_class, f"{config_name.title()}Config")
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config


def load(
    path_or_hf_repo: str,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    revision: Optional[str] = None,
    strict: bool = True,
    **kwargs,
) -> Tuple[nn.Module, ProcessorMixin]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``.
        quantize_activations (bool, optional): If True, convert QuantizedLinear layers
            to QQLinear layers for activation quantization. Only supported for models
            quantized with 'nvfp4' or 'mxfp8' modes. Default: ``False``.

    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    force_download = kwargs.get("force_download", False)
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
    model = load_model(model_path, lazy, strict=strict, **kwargs)
    if adapter_path is not None:
        model = apply_lora_layers(model, adapter_path)
        model.eval()

    image_processor = load_image_processor(model_path, **kwargs)

    # Get the eos_token_id from the model config
    eos_token_id = getattr(model.config, "eos_token_id", None)

    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)

    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def _weight_ship_enabled() -> bool:
    """Weight-shipping mode: the holder rank (rank 0) reads every *remote* rank's
    layer shards from its own disk, decodes/quantizes them, and ships the
    materialised parameter tensors over the distributed ring into the remote
    rank's unified memory. Remote ranks load **no weights from local disk** — they
    only need the tiny config/tokenizer metadata to build the graph.

    Gated so the default (every rank reads its own local shards) is unchanged for
    same-disk / shared-storage setups. Enable with ``MLX_VLM_WEIGHT_SHIP=1`` or the
    server flag ``--weight-source ship-from-rank0``."""
    return os.environ.get("MLX_VLM_WEIGHT_SHIP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "ship-from-rank0",
    }


def _pipeline_local_files(model, inner, model_path, config, pipeline_group):
    """The set of safetensors shard basenames whose tensors back THIS pipeline
    rank's live parameters. Mirrors the index lookup in :func:`sharded_load`."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists() and inf_decode.is_inf_checkpoint(model_path):
        index_path = inf_decode.write_index_json(model_path, config)
    if not index_path.exists():
        raise ValueError(
            "Pipeline loading requires model.safetensors.index.json for this "
            f"checkpoint. {model_path} has no index; INF/encoded checkpoints "
            "need a name-decode index pass before distributed loading."
        )
    with open(index_path, "r", encoding="utf-8") as fid:
        weight_index = json.load(fid)["weight_map"]
    local_files = set()
    flat_params = tree_flatten(model.parameters())
    if not flat_params:
        indexed_files = set(weight_index.values())
        if len(indexed_files) == 1:
            return indexed_files
        raise ValueError(
            "Pipeline loading could not infer rank-local safetensors files from "
            "the weightless graph. This fallback is only safe for single-file "
            f"checkpoints; {model_path} index references {len(indexed_files)} files."
        )
    for key, _ in flat_params:
        file_name = _lookup_sharded_weight_file(weight_index, key)
        if file_name is None:
            raise ValueError(
                "Pipeline loading is only supported for indexed MLX models; "
                f"missing safetensors index entry for {key!r}."
            )
        local_files.add(file_name)
    return local_files


def _materialize_pipeline_rank_model(model_path, target_rank, pipeline_group):
    """Build a fully-quantized, weight-loaded model on the holder rank that holds
    **only** ``target_rank``'s pipeline layers — read transiently from local disk.

    Two-pass like :func:`sharded_load`: (1) lazy no-weight graph, pipeline-split
    for ``target_rank`` so the off-rank layers become ``None``; (2) compute the
    shard files backing the surviving params and re-load only those, applying the
    INF decode + per-tensor mixed quantization. The off-rank layers never
    materialise, so the holder never needs the whole model resident."""

    class _SplitForRank:
        """A view of ``pipeline_group`` that reports ``target_rank`` so the model's
        ``pipeline()`` selects the remote rank's layer slice on the holder."""

        def __init__(self, group, rank):
            self._group = group
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return self._group.size()

    split = _SplitForRank(pipeline_group, target_rank)

    skel = load_model(model_path, lazy=True, strict=False, load_weights=False)
    config = skel.config.to_dict()
    inner = _pipeline_target(skel)
    inner.pipeline(split)
    local_files = _pipeline_local_files(skel, inner, model_path, config, pipeline_group)
    del skel, inner

    model = load_model(model_path, lazy=True, strict=False, weight_files=local_files)
    inner = _pipeline_target(model)
    inner.pipeline(split)
    return model


def _lazy_pipeline_rank_graph(model_path, target_rank, pipeline_group):
    """Build the WEIGHTLESS pipeline-split graph for ``target_rank`` on the holder —
    just the architecture (off-rank layers ``None``), no disk weight reads. Used by
    the streaming ship path to derive the quant manifest + send plan without the
    expensive eager materialize that previously preceded the first ring send."""

    class _SplitForRank:
        def __init__(self, group, rank):
            self._group = group
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return self._group.size()

    split = _SplitForRank(pipeline_group, target_rank)
    model = load_model(model_path, lazy=True, strict=False, load_weights=False)
    inner = _pipeline_target(model)
    inner.pipeline(split)
    return model


def _shipped_param_plan(model):
    """Deterministic, both-ranks-identical transfer order for a pipeline-split
    model's live parameters: ``[(path, shape, dtype_str), ...]`` sorted by path.

    Only real (non-``None``) layers contribute params, so a pipeline-split model
    yields exactly its rank's slice."""
    flat = tree_flatten(model.language_model.parameters())
    plan = []
    for path, arr in flat:
        plan.append((path, list(arr.shape), str(arr.dtype).split(".")[-1]))
    plan.sort(key=lambda t: t[0])
    return plan


class _InfShardReader:
    """On-demand single-tensor reader for an INF (inferencerlabs) checkpoint.

    The INF shards already store the *quantized* tensors (``.weight`` packed as
    ``uint32`` plus ``bfloat16`` ``.scales``/``.biases``), so streaming a remote
    rank's layers needs no re-quantization compute — only a per-tensor disk read.

    This reader decodes the XOR keystream once, indexes real-name -> (shard, byte
    range, on-disk dtype/shape) from the shard headers, and exposes
    :meth:`read` which seeks+reads *just one tensor's bytes* (never the whole
    shard) and reconstructs an ``mx.array`` (bf16 via a ``uint16`` bit-cast, which
    ``numpy`` cannot represent natively). Headers + the keystream are tiny; the big
    shard bytes are touched only when a tensor is read, so rank0 never holds the
    whole 78 GB resident while shipping."""

    _NP_DTYPE = {
        "U8": np.uint8,
        "U16": np.uint16,
        "U32": np.uint32,
        "U64": np.uint64,
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
    }
    # safetensors dtype tag -> mlx dtype for the types numpy cannot represent.
    _MX_BITCAST = {"BF16": (np.uint16, mx.bfloat16)}

    def __init__(self, model_path, config):
        self.model_path = Path(model_path)
        keystream = inf_decode.decode_keystream(self.model_path, config)
        # real-name -> (shard_path, data_start, data_end, dtype_tag, shape)
        self._index = {}
        for shard in sorted(self.model_path.glob("model-*.safetensors")):
            with open(shard, "rb") as fh:
                (hdr_len,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hdr_len))
            data_base = 8 + hdr_len
            name_map = inf_decode.build_name_map(
                self.model_path, keystream, shard=shard
            )
            for obf, info in header.items():
                if obf == "__metadata__" or not isinstance(info, dict):
                    continue
                real = name_map.get(obf)
                if real is None:
                    continue
                s, e = info["data_offsets"]
                self._index[real] = (
                    shard,
                    data_base + s,
                    data_base + e,
                    info["dtype"],
                    list(info["shape"]),
                )

    def has(self, real_name):
        return real_name in self._index

    def read(self, real_name, target_shape, target_dtype):
        """Read one tensor's raw bytes from its shard and return an ``mx.array``
        cast to ``target_dtype`` and reshaped to ``target_shape`` (the receiver's
        graph param). Element counts are guaranteed equal to the on-disk tensor, so
        the reshape only re-groups (e.g. ``wo_a`` 2D -> 3D); the cast handles the
        bf16->f32 scales/biases and int64->int32 routing-table conversions."""
        shard, start, end, dtype_tag, _shape = self._index[real_name]
        with open(shard, "rb") as fh:
            fh.seek(start)
            raw = fh.read(end - start)
        if dtype_tag in self._MX_BITCAST:
            np_view_dtype, mx_dtype = self._MX_BITCAST[dtype_tag]
            arr = mx.array(np.frombuffer(raw, dtype=np_view_dtype)).view(mx_dtype)
        else:
            np_dtype = self._NP_DTYPE.get(dtype_tag)
            if np_dtype is None:
                raise ValueError(
                    f"INF stream: unsupported on-disk dtype {dtype_tag!r} for "
                    f"{real_name!r}"
                )
            arr = mx.array(np.frombuffer(raw, dtype=np_dtype))
        return arr.astype(target_dtype).reshape(target_shape)


def _ship_quant_groups_from_disk(model, model_path, config):
    """Derive the per-tensor ``(bits, group_size, mode)`` -> sorted layer-path
    manifest for a *weightless* pipeline-split graph by reading only the INF shard
    *headers* (packed/scales last dims), with no eager weight materialisation.

    For each live ``*.weight`` param whose checkpoint tensor is ``uint32`` (i.e.
    quantized on disk) we recover the module's true input dim from the graph and
    compute ``bits = packed_last*32/in_dim`` and ``group_size = in_dim/scales_last``
    — the exact pair the receiver must apply with :func:`nn.quantize`."""
    reader = _InfShardReader(model_path, config)
    module_in_dim = {}
    for path, arr in tree_flatten(model.language_model.parameters()):
        if path.endswith(".weight") and len(getattr(arr, "shape", ())) >= 2:
            module_in_dim[path.rsplit(".weight", 1)[0]] = int(arr.shape[-1])
    triples = {}
    for path, _arr in tree_flatten(model.language_model.parameters()):
        if not path.endswith(".weight"):
            continue
        layer_path = path.rsplit(".weight", 1)[0]
        real = layer_path.removeprefix("language_model.") + ".weight"
        info = reader._index.get(real)
        if info is None or info[3] not in ("U32", "uint32"):
            continue
        scales_info = reader._index.get(
            layer_path.removeprefix("language_model.") + ".scales"
        )
        if scales_info is None:
            continue
        in_dim = module_in_dim.get(layer_path) or module_in_dim.get(
            layer_path.removeprefix("language_model.")
        )
        if not in_dim:
            continue
        packed_last = info[4][-1]
        scales_last = scales_info[4][-1]
        bits = round(packed_last * 32 / in_dim)
        group_size = round(in_dim / scales_last)
        triples.setdefault((bits, group_size, "affine"), set()).add(
            layer_path.removeprefix("language_model.")
        )
    return {"groups": [[list(k), sorted(v)] for k, v in triples.items()]}


def _apply_ship_quant_manifest(model, manifest_json):
    """Apply the INF mixed-quant groups from a manifest to a weightless graph so its
    param tree (packed uint32 ``.weight`` + ``.scales``/``.biases``) matches the
    holder's send plan. Shared by rank0 (its own derived manifest) and rank1 (the
    received manifest) so both graphs are quantized identically."""
    quant_groups = {tuple(k): set(v) for k, v in manifest_json["groups"]}
    for (bits, gsize, mode), layer_set in sorted(quant_groups.items()):

        def _predicate(p, m, _layer_set=layer_set):
            names = (p, f"language_model.{p}", p.removeprefix("language_model."))
            return any(name in _layer_set for name in names) and hasattr(
                m, "to_quantized"
            )

        nn.quantize(
            model,
            group_size=gsize,
            bits=bits,
            mode=mode,
            class_predicate=_predicate,
        )
    return model


def _manifest_from_quantized_model(model, config):
    """Derive the ``(bits, group_size, mode)`` -> sorted layer-path manifest from an ALREADY-QUANTIZED
    model — the holder's materialized remote-rank graph in the NON-INF (standard indexed) ship path —
    so a weightless receiver applies the IDENTICAL quantization (matching param tree) before recv.

    PREFERS the EXACT manifest ``load_model`` stashed (``_ship_quant_manifest_json``) — the precise
    groups it quantized from the loaded checkpoint weights (all 203 modules). Re-deriving from the
    post-quant model under-counted (leaf_modules: 144/203; missed nested MoE experts) -> the receiver
    left ~59 layers UNquantized -> shapes diverged -> per-tensor send/recv DESYNC + hang partway through
    shipping. The shape-inspection below is only a fallback if the stash is absent."""
    stashed = getattr(model, "_ship_quant_manifest_json", None)
    if stashed:
        return json.loads(stashed)
    quantization_config = (config or {}).get("quantization_config", {}) or {}
    group_size = quantization_config.get("group_size", 64)
    weights = dict(tree_flatten(model.language_model.parameters()))
    groups: dict = {}
    for wk, w_arr in weights.items():
        if not wk.endswith(".weight"):
            continue
        s_arr = weights.get(_quant_scales_key(wk))
        if s_arr is None or getattr(w_arr, "dtype", None) != mx.uint32 or w_arr.ndim not in (2, 3):
            continue
        layer_path = wk.rsplit(".weight", 1)[0].removeprefix("language_model.")
        cfg = quantization_config.get(layer_path, {})
        if not isinstance(cfg, dict):
            cfg = {}
        # A quantized module records its own group_size/bits; prefer those, then the per-module
        # override, then the global value (2026-09-10: the group size is per module).
        mod = _module_at_path(model, wk.rsplit(".weight", 1)[0])
        gs = int(getattr(mod, "group_size", 0) or cfg.get("group_size", group_size) or group_size)
        input_dim = s_arr.shape[-1] * gs
        bits = int(getattr(mod, "bits", 0) or round(w_arr.shape[-1] * 32 / input_dim))
        mode = str(getattr(mod, "mode", None) or cfg.get("mode", "affine"))
        groups.setdefault((bits, gs, mode), set()).add(layer_path)
    return {"groups": [[list(k), sorted(v)] for k, v in groups.items()]}


def _module_at_path(model, path):
    """Resolve a dotted parameter-module path (e.g. ``language_model.model.layers.3
    .attn.wq_a``) to the nn.Module instance, or None."""
    obj = model
    for part in path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (TypeError, IndexError, KeyError):
                return None
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _ship_weights_for_pipeline(model_path, pipeline_group):
    """rank0 (holder): STREAM every remote rank's layers over the ring with no long
    idle gap (the old eager-materialize-before-first-send caused the ring socket to
    time out -> EPIPE). For each remote rank we:

      1. build the remote rank's LAZY split graph (``load_weights=False`` — fast, no
         disk reads), derive its INF quant manifest from the shard *headers* only,
         and quantize the graph so its param tree matches the receiver's exactly;
      2. send the manifest + tensor count IMMEDIATELY (first ring traffic within ~a
         second of connect — keeps the socket alive);
      3. STREAM each tensor in the agreed path-sorted order: read just that one
         tensor's bytes from its shard on demand, cast/reshape to the graph param,
         ``mx.eval`` it, ``mx.distributed.send`` it, then free it — so the socket
         gets a send every fraction of a second and rank0 never holds the whole
         remote model resident.

    Non-INF indexed checkpoints fall back to the eager materialize path (their
    quant is config-baked and a per-tensor on-disk reader is INF-specific)."""
    size = pipeline_group.size()
    config = load_config(model_path)
    is_inf = inf_decode.is_inf_checkpoint(model_path)
    reader = _InfShardReader(model_path, config) if is_inf else None
    for dst in range(1, size):
        if is_inf:
            # --- True streaming path (INF): manifest first, then per-tensor reads.
            logging.info(
                "WEIGHT_SHIP: building lazy rank %d graph (no eager load)", dst
            )
            rmodel = _lazy_pipeline_rank_graph(model_path, dst, pipeline_group)
            manifest = _ship_quant_groups_from_disk(rmodel, model_path, config)
            _apply_ship_quant_manifest(rmodel, manifest)
            plan = _shipped_param_plan(rmodel)
            # First ring traffic: manifest + count, sent right away to keep the
            # socket alive (no multi-second idle before the first send).
            _send_json(manifest, dst, pipeline_group)
            # Send rank0's AUTHORITATIVE plan (paths/shapes/DTYPES) so the receiver recvs with the
            # holder's EXACT dtypes. rank0's non-INF path materializes a bf16 graph while the receiver
            # is weightless (f32) -> .biases dtype diverged -> byte-stream desync. The receiver's
            # load_weights() then adapts its graph to these tensors. (Replaces the bare count.)
            _send_json({"plan": [[p, list(s), d] for p, s, d in plan]}, dst, pipeline_group)
            logging.info(
                "WEIGHT_SHIP: streaming %d tensors to rank %d", len(plan), dst
            )
            del rmodel
            for i, (path, shape, dtype) in enumerate(plan):
                real = path.removeprefix("language_model.")
                arr = reader.read(real, tuple(shape), getattr(mx, dtype))
                mx.eval(mx.distributed.send(arr, dst, group=pipeline_group))
                del arr
                if (i + 1) % 100 == 0 or i + 1 == len(plan):
                    logging.info(
                        "WEIGHT_SHIP: rank %d streamed %d/%d tensors",
                        dst,
                        i + 1,
                        len(plan),
                    )
            mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
            logging.info("WEIGHT_SHIP: rank %d shipped", dst)
            continue

        # --- Non-INF fallback: eager materialize then per-tensor send. ----------
        logging.info("WEIGHT_SHIP: materializing rank %d layers from local disk", dst)
        rmodel = _materialize_pipeline_rank_model(model_path, dst, pipeline_group)
        # Derive the REAL quant manifest from the materialized (already-quantized) graph so the
        # weightless receiver quantizes IDENTICALLY (empty groups left it unquantized -> tensor-count
        # mismatch for mixed-quant checkpoints like V4-Flash: 560 offered vs 302 plain params).
        manifest = _manifest_from_quantized_model(rmodel, config)
        _send_json(manifest, dst, pipeline_group)
        plan = _shipped_param_plan(rmodel)
        params = dict(tree_flatten(rmodel.language_model.parameters()))
        logging.info("WEIGHT_SHIP: sending %d tensors to rank %d", len(plan), dst)
        # Authoritative plan with the holder's exact dtypes (see INF branch note above).
        _send_json({"plan": [[p, list(s), d] for p, s, d in plan]}, dst, pipeline_group)
        for path, _shape, _dtype in plan:
            arr = params[path]
            mx.eval(mx.distributed.send(arr, dst, group=pipeline_group))
        del rmodel, params
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        logging.info("WEIGHT_SHIP: rank %d shipped", dst)


def _recv_shipped_weights(model, pipeline_group):
    """remote rank: receive the holder's AUTHORITATIVE plan (paths/shapes/dtypes), then receive each
    parameter array with the holder's EXACT shape+dtype and load it into the graph. Using rank0's plan
    (not the local graph's) keeps the byte stream aligned even when the two ranks' graphs differ in a
    param's dtype (rank0 non-INF materialize = bf16 vs this rank's weightless build = f32 on .biases).
    ``load_weights(strict=False)`` then adapts the local graph's params to the received arrays; any
    local-graph param the holder didn't ship keeps its config-built value."""
    rank = pipeline_group.rank()
    plan = _recv_json(0, pipeline_group).get("plan", [])
    logging.info("WEIGHT_SHIP rank %d: receiving %d tensors from rank 0", rank, len(plan))
    received = []
    for i, (path, shape, dtype) in enumerate(plan):
        arr = mx.distributed.recv(tuple(shape), getattr(mx, dtype), 0, group=pipeline_group)
        mx.eval(arr)  # incremental: one Metal command buffer per tensor (watchdog-safe)
        received.append((path, arr))
        if (i + 1) % 100 == 0 or i + 1 == len(plan):
            logging.info(
                "WEIGHT_SHIP rank %d: received %d/%d tensors", rank, i + 1, len(plan)
            )
    model.language_model.load_weights(received, strict=False)
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    logging.info("WEIGHT_SHIP rank %d: %d tensors received into unified memory", rank, len(received))


# Number of heartbeat collectives the holder (rank 0) emits while it builds +
# materializes its OWN (large) local slice after the ship. Every remote rank runs
# exactly this many matching heartbeats so the ring sees continuous lockstep
# traffic instead of one long one-sided idle (the post-ship EPIPE window). Fixed
# constant => no count to negotiate; both sides agree by construction.
_WEIGHT_SHIP_HEARTBEATS = 64


def _materialize_with_heartbeats(language_model, pipeline_group):
    """rank0: materialize the holder's own slice in ``_WEIGHT_SHIP_HEARTBEATS``
    chunks, emitting one tiny ``all_sum`` heartbeat between chunks. The remote
    ranks call :func:`_heartbeat_wait` with the SAME count, so the ring never
    idles longer than a single chunk's eval (seconds) while rank0 does its big
    (tens-of-seconds) materialize. Without this the remote rank reaches the final
    barrier immediately and the socket idles for the whole materialize -> the
    holder's next ring send hits EPIPE (errno 32)."""
    params = [v for _, v in tree_flatten(language_model.parameters())]
    n = len(params)
    hb = _WEIGHT_SHIP_HEARTBEATS
    for c in range(hb):
        chunk = params[c * n // hb : (c + 1) * n // hb]
        if chunk:
            mx.eval(chunk)
        # Heartbeat: keep the ring warm + stay in lockstep with the remote ranks.
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    # Materialize any params the integer-split chunking left over (n < hb, etc.).
    mx.eval(language_model.parameters())


def _heartbeat_wait(pipeline_group):
    """Remote rank: mirror rank0's ``_WEIGHT_SHIP_HEARTBEATS`` materialize
    heartbeats so the ring stays in continuous lockstep while the holder builds
    its own slice. Pure collectives — no local work."""
    for _ in range(_WEIGHT_SHIP_HEARTBEATS):
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))


def _quantize_pipeline_graph_for_ship(model_path, model, pipeline_group):
    """Apply the holder's INF per-tensor mixed quantization to a *weightless*
    pipeline-split graph so its param tree (packed uint32 ``.weight`` plus
    ``.scales``/``.biases``) matches what the holder ships.

    CRUCIAL: a remote rank in ship mode holds **no weight shards** — only the tiny
    config/index/tokenizer metadata — so ``is_inf_checkpoint(model_path)`` is False
    here (it keys on obfuscated shard tensor names, which are absent). The decision
    to recv+apply the manifest must therefore NOT depend on the remote rank's own
    files: the holder (rank 0) ALWAYS sends a manifest first (empty ``groups`` for
    non-INF indexed checkpoints), and every remote rank ALWAYS recvs and applies it.
    Identical quantize on both ranks (same manifest groups) => identical send/recv
    plans. rank 0 derives the manifest from its own shards and ships it in
    :func:`_ship_weights_for_pipeline`, so nothing to do for rank 0 here."""
    rank = pipeline_group.rank()
    if rank == 0:
        return model
    # Remote rank: receive the quant-group manifest from the holder and apply it.
    manifest_json = _recv_json(0, pipeline_group)
    logging.info(
        "WEIGHT_SHIP rank %d: received quant manifest (%d groups, %d modules)",
        rank,
        len(manifest_json.get("groups", [])),
        sum(len(v) for _, v in manifest_json.get("groups", [])),
    )
    return _apply_ship_quant_manifest(model, manifest_json)


def _send_json(obj, dst, group):
    payload = json.dumps(obj).encode("utf-8")
    buf = np.frombuffer(payload, dtype=np.uint8).copy()
    length = mx.array([buf.size], dtype=mx.int32)
    mx.eval(mx.distributed.send(length, dst, group=group))
    mx.eval(mx.distributed.send(mx.array(buf), dst, group=group))


def _recv_json(src, group):
    length = mx.distributed.recv((1,), mx.int32, src, group=group)
    mx.eval(length)
    n = int(length.item())
    arr = mx.distributed.recv((n,), mx.uint8, src, group=group)
    mx.eval(arr)
    return json.loads(bytes(np.array(arr).astype(np.uint8)).decode("utf-8"))


def _pipeline_target(model):
    lm = getattr(model, "language_model", model)
    inner = lm.model if hasattr(lm, "model") else lm
    if hasattr(inner, "pipeline"):
        return inner
    if hasattr(lm, "pipeline"):
        return lm
    return inner


def sharded_load(
    repo,
    tensor_group: Optional[mx.distributed.Group] = None,
    pipeline_group: Optional[mx.distributed.Group] = None,
):
    # Get model path with everything but weight safetensors
    model_path = get_model_path(repo)

    # Lazy load the model graph without weights first. Pipeline loading must not
    # touch every safetensors shard before the rank-local shard set is known.
    model = load_model(model_path, lazy=True, strict=False, load_weights=False)
    config = model.config.to_dict()

    has_tensor_parallel = hasattr(model, "shard")
    inner = _pipeline_target(model)
    has_pipeline_parallel = hasattr(inner, "pipeline")

    if tensor_group is not None and not has_tensor_parallel:
        raise ValueError(
            "The model does not support tensor parallelism but a tensor_group was provided"
        )
    if pipeline_group is not None and not has_pipeline_parallel:
        raise ValueError(
            "The model does not support pipeline parallelism but a pipeline_group was provided"
        )

    if tensor_group is None and pipeline_group is None:
        if has_tensor_parallel:
            tensor_group = mx.distributed.init()
        elif has_pipeline_parallel:
            pipeline_group = mx.distributed.init()

    processor = load_processor(
        model_path, True, eos_token_ids=config.get("eos_token_id", None)
    )

    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor

    # ---- Weight-shipping path (gated) -------------------------------------
    # Holder rank reads every remote rank's shards from local disk and ships the
    # materialised params over the ring; remote ranks recv into unified memory and
    # touch NO local weight files. Only the tiny config/tokenizer metadata must be
    # present on the remote node.
    if pipeline_group is not None and pipeline_group.size() > 1 and _weight_ship_enabled():
        rank = pipeline_group.rank()
        if rank == 0:
            # Holder: SHIP the remote ranks' weights FIRST (continuous ring traffic
            # from ~1s after connect), THEN build + materialize its own layers. The
            # ring socket is closed by the peer after only a few seconds idle, so
            # the FIRST send (the quant manifest) must go out almost immediately
            # after rendezvous. We therefore ship BEFORE loading/quantizing rank0's
            # own graph (which alone takes ~5 s — long enough to drop the socket and
            # EPIPE the first send). _ship_weights_for_pipeline builds its own lazy
            # remote-rank graph internally, so it needs nothing from rank0's model.
            logging.info("WEIGHT_SHIP rank 0 PHASE: ship-start")
            _ship_weights_for_pipeline(model_path, pipeline_group)
            logging.info("WEIGHT_SHIP rank 0 PHASE: ship-done -> own-load-start")
            # Now build + materialize its own layers from local disk. The ship's
            # ALL_SUM #1 just synchronized both ranks; rank1 is now racing to the
            # heartbeat phase below while rank0 does this (potentially multi-second)
            # own-graph build. We keep the ring warm by heartbeating once the build
            # is done and DURING the chunked materialize (rank1 mirrors the count).
            inner.pipeline(pipeline_group)
            local_files = _pipeline_local_files(
                model, inner, model_path, config, pipeline_group
            )
            model = load_model(
                model_path, lazy=True, strict=False, weight_files=local_files
            )
            inner = _pipeline_target(model)
            inner.pipeline(pipeline_group)
            logging.info("WEIGHT_SHIP rank 0 PHASE: own-load-done -> materialize-heartbeat")
            print("Materializing (holder rank 0, own layers, heartbeat-chunked)")
            _materialize_with_heartbeats(model.language_model, pipeline_group)
            logging.info("WEIGHT_SHIP rank 0 PHASE: materialize-done -> final-barrier")
        else:
            # Remote: build the identical split graph with NO local weights, apply
            # the INF mixed quantization from the holder's manifest, then recv.
            inner.pipeline(pipeline_group)
            model = _quantize_pipeline_graph_for_ship(
                model_path, model, pipeline_group
            )
            inner = _pipeline_target(model)
            print("Receiving shipped weights (remote rank)")
            _recv_shipped_weights(model, pipeline_group)
            logging.info("WEIGHT_SHIP rank %d PHASE: recv-done -> heartbeat-wait", rank)
            # Mirror rank0's heartbeats so the ring keeps continuous traffic while
            # rank0 builds + materializes its own (much larger) local slice. Without
            # this, rank1 races to the final barrier and the ring socket idles for
            # the whole rank0 own-load/materialize window -> EPIPE on the next send.
            _heartbeat_wait(pipeline_group)
            logging.info("WEIGHT_SHIP rank %d PHASE: heartbeat-wait-done -> final-barrier", rank)
        model.model_path = model_path
        model.eval()
        # Final ring barrier so all ranks proceed together.
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        logging.info("WEIGHT_SHIP rank %d PHASE: final-barrier-passed (load complete)", rank)
        return model, processor

    local_files = None
    if pipeline_group is not None:
        inner.pipeline(pipeline_group)
        index_path = model_path / "model.safetensors.index.json"
        # INF/encoded checkpoints have no native index; decode the obfuscated
        # tensor names and materialise one so per-rank shard selection works.
        if not index_path.exists() and inf_decode.is_inf_checkpoint(model_path):
            index_path = inf_decode.write_index_json(model_path, config)
        if not index_path.exists():
            raise ValueError(
                "Pipeline loading requires model.safetensors.index.json for this "
                f"checkpoint. {model_path} has no index; INF/encoded checkpoints "
                "need a name-decode index pass before distributed loading."
            )
        with open(index_path, "r", encoding="utf-8") as fid:
            weight_index = json.load(fid)["weight_map"]

        local_files = set()
        for key, _ in tree_flatten(model.parameters()):
            file_name = _lookup_sharded_weight_file(weight_index, key)
            if file_name is None:
                raise ValueError(
                    "Pipeline loading is only supported for indexed MLX models; "
                    f"missing safetensors index entry for {key!r}."
                )
            local_files.add(file_name)

    model = load_model(
        model_path,
        lazy=True,
        strict=False,
        weight_files=local_files,
    )

    if tensor_group is not None:
        model.shard(tensor_group)

    if pipeline_group is not None:
        inner = _pipeline_target(model)
        inner.pipeline(pipeline_group)

    print("Materializing")
    mx.eval(model.language_model.parameters())
    model.eval()

    # Synchronize processes to avoid timeout
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))

    return model, processor


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(
            model_path,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )

    try:
        with open(model_path / "config.json", encoding="utf-8") as f:
            config = json.load(f)

        generation_config_file = model_path / "generation_config.json"
        if generation_config_file.exists():
            generation_config = {}
            try:
                with open(generation_config_file, "r") as f:
                    generation_config = json.load(f)
            except json.JSONDecodeError:
                pass

            if eos_token_id := generation_config.get("eos_token_id", False):
                config["eos_token_id"] = eos_token_id

        return config

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config not found at {model_path}") from exc


def load_image_processor(model_path: Union[str, Path], **kwargs) -> BaseImageProcessor:
    if isinstance(model_path, str):
        model_path = get_model_path(
            model_path,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )

    if not kwargs:
        config = load_config(model_path, trust_remote_code=True)
    else:
        config = load_config(model_path, **kwargs)

    try:
        model_class, _ = get_model_and_args(config)
    except ValueError:
        return None
    image_processor = None

    if hasattr(model_class, "ImageProcessor"):
        init_signature = inspect.signature(model_class.ImageProcessor.__init__)

        if "config" in init_signature.parameters:
            image_processor = model_class.ImageProcessor(config=config)
        else:
            image_processor = model_class.ImageProcessor()

    return image_processor


def load_processor(
    model_path, add_detokenizer=True, eos_token_ids=None, **kwargs
) -> ProcessorMixin:

    # Checkpoints that ship a CUSTOM processor class (Kimi-K2.x: kimi_k25_processor.py +
    # tokenization_kimi.py, registered via config auto_map) cannot be instantiated without this;
    # transformers raises "Unrecognized processing class ... Can't instantiate a processor, a
    # tokenizer, an image processor..." (measured 2026-08-11 on Kimi-K2.7-Code-MLX-Q3.5-INF).
    #
    # This matches what load_config() already does for the very same directory a few lines above,
    # so it introduces no new trust: the config path already executes that checkpoint's code.
    # setdefault, not a hard override, so a caller can still turn it off deliberately.
    kwargs.setdefault("trust_remote_code", True)
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except (ValueError, KeyError, OSError) as exc:
        # Some checkpoints ship a custom processor but DO NOT register it in config.auto_map, so
        # AutoProcessor cannot resolve it no matter what trust_remote_code says. Measured on
        # Kimi-K2.7-Code-MLX-Q3.5-INF: auto_map lists only AutoConfig / AutoModel /
        # AutoModelForCausalLM, and AutoProcessor raises "Unrecognized processing class".
        #
        # mlx-vlm already ships the right class for these architectures (kimi_k25 exports
        # `Processor` = KimiK25Processor), so fall back to the architecture module -- the same
        # resolution load_image_processor() uses for `ImageProcessor`.
        try:
            cfg = load_config(model_path, trust_remote_code=True)
            model_class, _ = get_model_and_args(cfg)
        except Exception:
            raise exc
        proc_class = getattr(model_class, "Processor", None)
        if proc_class is None:
            raise exc
        # Construct directly rather than via from_pretrained: ProcessorMixin.from_pretrained
        # resolves `tokenizer_class`/`image_processor_class` through the Auto* registry, which is
        # the very thing that failed above (it raised "stat: path ... not NoneType" on this
        # checkpoint). The class already defaults image_processor when None, and wants a real HF
        # tokenizer -- not mlx-vlm's TokenizerWrapper.
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = proc_class(image_processor=None, tokenizer=hf_tok)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)

        # Get the tokenizer object
        tokenizer_obj = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        # Instantiate the detokenizer
        processor.detokenizer = detokenizer_class(tokenizer_obj)

        # Determine the EOS token IDs, prioritizing the function argument
        final_eos_token_ids = (
            eos_token_ids
            or getattr(tokenizer_obj, "eos_token_ids", None)
            or getattr(tokenizer_obj, "eos_token_id", None)
        )

        # Create and assign the StoppingCriteria
        criteria = StoppingCriteria(final_eos_token_ids, tokenizer_obj)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.stopping_criteria = criteria
        else:
            processor.stopping_criteria = criteria

    return processor


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs
) -> Tuple[nn.Module, dict, ProcessorMixin]:
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path, **kwargs)
    processor = load_processor(
        model_path,
        add_detokenizer=False,
        eos_token_ids=config.get("eos_token_id", None),
        **kwargs,
    )
    return model, config, processor


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def create_model_card(
    path: Union[str, Path], hf_path: Optional[Union[str, Path]] = None
):
    """
    Create model card for a converted MLX model.

    Args:
        path (Union[str, Path]): Local path to the converted model.
        hf_path (Optional[Union[str, Path]]): Original Hugging Face repo id or local path used for conversion.
    """
    from huggingface_hub import ModelCard, ModelCardData

    if hf_path is None:
        card = ModelCard.from_template(ModelCardData(language="en"))
    else:
        card = ModelCard.load(hf_path)
    card.data.library_name = "mlx"
    if card.data.pipeline_tag is None:
        card.data.pipeline_tag = "image-text-to-text"
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    if hf_path is not None:
        card.data.base_model = str(hf_path)
    card.text = ""
    card.save(Path(path) / "README.md")


def upload_to_hub(path: str, upload_repo: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    logging.set_verbosity_info()
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)

    hf_path = card.data.base_model

    if hf_path is not None:
        provenance = f"""
        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path})
        using mlx-vlm version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        """
    else:
        provenance = ""

    card.text = dedent(
        f"""
        # {upload_repo}
        {provenance}
        ## Use with mlx

        ```bash
        pip install -U mlx-vlm
        ```

        ```bash
        python -m mlx_vlm.generate --model {upload_repo} --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
        ```
        """
    )
    card.save(card_path)

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def save_weights(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    weights = dict(tree_flatten(model.parameters()))

    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))

    weights.clear()
    del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("torch_dtype", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def load_image(image_source: Union[str, Path, BytesIO], timeout: int = 10):
    """
    Helper function to load an image from either a URL, file path, data URI,
    or BytesIO object.
    """
    import base64

    original_source = image_source
    try:
        if not isinstance(image_source, (str, Path, BytesIO)):
            raise ValueError(
                f"Unsupported image source type: {type(image_source).__name__}"
            )
        if isinstance(image_source, str) and image_source.startswith("data:image/"):
            if "," not in image_source:
                raise ValueError("Invalid data URI format - missing comma separator")
            _, data = image_source.split(",", 1)
            image_source = BytesIO(base64.b64decode(data))
        if isinstance(image_source, str) and image_source.startswith(
            ("http://", "https://")
        ):
            with requests.get(image_source, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                image_source = BytesIO(response.content)

        image = Image.open(image_source)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {original_source}: {e}") from e

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(img, max_size):

    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size)


def process_image(img, resize_shape, image_processor):
    if isinstance(img, str):
        img = load_image(img)
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    if resize_shape is not None and not isinstance(image_processor, BaseImageProcessor):
        img = resize_image(img, resize_shape)
    return img


def read_audio(file) -> tuple:
    """Read an audio file using miniaudio (or ffmpeg for m4a/aac/ogg/opus).

    Returns (samples_float32, sample_rate) where samples is always 2D (samples, channels).
    """
    import io as _io

    if isinstance(file, bytes):
        file = _io.BytesIO(file)

    # Check if ffmpeg is needed for certain formats
    use_ffmpeg = False
    if isinstance(file, (str, Path)):
        ext = Path(file).suffix.lstrip(".").lower()
        if ext in ("m4a", "aac", "ogg", "opus"):
            use_ffmpeg = True
    elif isinstance(file, _io.BytesIO):
        pos = file.tell()
        header = file.read(12)
        file.seek(pos)
        if header[4:8] == b"ftyp" or header[:4] == b"OggS":
            use_ffmpeg = True

    if use_ffmpeg:
        import json as _json
        import shutil
        import subprocess

        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found. Install it: brew install ffmpeg (macOS) "
                "or sudo apt install ffmpeg (Linux)"
            )

        if isinstance(file, _io.BytesIO):
            file.seek(0)
            input_data = file.read()
        else:
            input_data = None

        # Get info via ffprobe
        if ffprobe_path and input_data is not None:
            probe = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "a:0",
                    "-i",
                    "pipe:0",
                ],
                input=input_data,
                capture_output=True,
            )
        elif ffprobe_path:
            probe = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "a:0",
                    str(file),
                ],
                capture_output=True,
            )
        else:
            probe = None

        sample_rate, nchannels = 44100, 1
        if probe and probe.returncode == 0:
            info = _json.loads(probe.stdout)
            if info.get("streams"):
                stream = info["streams"][0]
                sample_rate = int(stream.get("sample_rate", 44100))
                nchannels = int(stream.get("channels", 1))

        # Decode via ffmpeg to raw PCM s16le
        cmd = [ffmpeg_path, "-v", "quiet"]
        if input_data is not None:
            cmd += ["-i", "pipe:0"]
        else:
            cmd += ["-i", str(file)]
        cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(nchannels), "pipe:1"]

        result = subprocess.run(cmd, input=input_data, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decoding failed: {result.stderr.decode()}")

        samples = np.frombuffer(result.stdout, dtype=np.int16)
    else:
        import miniaudio

        if isinstance(file, (str, Path)):
            info = miniaudio.get_file_info(str(file))
            decoded = miniaudio.decode_file(
                str(file),
                nchannels=info.nchannels,
                sample_rate=info.sample_rate,
            )
        elif isinstance(file, _io.BytesIO):
            file.seek(0)
            data = file.read()
            # Detect format from magic bytes
            if data[:4] == b"RIFF":
                info = miniaudio.wav_get_info(data)
            elif data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xfa"):
                info = miniaudio.mp3_get_info(data)
            elif data[:4] == b"fLaC":
                info = miniaudio.flac_get_info(data)
            else:
                info = miniaudio.vorbis_get_info(data)
            decoded = miniaudio.decode(
                data,
                nchannels=info.nchannels,
                sample_rate=info.sample_rate,
            )
        else:
            raise TypeError(f"Unsupported file type: {type(file)}")

        sample_rate = decoded.sample_rate
        nchannels = decoded.nchannels
        samples = np.array(decoded.samples, dtype=np.int16)

    # Reshape multi-channel and convert to float32
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels)
    audio = samples.astype(np.float32) / 32768.0

    # Ensure always 2D
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    return audio, sample_rate


def load_audio(
    file,
    sr: int,
    timeout: int = 10,
):
    """
    Helper function to load audio from either a URL, file path, or numpy array.
    """
    if isinstance(file, np.ndarray):
        audio = file.astype(np.float32, copy=False)
        return audio.mean(axis=1) if audio.ndim > 1 else audio

    from mlx_audio.audio_io import read as read_audio
    from mlx_audio.utils import resample_audio

    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, str) and file.startswith(("http://", "https://")):
        try:
            response = requests.get(file, stream=True, timeout=timeout)
            response.raise_for_status()
            audio, sample_rate = read_audio(BytesIO(response.content), dtype="float32")
        except Exception as e:
            raise ValueError(
                f"Failed to load audio from URL: {file} with error {e}"
            ) from e
    else:
        audio, sample_rate = read_audio(file, dtype="float32")

    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def normalize_audio_features(features: mx.array) -> mx.array:
    """Normalize mel spectrogram features for lossy audio formats (e.g., MP3)."""
    return (features - mx.mean(features)) / (mx.std(features) + 1e-6)


def load_video(
    video_path: str,
    fps: float = 2.0,
    nframes: Optional[int] = None,
    min_frames: int = 4,
    max_frames: int = 768,
    frame_factor: int = 2,
) -> Tuple[np.ndarray, float]:
    """Read a video file as a (T, C, H, W) numpy array.

    Uniformly samples ``nframes`` frames — either a fixed count or derived
    from ``fps`` — and returns the sampled frames alongside the effective
    sampling fps.
    """
    import cv2

    if video_path.startswith("file://"):
        video_path = video_path[7:]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    def _round(n):
        return round(n / frame_factor) * frame_factor

    def _floor(n):
        return math.floor(n / frame_factor) * frame_factor

    def _ceil(n):
        return math.ceil(n / frame_factor) * frame_factor

    if nframes is not None:
        n = _round(nframes)
    else:
        lo = _ceil(min_frames)
        hi = _floor(min(max_frames, total_frames))
        n = total_frames / video_fps * fps
        n = min(max(n, lo), hi, total_frames)
        n = _floor(n)
    if not (frame_factor <= n <= total_frames):
        cap.release()
        raise ValueError(
            f"nframes must be in [{frame_factor}, {total_frames}], got {n}."
        )

    indices = np.linspace(0, total_frames - 1, n).round().astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("No frames read from the video.")

    video_np = np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2))
    sample_fps = n / max(total_frames, 1e-6) * video_fps
    return video_np, sample_fps


def process_inputs(
    processor,
    prompts,
    images=None,
    audio=None,
    add_special_tokens=False,
    padding=True,
    padding_side="left",
    return_tensors="mlx",
    **kwargs,
):
    # Get the process method from the processor
    process_method = getattr(processor, "process", processor)
    parameters = inspect.signature(process_method).parameters

    # Prepare arguments
    args = {
        "text": prompts,
        "images": images,
        "padding": padding,
        "return_tensors": return_tensors,
    }
    if "padding_side" in parameters:
        args["padding_side"] = padding_side

    # Add special tokens if supported
    if "add_special_tokens" in parameters:
        args["add_special_tokens"] = add_special_tokens

    for param in parameters.keys():
        if param in kwargs.keys():
            args[param] = kwargs.get(param, None)

    # Add audio if provided and supported
    if audio is not None and len(audio) > 0:
        if "audio" in parameters:
            args["audio"] = audio
        elif "audios" in parameters:
            args["audios"] = audio
        else:
            raise ValueError(
                f"Processor {processor.__class__.__name__} does not support audio parameter"
            )

    return process_method(**args)


def process_inputs_with_fallback(
    processor,
    prompts,
    images,
    audio,
    add_special_tokens=False,
    return_tensors="mlx",
    **kwargs,
):
    # First attempt with specified return_tensors
    try:
        return process_inputs(
            processor,
            prompts=prompts,
            images=images,
            audio=audio,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Failed to process inputs with error: {e}")


def prepare_inputs(
    processor,
    images=None,
    audio=None,
    videos=None,
    prompts=None,
    image_token_index=None,
    resize_shape=None,
    add_special_tokens=False,
    padding=True,
    padding_side="left",
    pad_to_uniform_size=False,
    return_tensors="mlx",
    **kwargs,
):

    has_images = images is not None and (
        not hasattr(images, "__len__") or len(images) > 0
    )
    has_audio = audio is not None and (not hasattr(audio, "__len__") or len(audio) > 0)
    has_videos = videos is not None and (
        not hasattr(videos, "__len__") or len(videos) > 0
    )
    if not has_images and not has_audio and not has_videos:
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        # Ensure pad_token exists when padding text-only inputs
        if padding and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            prompts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            padding_side=padding_side,
            return_tensors=return_tensors,
        )
        input_ids = (
            inputs.input_ids
            if isinstance(inputs.input_ids, mx.array)
            else mx.array(inputs.input_ids)
        )
        mask = (
            inputs.attention_mask
            if isinstance(inputs.attention_mask, mx.array)
            else mx.array(inputs.attention_mask)
        )
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
        }

    # Process images
    if images is not None:
        if not isinstance(images, list):
            images = [images]

        image_processor = (
            processor.image_processor if hasattr(processor, "image_processor") else None
        )
        images = [process_image(img, resize_shape, image_processor) for img in images]

        # For batching, we need uniform image sizes. Instead of padding to the
        # largest image (which adds white borders that hurt accuracy), we resize
        # all images to the model's expected input size.
        if len(images) > 1 and pad_to_uniform_size:
            # Get target size from image processor if available
            target_size = None
            if image_processor is not None and hasattr(image_processor, "size"):
                size = image_processor.size
                if isinstance(size, tuple):
                    target_size = size
                elif isinstance(size, dict):
                    target_size = (size.get("height", 384), size.get("width", 384))
                elif isinstance(size, int):
                    target_size = (size, size)

            if target_size is not None:
                # Resize all images to the target size
                resized_images = []
                for img in images:
                    if img.size != (
                        target_size[1],
                        target_size[0],
                    ):  # PIL uses (width, height)
                        img = img.resize(
                            (target_size[1], target_size[0]), Image.Resampling.BICUBIC
                        )
                    resized_images.append(img)
                images = resized_images
            else:
                # Fallback: pad to largest size (original behavior)
                max_width = max(img.width for img in images)
                max_height = max(img.height for img in images)

                padded_images = []
                for img in images:
                    if img.width != max_width or img.height != max_height:
                        padded_img = Image.new(
                            "RGB", (max_width, max_height), (255, 255, 255)
                        )
                        x_offset = (max_width - img.width) // 2
                        y_offset = (max_height - img.height) // 2
                        padded_img.paste(img, (x_offset, y_offset))
                        padded_images.append(padded_img)
                    else:
                        padded_images.append(img)
                images = padded_images

    # Process audio
    audio_inputs = None
    audio_feature_lengths = None
    is_qwen3_omni_moe = False
    processor_class_name = (
        processor.__class__.__name__ if hasattr(processor, "__class__") else ""
    )
    if (
        "qwen3" in processor_class_name.lower()
        and "omni" in processor_class_name.lower()
    ):
        is_qwen3_omni_moe = True

    is_lossy_audio = False
    if audio is not None and len(audio) > 0:
        if not isinstance(audio, list):
            audio = [audio]

        # Check if any audio file is a lossy format (MP3, AAC, OGG, etc.)
        lossy_extensions = {".mp3", ".m4a"}
        is_lossy_audio = any(
            str(f).lower().endswith(tuple(lossy_extensions)) for f in audio
        )

        if len(audio) > 1:
            print(
                "\033[33mWarning\033[0m: Single prompt with multiple audio files is not supported yet. Using the first audio file.\n"
            )
            audio = audio[:1]

        if is_qwen3_omni_moe:
            audio_arrays = [
                load_audio(audio_file, sr=processor.feature_extractor.sampling_rate)
                for audio_file in audio
            ]
            audio_arrays = [
                audio_array.astype(np.float32) for audio_array in audio_arrays
            ]

            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is None:
                raise ValueError("Processor missing feature_extractor for audio prep.")

            audio_inputs = feature_extractor(
                audio_arrays,
                sampling_rate=feature_extractor.sampling_rate,
                padding=True,
                return_attention_mask=True,
            )

            audio_feature_lengths = np.sum(
                audio_inputs["attention_mask"], axis=-1, dtype=np.int32
            )
        else:
            feature_extractor = getattr(processor, "feature_extractor", None)
            sr = (
                getattr(feature_extractor, "sampling_rate", 16000)
                if feature_extractor is not None
                else 16000
            )
            audio = [load_audio(audio_file, sr=sr) for audio_file in audio]

    video_fps = None
    if has_videos:
        if not isinstance(videos, list):
            videos = [videos]
        fps_hint = kwargs.pop("fps", 2.0)
        loaded, video_fps = [], []
        for v in videos:
            arr, s_fps = (
                load_video(str(v), fps=fps_hint)
                if isinstance(v, (str, bytes))
                else (v, fps_hint)
            )
            loaded.append(arr)
            video_fps.append(s_fps)
        videos = loaded

    model_inputs = {}

    if hasattr(processor, "image_processor") and isinstance(
        processor.image_processor, BaseImageProcessor
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]

        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        text_chunks = [
            [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            for prompt in prompts
        ]

        # Find the maximum length for padding
        max_length = max(
            sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks
        )

        # Pad and create input_ids
        input_ids = []
        for chunks in text_chunks:
            ids = chunks[0] + [image_token_index] + chunks[1]
            padding = [processor.pad_token_id] * (max_length - len(ids))
            input_ids.append(mx.array(ids + padding))

        model_inputs["input_ids"] = mx.array(input_ids)
        pixel_values = processor.image_processor.preprocess(images=images)
        model_inputs["pixel_values"] = mx.array(np.stack(pixel_values))
        model_inputs["attention_mask"] = mx.array(
            [(ids != processor.pad_token_id) for ids in input_ids]
        ).astype(mx.int32)

    else:
        if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        extra = {}
        if has_videos:
            extra["videos"] = videos
            if video_fps is not None:
                extra["fps"] = video_fps
        inputs = process_inputs_with_fallback(
            processor,
            images=images,
            audio=audio,
            prompts=prompts,
            add_special_tokens=add_special_tokens,
            **extra,
            **kwargs,
        )

        if "images" in inputs:
            inputs["pixel_values"] = inputs["images"]
            inputs.pop("images")

        model_inputs["attention_mask"] = (
            mx.array(inputs["attention_mask"]) if "attention_mask" in inputs else None
        )

        # Convert inputs to model_inputs with mx.array if present
        for key, value in inputs.items():
            if key not in model_inputs:
                if value is None:
                    model_inputs[key] = value
                elif isinstance(value, (str, list, mx.array)):
                    model_inputs[key] = value
                else:
                    model_inputs[key] = mx.array(value)

    if audio_inputs is not None:
        model_inputs["input_features"] = mx.array(audio_inputs["input_features"])
        model_inputs["feature_attention_mask"] = mx.array(
            audio_inputs["attention_mask"]
        ).astype(mx.int32)
        model_inputs["audio_feature_lengths"] = mx.array(
            audio_feature_lengths, dtype=mx.int32
        )

    if is_lossy_audio and "input_features" in model_inputs:
        f = model_inputs["input_features"]
        if isinstance(f, list):
            model_inputs["input_features"] = [
                normalize_audio_features(mx.array(x)) for x in f
            ]
        else:
            model_inputs["input_features"] = normalize_audio_features(f)

    return model_inputs


def group_images_by_shape(
    images: List[Image.Image],
    disable_grouping: bool = False,
) -> Tuple[Dict[Tuple[int, int], List[Image.Image]], Dict[Tuple[int, int], List[int]]]:
    """
    Group images by their dimensions for efficient batch processing.

    Images with the same dimensions can be stacked and processed together,
    which is much faster than processing individually (especially on GPU).

    Args:
        images: List of PIL images to group
        disable_grouping: If True, each image gets its own group (useful for debugging)

    Returns:
        grouped_images: Dict mapping shape -> list of images with that shape
        grouped_indices: Dict mapping shape -> list of original indices

    Example:
        >>> images = [img_400x300, img_800x600, img_400x300_2]
        >>> grouped, indices = group_images_by_shape(images)
        >>> grouped
        {(300, 400): [img_400x300, img_400x300_2], (600, 800): [img_800x600]}
        >>> indices
        {(300, 400): [0, 2], (600, 800): [1]}
    """
    if disable_grouping:
        # Each image in its own group
        grouped_images = {}
        grouped_indices = {}
        for i, img in enumerate(images):
            shape = (img.height, img.width)
            # Make each shape unique by adding index
            unique_shape = (img.height, img.width, i)
            grouped_images[unique_shape] = [img]
            grouped_indices[unique_shape] = [i]
        return grouped_images, grouped_indices

    grouped_images: Dict[Tuple[int, int], List[Image.Image]] = {}
    grouped_indices: Dict[Tuple[int, int], List[int]] = {}

    for i, img in enumerate(images):
        shape = (img.height, img.width)
        if shape not in grouped_images:
            grouped_images[shape] = []
            grouped_indices[shape] = []
        grouped_images[shape].append(img)
        grouped_indices[shape].append(i)

    return grouped_images, grouped_indices


class StoppingCriteria:
    def __init__(self, eos_token_ids: List[int], tokenizer=None):

        if isinstance(eos_token_ids, int):
            self.eos_token_ids = [eos_token_ids]
        else:
            self.eos_token_ids = eos_token_ids

        self.tokenizer = tokenizer

    def add_eos_token_ids(self, new_eos_token_ids: Union[int, List[int]] = None):
        """
        Add new token IDs to the list of EOS token IDs.

        Args:
            new_eos_token_ids: Integer, string, or list of integers/strings representing token IDs to add.
                               If strings are provided, they will be converted to integers if possible.
        """
        if new_eos_token_ids is None:
            return

        if self.tokenizer is None:
            raise ValueError("Processor is not provided")

        if new_eos_token_ids is not None:
            if isinstance(new_eos_token_ids, (str, int)):
                new_eos_token_ids = [new_eos_token_ids]
            resolved = []
            for token in new_eos_token_ids:
                if isinstance(token, int):
                    resolved.append(token)
                elif isinstance(token, str):
                    resolved.append(
                        self.tokenizer.encode(" " + token, add_special_tokens=False)[-1]
                    )
            self.eos_token_ids.extend(resolved)

    def reset(self, eos_token_ids: List[int] = None):
        eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else self.tokenizer.eos_token_ids
        )

        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        if self.eos_token_ids != eos_token_ids:
            self.eos_token_ids = eos_token_ids

    def __call__(self, input_ids: mx.array) -> bool:
        return input_ids in self.eos_token_ids


class ThinkingBudgetCriteria:
    """
    Enforces a budget on thinking tokens.

    Tracks tokens within thinking blocks (between start and end tokens) and
    forces a closing sequence (e.g. ``\\n</think>``) when budget is exceeded.
    """

    def __init__(
        self,
        tokenizer,
        thinking_budget: int,
        thinking_end_token: str = "</think>",
        thinking_start_token: Optional[str] = None,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking

        # Resolve token IDs from strings
        self.thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]

        self.thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]

        self._forced_sequence: List[int] = []
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            self._forced_sequence.append(newline_ids[-1])
        self._forced_sequence.append(self.thinking_end_token_id)
        self._forced_index = 0

        self.in_thinking = self.enable_thinking
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self.forced_token_id = None

    def reset_thinking_state(self):
        """Reset thinking state between generations."""
        self.in_thinking = self.enable_thinking
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self._forced_index = 0

    def __call__(self, token_id: int) -> Optional[int]:
        """Process a token and return a forced token ID if budget exceeded, else None."""
        if self.enable_thinking and token_id == self.thinking_start_token_id:
            self.in_thinking = True
            return None

        if token_id == self.thinking_end_token_id:
            self.in_thinking = False
            self.budget_exceeded = False
            self._forced_index = 0
            return None

        if self.in_thinking:
            self.thinking_token_count += 1
            if self.thinking_token_count > self.thinking_budget:
                self.budget_exceeded = True

        if self.budget_exceeded and self._forced_index < len(self._forced_sequence):
            forced = self._forced_sequence[self._forced_index]
            self._forced_index += 1
            self.forced_token_id = forced
            return forced

        self.forced_token_id = None
        return None

    def apply_forced_token(self, next_y: mx.array) -> Optional[mx.array]:
        if self.forced_token_id is not None and self.enable_thinking:
            next_y = mx.array([self.forced_token_id])
            self.forced_token_id = None
            return next_y
        return next_y


def print_array_report(t: mx.array, label: Optional[str]) -> dict:
    """
    Return a dictionary report of an MLX array similar to PyTorch's tensor representation.
    Args:
        arr: MLX array to analyze
    Returns:
        Dictionary containing shape, dtype, value representation, and statistics
    """

    # Get basic statistics
    mean_val = mx.mean(t)
    std_val = mx.std(t)
    min_val = mx.min(t)
    max_val = mx.max(t)

    report = {
        "shape": f"{tuple(t.shape)}",
        "dtype": str(t.dtype),
        "value": repr(t),
        "mean": f"array({mean_val}, dtype={t.dtype})",
        "std": f"array({std_val}, dtype={t.dtype})",
        "min": f"array({min_val}, dtype={t.dtype})",
        "max": f"array({max_val}, dtype={t.dtype})",
        "label": label if label else "array",
    }

    # Print each field, handling 'value' specially
    print("{")
    for key, value in report.items():
        if key == "value":
            print(f" '{key}': {value},")  # No quotes around value
        else:
            print(f" '{key}': {repr(value)},")
    print("}")
    return report
