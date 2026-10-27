"""Split the DSpark draft head (``mtp.*`` tensors) out of a DeepSeek-V4-Flash
checkpoint into a standalone MLX drafter folder.

Mirrors ``deepseek_v4_mtp/split.py`` but selects the DSpark surface and writes a
``deepseek_v4_dspark`` drafter config carrying the dspark_* hyper-parameters
(block_size / noise_token_id / target_layer_ids / markov_rank / n_mtp_layers)
read from the source ``config.json`` (they live either at top level or inside
``text_config``).

Weight-free / not executed here: run on the Mac against the downloaded release.
"""

import argparse
import glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config
from ....utils import _load_safetensors, get_model_path
from .config import DeepseekV4DSparkConfig
from .deepseek_v4_dspark import DeepseekV4DSparkDraftModel


def _safetensor_files(model_path: Path) -> list:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def _iter_mtp_keys(model_path: Path) -> Iterable:
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, list] = {}
        for key, filename in weight_map.items():
            if key.startswith("mtp."):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return
    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if key.startswith("mtp.")]
        if keys:
            yield file, keys


def _load_selected_tensors(file: Path, keys: list) -> Dict[str, mx.array]:
    tensors = {}
    try:
        with safe_open(file, framework="mlx") as f:
            for key in keys:
                tensors[key] = mx.array(f.get_tensor(key))
    except (AttributeError, RuntimeError, TypeError):
        shard = _load_safetensors(str(file))
        tensors = {key: shard[key] for key in keys}
    return tensors


def _text_config(source_config: dict) -> dict:
    return dict(source_config.get("text_config") or source_config)


def _dspark_params(source_config: dict, text_config: dict) -> dict:
    def pick(key, default):
        if key in source_config:
            return source_config[key]
        if key in text_config:
            return text_config[key]
        return default

    n_mtp = int(pick("n_mtp_layers", pick("num_nextn_predict_layers", 3)))
    return {
        "block_size": int(pick("dspark_block_size", 5)),
        "noise_token_id": int(pick("dspark_noise_token_id", 128799)),
        "target_layer_ids": list(pick("dspark_target_layer_ids", [40, 41, 42])),
        "markov_rank": int(pick("dspark_markov_rank", 256)),
        "n_draft_layers": n_mtp,
    }


def _quantization_from_weights(weights: Dict[str, mx.array]) -> Optional[dict]:
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {"group_size": 64, "bits": 8, "mode": "affine"}
    for key in weights:
        if not key.endswith(".scales"):
            continue
        module = key[: -len(".scales")]
        if ".ffn.switch_mlp." in module and module.endswith("_proj"):
            quantization[module] = mxfp4
        elif (
            ".ffn.shared_experts." in module
            or ".attn.w" in module
            or module.endswith(".main_proj")
        ):
            quantization[module] = mxfp8
    return quantization if len(quantization) > 3 else None


def split_deepseek_v4_dspark(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    confidence_threshold: float = 0.0,
    force_download: bool = False,
) -> Path:
    source_path = get_model_path(source, revision=revision, force_download=force_download)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = _text_config(source_config)
    dspark = _dspark_params(source_config, text_config)

    selected = {}
    for file, keys in _iter_mtp_keys(source_path):
        selected.update(_load_selected_tensors(file, keys))
    if not selected:
        raise ValueError(f"No mtp.* (DSpark) tensors found in {source_path}.")

    drafter_config = DeepseekV4DSparkConfig.from_dict(
        {"text_config": text_config, **dspark}
    )
    sanitize_context = SimpleNamespace(
        config=drafter_config,
        args=DeepseekV4Config.from_dict(text_config),
        stages=[None] * dspark["n_draft_layers"],
    )
    selected = DeepseekV4DSparkDraftModel.sanitize(sanitize_context, selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"), selected, metadata={"format": "mlx"}
    )

    draft_config = {
        "model_type": "deepseek_v4_dspark",
        "text_config": text_config,
        "confidence_threshold": float(confidence_threshold),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
        **dspark,
    }
    quantization = _quantization_from_weights(selected)
    if quantization is not None:
        draft_config["quantization"] = quantization
        draft_config["quantization_config"] = quantization
    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-V4-Flash DSpark tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v4_dspark(**vars(args))
    print(f"Wrote DeepSeek-V4 DSpark drafter to {output}")


if __name__ == "__main__":
    main()
