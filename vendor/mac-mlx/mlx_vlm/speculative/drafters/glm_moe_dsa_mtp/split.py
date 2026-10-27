import argparse
import glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.glm_moe_dsa.config import ModelConfig as GlmConfig
from ....utils import _load_safetensors, get_model_path
from .glm_moe_dsa_mtp import GlmMoeDsaMTPDraftModel


def _safetensor_files(model_path: Path) -> list[Path]:
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
        data = json.load(f)
    return data.get("weight_map", {})


def _iter_layer_keys(
    model_path: Path,
    source_layer: int,
) -> Iterable[tuple[Path, list[str]]]:
    layer_prefix = f"model.layers.{source_layer}."
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith(layer_prefix):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return

    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if key.startswith(layer_prefix)]
        if keys:
            yield file, keys


def _load_selected_tensors(file: Path, keys: list[str]) -> Dict[str, mx.array]:
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


def split_glm_moe_dsa_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    source_layer: Optional[int] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write GLM native layer-78/NextN tensors into a standalone MTP drafter."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = _text_config(source_config)
    source_layer = int(source_layer or text_config.get("num_hidden_layers", 78))

    selected = {}
    for file, keys in _iter_layer_keys(source_path, source_layer):
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No model.layers.{source_layer}.* tensors found in {source_path}.")

    sanitize_context = SimpleNamespace(
        args=GlmConfig.from_dict(text_config),
        source_layer=source_layer,
    )
    selected = GlmMoeDsaMTPDraftModel.sanitize(sanitize_context, selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        selected,
        metadata={"format": "mlx"},
    )

    config = {
        "model_type": "glm_moe_dsa_mtp",
        "text_config": text_config,
        "source_layer": source_layer,
        "block_size": int(
            block_size or int(text_config.get("num_nextn_predict_layers", 1)) + 1
        ),
        "compute_dtype": "float16",
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(config.items())), f, indent=2)

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
        description="Split GLM native layer-78 tensors into a standalone MLX MTP drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-layer", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_glm_moe_dsa_mtp(**vars(args))
    print(f"Wrote GLM MTP drafter to {output}")


if __name__ == "__main__":
    main()
