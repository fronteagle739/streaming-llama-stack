"""Pure-python expected checkpoint layout for DeepSeek-V4-Flash(-DSpark).

Builds the exact HF-checkpoint tensor inventory (key -> (dtype, shape)) that
the official ``deepseek-ai/DeepSeek-V4-Flash-DSpark`` release ships, derived
only from ``config.json`` values. This is validated bit-for-bit against the
vendored real-checkpoint manifest (``dsv4_flash_checkpoint_manifest.json``,
generated from the release's safetensors index + per-shard headers) by
``test_checkpoint_layout.py`` — giving weight-free, platform-independent
structural coverage of:

  * the vanilla DeepseekV4 target module's weight surface (``layers.*``), and
  * the DSpark draft head's weight surface (``mtp.*``) consumed by
    ``mlx_vlm.speculative.drafters.deepseek_v4_dspark``.

Dtype/packing rules mirror the official ``inference/model.py`` reference:
  * fp8 linears: weight F8_E4M3 [out, in], scale F8_E8M0 in 128x128 blocks
  * fp4 (routed experts): weight I8 [out, in//2] (two e2m1 nibbles per byte),
    scale F8_E8M0 [out, in//32] (one scale per 32 fp4 elements)
  * norms / gates / markov / lm-head / embeddings: BF16
  * hyper-connection params, attention sinks, gate bias, compressor APE: F32
  * hash-router tid2eid: I64 [vocab, top_k]

No numpy / mlx dependency: runnable anywhere (including the Windows dev box).
"""

import json
import math
import os
from typing import Dict, Tuple

LayoutT = Dict[str, Tuple[str, Tuple[int, ...]]]

_HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(_HERE, "dsv4_flash_checkpoint_manifest.json")
FLASH_CONFIG_PATH = os.path.join(_HERE, "dsv4_flash_config.json")


def load_flash_config() -> dict:
    with open(FLASH_CONFIG_PATH) as f:
        return json.load(f)


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def expand_manifest(manifest: dict) -> LayoutT:
    """Expand the compact manifest back into the full key -> (dtype, shape) map."""
    out: LayoutT = {}
    for e in manifest["entries"]:
        dtype, shape = e["dtype"], tuple(e["shape"])
        ix = e["indices"]
        if ix is None:
            out[e["pattern"]] = (dtype, shape)
            continue
        if isinstance(ix, dict) and "range" in ix:
            tuples = [(i,) for i in range(ix["range"][0], ix["range"][1])]
        elif isinstance(ix, dict) and "grid" in ix:
            tuples = [()]
            for lo, hi in ix["grid"]:
                tuples = [t + (i,) for t in tuples for i in range(lo, hi)]
        elif ix and isinstance(ix[0], int):
            tuples = [(i,) for i in ix]
        else:
            tuples = [tuple(t) for t in ix]
        for t in tuples:
            k = e["pattern"]
            for i in t:
                k = k.replace("{}", str(i), 1)
            out[k] = (dtype, shape)
    return out


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _fp8(layout: LayoutT, key: str, out_features: int, in_features: int) -> None:
    layout[key + ".weight"] = ("F8_E4M3", (out_features, in_features))
    layout[key + ".scale"] = (
        "F8_E8M0",
        (_ceil_div(out_features, 128), _ceil_div(in_features, 128)),
    )


def _fp4(layout: LayoutT, key: str, out_features: int, in_features: int) -> None:
    layout[key + ".weight"] = ("I8", (out_features, in_features // 2))
    layout[key + ".scale"] = ("F8_E8M0", (out_features, in_features // 32))


def _bf16(layout: LayoutT, key: str, *shape: int) -> None:
    layout[key] = ("BF16", tuple(shape))


def _f32(layout: LayoutT, key: str, *shape: int) -> None:
    layout[key] = ("F32", tuple(shape))


def _expert_ffn(layout: LayoutT, prefix: str, cfg: dict, quant) -> None:
    dim = cfg["hidden_size"]
    inter = cfg["moe_intermediate_size"]
    quant(layout, f"{prefix}.w1", inter, dim)
    quant(layout, f"{prefix}.w2", dim, inter)
    quant(layout, f"{prefix}.w3", inter, dim)


def _attention(layout: LayoutT, prefix: str, cfg: dict) -> None:
    """MLA-family MQA attention: low-rank Q, single shared KV head,
    grouped low-rank output projection (o_groups x o_lora_rank)."""
    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    q_lora = cfg["q_lora_rank"]
    o_groups = cfg["o_groups"]
    o_lora = cfg["o_lora_rank"]

    _f32(layout, f"{prefix}.attn_sink", n_heads)
    _fp8(layout, f"{prefix}.wq_a", q_lora, dim)
    _bf16(layout, f"{prefix}.q_norm.weight", q_lora)
    _fp8(layout, f"{prefix}.wq_b", n_heads * head_dim, q_lora)
    _fp8(layout, f"{prefix}.wkv", head_dim, dim)
    _bf16(layout, f"{prefix}.kv_norm.weight", head_dim)
    _fp8(layout, f"{prefix}.wo_a", o_groups * o_lora, n_heads * head_dim // o_groups)
    _fp8(layout, f"{prefix}.wo_b", dim, o_groups * o_lora)


def _compressor(layout: LayoutT, prefix: str, cfg: dict, ratio: int, head_dim: int) -> None:
    dim = cfg["hidden_size"]
    coff = 2 if ratio == 4 else 1  # overlapping windows only at ratio 4
    _f32(layout, f"{prefix}.ape", ratio, coff * head_dim)
    _bf16(layout, f"{prefix}.wkv.weight", coff * head_dim, dim)
    _bf16(layout, f"{prefix}.wgate.weight", coff * head_dim, dim)
    _bf16(layout, f"{prefix}.norm.weight", head_dim)


def _hc(layout: LayoutT, prefix: str, cfg: dict) -> None:
    hc = cfg["hc_mult"]
    dim = cfg["hidden_size"]
    mix = (2 + hc) * hc
    for sub in ("attn", "ffn"):
        _f32(layout, f"{prefix}.hc_{sub}_fn", mix, hc * dim)
        _f32(layout, f"{prefix}.hc_{sub}_base", mix)
        _f32(layout, f"{prefix}.hc_{sub}_scale", 3)


def _moe(layout: LayoutT, prefix: str, cfg: dict, hash_routed: bool) -> None:
    dim = cfg["hidden_size"]
    n_exp = cfg["n_routed_experts"]
    _bf16(layout, f"{prefix}.gate.weight", n_exp, dim)
    if hash_routed:
        layout[f"{prefix}.gate.tid2eid"] = (
            "I64",
            (cfg["vocab_size"], cfg["num_experts_per_tok"]),
        )
    else:
        _f32(layout, f"{prefix}.gate.bias", n_exp)
    for e in range(n_exp):
        _expert_ffn(layout, f"{prefix}.experts.{e}", cfg, _fp4)
    _expert_ffn(layout, f"{prefix}.shared_experts", cfg, _fp8)


def _block(layout: LayoutT, prefix: str, cfg: dict, ratio: int, hash_routed: bool) -> None:
    dim = cfg["hidden_size"]
    _attention(layout, f"{prefix}.attn", cfg)
    if ratio:
        _compressor(layout, f"{prefix}.attn.compressor", cfg, ratio, cfg["head_dim"])
    if ratio == 4:
        # DSA-style lightning indexer, only on ratio-4 (sparse) layers
        idx = f"{prefix}.attn.indexer"
        _fp8(layout, f"{idx}.wq_b", cfg["index_n_heads"] * cfg["index_head_dim"], cfg["q_lora_rank"])
        _bf16(layout, f"{idx}.weights_proj.weight", cfg["index_n_heads"], dim)
        _compressor(layout, f"{idx}.compressor", cfg, ratio, cfg["index_head_dim"])
    _bf16(layout, f"{prefix}.attn_norm.weight", dim)
    _bf16(layout, f"{prefix}.ffn_norm.weight", dim)
    _moe(layout, f"{prefix}.ffn", cfg, hash_routed)
    _hc(layout, prefix, cfg)


def num_draft_layers(cfg: dict) -> int:
    """The release appends one compress_ratios entry per DSpark draft layer
    (all zero) after the ``num_hidden_layers`` target entries; the official
    inference config states this explicitly as ``n_mtp_layers: 3``."""
    return len(cfg["compress_ratios"]) - cfg["num_hidden_layers"]


def target_checkpoint_layout(cfg: dict) -> LayoutT:
    """Expected checkpoint inventory of the 43-layer target model (non-mtp keys)."""
    layout: LayoutT = {}
    dim = cfg["hidden_size"]
    hc = cfg["hc_mult"]

    _bf16(layout, "embed.weight", cfg["vocab_size"], dim)
    _bf16(layout, "head.weight", cfg["vocab_size"], dim)
    _bf16(layout, "norm.weight", dim)
    _f32(layout, "hc_head_fn", hc, hc * dim)
    _f32(layout, "hc_head_base", hc)
    _f32(layout, "hc_head_scale", 1)

    for i in range(cfg["num_hidden_layers"]):
        _block(
            layout,
            f"layers.{i}",
            cfg,
            ratio=cfg["compress_ratios"][i],
            hash_routed=i < cfg["num_hash_layers"],
        )
    return layout


def dspark_checkpoint_layout(cfg: dict) -> LayoutT:
    """Expected checkpoint inventory of the DSpark draft head (``mtp.*`` keys).

    Per the official reference (``inference/model.py`` DSparkBlock):
      * every stage is a full DSV4 block with ratio-0 (sliding-window)
        attention and bias-routed (non-hash) MoE,
      * stage 0 additionally owns ``main_proj`` (concat of the target-layer
        context features -> hidden) + ``main_norm``,
      * the last stage additionally owns the draft's own hyper-connection
        head, final norm, markov head (W1/W2, rank ``dspark_markov_rank``)
        and confidence head (Linear(dim + markov_rank -> 1, no bias)).
    """
    layout: LayoutT = {}
    dim = cfg["hidden_size"]
    hc = cfg["hc_mult"]
    n_mtp = num_draft_layers(cfg)
    rank = cfg["dspark_markov_rank"]

    for s in range(n_mtp):
        prefix = f"mtp.{s}"
        _block(layout, prefix, cfg, ratio=0, hash_routed=False)
        if s == 0:
            _fp8(layout, f"{prefix}.main_proj", dim, dim * len(cfg["dspark_target_layer_ids"]))
            _bf16(layout, f"{prefix}.main_norm.weight", dim)
        if s == n_mtp - 1:
            _f32(layout, f"{prefix}.hc_head_fn", hc, hc * dim)
            _f32(layout, f"{prefix}.hc_head_base", hc)
            _f32(layout, f"{prefix}.hc_head_scale", 1)
            _bf16(layout, f"{prefix}.norm.weight", dim)
            _bf16(layout, f"{prefix}.markov_head.markov_w1.weight", cfg["vocab_size"], rank)
            # markov_w2 is Linear(rank -> vocab): weight stored [vocab, rank]
            _bf16(layout, f"{prefix}.markov_head.markov_w2.weight", cfg["vocab_size"], rank)
            _bf16(layout, f"{prefix}.confidence_head.proj.weight", 1, dim + rank)
    return layout


def full_checkpoint_layout(cfg: dict) -> LayoutT:
    layout = target_checkpoint_layout(cfg)
    layout.update(dspark_checkpoint_layout(cfg))
    return layout


def yarn_correction_range(dim: int, base: float, orig_len: int, beta_fast: float, beta_slow: float):
    """YaRN correction-dim range (pure-python mirror of the official
    precompute_freqs_cis; used by tests to cross-check the mlx RoPE)."""

    def correction_dim(num_rotations: float) -> float:
        return dim * math.log(orig_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    return low, high
