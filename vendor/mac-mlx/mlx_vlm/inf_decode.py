"""Inferencerlabs ``INF`` checkpoint name decoder.

Some inferencerlabs MLX exports (the ``*-INF`` variants, e.g.
``DeepSeek-V4-Pro-MLX-2.8bit-INF``) ship with **obfuscated tensor names** and a
binary ``model.inf`` sidecar *instead of* a ``model.safetensors.index.json``.
Every tensor key in the ``*.safetensors`` shards is a base64 string whose decoded
bytes are the real tensor name XOR-ed with a single fixed keystream that is shared
across all shards of the checkpoint.

This module recovers that keystream **from the checkpoint itself** (no reliance on
the proprietary ``model.inf`` blob) and exposes:

* :func:`is_inf_checkpoint` — detect an INF checkpoint dir.
* :func:`decode_keystream` — recover the XOR keystream from the shard headers.
* :func:`build_name_map` — obfuscated-name -> real-name for one shard or all.
* :func:`build_weight_map` — real-name -> shard-basename index (the equivalent of
  ``model.safetensors.index.json``'s ``weight_map``, needed by the distributed /
  pipeline loader).
* :func:`write_index_json` — materialise a real ``model.safetensors.index.json``
  so the existing indexed-load path (incl. pipeline rank shard selection) works
  unchanged.

The recovery is anchored on the embedding-table weight, which is the only
``uint32`` (MLX-packed) tensor whose first dimension equals ``vocab_size`` — a
shape that is unambiguous in every supported architecture. From that single
(ciphertext, known-plaintext) pair we obtain the leading keystream bytes, then we
extend the keystream to full length by majority vote over every shard's headers
using the architecture's deterministic, non-ambiguous tensor-name grammar. The
result is validated (every decoded name must be clean ASCII in the expected
namespace) before it is used; a checkpoint that fails validation raises rather
than loading garbage.
"""
from __future__ import annotations

import base64
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Real names always live in this namespace once decoded. Used both to anchor the
# keystream search and to validate the final decode.
_CLEAN_NAME = re.compile(r"^[A-Za-z0-9_.]+$")
# A VLM-wrapper checkpoint also carries the vision stack, which lives OUTSIDE the text
# namespace. Measured 2026-08-11 on Kimi-K2.7-Code-MLX-Q3.5-INF: with the Kimi text grammar in
# place, 2558 of 2897 names decoded and exactly 339 kept failing validation regardless of which
# text names were added -- those are the vision-tower / projector tensors, which decode fine but
# were rejected by this namespace whitelist.
_ROOT_PREFIXES = (
    "model.", "lm_head", "language_model.",
    # Vision stack of a VLM wrapper. `mm_projector.` is the name Kimi-K2.x actually uses --
    # confirmed by decoding the residual failures rather than guessing: they came out as clean
    # 'mm_projector.proj.0.weight' / 'mm_projector.pre_norm.bias' etc., i.e. the keystream was
    # already correct and only this whitelist was short.
    "vision_tower.", "vision_model.", "mm_projector.", "multi_modal_projector.",
)


def _b64decode(name: str) -> bytes:
    # Inferencer names are standard base64 (with ``+`` and ``/``); pad as needed.
    return base64.b64decode(name + "=" * ((4 - len(name) % 4) % 4))


def _read_safetensors_header(path: Path) -> dict:
    """The shard's header, unmasking it first if this is the Q4i container.

    Every INF entry point starts here -- is_inf_checkpoint, decode_keystream, build_name_map -- so
    teaching this one function about the masked header is what makes the whole module work on a Q4i
    checkpoint. Unmasked shards take the plain path and are unaffected; see read_shard_header.
    """
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        raw = fh.read(hdr_len)
    try:
        return json.loads(raw)
    except Exception:
        header, _mask, _n = read_shard_header(path)
        return header


def _shard_files(model_path: Path) -> List[Path]:
    return sorted(model_path.glob("model-*.safetensors"))


def is_inf_checkpoint(model_path: Path) -> bool:
    """True for an inferencerlabs INF dir: the tensor keys in the safetensors are
    obfuscated (base64 of the XOR-encoded real name).

    Detection keys on the *obfuscated tensor names*, NOT on the absence of a
    safetensors index: we materialise an index from the decode (so the pipeline
    loader can select shards), yet the keys inside the shards stay obfuscated and
    must still be renamed on every load. An existing ``model.inf`` sidecar or an
    ``-INF`` directory name is corroborating but not required."""
    model_path = Path(model_path)
    shards = _shard_files(model_path)
    if not shards:
        return False
    try:
        hdr = _read_safetensors_header(shards[0])
    except Exception:
        return False
    keys = [k for k in hdr if k != "__metadata__"]
    if not keys:
        return False
    # Obfuscated names are pure base64; real weight names always contain dots.
    looks_obfuscated = all("." not in k for k in keys[:8]) and all(
        re.fullmatch(r"[A-Za-z0-9+/=]+", k) for k in keys[:8]
    )
    return looks_obfuscated


def index_keystream(model_path: Path) -> Optional[bytes]:
    """Return the stored INF keystream from an INF-derived index, if present."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        meta = json.loads(index_path.read_text()).get("metadata", {})
        raw = meta.get("inf_keystream_hex")
        if meta.get("inf_decoded") and raw:
            return bytes.fromhex(raw)
    except Exception:
        return None
    return None


def _anchor_pairs(model_path: Path, config: dict) -> List[Tuple[bytes, bytes]]:
    """Return (ciphertext, plaintext) anchor pairs we are certain of.

    The embedding table is the unambiguous anchor: the only packed ``uint32``
    tensor whose leading dim is ``vocab_size``. We anchor on its ``.weight`` and,
    when present, the matching ``.scales`` / ``.biases``.

    The ciphertext length identifies WHICH plaintext name it is, so a VLM-wrapper
    checkpoint has to be handled explicitly: its tensors are prefixed
    ``language_model.``, making the embedding name 40 bytes rather than 25.

    Measured 2026-08-11 on Kimi-K2.7-Code-MLX-Q3.5-INF (``KimiK25ForConditionalGeneration``,
    text stack nested in ``text_config``): all three leading-dim-163840 tensors --
    ``[163840, 1792] U32`` plus two ``[163840, 224] BF16`` scale/bias -- decode to a
    **40-byte** ciphertext. The old ``len(c) != 25`` filter skipped every one of them, so
    ``_anchor_pairs`` returned empty and the loader failed with the misleading
    "could not find the embedding-table anchor tensor (vocab_size=None)" -- misleading twice
    over, since ``vocab`` had in fact resolved correctly from ``text_config`` and only the
    error string printed the absent top-level key."""
    vocab = int(config.get("vocab_size") or config.get("text_config", {}).get("vocab_size") or 0)
    pairs: List[Tuple[bytes, bytes]] = []
    if not vocab:
        return pairs
    # plaintext embedding name -> its exact byte length, longest first so the more specific
    # (prefixed) form is preferred when a checkpoint somehow contains both.
    candidates = [
        b"language_model.model.embed_tokens.weight",  # VLM wrapper (Kimi-K2.5/K2.7, 40 bytes)
        b"model.embed_tokens.weight",                 # text-only (25 bytes)
    ]
    by_len = {len(name): name for name in candidates}
    hdr = _read_safetensors_header(_shard_files(model_path)[0])
    by_role: Dict[int, Dict[str, bytes]] = {}
    for k, info in hdr.items():
        if k == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        dtype = info.get("dtype")
        if not shape or shape[0] != vocab:
            continue
        c = _b64decode(k)
        if len(c) not in by_len:
            continue
        slot = by_role.setdefault(len(c), {})
        if dtype in ("U32", "uint32"):
            slot["weight"] = c
        else:
            slot.setdefault("_other", c)  # scales/biases resolved via keystream below
    for clen, name in sorted(by_len.items(), key=lambda kv: -kv[0]):
        slot = by_role.get(clen) or {}
        if "weight" in slot:
            pairs.append((slot["weight"], name))
            break
    return pairs


def _expected_name_grammar(config: dict) -> List[bytes]:
    """Deterministic, NON-expert tensor names for this architecture.

    These names have unique stems per layer, so they let us extend the keystream
    one position at a time by unambiguous majority vote. Expert/stacked tensors are
    intentionally excluded here (decoded later once the keystream is complete)."""
    n_layers = int(config.get("num_hidden_layers") or config.get("text_config", {}).get("num_hidden_layers") or 0)

    # ---------------------------------------------------------------------------------------
    # KIMI / deepseek_v3 VLM-WRAPPER CHECKPOINTS (e.g. Kimi-K2.7-Code-MLX-Q3.5-INF).
    #
    # The grammar below this branch describes DSV4 (attn.wq_a, compressor.wkv, indexer.*,
    # hc_head.*). Kimi-K2.x is a different architecture entirely: mlx_vlm's kimi_k25
    # LanguageModel wraps mlx_lm.models.deepseek_v3.DeepseekV3Model, and every tensor is
    # additionally prefixed "language_model.". Feeding DSV4 names to a Kimi checkpoint made
    # 2890 of 2897 names fail to decode (measured 2026-08-11), because the keystream can only
    # be extended by names that actually occur.
    #
    # Names derived from the module definitions, not guessed:
    #   deepseek_v3.DeepseekV3Attention -> q_a_proj / q_a_layernorm / q_b_proj /
    #       kv_a_proj_with_mqa / kv_a_layernorm / o_proj  (q_lora_rank is set for K2.x, so the
    #       q_a/q_b pair exists rather than a single q_proj)
    #   DeepseekV3MLP (dense layers)    -> gate_proj / up_proj / down_proj
    #   DeepseekV3MoE                   -> gate.weight, shared_experts.{gate,up,down}_proj
    #   DeepseekV3Model                 -> embed_tokens, norm; plus top-level lm_head
    # Stacked/expert tensors (switch_mlp.*) are deliberately EXCLUDED here for the same reason
    # the DSV4 grammar excludes them: they are resolved once the keystream is complete.
    # ---------------------------------------------------------------------------------------
    tc = config.get("text_config") or {}
    mt = str(tc.get("model_type") or config.get("model_type") or "").lower()

    # ---------------------------------------------------------------------------------------
    # GLM-5.3 / glm_moe_dsa (e.g. GLM-5.3-MLX-Q4i).
    #
    # Same failure mode as the Kimi note below: feeding the DSV4 grammar to a GLM checkpoint left
    # 3474 of 3481 names undecodable and the keystream stuck at 25 bytes against names up to 51,
    # because a keystream can only be extended by names that actually occur.
    #
    # Names taken from the module definitions in models/glm_moe_dsa/_arch_v32_0307.py, not guessed.
    # The classes there are literally DeepseekV32*, so the layout matches the Kimi branch below with
    # two differences: there is NO "language_model." prefix, and GLM carries the sparse INDEXER
    # (Indexer: wq_b / wk / k_norm / weights_proj). k_norm is an nn.LayerNorm, so it has a bias as
    # well as a weight; the RMSNorms do not. switch_mlp.* is excluded for the same reason as
    # everywhere else here -- stacked expert tensors are resolved once the keystream is complete.
    # ---------------------------------------------------------------------------------------
    if mt.startswith("glm_moe") or mt.startswith("glm_moe_dsa"):
        cfg = tc or config
        n_layers = int(cfg.get("num_hidden_layers") or n_layers or 0)
        n_dense = int(cfg.get("first_k_dense_replace") or cfg.get("leading_dense_block_count") or 1)
        q = ("weight", "scales", "biases")
        gn: List[str] = [
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases", "model.norm.weight",
        ]
        for L in range(n_layers):
            b = f"model.layers.{L}"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj",
                         "embed_q", "unembed_out"):
                for s in q:
                    gn.append(f"{b}.self_attn.{proj}.{s}")
            gn.append(f"{b}.self_attn.q_a_layernorm.weight")
            gn.append(f"{b}.self_attn.kv_a_layernorm.weight")
            for proj in ("wq_b", "wk", "weights_proj"):
                for s in q:
                    gn.append(f"{b}.self_attn.indexer.{proj}.{s}")
            gn.append(f"{b}.self_attn.indexer.k_norm.weight")
            gn.append(f"{b}.self_attn.indexer.k_norm.bias")
            gn.append(f"{b}.input_layernorm.weight")
            gn.append(f"{b}.post_attention_layernorm.weight")
            if L < n_dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for s in q:
                        gn.append(f"{b}.mlp.{proj}.{s}")
            else:
                gn.append(f"{b}.mlp.gate.weight")
                gn.append(f"{b}.mlp.gate.e_score_correction_bias")
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for s in q:
                        gn.append(f"{b}.mlp.shared_experts.{proj}.{s}")
        return [n.encode("ascii") for n in gn]

    if tc and (mt.startswith("kimi") or mt.startswith("deepseek_v3")):
        p = "language_model."
        n_dense = int(tc.get("first_k_dense_replace") or tc.get("leading_dense_block_count") or 1)
        q = ("weight", "scales", "biases")
        kn: List[str] = [
            f"{p}lm_head.weight", f"{p}lm_head.scales", f"{p}lm_head.biases",
            f"{p}model.embed_tokens.weight", f"{p}model.embed_tokens.scales",
            f"{p}model.embed_tokens.biases", f"{p}model.norm.weight",
        ]
        for L in range(n_layers):
            b = f"{p}model.layers.{L}"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
                for s in q:
                    kn.append(f"{b}.self_attn.{proj}.{s}")
            # MultiLinear (per-head stacked) -- present in mlx_lm's DeepseekV3Attention as
            # embed_q / unembed_out. Omitting them left 339 of 2897 names undecodable.
            for proj in ("embed_q", "unembed_out"):
                for s in q:
                    kn.append(f"{b}.self_attn.{proj}.{s}")
            kn.append(f"{b}.self_attn.q_a_layernorm.weight")
            kn.append(f"{b}.self_attn.kv_a_layernorm.weight")
            kn.append(f"{b}.input_layernorm.weight")
            kn.append(f"{b}.post_attention_layernorm.weight")
            if L < n_dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for s in q:
                        kn.append(f"{b}.mlp.{proj}.{s}")
            else:
                kn.append(f"{b}.mlp.gate.weight")
                kn.append(f"{b}.mlp.gate.e_score_correction_bias")
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for s in q:
                        kn.append(f"{b}.mlp.shared_experts.{proj}.{s}")
        return [n.encode("ascii") for n in kn]

    names: List[str] = [
        "lm_head.weight", "lm_head.scales", "lm_head.biases",
        "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
        "model.norm.weight", "model.hc_head.fn", "model.hc_head.base", "model.hc_head.scale",
    ]
    for L in range(n_layers):
        b = f"model.layers.{L}"
        for w in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b",
                  "compressor.wkv", "compressor.wgate",
                  "indexer.wq_b", "indexer.weights_proj",
                  "indexer.compressor.wkv", "indexer.compressor.wgate"):
            for s in ("weight", "scales", "biases"):
                names.append(f"{b}.attn.{w}.{s}")
        for n in ("attn.q_norm.weight", "attn.kv_norm.weight", "attn_norm.weight",
                  "ffn_norm.weight", "attn.attn_sink",
                  "attn.compressor.ape", "attn.compressor.norm.weight",
                  "attn.indexer.compressor.ape", "attn.indexer.compressor.norm.weight"):
            names.append(f"{b}.{n}")
        for hc in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                names.append(f"{b}.{hc}.{p}")
        names.append(f"{b}.ffn.gate.weight")
        names.append(f"{b}.ffn.gate.e_score_correction_bias")
        names.append(f"{b}.ffn.gate.tid2eid")
        for me in ("gate_proj", "up_proj", "down_proj"):
            for s in ("weight", "scales", "biases"):
                names.append(f"{b}.ffn.shared_experts.{me}.{s}")
                names.append(f"{b}.ffn.switch_mlp.{me}.{s}")
    return [n.encode("ascii") for n in names]


def decode_keystream(model_path: Path, config: dict) -> bytes:
    """Recover the full XOR keystream for an INF checkpoint from its shard headers.

    Raises ``ValueError`` if a self-consistent keystream cannot be recovered."""
    model_path = Path(model_path)
    pairs = _anchor_pairs(model_path, config)
    if not pairs:
        raise ValueError(
            "INF decode: could not find the embedding-table anchor tensor "
            f"(vocab_size={config.get('vocab_size')}) in {model_path}."
        )
    cipher, plain = pairs[0]
    ks = bytearray(cipher[i] ^ plain[i] for i in range(len(plain)))

    # Gather all unique ciphertext keys across shards.
    ciphers = set()
    for f in _shard_files(model_path):
        for k in _read_safetensors_header(f):
            if k != "__metadata__":
                ciphers.add(_b64decode(k))
    ciphers = list(ciphers)
    if not ciphers:
        raise ValueError(f"INF decode: no tensors found in {model_path}.")
    max_len = max(len(c) for c in ciphers)

    expected = _expected_name_grammar(config)
    # Extend the keystream one byte at a time: at length L, a ciphertext longer
    # than L contributes a vote for ks[L] only if every grammar name sharing the
    # already-decoded prefix agrees on the next byte (unambiguous).
    while len(ks) < max_len:
        L = len(ks)
        votes: Counter = Counter()
        for c in ciphers:
            if len(c) <= L:
                continue
            prefix = bytes(c[i] ^ ks[i] for i in range(L))
            next_chars = set()
            for nb in expected:
                if len(nb) == len(c) and nb[:L] == prefix:
                    next_chars.add(nb[L])
                    if len(next_chars) > 1:
                        break
            if len(next_chars) == 1:
                votes[c[L] ^ next(iter(next_chars))] += 1
        if not votes:
            # Per-byte voting stalled. Try a stronger step: if some ciphertext's
            # already-decoded prefix matches exactly one grammar name (of its
            # length), that whole name is known — fill every remaining keystream
            # byte it covers at once. This unsticks positions where the only
            # same-length candidates differ at the very next byte but agree
            # nowhere useful for single-byte voting.
            advanced = False
            for c in ciphers:
                if len(c) <= L:
                    continue
                prefix = bytes(c[i] ^ ks[i] for i in range(L))
                matches = [nb for nb in expected if len(nb) == len(c) and nb[:L] == prefix]
                if len(matches) == 1:
                    full = matches[0]
                    for i in range(L, len(full)):
                        ks.append(c[i] ^ full[i])
                    advanced = True
                    break
            if not advanced:
                # No grammar evidence for this position; cannot safely continue.
                break
        else:
            ks.append(votes.most_common(1)[0][0])

    keystream = bytes(ks)
    _validate_keystream(ciphers, keystream)
    return keystream


def _validate_keystream(ciphers: List[bytes], keystream: bytes) -> None:
    """Every tensor name must decode to clean ASCII in the expected namespace."""
    bad = 0
    total = 0
    # Report WHAT failed, not just how many. A bare count says nothing about whether the
    # keystream is wrong (garbage everywhere) or the namespace whitelist is merely incomplete
    # (a handful of clean-looking names) -- which is exactly the distinction needed to fix it.
    samples: List[str] = []
    for c in ciphers:
        total += 1
        if len(c) > len(keystream):
            bad += 1
            if len(samples) < 10:
                samples.append(f"<too-long len={len(c)} ks={len(keystream)}>")
            continue
        try:
            name = bytes(c[i] ^ keystream[i] for i in range(len(c))).decode("ascii")
        except UnicodeDecodeError:
            bad += 1
            if len(samples) < 10:
                samples.append("<non-ascii>")
            continue
        if not _CLEAN_NAME.match(name) or not name.startswith(_ROOT_PREFIXES):
            bad += 1
            if len(samples) < 10:
                samples.append(repr(name))
    if bad:
        raise ValueError(
            f"INF decode: keystream failed validation ({bad}/{total} tensor names "
            f"did not decode to clean weight names). Refusing to load to avoid garbage. "
            f"First failures: {'; '.join(samples)}"
        )


def build_name_map(model_path: Path, keystream: bytes, shard: Optional[Path] = None) -> Dict[str, str]:
    """Map obfuscated key -> real name for one shard (or all shards if ``shard`` is None)."""
    files = [shard] if shard is not None else _shard_files(Path(model_path))
    out: Dict[str, str] = {}
    for f in files:
        for k in _read_safetensors_header(Path(f)):
            if k == "__metadata__":
                continue
            c = _b64decode(k)
            out[k] = bytes(c[i] ^ keystream[i] for i in range(len(c))).decode("ascii")
    return out


def quant_groups_from_model(model, weights) -> Dict[tuple, set]:
    """Per-tensor ``(bits, group_size, mode)`` -> set of layer paths for an INF model.

    INF checkpoints use **per-tensor mixed** quantization (e.g. 4-bit/gs=128 for
    attention & shared experts, 2-bit/gs=32 or gs=64 for the routed switch-MLP
    experts), and ``config.json`` carries none of it. We recover each quantized
    module's true ``input_dim`` from the *model graph* (the un-quantized Linear's
    weight shape), then derive its bit width and group size exactly:

        input_dim   = module.weight.shape[-1]
        bits        = packed_last * 32 / input_dim
        group_size  = input_dim / scales_last

    Returns groups keyed by ``(bits, group_size, mode)`` so the caller can issue one
    ``nn.quantize`` per distinct (bits, group_size) combination."""
    import mlx.core as _mx  # local import keeps this module mlx-free until used

    # Map every Linear-ish module path -> its true input dim from the graph.
    from mlx.utils import tree_flatten as _tf

    module_in_dim: Dict[str, int] = {}
    for path, arr in _tf(model.parameters()):
        if path.endswith(".weight") and hasattr(arr, "shape") and len(arr.shape) >= 2:
            module_in_dim[path.rsplit(".weight", 1)[0]] = int(arr.shape[-1])

    groups: Dict[tuple, set] = {}
    for wk, w_arr in weights.items():
        if not wk.endswith(".weight"):
            continue
        sk = wk.rsplit(".weight", 1)[0] + ".scales"
        s_arr = weights.get(sk)
        if s_arr is None or w_arr.dtype != _mx.uint32 or w_arr.ndim not in (2, 3):
            continue
        layer_path = wk.rsplit(".weight", 1)[0]
        in_dim = module_in_dim.get(layer_path) or module_in_dim.get(
            "language_model." + layer_path
        ) or module_in_dim.get(layer_path.removeprefix("language_model."))
        if not in_dim:
            # Fall back to deriving in_dim from scales when the graph lacks it
            # (e.g. stacked SwitchLinear stored as a single 3D param): assume the
            # smallest group size that yields integer bits.
            in_dim = _infer_in_dim(int(w_arr.shape[-1]), int(s_arr.shape[-1]))
            if not in_dim:
                continue
        bits = round(int(w_arr.shape[-1]) * 32 / in_dim)
        group_size = round(in_dim / int(s_arr.shape[-1]))
        groups.setdefault((bits, group_size, "affine"), set()).add(layer_path)
    return groups


def _infer_in_dim(packed_last: int, scales_last: int) -> Optional[int]:
    """Recover input_dim from packed/scales last dims when the graph dim is absent.

    ``packed_last*32/scales_last == bits*group_size``; pick the (bits, gs) with
    bits in {2,3,4,5,6,8} and gs in {32,64,128} — unique in practice."""
    prod = packed_last * 32 / scales_last
    for gs in (32, 64, 128):
        bits = prod / gs
        if abs(bits - round(bits)) < 1e-6 and round(bits) in {2, 3, 4, 5, 6, 8}:
            return scales_last * gs
    return None


def detect_group_size(model_path: Path, keystream: bytes) -> Optional[int]:
    """Infer the MLX quantization ``group_size`` from the shard-0 header.

    INF checkpoints leave ``config.json``'s ``quantization_config`` empty, but the
    per-tensor mixed-quant loader needs the group size to derive each tensor's bit
    width from its packed shapes. For an affine-quantized ``uint32`` weight,
    ``scales_last == input_dim / group_size`` and ``packed_last == input_dim * bits / 32``,
    so for the integer bit widths MLX supports (2/3/4/5/6/8) the only ``group_size``
    in {32, 64, 128} that makes *every* quantized tensor resolve to an integer bit
    width is the true one. We pick that (preferring the smallest, since inferencer
    builds use gs=32 with mixed 2/4-bit)."""
    hdr = _read_safetensors_header(_shard_files(Path(model_path))[0])
    name_of = {}
    for k in hdr:
        if k == "__metadata__":
            continue
        c = _b64decode(k)
        name_of[k] = bytes(c[i] ^ keystream[i] for i in range(len(c))).decode("ascii")
    pairs = []  # (packed_last, scales_last)
    for k, info in hdr.items():
        if k == "__metadata__" or not isinstance(info, dict):
            continue
        if info.get("dtype") not in ("U32", "uint32"):
            continue
        weight_name = name_of.get(k, "")
        scales_name = weight_name.rsplit(".", 1)[0] + ".scales"
        sc = next((kk for kk, nm in name_of.items() if nm == scales_name), None)
        if sc is None:
            continue
        pl = info.get("shape", [0])[-1]
        sl = hdr[sc].get("shape", [0])[-1]
        if pl and sl:
            pairs.append((pl, sl))
    if not pairs:
        return None
    valid_bits = {2, 3, 4, 5, 6, 8}
    for gs in (32, 64, 128):
        ok = True
        for pl, sl in pairs:
            bits = pl * 32 / (sl * gs)
            if abs(bits - round(bits)) > 1e-6 or round(bits) not in valid_bits:
                ok = False
                break
        if ok:
            return gs
    return None


def apply_to_config(model_path: Path, config: dict) -> dict:
    """Populate ``config['quantization_config']['group_size']`` for an INF checkpoint
    so the per-tensor mixed-quant loader can derive each tensor's bit width. No-op if
    the group size is already present or cannot be inferred. Returns ``config``."""
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        qc = {}
        config["quantization_config"] = qc
    if qc.get("group_size"):
        return config
    keystream = decode_keystream(model_path, config)
    gs = detect_group_size(model_path, keystream)
    if gs:
        qc["group_size"] = gs
    return config


def build_weight_map(model_path: Path, keystream: bytes) -> Dict[str, str]:
    """Real-name -> shard-basename, the equivalent of the safetensors index weight_map."""
    out: Dict[str, str] = {}
    for f in _shard_files(Path(model_path)):
        name_map = build_name_map(model_path, keystream, shard=f)
        for real in name_map.values():
            out[real] = f.name
    return out


def decode(model_path: Path, config: dict) -> Tuple[bytes, Dict[str, str]]:
    """Convenience: recover keystream + obf->real map for the whole checkpoint."""
    keystream = decode_keystream(model_path, config)
    return keystream, build_name_map(model_path, keystream)


def write_index_json(model_path: Path, config: dict, force: bool = False) -> Path:
    """Materialise a real ``model.safetensors.index.json`` from the INF decode.

    Idempotent: skips work if a valid INF-derived index already exists. Returns the
    index path. This lets the standard indexed / pipeline load path consume the
    checkpoint with no further INF awareness."""
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists() and not force:
        try:
            meta = json.loads(index_path.read_text())
            if meta.get("metadata", {}).get("inf_decoded"):
                return index_path
        except Exception:
            pass
    keystream = decode_keystream(model_path, config)
    weight_map = build_weight_map(model_path, keystream)
    total_size = sum(p.stat().st_size for p in _shard_files(model_path))
    payload = {
        "metadata": {
            "total_size": total_size,
            "inf_decoded": True,
            "inf_keystream_hex": keystream.hex(),
        },
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(payload))
    return index_path


# --- Q4i: masked safetensors headers --------------------------------------------------------
# The newer inferencerlabs container (GLM-5.3-MLX-Q4i and siblings) additionally XOR-masks the
# safetensors HEADER with a fixed repeating byte sequence. Tensor DATA is left plain — verified on
# the real checkpoint: a sampled BF16 tensor has mean -0.00001, std 0.00101, 100% finite.
#
# Everything above assumes the older shape: a readable JSON header whose tensor NAMES are
# obfuscated. With a masked header, every entry point that starts by reading it fails identically —
# is_inf_checkpoint, decode_keystream, and mx.load itself — with
# "[json.exception.parse_error.101] ... last read: '<U+0012>'" at load time.
#
# Recovering the mask needs nothing secret. The plaintext is JSON, so byte 0 is '{'; only 128
# distinct byte values occur across a 25,825-byte header, so mask and plaintext are both 7-bit; and
# the mask repeats (autocorrelation peaks at 126/147/252/273/399/525/651/672/798/819 — all multiples
# of 21). Solving per residue class and keeping the most JSON-like decode is SELF-CHECKING: the
# header either parses and yields data_offsets that fit inside the file, or it does not.
#
# Measured on GLM-5.3-MLX-Q4i: a 21-byte mask reads 47 of 47 shards with every extent in range.
_Q4I_PERIODS = (21, 42, 63, 7, 14, 126, 147)
_JSONISH = set(b'{}[]",:0123456789_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- /')


def _jsonish_score(buf: bytes) -> int:
    """How much a decoded column looks like safetensors JSON."""
    return sum(1 for v in buf if v in _JSONISH) - 4 * sum(1 for v in buf if v < 32 or v > 126)


def recover_header_mask(cipher: bytes, period: int) -> bytes:
    """Best repeating mask of ``period`` bytes, chosen column-wise by JSON-likeness."""
    mask = bytearray()
    for r in range(period):
        col = cipher[r::period]
        best_k, best_s = 0, -(10 ** 9)
        for k in range(128):
            s = _jsonish_score(bytes(v ^ k for v in col))
            if s > best_s:
                best_s, best_k = s, k
        mask.append(best_k)
    return bytes(mask)


def read_shard_header(path) -> tuple:
    """``(header_dict, mask, header_len)``. ``mask`` is None when the header was never masked."""
    path = Path(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cipher = f.read(n)
    try:
        return json.loads(cipher), None, n
    except Exception:
        pass
    for period in _Q4I_PERIODS:
        mask = recover_header_mask(cipher, period)
        try:
            plain = bytes(v ^ mask[i % period] for i, v in enumerate(cipher))
            return json.loads(plain), mask, n
        except Exception:
            continue
    raise ValueError(f"{path}: header is neither JSON nor a recognised masked header")


def has_masked_header(model_path) -> bool:
    """True when this checkpoint's shards carry a masked header (the Q4i container)."""
    shards = _shard_files(Path(model_path))
    if not shards:
        return False
    try:
        _hdr, mask, _n = read_shard_header(shards[0])
    except Exception:
        return False
    return mask is not None


def load_masked_safetensors(path) -> dict:
    """``mx.load`` equivalent for a masked-header shard.

    Only the header is masked, so tensor bytes are read straight out of the file. BF16 is viewed
    rather than widened: going via float32 would double peak memory for every tensor, which is not
    affordable on a 426 GB checkpoint.
    """
    import mlx.core as mx
    import numpy as np

    path = Path(path)
    hdr, mask, n = read_shard_header(path)
    if mask is None:
        return mx.load(str(path))

    np_dtypes = {
        "F64": np.float64, "F32": np.float32, "F16": np.float16,
        "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
        "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
        "BOOL": np.bool_,
    }
    base = 8 + n
    mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = mm[base + a: base + b]
        dtype, shape = info["dtype"], info["shape"]
        if dtype == "BF16":
            # numpy has no bfloat16: carry the bits as uint16 and reinterpret in MLX.
            arr = mx.array(np.ascontiguousarray(raw.view(np.uint16)).reshape(shape))
            out[name] = arr.view(mx.bfloat16)
        else:
            if dtype not in np_dtypes:
                raise ValueError(f"{path}: unsupported dtype {dtype!r} for tensor {name!r}")
            arr = np.ascontiguousarray(raw.view(np_dtypes[dtype])).reshape(shape)
            out[name] = mx.array(arr)
    return out
