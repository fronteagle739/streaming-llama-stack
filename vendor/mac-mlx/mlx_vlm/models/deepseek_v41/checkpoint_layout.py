"""Layout of the ORIGINAL fp8/fp4 release (not a requantized checkpoint).

Names are relative to LanguageModel; the VLM wrapper adds ``language_model.``.
Shapes/dtypes describe the destination, including the full stack for experts.
An engram entry's ``outputs`` enumerates its chunks; ``dst`` names chunk zero.
Drop entries retain the source shape/dtype and have no destination.
Planning and expected_parameters require neither MLX nor checkpoint data files.

The target constructs DSpark when the config describes its stages. The same
stage-count predicate controls construction and the ``mtp.*`` load contract.
Vision is handled by the VLM wrapper before this text-only planner is called.
"""

from collections import defaultdict

DSPARK_PREFIX = "dspark.stages."


def _plain(name, shape, dtype="BF16"):
    yield name, dtype, list(shape)


def _fp8(name, out, inp):
    if out % 32 or inp % 32:
        raise ValueError(f"{name}: fp8 dimensions must be divisible by 32")
    yield name + ".weight", "F8_E4M3", [out, inp]
    yield name + ".scale", "F8_E8M0", [out // 32, inp // 32]


def dspark_stage_count(c) -> int:
    """Number of DSpark stages this config describes, 0 when it describes none.

    The release appends ``num_nextn_predict_layers`` compress ratios (all 0 --
    the DSpark blocks are sliding-window only) after the backbone's, so a
    ``compress_ratios`` that stops at ``num_hidden_layers`` means no MTP stack.
    """
    extra = len(c.compress_ratios) - c.num_hidden_layers
    if extra <= 0 or not getattr(c, "dspark_block_size", 0):
        return 0
    if extra != c.num_nextn_predict_layers:
        raise ValueError("compress_ratios must cover every MTP layer or none of them")
    if any(c.compress_ratios[c.num_hidden_layers + s] != 0 for s in range(extra)):
        raise ValueError("DSpark stages must be sliding-window only (compress ratio 0)")
    return extra


def _dspark_source_parameters(c):
    """Yield the release's ``mtp.<stage>.*`` (name, dtype, shape) triples.

    Every stage is a full V4.1 block with the narrower MTP MoE
    (``dspark_n_routed_experts`` experts). Stage 0 adds ``main_proj``/
    ``main_norm``; the last stage adds ``norm``, ``markov_head`` and
    ``confidence_head``. No engram, no compressor, no indexer: the DSpark blocks
    are ratio-0 and the reference never gives them an engram layout.
    """
    stages = dspark_stage_count(c)
    d, h, q = c.hidden_size, c.head_dim, c.q_lora_rank
    m, e = c.moe_intermediate_size, c.dspark_n_routed_experts
    for s in range(stages):
        p = f"mtp.{s}."
        for name, out, inp in (
            ("wq_a", q, d), ("wq_b", c.num_attention_heads * h, q),
            ("wkv", h, d),
            ("wo_a", c.o_groups * c.o_lora_rank, c.num_attention_heads * h // c.o_groups),
            ("wo_b", d, c.o_groups * c.o_lora_rank),
        ):
            yield from _fp8(p + "attn." + name, out, inp)
        yield from _plain(p + "attn.q_norm.weight", [q])
        yield from _plain(p + "attn.kv_norm.weight", [h])
        yield from _plain(p + "attn.attn_sink", [c.num_attention_heads], "F32")
        for sub in ("attn", "ffn"):
            yield from _plain(p + sub + "_norm.weight", [d])
            hc = c.hc_mult * (c.hc_mult + 2)
            for name, shape in (("fn", [hc, c.hc_mult * d]), ("base", [hc]), ("scale", [3])):
                yield from _plain(p + f"hc_{sub}_{name}", shape, "F32")
        yield from _plain(p + "ffn.gate.weight", [e, d])
        for name in ("bias", "bias_vl"):
            yield from _plain(p + "ffn.gate." + name, [e], "F32")
        for name, out, inp in (("w1", m, d), ("w2", d, m), ("w3", m, d)):
            sm = c.n_shared_experts
            yield from _fp8(p + "ffn.shared_experts." + name,
                            out if name == "w2" else out * sm,
                            inp * sm if name == "w2" else inp)
            if inp % 32:
                raise ValueError("Expert input dimensions must be divisible by 32")
            for expert in range(e):
                ep = p + f"ffn.experts.{expert}.{name}"
                yield ep + ".weight", "I8", [out, inp // 2]
                yield ep + ".scale", "F8_E8M0", [out, inp // 32]
        if s == 0:
            taps = len(c.dspark_target_layer_ids)
            if not taps:
                raise ValueError("DSpark needs target layers")
            yield from _fp8(p + "main_proj", d, d * taps)
            yield from _plain(p + "main_norm.weight", [d])
        if s == stages - 1:
            yield from _plain(p + "norm.weight", [d])
            for name in ("embed", "head"):
                yield from _plain(p + f"markov_head.{name}.weight",
                                  [c.vocab_size, c.dspark_markov_rank])
            yield from _plain(p + "confidence_head.proj.weight",
                              [1, d + c.dspark_markov_rank])


def _source_parameters(c):
    """Yield (source name, source dtype, source shape), from architecture alone."""
    d, h, q = c.hidden_size, c.head_dim, c.q_lora_rank
    m, e = c.moe_intermediate_size, c.n_routed_experts
    plain, fp8 = _plain, _fp8

    yield from plain("embed.weight", [c.vocab_size, d])
    yield from plain("norm.weight", [d])
    yield from plain("head.weight", [c.vocab_size, d])
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        for name, out, inp in (
            ("wq_a", q, d), ("wq_b", c.num_attention_heads * h, q),
            ("wkv", h, d),
            ("wo_a", c.o_groups * c.o_lora_rank, c.num_attention_heads * h // c.o_groups),
            ("wo_b", d, c.o_groups * c.o_lora_rank),
        ):
            yield from fp8(p + "attn." + name, out, inp)
        yield from plain(p + "attn.q_norm.weight", [q])
        yield from plain(p + "attn.kv_norm.weight", [h])
        yield from plain(p + "attn.attn_sink", [c.num_attention_heads], "F32")
        if i in c.kv_source_layer_ids:
            yield from plain(p + "attn.compressor.wkv.weight", [h, d])
            yield from plain(p + "attn.compressor.norm.weight", [h])
            if c.compress_ratios[i] > 1:
                yield from plain(p + "attn.compressor.wgate.weight", [h, d])
        if i in c.index_source_layer_ids:
            yield from fp8(p + "attn.indexer.wq_b", c.index_n_heads * c.index_head_dim, q)
            yield from plain(p + "attn.indexer.weights_proj.weight", [c.index_n_heads, d])
            if i in c.kv_source_layer_ids:
                yield from plain(p + "attn.indexer.wk.weight", [c.index_head_dim, h])
                yield from plain(p + "attn.indexer.k_norm.weight", [c.index_head_dim])
        for sub in ("attn", "ffn"):
            yield from plain(p + sub + "_norm.weight", [d])
            hc = c.hc_mult * (c.hc_mult + 2)
            for name, shape in (("fn", [hc, c.hc_mult * d]), ("base", [hc]), ("scale", [3])):
                yield from plain(p + f"hc_{sub}_{name}", shape, "F32")
        yield from plain(p + "ffn.gate.weight", [e, d])
        for name in ("bias", "bias_vl"):
            yield from plain(p + "ffn.gate." + name, [e], "F32")
        for name, out, inp in (("w1", m, d), ("w2", d, m), ("w3", m, d)):
            sm = c.n_shared_experts
            yield from fp8(p + "ffn.shared_experts." + name,
                          out if name == "w2" else out * sm,
                          inp * sm if name == "w2" else inp)
            if inp % 32:
                raise ValueError("Expert input dimensions must be divisible by 32")
            for expert in range(e):
                ep = p + f"ffn.experts.{expert}.{name}"
                yield ep + ".weight", "I8", [out, inp // 2]
                yield ep + ".scale", "F8_E8M0", [out, inp // 32]
        if i in c.engram_layer_ids:
            rows = c.engram_num_embeddings[c.engram_layer_ids.index(i)]
            yield p + "engram.embed.weight", "F8_E4M3", [rows, c.engram_head_dim]
            yield p + "engram.embed.scale", "F8_E8M0", [rows, c.engram_head_dim // 32]
            cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads
            yield from fp8(p + "engram.wkv", d * (c.hc_mult + 1), cols * c.engram_head_dim)
            for name in ("q_weight", "k_weight"):
                yield from plain(p + "engram." + name, [c.hc_mult, d])


def _check_source(source):
    if source not in ("ram", "nvme", "hybrid"):
        raise ValueError("engram_source must be ram, nvme or hybrid")


def _entry(key, dtype, shape, c, source):
    top = {"embed.weight": "model.embed_tokens.weight",
           "norm.weight": "model.norm.weight", "head.weight": "lm_head.weight"}
    if key.startswith("mtp."):
        # ``mtp.<stage>.<rest>`` -> ``dspark.stages.<stage>.<rest>``: a sibling of
        # ``model``/``lm_head``, as the reference's ``Transformer.mtp`` is of
        # ``layers``/``head``. Its MoE is the narrower DSpark one.
        stage, rest = key[len("mtp."):].split(".", 1)
        dst = DSPARK_PREFIX + f"{stage}.{rest}"
        experts = c.dspark_n_routed_experts
    else:
        dst = top.get(key, "model." + key)
        experts = c.n_routed_experts
    op = "rename" if dst != key else "copy"
    out_dtype = {"BF16": "bfloat16", "F32": "float32"}.get(dtype, dtype)
    out_shape = list(shape)
    extra = {}
    if ".engram.embed." in key:
        if source != "ram":
            return dict(dst=None, op="drop", expect_shape=list(shape), expect_dtype=dtype)
        is_weight = key.endswith(".weight")
        out_dtype = "uint32" if is_weight else "uint8"
        width = shape[1] // 4 if is_weight else shape[1]
        base = dst if is_weight else dst + "s"
        chunk_rows = -(-shape[0] // c.engram_ram_chunks)
        outputs = []
        for j in range(c.engram_ram_chunks):
            start = j * chunk_rows
            rows = min(chunk_rows, shape[0] - start)
            if rows < 0:
                raise ValueError("engram_ram_chunks creates negative-sized chunks")
            outputs.append(dict(dst=f"{base}.{j}", expect_shape=[rows, width],
                                expect_dtype=out_dtype, row_start=start, row_stop=start + rows))
        return dict(dst=outputs[0]["dst"], op="engram_chunk",
                    expect_shape=outputs[0]["expect_shape"], expect_dtype=out_dtype, outputs=outputs)
    if ".ffn.experts." in key:
        parts = key.split(".")
        proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[parts[5]]
        kind = "weight" if parts[6] == "weight" else "scales"
        dst = dst.split(".ffn.experts.")[0] + f".ffn.switch_mlp.{proj}.{kind}"
        out_dtype = "uint32" if kind == "weight" else "uint8"
        out_shape = [experts, shape[0], shape[1] // 4 if kind == "weight" else shape[1]]
        op, extra = "expert_stack", {"index": int(parts[4])}
    else:
        for sub in ("attn", "ffn"):
            for param in ("fn", "base", "scale"):
                dst = dst.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
        if dst.endswith(".ffn.gate.bias"):
            dst = dst[:-4] + "e_score_correction_bias"
        for old, new in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            dst = dst.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
        if dtype in ("F8_E4M3", "F8_E8M0"):
            if ".attn.wo_a." in key:
                return dict(
                    dst=dst.rsplit(".", 1)[0] + ".weight",
                    op="dequant_bf16_reshape", expect_dtype="bfloat16",
                    expect_shape=[c.o_groups, c.o_lora_rank,
                                  c.num_attention_heads * c.head_dim // c.o_groups],
                )
            op = "view_u32_scale32"
            if key.endswith(".scale"):
                dst += "s"
                out_shape, out_dtype = [shape[0] * 32, shape[1]], "uint8"
            else:
                out_shape, out_dtype = [shape[0], shape[1] // 4], "uint32"
    return dict(dst=dst, op=op, expect_shape=out_shape, expect_dtype=out_dtype, **extra)


def _outputs(entry):
    return entry.get("outputs", [entry]) if entry["dst"] is not None else []


def expected_parameters(text_config, engram_source):
    """Map runtime parameter names to (MLX dtype name, shape list), config only."""
    _check_source(engram_source)
    result = {}
    for key, dtype, shape in _source_parameters(text_config):
        for out in _outputs(_entry(key, dtype, shape, text_config, engram_source)):
            result[out["dst"]] = (out["expect_dtype"], out["expect_shape"])
    result.update(dspark_expected_parameters(text_config))
    return result


def plan_v41(tensor_meta: dict, text_config, engram_source: str) -> dict:
    """Validate a full release manifest and plan every key, rejecting unknown keys.

    Metadata values start with [safetensors dtype, shape]; shard/byte fields may
    follow. Expert entries name the complete stacked output plus their index.
    """
    _check_source(engram_source)
    specs = {k: (d, s) for k, d, s in _source_parameters(text_config)}
    specs.update({k: (d, s) for k, d, s in _dspark_source_parameters(text_config)})
    missing = specs.keys() - tensor_meta.keys()
    if missing:
        raise ValueError(f"Missing checkpoint keys: {sorted(missing)[:8]}")
    dropped = ("vision.", "aligner.")
    if not dspark_stage_count(text_config):
        dropped += ("mtp.",)
    plan = {}
    for key, meta in tensor_meta.items():
        dtype, shape = meta[:2]
        if key.startswith(tuple(dropped)) or key in ("image_start", "image_end", "image_newline"):
            plan[key] = dict(dst=None, op="drop", expect_shape=list(shape), expect_dtype=dtype)
            continue
        if key not in specs:
            raise ValueError(f"Unrecognized checkpoint key: {key}")
        if (dtype, list(shape)) != specs[key]:
            raise ValueError(f"{key}: got {(dtype, shape)}, expected {specs[key]}")
        plan[key] = _entry(key, dtype, shape, text_config, engram_source)
    return plan


def sanitize_weights_v41(weights: dict, text_config, engram_source: str) -> dict:
    """Execute the layout, materializing wo_a dequantization one layer at a time.

    MLX loads F8_E4M3/F8_E8M0 as byte payloads. Use views (not numeric
    conversions) so exponent bytes and fp4 nibbles remain bit-exact.
    Already sanitized complete dictionaries are accepted for loader re-entry.
    """
    import mlx.core as mx

    expected = expected_parameters(text_config, engram_source)

    def validate(result):
        if result.keys() != expected.keys():
            raise ValueError("Sanitized parameter keys do not match V4.1 architecture")
        for key, (dtype, shape) in expected.items():
            v = result[key]
            if list(v.shape) != shape or v.dtype != getattr(mx, dtype):
                raise ValueError(f"{key}: expected {dtype} {shape}, got {v.dtype} {v.shape}")
        return result

    if weights.keys() == expected.keys():
        return validate(weights)
    specs = {k: d for k, d, _ in _source_parameters(text_config)}
    specs.update({k: d for k, d, _ in _dspark_source_parameters(text_config)})
    actual_types = {"BF16": (mx.bfloat16,), "F32": (mx.float32,),
                    "I8": (mx.int8, mx.uint8), "F8_E4M3": (mx.uint8,),
                    "F8_E8M0": (mx.uint8,)}
    metadata = {}
    for key, value in weights.items():
        dtype = specs.get(key)
        if dtype is not None and value.dtype not in actual_types[dtype]:
            raise ValueError(f"{key}: expected raw {dtype} payload, got {value.dtype}")
        metadata[key] = [dtype or str(value.dtype), list(value.shape)]
    plan = plan_v41(metadata, text_config, engram_source)
    return validate(_execute_plan(plan, weights, text_config, mx))


def _execute_plan(plan: dict, weights: dict, text_config, mx) -> dict:
    """Apply one planned layout to raw payloads. Shared by the full-checkpoint
    sanitizer and the DSpark-only one, so both do the same byte operations."""
    result, groups = {}, defaultdict(list)
    for key, entry in plan.items():
        op, dst = entry["op"], entry["dst"]
        if op == "drop":
            continue
        if op == "expert_stack":
            groups[dst].append((entry["index"], key))
            continue
        if op == "dequant_bf16_reshape":
            # Both source entries name the same output; consume the pair once.
            if key.endswith(".scale"):
                continue
            scale = weights[key[:-6] + "scale"]
            w = mx.dequantize(weights[key].view(mx.uint32),
                              mx.repeat(scale, 32, axis=0),
                              group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
            w = w.reshape(text_config.o_groups, text_config.o_lora_rank, -1)
            mx.eval(w)
            result[dst] = w
            continue
        v = weights[key]
        if op == "engram_chunk":
            for out in entry["outputs"]:
                chunk = v[out["row_start"]:out["row_stop"]]
                result[out["dst"]] = chunk.view(getattr(mx, out["expect_dtype"]))
            continue
        if op == "view_u32_scale32":
            v = mx.repeat(v.view(mx.uint8), 32, axis=0) if key.endswith(".scale") else v.view(mx.uint32)
        result[dst] = v
    for dst, members in groups.items():
        members.sort()
        count = (text_config.dspark_n_routed_experts if dst.startswith(DSPARK_PREFIX)
                 else text_config.n_routed_experts)
        if [i for i, _ in members] != list(range(count)):
            raise ValueError(f"Incomplete expert stack: {dst}")
        dtype = mx.uint32 if dst.endswith(".weight") else mx.uint8
        result[dst] = mx.stack([weights[key].view(dtype) for _, key in members])
    return result


def dspark_expected_parameters(text_config, prefix: str = DSPARK_PREFIX) -> dict:
    """The DSpark stages' runtime parameters alone, under ``prefix``."""
    cut = len(DSPARK_PREFIX)
    return {prefix + out["dst"][cut:]: (out["expect_dtype"], out["expect_shape"])
            for key, dtype, shape in _dspark_source_parameters(text_config)
            for out in _outputs(_entry(key, dtype, shape, text_config, "hybrid"))}


def _normalize_dspark_key(key: str) -> str:
    """Accept the release (``mtp.<s>.``), the target tree (``dspark.stages.<s>.``)
    and a split drafter's (``stages.<s>.``) spelling of the same tensor."""
    for prefix in (DSPARK_PREFIX, "stages."):
        if key.startswith(prefix):
            return "mtp." + key[len(prefix):]
    return key


def sanitize_dspark_weights(weights: dict, text_config, prefix: str = DSPARK_PREFIX) -> dict:
    """Map the release's raw ``mtp.<stage>.*`` payloads onto ``prefix<stage>.*``.

    Same fp8 pair / fp4 expert-stack / wo_a dequantization byte operations as the
    backbone, from the same layout operations. The separate drafter can read
    either a full release dictionary or a checkpoint containing only MTP.
    Already-sanitized dictionaries are returned untouched (loader re-entry).
    """
    import mlx.core as mx

    expected = dspark_expected_parameters(text_config, prefix)
    if weights.keys() == expected.keys():
        return weights
    specs = {k: (d, s) for k, d, s in _dspark_source_parameters(text_config)}
    meta, payloads = {}, {}
    for key, value in weights.items():
        norm = _normalize_dspark_key(key)
        if norm not in specs:
            continue
        dtype, shape = specs[norm]
        if list(value.shape) != shape:
            raise ValueError(f"{key}: expected {dtype} {shape}, got {value.shape}")
        meta[norm], payloads[norm] = [dtype, shape], value
    missing = specs.keys() - meta.keys()
    if missing:
        raise ValueError(f"Missing DSpark keys: {sorted(missing)[:8]}")
    plan = {k: _entry(k, *meta[k], text_config, "hybrid") for k in meta}
    result = _execute_plan(plan, payloads, text_config, mx)
    cut = len(DSPARK_PREFIX)
    result = {prefix + k[cut:]: v for k, v in result.items()}
    if result.keys() != expected.keys():
        raise ValueError("Sanitized DSpark keys do not match the architecture")
    return result


def make_quantization_config_v41(model):
    """Select native quantized modules after the graph has been constructed."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    config = {"group_size": 64, "bits": 8, "mode": "affine"}
    for key, _ in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if key.startswith(("vision.", "aligner.")):
            # All release vision/aligner headers are BF16, with no scales.
            # Explicit graph entries prevent the language attn.w* rule or a
            # global fallback from quantizing these weights.
            config[key] = False
        elif ".ffn.switch_mlp." in key and key.endswith("_proj"):
            config[key] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        elif (".ffn.shared_experts." in key or (".attn.w" in key and not key.endswith(".attn.wo_a"))
              or ".attn.indexer.wq" in key or key.endswith(".engram.wkv")
              or key.endswith(".main_proj")):
            config[key] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        elif ".markov_head." in key or ".confidence_head." in key:
            # bf16 in the release; quantizing a 129280x256 table would change
            # the draft logits, and the confidence head is scored in fp32.
            config[key] = False
    return config
