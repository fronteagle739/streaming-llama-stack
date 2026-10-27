"""CPU-readable, exact layout of the original BF16 vision checkpoint."""

MARKERS = ("image_start", "image_end", "image_newline")


def is_vision_key(key):
    return key.startswith(("vision.", "aligner.")) or key in MARKERS


def expected_vision_parameters(config, text_dim):
    c, d = config, config.hidden_size
    result = {
        "vision.patch_embed.proj.weight": ("BF16", [d, 3 * c.patch_size**2]),
        "vision.patch_embed.proj.bias": ("BF16", [d]),
        "vision.norm.weight": ("BF16", [d]),
        "aligner.w1.weight": ("BF16", [text_dim, d * c.downsample_ratio**2]),
        "aligner.w1.bias": ("BF16", [text_dim]),
        "aligner.w2.weight": ("BF16", [text_dim, text_dim]),
        "aligner.w2.bias": ("BF16", [text_dim]),
        **{k: ("BF16", [text_dim]) for k in MARKERS},
    }
    for i in range(c.num_hidden_layers):
        for name, shape in {
            "norm1.weight": [d], "norm2.weight": [d],
            "attn.wqkv.weight": [3 * d, d], "attn.wqkv.bias": [3 * d],
            "attn.wo.weight": [d, d], "attn.wo.bias": [d],
            "mlp.w1.weight": [2 * c.intermediate_size, d],
            "mlp.w2.weight": [d, c.intermediate_size],
        }.items():
            result[f"vision.blocks.{i}.{name}"] = ("BF16", shape)
    return result


def validate_vision_metadata(metadata, config, text_dim):
    expected = expected_vision_parameters(config, text_dim)
    if metadata.keys() != expected.keys():
        raise ValueError(f"Vision keys: missing={sorted(expected.keys() - metadata.keys())}, "
                         f"unexpected={sorted(metadata.keys() - expected.keys())}")
    for key, (dtype, shape) in expected.items():
        actual = metadata[key]
        if (actual["dtype"], list(actual["shape"])) != (dtype, shape):
            raise ValueError(f"{key}: expected {dtype} {shape}, got {actual}")


def sanitize_vision_weights(weights, config, text_dim):
    # The release has no packed vision weights or scales. Preserve every BF16
    # payload and its layout; never reshape or requantize to fit a language rule.
    import mlx.core as mx

    metadata = {k: {"dtype": "BF16" if v.dtype == mx.bfloat16 else str(v.dtype),
                    "shape": list(v.shape)} for k, v in weights.items()}
    validate_vision_metadata(metadata, config, text_dim)
    return weights
