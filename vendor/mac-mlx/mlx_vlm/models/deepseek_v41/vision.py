"""DeepSeek-V4.1 ViT and aligner; names/layout match the original BF16 tensors.

See vision_checkpoint_metadata.json for all 266 index/header records and
VISION_NOTES.md for the reference arithmetic and preprocessing contract.
"""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.bfloat16)

    def __call__(self, x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
                * self.weight.astype(mx.float32)).astype(x.dtype)


def rotary_frequencies(n_h, n_w, head_dim, theta):
    dim = head_dim // 2
    inv = theta ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    h = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    w = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    freq = (mx.stack([h, w], axis=-1).reshape(-1, 2, 1) * inv).reshape(-1, dim)
    return mx.cos(freq)[:, None, :], mx.sin(freq)[:, None, :]


def apply_rotary(x, cos, sin):
    a, b = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Linear(3 * c.patch_size**2, c.hidden_size)

    def __call__(self, patches):
        return self.proj(patches.reshape(patches.shape[0], -1))


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads = c.num_attention_heads
        self.head_dim = c.hidden_size // self.n_heads
        self.wqkv = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.wo = nn.Linear(c.hidden_size, c.hidden_size)

    def __call__(self, x, cos, sin):
        n = x.shape[0]
        q, k, v = [t.reshape(n, self.n_heads, self.head_dim)
                   for t in mx.split(self.wqkv(x), 3, axis=-1)]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None], scale=self.head_dim**-0.5)
        return self.wo(o[0].transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.hidden_size, 2 * c.intermediate_size, bias=False)
        self.w2 = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.hidden_size), RMSNorm(c.hidden_size)
        self.attn, self.mlp = Attention(c), MLP(c)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.hidden_size % c.num_attention_heads or (c.hidden_size // c.num_attention_heads) % 4:
            raise ValueError("Vision head dimension must be divisible by four for 2-D RoPE")
        self.config = c
        self.patch_embed = PatchEmbed(c)
        self.blocks = [Block(c) for _ in range(c.num_hidden_layers)]
        self.norm = RMSNorm(c.hidden_size)

    def __call__(self, patches, n_h, n_w):
        c = self.config
        if n_h < 1 or n_w < 1 or patches.shape != (n_h * n_w, 3, c.patch_size, c.patch_size):
            raise ValueError("Vision patches must match [height*width, 3, patch_size, patch_size]")
        x = self.patch_embed(patches)
        cos, sin = rotary_frequencies(n_h, n_w, c.hidden_size // c.num_attention_heads, c.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


def downsample_grid(x, n_h, n_w, ratio):
    """F.unfold ordering: channel, within-cell row, within-cell column."""
    d = x.shape[-1]
    x = x.reshape(n_h, n_w, d)
    x = mx.pad(x, [(0, -n_h % ratio), (0, -n_w % ratio), (0, 0)])
    h, w = x.shape[:2]
    return x.reshape(h // ratio, ratio, w // ratio, ratio, d).transpose(
        0, 2, 4, 1, 3).reshape(-1, d * ratio**2)


class Aligner(nn.Module):
    def __init__(self, c, text_dim):
        super().__init__()
        self.downsample_ratio = c.downsample_ratio
        self.w1 = nn.Linear(c.hidden_size * c.downsample_ratio**2, text_dim)
        self.w2 = nn.Linear(text_dim, text_dim)

    def __call__(self, x, n_h, n_w):
        x = downsample_grid(x, n_h, n_w, self.downsample_ratio)
        # nn.gelu is the exact erf form, as in torch F.gelu (approximate='none').
        return self.w2(nn.gelu(self.w1(x)))


def merge_image_embeddings(model, input_ids, embeddings, images):
    """Batch-one spans from encoding.prepare_vision_inputs, in prompt order."""
    from .encoding_dsv41 import IMAGE, image_token_types

    if input_ids.shape[0] != 1:
        raise ValueError("Vision currently supports batch one")
    pieces, cursor = [], 0
    ids = input_ids[0].tolist()
    for img in images:
        r = model.config.vision_config.downsample_ratio
        h, w = (img.n_vit_h + r - 1) // r, (img.n_vit_w + r - 1) // r
        types = image_token_types(h, w)
        end = img.start + len(types)
        if list(img.types) != types or img.start < cursor or end > len(ids):
            raise ValueError("Invalid image span layout or overlapping spans")
        if ids[img.start:end] != [model.config.image_token_id] * len(types):
            raise ValueError("Image span must contain only configured image token ids")
        patches = mx.array(img.patches).astype(embeddings.dtype)
        rows = model.aligner(model.vision(patches, img.n_vit_h, img.n_vit_w),
                             img.n_vit_h, img.n_vit_w).astype(embeddings.dtype)
        markers = [model.image_start, None, model.image_newline, model.image_end]
        row = 0
        span = []
        for kind in types:
            span.append(rows[row] if kind == IMAGE else markers[kind].astype(embeddings.dtype))
            row += kind == IMAGE
        pieces.extend([embeddings[:, cursor:img.start], mx.stack(span)[None]])
        cursor = end
    pieces.append(embeddings[:, cursor:])
    return mx.concatenate(pieces, axis=1)
