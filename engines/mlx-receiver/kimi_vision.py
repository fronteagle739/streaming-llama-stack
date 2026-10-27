from __future__ import annotations
import hashlib
import io
import struct
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple
PSEUDO_BASE = 1 << 30
PSEUDO_SPAN = (1 << 31) - PSEUDO_BASE - (1 << 20)
MEDIA_PAD = '<|media_pad|>'
MIMG_MAGIC = b'MIMG'

def split_images(raw: bytes) -> List[bytes]:
    raw = bytes(raw)
    if raw[:4] != MIMG_MAGIC:
        return [raw]
    n = struct.unpack_from('<I', raw, 4)[0]
    sizes = [struct.unpack_from('<I', raw, 8 + 4 * i)[0] for i in range(n)]
    off = 8 + 4 * n
    out = []
    for size in sizes:
        out.append(raw[off:off + size])
        off += size
    if off != len(raw):
        raise ValueError(f'MIMG blob sizes do not add up: {off} != {len(raw)}')
    return out

def pack_images(images: Sequence[bytes]) -> bytes:
    if len(images) == 1:
        return bytes(images[0])
    head = MIMG_MAGIC + struct.pack('<I', len(images)) + b''.join((struct.pack('<I', len(b)) for b in images))
    return head + b''.join((bytes(b) for b in images))

def pseudo_base(image_bytes: bytes) -> int:
    digest = hashlib.sha256(image_bytes).digest()
    return PSEUDO_BASE + int.from_bytes(digest[:8], 'little') % PSEUDO_SPAN

def is_pseudo(token_id: int) -> bool:
    return token_id >= PSEUDO_BASE

class KimiVision:

    def __init__(self, root_config: dict, weights: Dict[str, 'object'], feature_cache_size: int=8):
        import mlx.core as mx
        from mlx_vlm.models.kimi_k25.config import ModelConfig
        from mlx_vlm.models.kimi_k25.kimi_k25 import KimiK25MultiModalProjector
        from mlx_vlm.models.kimi_k25.processing_kimi_k25 import KimiK25ImageProcessor
        from mlx_vlm.models.kimi_k25.vision import VisionModel
        cfg = dict(root_config)
        cfg.setdefault('text_config', {})
        self.config = ModelConfig.from_dict(cfg)
        self.media_pad_id = int(root_config.get('media_placeholder_token_id', self.config.image_token_index))
        self.tower = VisionModel(self.config.vision_config)
        self.projector = KimiK25MultiModalProjector(self.config)
        tower_w = {k[len('vision_tower.'):]: v for k, v in weights.items() if k.startswith('vision_tower.')}
        proj_w = {k[len('mm_projector.'):]: v for k, v in weights.items() if k.startswith('mm_projector.')}
        if not tower_w or not proj_w:
            raise ValueError(f'checkpoint has no vision weights (tower={len(tower_w)} projector={len(proj_w)})')
        tower_w = {k.replace('encoder.', ''): v for k, v in tower_w.items()}
        tower_w = {k[len('vision_tower.'):]: v for k, v in self.tower.sanitize({'vision_tower.' + k: v for k, v in tower_w.items()}).items()}
        self.tower.load_weights(list(tower_w.items()), strict=True)
        self._quantize_like_checkpoint(self.projector.proj, proj_w, 'proj.')
        self.projector.load_weights(list(proj_w.items()), strict=True)
        self.tower.eval()
        self.projector.eval()
        mx.eval(self.tower.parameters(), self.projector.parameters())
        self.processor = KimiK25ImageProcessor(config=root_config)
        self.merge_area = int(self.processor.merge_kernel_size[0] * self.processor.merge_kernel_size[1])
        self._features: 'OrderedDict[int, object]' = OrderedDict()
        self._cache_size = int(feature_cache_size)
        self.encoded_images = 0

    @staticmethod
    def _quantize_like_checkpoint(layers, weights, prefix):
        import mlx.nn as nn
        for i, layer in enumerate(layers):
            scales = weights.get(f'{prefix}{i}.scales')
            if scales is None or not isinstance(layer, nn.Linear):
                continue
            out_f, in_f = layer.weight.shape
            packed = weights[f'{prefix}{i}.weight'].shape[1]
            bits = packed * 32 // in_f
            group = in_f // int(scales.shape[1])
            if packed * 32 != bits * in_f or group * int(scales.shape[1]) != in_f:
                raise ValueError(f'cannot infer quantization of {prefix}{i}: weight {packed} scales {scales.shape}')
            layers[i] = nn.QuantizedLinear.from_linear(layer, group_size=group, bits=bits)

    def preprocess(self, image_bytes: bytes):
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        feats = self.processor([image])
        grid = feats['image_grid_hws']
        h, w = (int(x) for x in grid.tolist()[0])
        return (feats['pixel_values'], grid, h * w // self.merge_area)

    def features(self, image_bytes: bytes, base: int):
        import mlx.core as mx
        hit = self._features.get(base)
        if hit is not None:
            self._features.move_to_end(base)
            return hit
        import time
        t0 = time.time()
        pixels, grid, cells = self.preprocess(image_bytes)
        hidden = self.tower(pixels.transpose(0, 2, 3, 1), output_hidden_states=True, grid_thw=grid)
        feats = mx.concatenate(self.projector(hidden), axis=0)
        mx.eval(feats)
        log(f'encoded image {base:#x}: {cells} cells in {time.time() - t0:.2f}s')
        if int(feats.shape[0]) != cells:
            raise ValueError(f'vision tower returned {feats.shape[0]} rows for {cells} image cells')
        self.encoded_images += 1
        self._features[base] = feats
        while len(self._features) > self._cache_size:
            self._features.popitem(last=False)
        return feats

class VisionPrompt:

    def __init__(self, token_ids: List[int], images: List[Tuple[int, int, int, bytes]]):
        self.token_ids = token_ids
        self.images = images

    def image_at(self, token_id: int):
        for pos, cells, base, data in self.images:
            if base <= token_id < base + cells:
                return (base, data, token_id - base)
        raise KeyError(f'pseudo id {token_id} belongs to no image of this prompt')

def build_vision_prompt(vision: KimiVision, text_ids: Sequence[int], image_blobs: Sequence[bytes]) -> VisionPrompt:
    pads = [i for i, t in enumerate(text_ids) if int(t) == vision.media_pad_id]
    if len(pads) != len(image_blobs):
        raise ValueError(f'prompt has {len(pads)} image placeholders ({MEDIA_PAD}) but {len(image_blobs)} images were sent')
    out: List[int] = []
    images = []
    bases = []
    img = 0
    for t in text_ids:
        t = int(t)
        if is_pseudo(t):
            raise ValueError(f'prompt token {t} collides with the image pseudo-id range')
        if t != vision.media_pad_id:
            out.append(t)
            continue
        data = bytes(image_blobs[img])
        img += 1
        _pixels, _grid, cells = vision.preprocess(data)
        base = pseudo_base(data)
        for other_base, other_cells, other_data in bases:
            overlap = base < other_base + other_cells and other_base < base + cells
            if overlap and other_data != data:
                raise ValueError('two different images hashed to overlapping pseudo-id ranges')
        bases.append((base, cells, data))
        images.append((len(out), cells, base, data))
        out.extend(range(base, base + cells))
    return VisionPrompt(out, images)

def embed_rows(vision: KimiVision, prompt: VisionPrompt, embed_tokens, ids: Sequence[int]):
    import mlx.core as mx
    ids = [int(t) for t in ids]
    text = mx.array([[0 if is_pseudo(t) else t for t in ids]], dtype=mx.int32)
    rows = embed_tokens(text)
    i = 0
    while i < len(ids):
        if not is_pseudo(ids[i]):
            i += 1
            continue
        base, data, first = prompt.image_at(ids[i])
        j = i
        while j < len(ids) and ids[j] == base + first + (j - i):
            j += 1
        feats = vision.features(data, base)
        rows[:, i:j, :] = feats[first:first + (j - i)].astype(rows.dtype)
        i = j
    return rows

def forward_embeddings(lang, rows, cache):
    from mlx_lm.models.base import create_attention_mask
    inner = lang.model
    if getattr(inner, 'pipeline_size', 1) != 1:
        raise RuntimeError('image prefill does not support a pipelined Kimi model')
    h = rows
    mask = create_attention_mask(h, cache[0], return_array=True)
    for layer, c in zip(inner.pipeline_layers, cache):
        h = layer(h, mask, cache=c)
    return lang.lm_head(inner.norm(h))

def log(msg: str) -> None:
    sys.stderr.write(f'[kimi-vision] {msg}\n')
    sys.stderr.flush()
