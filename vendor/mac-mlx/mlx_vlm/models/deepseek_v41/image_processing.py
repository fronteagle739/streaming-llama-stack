"""CPU image preparation matching the V4.1 reference image_processor.py."""

import base64
import io
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps


def plan_image_grid(width, height, c):
    if width <= 0 or height <= 0 or c.max_image_tokens < 4:
        raise ValueError("Positive image dimensions and at least four image tokens are required")
    p, r, limit = c.patch_size, c.downsample_ratio, c.max_image_tokens
    if c.max_wh_ratio is not None and width > height * c.max_wh_ratio:
        width = height * c.max_wh_ratio
    if width * height < c.min_pixels:
        ratio = (c.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    bw, bh = math.ceil(width / p) * p, math.ceil(height / p) * p
    gh, gw = math.ceil((bh // p) / r), math.ceil((bw // p) / r)
    if gh * (gw + 1) + 2 > limit:
        aspect = height / width
        max_w = math.sqrt((limit - 2) / aspect + 0.25) - 0.5
        max_h, cell = max_w * aspect, p * r
        if max_w < 1:
            bh, bw = (limit - 2) // 2 * cell, cell
        elif max_h < 1:
            bh, bw = cell, (limit - 3) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            bh, bw = math.floor(height * beta / p) * p, math.floor(width * beta / p) * p
        gh, gw = math.ceil((bh // p) / r), math.ceil((bw // p) / r)
    if gh * (gw + 1) + 2 > limit or min(gh, gw) < 1:
        raise ValueError("Image grid does not fit the configured token budget")
    return gh, gw, bh, bw


def load_image_bytes(record):
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    source = record.get("source")
    if isinstance(source, dict):
        return load_image_bytes(source)
    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError("Image data URLs must use base64")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        return Path(url).read_bytes()
    raise ValueError("Image record requires data, source, or url")


def load_image(record, c):
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    gh, gw, bh, bw = plan_image_grid(image.width, image.height, c)
    if c.max_wh_ratio is not None and image.width >= c.max_wh_ratio * image.height:
        image = image.resize((bw, bh), resample=Image.Resampling.BICUBIC)
    else:
        image = ImageOps.pad(image, (bw, bh), method=Image.Resampling.BICUBIC, color=(127, 127, 127))
    # Remain float32 on CPU; the vision boundary casts to checkpoint BF16.
    x = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    p = c.patch_size
    vh, vw = bh // p, bw // p
    patches = x.reshape(3, vh, p, vw, p).transpose(1, 3, 0, 2, 4).reshape(vh * vw, 3, p, p)
    return patches, vh, vw, gh, gw
