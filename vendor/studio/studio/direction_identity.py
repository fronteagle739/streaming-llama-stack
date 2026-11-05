"""Quant-independent model binding from bounded GGUF metadata and tensor shapes."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    before = path.stat()
    limit = min(before.st_size, 64 * 1024 * 1024)
    with path.open("rb") as f:
        def take(n):
            if n < 0 or f.tell() + n > limit:
                raise ValueError("GGUF identity header exceeds 64 MiB")
            data = f.read(n)
            if len(data) != n:
                raise ValueError("Truncated GGUF identity header")
            return data
        def unpack(fmt):
            return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]
        def string():
            return take(unpack("Q")).decode("utf-8", "strict")
        formats = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}
        def value(kind, depth=0):
            if depth > 4:
                raise ValueError("Nested GGUF metadata exceeds bound")
            if kind == 8:
                return string()
            if kind == 9:
                element, count = unpack("I"), unpack("Q")
                if count > 2_000_000:
                    raise ValueError("GGUF metadata array exceeds bound")
                return [value(element, depth + 1) for _ in range(count)]
            if kind not in formats:
                raise ValueError("Unsupported GGUF metadata type")
            return unpack(formats[kind])
        if take(4) != b"GGUF" or unpack("I") not in (2, 3):
            raise ValueError("GGUF v2/v3 model required")
        tensors, keys = unpack("Q"), unpack("Q")
        if tensors > 200_000 or keys > 100_000:
            raise ValueError("GGUF identity count exceeds bound")
        metadata, native_records = {}, []
        for _ in range(keys):
            key = string()
            if key in metadata:
                raise ValueError("Duplicate GGUF metadata key")
            kind = unpack("I")
            start = f.tell()
            metadata[key] = value(kind)
            end = f.tell()
            if key.startswith("tokenizer.") or key in ("general.architecture", "general.name", "general.source.url", "general.source.revision"):
                f.seek(start)
                raw = take(end - start)
                native_records.append((b"K" + key.encode(), hashlib.sha256(struct.pack("<I", kind) + raw).digest()))
        shapes, types = {}, Counter()
        for _ in range(tensors):
            name, dimensions = string(), unpack("I")
            if name in shapes or not 1 <= dimensions <= 4:
                raise ValueError("Invalid GGUF tensor inventory")
            shapes[name] = [unpack("Q") for _ in range(dimensions)]
            native_records.append((b"T" + name.encode(), hashlib.sha256(struct.pack("<I" + "Q" * dimensions, dimensions, *shapes[name])).digest()))
            types[unpack("I")] += 1
            unpack("Q")  # Storage offset/quant are deliberately outside model identity.
        header_bytes = f.tell()
    after = path.stat()
    if any(getattr(after, key) != getattr(before, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("Model changed while reading identity")
    architecture = metadata.get("general.architecture")
    name = metadata.get("general.name")
    revision = metadata.get("general.source.revision")
    source = metadata.get("general.source.url")
    tokenizer = {k:v for k,v in metadata.items() if k.startswith("tokenizer.")}
    if not all(isinstance(v, str) and v.strip() for v in (architecture, name, revision, source)) or not tokenizer or not shapes:
        raise ValueError("Model identity requires architecture, base name, source revision/URL, tokenizer and tensor shapes")
    tokenizer_digest = hashlib.sha256(b"direction-tokenizer-v1\0")
    for key, hashed in sorted(native_records):
        if key.startswith(b"Ktokenizer."):
            tokenizer_digest.update(struct.pack("<I", len(key)) + key + hashed)
    identity = dict(schema="direction-model-v1", architecture=architecture,
        base_name=name, revision=revision, source=source,
        tokenizer_sha256=tokenizer_digest.hexdigest(), tensor_shapes_sha256=digest(shapes))
    label = re.search(r"(?:^|[-_])(Q\d+(?:_[A-Z0-9]+)*|IQ\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)(?:[-_.]|$)", path.name, re.I)
    uniform_quant = {0: 'F32', 1: 'F16', 8: 'Q8_0', 10: 'Q2_K', 12: 'Q4_K', 30: 'BF16'}
    if 39 in types:
        # The tensors decide, not a leftover source quant in the file name.
        quant_note = 'FP4 speed proxy' if 'speed-proxy' in path.name.lower() else 'FP4'
    elif label:
        quant_note = label.group(1).upper()
    elif len(types) == 1:
        quant_note = uniform_quant.get(next(iter(types)), 'mixed GGUF')
    else:
        quant_note = 'mixed GGUF'
    native = hashlib.sha256(b"direction-model-native-v1\0")
    for key, hashed in sorted(native_records):
        native.update(struct.pack("<I", len(key)) + key + hashed)
    layers=metadata.get(architecture + ".block_count")
    hidden=metadata.get(architecture + ".embedding_length")
    if architecture == 'deepseek41':
        # The native 1017 GGUF can omit architectural scalar metadata. Derive
        # geometry only from a complete, consistent inventory of actual writers.
        writers={}
        for tensor, shape in shapes.items():
            match=re.fullmatch(r'blk\.(\d+)\.(attn_output_b|ffn_down_shexp|ffn_down_exps)\.weight',tensor)
            if match:
                writers.setdefault(int(match[1]),{})[match[2]]=shape
        expected={'attn_output_b','ffn_down_shexp','ffn_down_exps'}
        if not writers or sorted(writers)!=list(range(len(writers))) or any(set(row)!=expected for row in writers.values()):
            raise ValueError('DSV4.1 model requires a complete contiguous writer inventory')
        widths={shape[1] for row in writers.values() for shape in row.values() if len(shape) in (2,3)}
        if (len(widths)!=1 or any(len(shape)!=(3 if role=='ffn_down_exps' else 2)
                for row in writers.values() for role,shape in row.items())):
            raise ValueError('DSV4.1 writer geometry is inconsistent')
        derived_hidden=next(iter(widths))
        token_shape=shapes.get('token_embd.weight')
        if not token_shape or len(token_shape)!=2 or token_shape[0]!=derived_hidden:
            raise ValueError('DSV4.1 token embeddings and writer width disagree')
        if (layers is not None and layers!=len(writers)) or (hidden is not None and hidden!=derived_hidden):
            raise ValueError('DSV4.1 scalar metadata and tensor geometry disagree')
        layers,hidden=len(writers),derived_hidden
    return dict(identity=identity, model_id=digest(identity), native_id=native.hexdigest(), source_quant=quant_note,
        quant_types=dict(types), tensor_count=len(shapes), header_bytes=header_bytes,
        layers=layers, hidden=hidden)


def active_path():
    from . import llm_launch, preset_store
    preset = preset_store.get_active(strict=True) or {}
    if llm_launch.external_backend(preset):
        raise ValueError("Direction extraction requires a local GGUF model")
    path = (preset.get("rtx") or {}).get("model_path")
    if not path:
        raise ValueError("Select a GGUF model in Connect")
    return Path(llm_launch._resolve_gguf_model_path(path))


def active():
    return read(active_path())
