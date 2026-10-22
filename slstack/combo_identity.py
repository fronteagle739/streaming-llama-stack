"""Bind final native IDs/template and RoPE to the existing original GGUF.

Use the vendored GGUF reader; never hash tensor payloads into a new model or
materialize weights. The original whole-model SHA receipt is verified first.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import sys
from .config import ROOT, path
from .native_broker import model_receipt, vendor
from .journal import sha256


def _fields_digest(fields,names):
    digest=hashlib.sha256()
    for name in sorted(names):
        value=fields[name]
        encoded=name.encode('utf-8');digest.update(struct.pack('<I',len(encoded)));digest.update(encoded)
        for part in value.parts:
            raw=memoryview(part).cast('B');digest.update(struct.pack('<Q',len(raw)))
            for off in range(0,len(raw),65536):digest.update(raw[off:off+65536])
    return digest.hexdigest()


def identity(cfg):
    receipt=model_receipt(cfg)
    vendor()
    from studio.dsv41_session_contract import MODEL_SHA256
    location=str(ROOT/'vendor'/'engines')
    if location not in sys.path:sys.path.insert(0,location)
    from gguf import GGUFReader
    model=path(cfg['model']);before=model.stat()
    reader=GGUFReader(model,'r')
    if 'tokenizer.ggml.tokens' not in reader.fields:raise ValueError('Original GGUF tokenizer metadata is missing')
    tokenizer=[name for name in reader.fields if name.startswith('tokenizer.') and 'chat_template' not in name]
    rope=[name for name in reader.fields if 'rope' in name]
    if not rope:raise ValueError('Original GGUF RoPE metadata is missing')
    tokenizer_sha=_fields_digest(reader.fields,tokenizer)
    rope_metadata=_fields_digest(reader.fields,rope)
    native_sha=sha256(path(cfg['ds4_command'][0]))
    producer_sha=sha256(path(cfg['llama_producer']))
    # Native built-in rendering and both compiled RoPE implementations are
    # immutable binary inputs in addition to the model's metadata.
    template_sha=hashlib.sha256(('native-renderer/1:'+native_sha).encode()).hexdigest()
    rope_sha=hashlib.sha256((':'.join(('original-rope/1',rope_metadata,native_sha,producer_sha))).encode()).hexdigest()
    after=model.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Original model changed during metadata binding')
    return dict(model_sha256=MODEL_SHA256,tokenizer_sha256=tokenizer_sha,template_sha256=template_sha,rope_sha256=rope_sha),dict(
        model_receipt=str(receipt),native_sha256=native_sha,producer_sha256=producer_sha,
        tokenizer_fields=len(tokenizer),rope_fields=len(rope),rope_metadata_sha256=rope_metadata,
        whole_model_rehashed=False,weights_copied=False)
