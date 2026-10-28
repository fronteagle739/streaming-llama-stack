"""Local, streamed native checkpoints, including YOCO replay provenance.

These snapshots are deliberately separate from the exact RTX/Mac transfer
contract. No weights are copied and no full payload is staged in host RAM.
Files are retained when the configured budget is reached; nothing is deleted.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from .combo_contract import validate
from .combo_state import contract_from_checkpoint, payload_bytes, verify_payload
from .generation import publish
from .journal import sha256

VERSION = 'slstack-local-state/1'


def options(cfg):
    location = Path(cfg.get('stack_preferences_file') or Path(cfg['runtime_root']) / 'settings' / 'stack.local.json')
    row = dict(cache_media=True, cache_conversations=True, cache_budget_gib=16)
    if location.is_file():
        saved = json.loads(location.read_text(encoding='utf-8'))
        row.update({k: saved[k] for k in row if k in saved})
    return location.parent.parent / 'checkpoints', row


def validate_contract(row):
    standard = dict(row);local = standard.pop('local_state')
    validate(standard, local=True)
    if not isinstance(local, dict) or set(local) != {'version', 'yoco_keep', 'cache_precision', 'producer_sha256'}:
        raise ValueError('Malformed local snapshot provenance')
    if local['version'] != VERSION or type(local['yoco_keep']) is not int or local['yoco_keep'] not in (0, 140, 2541):
        raise ValueError('Unsupported local snapshot/YOCO version')
    if local['cache_precision'] not in ('f32', 'bf16') or not isinstance(local['producer_sha256'], str) or len(local['producer_sha256']) != 64:
        raise ValueError('Malformed local snapshot build/precision')
    return row


class Payload:
    def __init__(self, path):
        self.path = Path(path);self.capacity = self.path.stat().st_size;self.stream = None

    @contextmanager
    def lease(self):
        if self.stream is not None:
            yield self;return
        with self.path.open('rb') as stream:
            self.stream = stream
            try:yield self
            finally:self.stream = None

    def read(self, offset, count):
        if self.stream is None:
            with self.lease():return self.read(offset, count)
        if not 0 <= offset <= self.capacity or not 0 <= count <= min(65536, self.capacity-offset):
            raise ValueError('Local checkpoint read outside its bounded lease')
        self.stream.seek(offset);block = self.stream.read(count)
        if len(block) != count:raise ValueError('Truncated local checkpoint')
        return block


class Cache:
    def __init__(self, cfg):
        self.cfg = cfg;self._identity = None

    def identity(self):
        if self._identity is None:
            from .native_broker import model_receipt, model_digest, vendor
            from . import ds4_render
            model_receipt(self.cfg);vendor()
            MODEL_SHA256 = model_digest(self.cfg)
            def digest(text):return hashlib.sha256(text.encode()).hexdigest()
            # The receipt binds all GGUF bytes, including tokenizer and RoPE.
            # Distinct domains prevent treating these as an exact-remote identity.
            self._identity = dict(model_sha256=MODEL_SHA256,
                tokenizer_sha256=digest('local-model-tokenizer/1:'+MODEL_SHA256),
                template_sha256=sha256(Path(ds4_render.__file__)),
                rope_sha256=digest('local-model-rope/1:'+MODEL_SHA256+':'+self.cfg['producer_sha256']))
        return dict(self._identity)

    def provenance(self):
        return dict(version=VERSION, yoco_keep=self.cfg.get('yoco_keep', 0),
                    cache_precision=self.cfg.get('ctx_cache_precision', 'f32'), producer_sha256=self.cfg['producer_sha256'])

    def compatible(self, manifest):
        row = validate_contract(manifest['contract'])
        prefix=[{k:span[k] for k in ('position','count','sha256')} for span in self.cfg.get('_vision_spans',[]) if span['position']<row['position']]
        return manifest.get('images',[])==prefix and row['context'] == self.cfg['context'] and row['local_state'] == self.provenance() and all(
            row[k] == v for k, v in self.identity().items())

    def save(self, producer, tokens, tool_memory, cut_turn):
        if producer.process.ready.get('local_snapshot_version') != 1:
            raise ValueError('Producer needs the local KV/YOCO snapshot build')
        if len(tokens) != producer.checkpoint['position']:raise ValueError('Local token frontier differs')
        contract = contract_from_checkpoint(producer.checkpoint, self.identity(), producer.session, producer.epoch, 1, 'phase-bias', local=True)
        contract['local_state'] = self.provenance();validate_contract(contract)
        root, prefs = options(self.cfg);root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        target = root / (key + '.json');payload = root / (key + '.kv')
        if target.is_file():
            saved = json.loads(target.read_text(encoding='utf-8'))
            self.verify(saved, payload);return saved
        size = payload_bytes(len(tokens), local=True)
        used = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        if used + size > int(prefs['cache_budget_gib']) * (1 << 30):
            raise MemoryError('KV checkpoint budget is full; existing checkpoints are preserved')
        if shutil.disk_usage(root).free < size + (2 << 30):
            raise MemoryError('KV checkpoint needs its payload plus 2 GiB free disk headroom')
        started = time.monotonic()
        # Exclusive creation retains incomplete evidence after any failure.
        with payload.open('xb') as output:
            try:
                meta = producer.process.begin_export(producer.command('export_local', base=producer.checkpoint))
                if meta['bytes'] != size or meta['yoco_keep'] != contract['local_state']['yoco_keep'] or meta['cache_precision'] != contract['local_state']['cache_precision']:
                    raise ValueError('Local snapshot extent/preset differs')
                while block := producer.process.read_export():output.write(block)
                producer.process.end_export();output.flush();os.fsync(output.fileno())
            except BaseException:
                producer.process.abort_transfer();raise
        standard = dict(contract);standard.pop('local_state')
        proof = verify_payload(Payload(payload), standard, local=True)
        saved = dict(version=VERSION, contract=contract, payload=payload.name, payload_sha256=proof['payload_sha256'],
                     tokens=list(tokens), tool_memory=deepcopy(tool_memory), cut_turn=deepcopy(cut_turn),
                     images=[{k:span[k] for k in ('position','count','sha256')} for span in self.cfg.get('_vision_spans',[]) if span['position']<len(tokens)],
                     bytes=size, save_seconds=time.monotonic()-started, created=time.time())
        publish(target, saved)
        return saved

    def verify(self, saved, payload):
        if saved.get('version') != VERSION or not self.compatible(saved):raise ValueError('Incompatible local KV/YOCO checkpoint')
        contract = dict(saved['contract']);contract.pop('local_state')
        reader = Payload(payload)
        if reader.capacity != payload_bytes(contract['position'], local=True):raise ValueError('Local snapshot file extent differs')
        verify_payload(reader, contract, saved['payload_sha256'], local=True)
        digest = hashlib.sha256()
        for offset in range(0, len(saved['tokens']), 8192):
            part = saved['tokens'][offset:offset+8192];digest.update(struct.pack('<'+'I'*len(part), *part))
        if len(saved['tokens']) != contract['position'] or digest.hexdigest() != contract['tokens_sha256']:
            raise ValueError('Local snapshot prompt token identity differs')
        return reader

    def candidates(self):
        root, _ = options(self.cfg)
        if not root.is_dir():return []
        result = []
        for path in root.glob('*.json'):
            if path.stat().st_size > 64 << 20:continue
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                payload = root / saved['payload']
                if payload.parent != root or not payload.is_file() or not self.compatible(saved):continue
                result.append((saved, payload))
            except (OSError, ValueError, KeyError, TypeError):continue
        return sorted(result, key=lambda item:(len(item[0]['tokens']),item[0]['created']), reverse=True)
