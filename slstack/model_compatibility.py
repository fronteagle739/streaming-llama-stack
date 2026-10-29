"""Quant-independent native state compatibility, derived from bounded GGUF headers.

Weight SHA receipts continue to identify each endpoint separately. A compatibility
digest identifies the checkpoint config, tokenizer and every logical tensor shape.
Storage types/offsets and quantization provenance are deliberately separate.
"""
import hashlib
import json
import re
from collections import Counter
from .gguf_header import read_header, TYPE_NAMES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def describe(header, weight_sha256, keep=0):
    from .yoco import state_version
    if not re.fullmatch('[0-9a-f]{64}', weight_sha256):
        raise ValueError('Invalid endpoint weight SHA')
    meta = header['metadata']
    if meta.get('general.architecture') != 'deepseek41':
        raise ValueError('Native state requires deepseek41')
    # These keys describe numerical storage, not network geometry. Every other
    # architecture key, including revision, Engram and compression config, counts.
    storage = {'deepseek41.quantization', 'deepseek41.calibration',
               'deepseek41.expert_layout', 'deepseek41.dense_layout', 'deepseek41.speed_proxy'}
    config = {k:v for k,v in meta.items()
              if k == 'general.architecture' or k.startswith('general.source.') or
              (k.startswith('deepseek41.') and k not in storage)}
    tokenizer = {k:v for k,v in meta.items() if k.startswith('tokenizer.')}
    shapes = sorted((t['name'], t['shape']) for t in header['tensors'])
    if not tokenizer or not shapes or 'deepseek41.num_hidden_layers' not in config:
        raise ValueError('Incomplete architecture/tokenizer inventory')
    contract = dict(schema='dsv41-state-compatibility-v1', config_sha256=digest(config),
                    tokenizer_sha256=digest(tokenizer), shapes_sha256=digest(shapes),
                    state_format='dsv41-native-c-v2.1', wire_version=2, mode=3,
                    context=1000001, yoco_keep=keep, yoco_state_version=state_version(keep))
    return dict(contract=contract, compatibility_sha256=digest(contract),
                model_sha256=weight_sha256,
                quant=dict(sorted(Counter(TYPE_NAMES.get(t['type'], str(t['type']))
                                          for t in header['tensors']).items())))


def model_description(path, weight_sha256, keep=0):
    return describe(read_header(path), weight_sha256, keep)


def require_compatible(local, remote):
    for row in (local, remote):
        if (not isinstance(row, dict) or set(row) != {'contract', 'compatibility_sha256', 'model_sha256', 'quant'}
                or not isinstance(row['contract'], dict)
                or row['compatibility_sha256'] != digest(row['contract'])
                or not re.fullmatch('[0-9a-f]{64}', str(row['model_sha256']))
                or not isinstance(row['quant'], dict) or not row['quant']):
            raise ValueError('Invalid native model compatibility description')
    if local['contract'] != remote['contract']:
        differences = [k for k in set(local['contract']) | set(remote['contract'])
                       if local['contract'].get(k) != remote['contract'].get(k)]
        raise ValueError('Incompatible native model/state config: ' + ', '.join(sorted(differences)))
    return local['model_sha256'] != remote['model_sha256']


def continuation_kl(control, candidate, vocab=129280):
    """Full-vocabulary KL(Mac-only || paired), on an identical forced prefix."""
    import numpy as np
    a = np.frombuffer(control, dtype='<f4').astype(np.float64)
    b = np.frombuffer(candidate, dtype='<f4').astype(np.float64)
    if not len(a) or len(a) != len(b) or len(a) % vocab or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing/incomplete/nonfinite continuation logits')
    a, b = a.reshape(-1,vocab), b.reshape(-1,vocab)
    def logsoftmax(x):
        x = x - x.max(axis=1,keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1,keepdims=True))
    a, b = logsoftmax(a), logsoftmax(b)
    rows = (np.exp(a)*(a-b)).sum(axis=1)
    return dict(direction='KL(Mac-only || paired)', rows=len(rows), vocab=vocab,
                mean_nats=float(rows.mean()), max_nats=float(rows.max()), per_token_nats=rows.tolist())
