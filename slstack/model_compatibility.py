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
    """Full-vocabulary KL(Mac-only || paired), on an identical forced prefix.

    Standard library only: the stack's core runtime (requirements.txt) has no numpy, and
    the receiver runs this at pairing (8 rows), where it takes about a second."""
    import array, math, sys
    a, b = array.array('f'), array.array('f')
    a.frombytes(control); b.frombytes(candidate)
    if sys.byteorder != 'little':
        a.byteswap(); b.byteswap()
    if (not len(a) or len(a) != len(b) or len(a) % vocab
            or not all(map(math.isfinite, a)) or not all(map(math.isfinite, b))):
        raise ValueError('Missing/incomplete/nonfinite continuation logits')

    def logsoftmax(row):
        top = max(row)
        total = math.fsum(math.exp(x - top) for x in row)
        shift = top + math.log(total)
        return [x - shift for x in row]
    rows = []
    for start in range(0, len(a), vocab):
        p = logsoftmax(a[start:start + vocab])
        q = logsoftmax(b[start:start + vocab])
        rows.append(math.fsum(math.exp(x) * (x - y) for x, y in zip(p, q)))
    return dict(direction='KL(Mac-only || paired)', rows=len(rows), vocab=vocab,
                mean_nats=math.fsum(rows) / len(rows), max_nats=max(rows), per_token_nats=rows)
