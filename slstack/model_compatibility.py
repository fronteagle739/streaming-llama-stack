"""Quant-independent native state compatibility, derived from bounded GGUF headers.

Weight SHA receipts continue to identify each endpoint separately. A compatibility
digest identifies the checkpoint config, tokenizer and every logical tensor shape.
Storage types/offsets and quantization provenance are deliberately separate.
"""
import hashlib
import json
import re
import struct
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
    if meta.get('general.architecture') == 'qwen4exp':
        return describe_qfn(header, weight_sha256, keep)
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
    from pathlib import Path
    with Path(path).open('rb') as source:magic=source.read(8)
    if magic == b'SLPACK\0\0':
        from scripts.qfn_mac_convert import plan
        _,_,header,_=plan(Path(path))
        # The converter serializes floating GGUF metadata as IEEE float32.
        # Match those actual wire values instead of hashing the source JSON's
        # decimal spelling (e.g. 0.6 versus the stored 0.6000000238418579).
        def wire_value(v):
            if isinstance(v,float):return struct.unpack('<f',struct.pack('<f',v))[0]
            if isinstance(v,list):return [wire_value(x) for x in v]
            return v
        header['metadata']={k:wire_value(v) for k,v in header['metadata'].items()}
        # The same logical tensor map is materialized by the Mac converter;
        # source and receiver numerical storage inventories remain distinct.
        for t in header['tensors']:
            if t['operation']=='expert':t['type']=40
            elif t['operation']=='mtp_expert':t['type']=42
            else:t['type']=t.get('source_type',t['type'])
        return describe_qfn(header,weight_sha256,keep)
    return describe(read_header(path), weight_sha256, keep)


def describe_qfn(header, weight_sha256, keep=0):
    """QFN's independent family/geometry contract; never relabel as DeepSeek.

    Precision differences remain visible in endpoint quant inventories and need
    numerical continuation qualification before the pair may be accepted.
    """
    if keep != 0:
        raise ValueError('QFN pairing requires its complete state; DeepSeek YOCO is a different contract')
    meta = header['metadata']
    config = {k:v for k,v in meta.items() if k.startswith('qwen4exp.')
              or k.startswith('general.source.') or k == 'general.architecture'}
    tokens = {k:v for k,v in meta.items() if k.startswith('tokenizer.') or k.startswith('general.sampling.')}
    required = ('block_count','embedding_length','attention.key_length','attention.head_count_kv',
                'ssm.state_size','ssm.time_step_rank','ssm.conv_kernel','full_attention_interval',
                'hyper_connection.count','attention.indexer.key_length','attention.compress_ratios',
                'ple.ngram_size','ple.conv_kernel','ple.eos_token_id')
    if any('qwen4exp.'+k not in config for k in required) or not tokens or not header['tensors']:
        raise ValueError('Incomplete QFN state/tokenizer inventory')
    if not re.fullmatch('[0-9a-f]{64}', str(weight_sha256)):
        raise ValueError('Invalid QFN endpoint weight SHA')
    shapes = sorted((t['name'],t['shape']) for t in header['tensors'])
    if len({n for n,_ in shapes}) != len(shapes):
        raise ValueError('Duplicate QFN tensor name')
    contract = dict(schema='qfn-state-compatibility-v1', config_sha256=digest(config),
        tokenizer_sha256=digest(tokens), shapes_sha256=digest(shapes), state_format='qfn-native-c-v2',
        family_tag=0x51573802, wire_version=2, kv_bits=16, recurrence_bits=32,
        index_storage='fp16-rounded-widened-fp32', histories='time-major',
        mtp_alignment='hidden-q-token-q-plus-1',
        position_format='mrope-4xi32', yoco_keep=0, yoco_state_version=0)
    names={**TYPE_NAMES,40:'NVFP4',42:'Q2_0'}
    return dict(contract=contract, compatibility_sha256=digest(contract), model_sha256=weight_sha256,
        quant=dict(sorted(Counter(names.get(t['type'],str(t['type'])) for t in header['tensors']).items())))


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


def continuation_kl_files(control, candidate, vocab, control_skip=0, candidate_skip=0, progress=None):
    """Keep one complete vocabulary row resident, preserving every row/metric."""
    import array, math, sys
    from pathlib import Path
    if vocab<=0 or min(control_skip,candidate_skip)<0:raise ValueError('Invalid row geometry')
    paths=[Path(control),Path(candidate)];width=vocab*4
    sizes=[p.stat().st_size for p in paths]
    if any(n%width for n in sizes):raise ValueError('Partial vocabulary row')
    count=sizes[0]//width-control_skip
    if count<=0 or count!=sizes[1]//width-candidate_skip:raise ValueError('Matched computed row extents differ')
    kl=[];top=[]
    with paths[0].open('rb') as left,paths[1].open('rb') as right:
        left.seek(control_skip*width);right.seek(candidate_skip*width)
        for i in range(count):
            x,y=left.read(width),right.read(width)
            if len(x)!=width or len(y)!=width:raise ValueError('Continuation row ended early')
            row=continuation_kl(x,y,vocab)
            a,b=array.array('f'),array.array('f');a.frombytes(x);b.frombytes(y)
            if sys.byteorder!='little':a.byteswap();b.byteswap()
            top.append(max(range(vocab),key=a.__getitem__)==max(range(vocab),key=b.__getitem__))
            kl.append(row['mean_nats'])
            if progress:progress(i+1,count,math.fsum(kl)/len(kl),sum(top)/len(top))
        if left.read(1) or right.read(1):raise ValueError('Unexpected trailing continuation rows')
    return dict(rows=count,vocab=vocab,mean_nats=math.fsum(kl)/count,max_nats=max(kl),
        per_token_nats=kl,top1=sum(top)/count,top1_per_row=top,
        control_skipped_rows=control_skip,candidate_skipped_rows=candidate_skip)
