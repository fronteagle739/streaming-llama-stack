"""Bounded GGUF header inspection. Never mmap or read tensor payloads."""
from pathlib import Path
import re
import struct

FORMATS = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
TYPE_NAMES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
              10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',
              17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',
              24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',34:'TQ1_0',35:'TQ2_0',39:'MXFP4'}


def read_header(path, *, limit=16 << 20):
    """Return metadata and tensor descriptors; reject malformed/oversized input."""
    with Path(path).open('rb') as source:
        def read(count):
            if count < 0 or source.tell() + count > limit:raise ValueError('GGUF header exceeds inspection budget')
            data = source.read(count)
            if len(data) != count:raise ValueError('Truncated GGUF header')
            return data
        def unpack(fmt):return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
        def string():return read(unpack('Q')).decode('utf-8')
        def value(kind):
            if kind == 8:return string()
            if kind == 9:
                subtype, count = unpack('I'), unpack('Q')
                if subtype == 9 or count > 1_000_000:raise ValueError('Invalid GGUF array')
                return [value(subtype) for _ in range(count)]
            if kind not in FORMATS:raise ValueError(f'Unknown GGUF metadata type {kind}')
            return unpack(FORMATS[kind])
        if read(4) != b'GGUF' or unpack('I') not in (2,3):raise ValueError('Expected GGUF v2/v3')
        count, keys = unpack('Q'), unpack('Q')
        if count > 100_000 or keys > 100_000:raise ValueError('GGUF inventory exceeds inspection budget')
        metadata = {}
        for _ in range(keys):
            key = string()
            if key in metadata:raise ValueError('Duplicate GGUF metadata key '+key)
            metadata[key] = value(unpack('I'))
        tensors, names = [], set()
        for _ in range(count):
            name, dimensions = string(), unpack('I')
            if name in names or not 1 <= dimensions <= 4:raise ValueError('Invalid GGUF tensor '+name)
            names.add(name)
            shape = [unpack('Q') for _ in range(dimensions)]
            if not all(shape):raise ValueError('Zero GGUF dimension '+name)
            tensors.append(dict(name=name,shape=shape,type=unpack('I'),offset=unpack('Q')))
        return dict(metadata=metadata,tensors=tensors,header_bytes=source.tell())


def tensor_map(header):
    return {t['name']:dict(type=t['type'],shape=t['shape']) for t in header['tensors']}


def validate_dsv41_types(header,*,expert_layout=None,dense_layout=None):
    """Reject unsupported types before hashing or starting a CUDA process."""
    meta=header['metadata']
    packed=meta.get('deepseek41.expert_layout')
    native_dense=meta.get('deepseek41.dense_layout')
    if native_dense is not None and (native_dense!='fp8-e4m3-e8m0-32x32-v1' or native_dense!=dense_layout or packed is None):
        raise ValueError('Native FP8 dense requires the explicit matching prefill-only loader and native expert bundle')
    if packed is not None:
        if packed!='flashinfer-sm120-mxfp8-mxfp4-v1' or packed!=expert_layout:
            raise ValueError('Kernel-native expert layout requires its explicit matching prefill loader; canonical/decode paths refused')
        alignment=meta.get('general.alignment',32)
        if type(alignment) is not int or alignment<128 or alignment>65536 or alignment&(alignment-1):
            raise ValueError('Kernel-native expert alignment must be a power of two >=128')
    if meta.get('general.architecture')!='deepseek41':raise ValueError('Expected native deepseek41 GGUF')
    if meta.get('split.count',1)!=1:
        raise ValueError('Split GGUF: losslessly assemble the local shards with antirez/ds4 merge_deepseek41_shards.py first; no requantization')
    by_name={t['name']:t for t in header['tensors']}
    for tensor in header['tensors']:
        name,kind=tensor['name'],tensor['type']
        role=re.sub(r'^blk\.\d+\.','',name)
        if role in ('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight'):allowed=(12,16,10,39)
        elif role in ('output.weight','attn_q_a.weight','attn_q_b.weight','attn_kv.weight',
                      'attn_output_a.weight','attn_output_b.weight','ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight'):allowed=(8,12,2)
        elif role=='engram_embd.weight':allowed=(24,)
        elif role in ('token_embd.weight','hc_attn_fn.weight','hc_ffn_fn.weight','attn_compressor_kv.weight',
                      'attn_compressor_gate.weight','indexer.attn_k.weight','indexer.attn_q_b.weight','indexer.proj.weight','engram_kv.weight'):allowed=(1,)
        else:allowed=(0,)
        if kind not in allowed:
            raise ValueError(f'Unsupported DSV4.1 tensor {name} type {TYPE_NAMES.get(kind,"unknown")} ({kind}); no common prefill/decode kernel')
        if packed and role in ('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight'):
            expected_shape=[2304,5120,384] if role=='ffn_down_exps.weight' else [5120,2304,384]
            if kind!=39 or tensor['shape']!=expected_shape or tensor['offset']%128:
                raise ValueError('Kernel-native expert tensor type/shape/alignment mismatch: '+name)
        if native_dense and role in ('attn_q_a.weight','attn_q_b.weight','attn_kv.weight','attn_output_b.weight',
                                      'ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight'):
            if kind!=8 or len(tensor['shape'])!=2 or any(n%128 for n in tensor['shape']) or tensor['offset']%128:
                raise ValueError('Native FP8 dense tensor type/geometry mismatch: '+name)
    for name,t in by_name.items():
        if name.endswith('ffn_gate_exps.weight'):
            prefix=name.removesuffix('ffn_gate_exps.weight')
            group=[by_name.get(prefix+'ffn_'+part+'_exps.weight',{}).get('type') for part in ('gate','up','down')]
            if tuple(group) not in ((12,12,12),(16,16,10),(39,39,39)):
                raise ValueError('Unsupported DSV4.1 expert triplet '+prefix+': '+', '.join(
                    'ffn_'+part+'_exps.weight='+TYPE_NAMES.get(kind,str(kind)) for part,kind in zip(('gate','up','down'),group)))
    return header
