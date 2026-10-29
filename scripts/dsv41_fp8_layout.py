"""Native E4M3/E8M0 32x32 dense operands in metadata-gated Q8-sized slots."""
KEY='deepseek41.dense_layout'
LAYOUT='fp8-e4m3-e8m0-32x32-v1'
SUFFIXES=('.attn_q_a.weight','.attn_q_b.weight','.attn_kv.weight','.attn_output_b.weight',
          '.ffn_gate_shexp.weight','.ffn_up_shexp.weight','.ffn_down_shexp.weight')

def selected(item):
    return item.name.startswith('blk.') and item.name.endswith(SUFFIXES) and item.qtype==8

def pack_source(db,name,shape):
    import numpy as np
    k,n=shape
    if k%128 or n%128:raise ValueError('Native FP8 dense requires128-aligned N,K')
    info=db.info(name);scale_name=name.removesuffix('.weight')+'.scale';sf=db.info(scale_name)
    if info['dtype']!='F8_E4M3' or info['shape']!=[n,k] or sf['dtype']!='F8_E8M0' or sf['shape']!=[n//32,k//32]:
        raise ValueError('Native FP8 requires E4M3 [N,K] + E8M0 [N/32,K/32]')
    values,scales=db.read(name),db.read(scale_name)
    if len(values)!=n*k or len(scales)!=n*k//1024:raise ValueError('Native FP8 payload byte count mismatch')
    if np.any((np.frombuffer(values,np.uint8)&127)==127) or np.any(np.frombuffer(scales,np.uint8)==255):
        raise ValueError('Nonfinite native FP8 value or scale')
    # Keep GGUF's Q8_0 byte extent so logical dimensions/row alignment remain
    # compatible with the streamer. The flag makes this a distinct encoding.
    padding=n*k*34//32-len(values)-len(scales)
    return values+scales+bytes(padding)
