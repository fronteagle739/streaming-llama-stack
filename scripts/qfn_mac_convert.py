"""Bounded, offline QFN SLPACK -> native Metal GGUF, preserving PLE/MTP/vision.

The source pack is the existing verified NVIDIA import, never a model download.
Q8_0 experts, original Q8_0 PLE and F16 dense projections; F32 recurrence/16-bit
KV are runtime choices and are not quantized by this weight conversion.
Run under qfn_dual_guard.py (or Studio GuardedJob on the Mac), niced there.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.slpack import ModelFile

ALIGN=16384
CHUNK=4<<20
SIZES={0:(1,4),1:(1,2),2:(32,18),8:(32,34),14:(256,210),30:(1,2)}
MTP_NAMES={
 'pre_fc_norm_embedding.weight':'nextn.enorm.weight', 'pre_fc_norm_hidden.weight':'nextn.hnorm.weight',
 'hyper_connection_mixer.hc_norm.weight':'nextn.hc_head_norm.weight',
 'hyper_connection_mixer.input_mix_weight_down.weight':'nextn.hc_head_down.weight',
 'hyper_connection_mixer.input_mix_weight_up.weight':'nextn.hc_head_up.weight',
 'attn_hyper_connection.hc_norm.weight':'hc_attn_norm.weight',
 'attn_hyper_connection.input_mix_weight_down.weight':'hc_attn_down.weight',
 'attn_hyper_connection.input_mix_weight_up.weight':'hc_attn_up.weight',
 'attn_hyper_connection.block_inject_weight.weight':'hc_attn_inject.weight',
 'mlp_hyper_connection.hc_norm.weight':'hc_ffn_norm.weight',
 'mlp_hyper_connection.input_mix_weight_down.weight':'hc_ffn_down.weight',
 'mlp_hyper_connection.input_mix_weight_up.weight':'hc_ffn_up.weight',
 'mlp_hyper_connection.block_inject_weight.weight':'hc_ffn_inject.weight',
 'mlp.gate.weight':'ffn_gate_inp.weight', 'mlp.shared_expert_gate.weight':'ffn_gate_inp_shexp.weight',
 'mlp.shared_expert.gate_proj.weight':'ffn_gate_shexp.weight',
 'mlp.shared_expert.up_proj.weight':'ffn_up_shexp.weight',
 'mlp.shared_expert.down_proj.weight':'ffn_down_shexp.weight',
 'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight',
 'self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
 'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
 'self_attn.indexer.q_layernorm.weight':'indexer.q_norm.weight',
 'self_attn.indexer.k_layernorm.weight':'indexer.k_norm.weight',
}


def align(n):return (n+ALIGN-1)//ALIGN*ALIGN
def sha(b):return hashlib.sha256(b).hexdigest()
def identity(p):
    s=p.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)
def text(s):
    b=s.encode('utf8');return struct.pack('<Q',len(b))+b
def value(v,typed=True):
    if isinstance(v,bool):kind,data=7,struct.pack('<?',v)
    elif isinstance(v,int):kind,data=(10,struct.pack('<Q',v)) if v>=0 else (11,struct.pack('<q',v))
    elif isinstance(v,float):kind,data=6,struct.pack('<f',v)
    elif isinstance(v,str):kind,data=8,text(v)
    elif isinstance(v,list) and v:
        encoded=[value(x) for x in v];kinds={b[:4] for b in encoded}
        if len(kinds)!=1:raise ValueError('GGUF array must be homogeneous')
        kind,data=9,encoded[0][:4]+struct.pack('<Q',len(v))+b''.join(b[4:] for b in encoded)
    else:raise ValueError('Unsupported metadata value '+str(type(v)))
    return (struct.pack('<I',kind) if typed else b'')+data


def plan(source,expert_bits=8):
    if expert_bits not in (4,8):raise ValueError('Main expert bits must be 4 or 8')
    pack=ModelFile(source);meta=json.loads(pack.read('meta'))
    kv=dict(meta['gguf_kv']);A='qwen4exp.'
    if kv.get('general.architecture')!='qwen4exp':raise ValueError('Expected QFN pack')
    layers=int(kv[A+'block_count']);E=int(kv[A+'embedding_length']);FF=int(kv[A+'expert_feed_forward_length']);NE=int(kv[A+'expert_count'])
    if (layers,E,FF,NE)!=(48,2560,640,512):raise ValueError('Unqualified source expert geometry')
    with Path(source).open('rb') as source_header:header_hash=sha(source_header.read(65536))
    kv.update({'general.alignment':ALIGN,A+'block_count':layers+1,A+'nextn_predict_layers':1,
        'general.source.qfn_checkpoint_config_sha256':meta['checkpoint']['config_sha256'],
        'general.source.qfn_pack_header_sha256':header_hash,
        'general.source.qfn_mtp_sha256':meta['mtp']['source_sha256'],
        'qfn.conversion.weight_format':('Q8_0 experts; original Q8_0 PLE; F16 dense' if expert_bits==8 else
            'Q4_0 main experts; Q8_0 MTP and PLE; F16 dense'),
        'qfn.conversion.original_metadata':json.dumps({k:v for k,v in meta.items() if k not in ('gguf_kv','tensors')},sort_keys=True,separators=(',',':'))})
    kv[A+'attention.compress_ratios']=list(kv[A+'attention.compress_ratios'])+[4]
    if A+'attention.recurrent_layers' in kv:kv[A+'attention.recurrent_layers']=list(kv[A+'attention.recurrent_layers'])+[False]
    tensors=[];goff,gsize=pack.section('gguf')
    for name,t in meta['tensors'].items():
        ty=t['type_id'];target=1 if ty in (14,30) else ty
        if ty not in SIZES:raise ValueError('Unsupported dense tensor type '+str(ty)+': '+name)
        if t['offset']+t['bytes']>gsize:raise ValueError('Dense tensor outside source section')
        tensors.append(dict(name=name,shape=t['shape'],type=target,source_type=ty,
            source_offset=goff+t['offset'],source_bytes=t['bytes'],operation='dense'))
    if meta['tensors']['per_layer_token_embd.weight']['type_id']!=8:raise ValueError('This recipe preserves the qualified Q8_0 PLE')
    for il in range(layers):
        for role,shape in [('gate',[E,FF,NE]),('up',[E,FF,NE]),('down',[FF,E,NE])]:
            tensors.append(dict(name=f'blk.{il}.ffn_{role}_exps.weight',shape=shape,type=2 if expert_bits==4 else 8,operation='expert',layer=il,role=role))
    index={}
    for line in pack.read('mtp/dense.index').decode().splitlines():
        if not line or line.startswith('#'):continue
        n,ty,rows,cols,off,nb=line.split();index[n]=dict(source_type={'q8_0':8,'bf16':30,'f32':0,'f16':1}[ty],
            source_offset=pack.section('mtp/dense')[0]+int(off),source_bytes=int(nb),shape=[int(cols)] if int(rows)==1 else [int(cols),int(rows)])
    expected=set(MTP_NAMES)|{'fc_embedding.weight','fc_hidden.weight','self_attn.indexer.index_qk_proj.weight'}
    if set(index)!=expected:raise ValueError('MTP inventory differs; refusing to drop unknown/missing tensors: '+str(set(index)^expected))
    for n,suffix in MTP_NAMES.items():
        t=index[n];tensors.append(dict(t,name=f'blk.{layers}.{suffix}',type=1 if t['source_type']==30 else t['source_type'],operation='dense'))
    a,b=index['fc_embedding.weight'],index['fc_hidden.weight']
    if any(t['source_type']!=8 or t['shape']!=[E,E] for t in (a,b)):raise ValueError('MTP projection geometry differs')
    tensors.append(dict(name=f'blk.{layers}.nextn.eh_proj.weight',shape=[2*E,E],type=8,operation='mtp_projection',parts=[a,b]))
    t=index['self_attn.indexer.index_qk_proj.weight'];qrows=kv[A+'attention.indexer.head_count']*kv[A+'attention.indexer.key_length']
    for role,start,rows in [('q',0,qrows),('k',qrows,kv[A+'attention.indexer.key_length'])]:
        rb=E//32*34
        if t['source_type']!=8 or t['shape']!=[E,qrows+rows] and role=='k':raise ValueError('MTP index projection differs')
        tensors.append(dict(name=f'blk.{layers}.indexer.{role}_proj.weight',shape=[E,rows],type=8,operation='dense',
            source_type=8,source_offset=t['source_offset']+start*rb,source_bytes=rows*rb))
    for role,shape in [('gate',[E,FF,NE]),('up',[E,FF,NE]),('down',[FF,E,NE])]:
        tensors.append(dict(name=f'blk.{layers}.ffn_{role}_exps.weight',shape=shape,type=8,operation='mtp_expert',role=role))
    # Keep the disk-only n-grams last and page-aligned, as the native Mac loader requires.
    tensors.sort(key=lambda t:(t['name']=='per_layer_token_embd.weight',t['name']))
    off=0
    for t in tensors:
        block,size=SIZES[t['type']];elements=math.prod(t['shape'])
        if elements%block:raise ValueError('Tensor is not block aligned: '+t['name'])
        t.update(offset=align(off),bytes=elements//block*size);off=t['offset']+t['bytes']
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(kv))
    header+=b''.join(text(k)+value(v) for k,v in sorted(kv.items()))
    for t in tensors:header+=text(t['name'])+struct.pack('<I',len(t['shape']))+struct.pack('<'+'Q'*len(t['shape']),*t['shape'])+struct.pack('<IQ',t['type'],t['offset'])
    data_start=align(len(header));header+=bytes(data_start-len(header))
    return pack,meta,dict(metadata=kv,tensors=tensors,data_start=data_start,total_bytes=data_start+off,header_sha256=sha(header),source_identity=identity(Path(source))),header


def dequant_nvfp4(raw,n,k,scale,packed):
    import numpy as np
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid NVFP4 weight global')
    from scripts.dsv41_fp4_layout import unpack_nvfp4_matrix
    if packed:raw=unpack_nvfp4_matrix(raw,n,k)
    blocks=np.frombuffer(raw,dtype=np.uint8).reshape(n,k//64,36)
    codes=blocks[:,:,4:].reshape(n,k//64,4,8)
    v=np.concatenate((codes&15,codes>>4),axis=3)
    lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
    b=blocks[:,:,:4]
    if (b>=127).any():raise ValueError('Invalid UE4M3 source scale')
    exp=(b>>3).astype(np.int32);mant=b&7
    sf=np.where(exp==0,mant/8.0*2.0**-6,(1+mant/8.0)*2.0**(exp-7)).astype(np.float32)
    out=(lut[v]*sf[:,:,:,None])*np.float32(scale)
    if not np.isfinite(out).all() or not math.isfinite(scale) or scale<=0:raise ValueError('Nonfinite NVFP4 source values')
    return out.reshape(n,k)


def ordered_parallel(fn,items,workers):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque();it=iter(items)
        for _ in range(workers*2):
            try:pending.append(pool.submit(fn,next(it)))
            except StopIteration:break
        while pending:
            yield pending.popleft().result()
            try:pending.append(pool.submit(fn,next(it)))
            except StopIteration:pass


def quantize_q4(values):
    """One quantization directly from decoded source FP4, with GGML Q4_0 layout."""
    import numpy as np
    values=np.asarray(values,dtype=np.float32).reshape(-1,32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite Q4 source values')
    indices=np.argmax(np.abs(values),axis=1)
    peak=values[np.arange(len(values)),indices]
    delta=peak/-8.
    inverse=np.divide(1.,delta,out=np.zeros_like(delta),where=delta!=0)
    codes=np.minimum(15,np.floor(values*inverse[:,None]+8.5)).astype(np.uint8)
    stored=delta.astype('<f2')
    if not np.isfinite(stored).all():raise ValueError('Q4 scale overflow')
    out=np.empty((len(values),18),np.uint8)
    out[:,:2]=stored.view(np.uint8).reshape(-1,2)
    out[:,2:]=codes[:,:16]|(codes[:,16:]<<4)
    return out.tobytes()


def chunks(pack,meta,t,workers=1):
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts/slpack_qfn_source'))
    import ggufq
    from model_sections import q8_0
    with open(pack.path,'rb') as f:
        def read(off,n):
            f.seek(off);b=f.read(n)
            if len(b)!=n:raise ValueError('Short source tensor')
            return b
        op=t['operation']
        if op=='dense':
            ty=t['source_type'];block,size=SIZES[ty];step=CHUNK//size*size
            for off in range(0,t['source_bytes'],step):
                raw=read(t['source_offset']+off,min(step,t['source_bytes']-off))
                if ty==t['type']:yield raw
                else:
                    x=ggufq.dequantize(np.frombuffer(raw,np.uint8),ggufq.GGMLQuantizationType(ty)).reshape(-1)
                    y=x.astype('<f2')
                    if not np.isfinite(y).all():raise ValueError('Dense F16 overflow')
                    yield y.tobytes()
        elif op=='mtp_projection':
            rb=2560//32*34
            for r in range(2560):yield b''.join(read(x['source_offset']+r*rb,rb) for x in t['parts'])
        elif op=='expert':
            role=('gate','up','down').index(t['role']);n,k=((640,2560),(640,2560),(2560,640))[role]
            packed='qfn_native_nvfp4' in meta
            def one(e):
                base=pack.section('experts')[0]+(t['layer']*512+e)*2764816
                with open(pack.path,'rb') as src:
                    src.seek(base+2764800);scale=struct.unpack('<4f',src.read(16))[role]
                    src.seek(base+role*921600);raw=src.read(921600)
                values=dequant_nvfp4(raw,n,k,scale,packed)
                return quantize_q4(values) if t['type']==2 else q8_0(values)
            yield from ordered_parallel(one,range(512),workers)
        elif op=='mtp_expert':
            role=('gate','up','down').index(t['role']);ne=1280*40;nd=2560*10
            for e in range(512):
                b=read(pack.section('mtp/experts')[0]+e*1382400,1382400)
                if role<2:
                    codes=np.frombuffer(b,np.uint8,count=ne*16).reshape(1280,40,16)[role::2]
                    ds=np.frombuffer(b,np.uint8,count=ne*2,offset=(ne+nd)*16).reshape(1280,40,2)[role::2]
                else:
                    codes=np.frombuffer(b,np.uint8,count=nd*16,offset=ne*16).reshape(2560,10,16)
                    ds=np.frombuffer(b,np.uint8,count=nd*2,offset=(ne+nd)*16+ne*2).reshape(2560,10,2)
                scale=ds.copy().view('<f2').astype(np.float32)
                q=np.stack([(codes>>(2*j))&3 for j in range(4)],axis=-1).reshape(*codes.shape[:2],64).astype(np.float32)-1
                yield q8_0((q*scale).reshape(codes.shape[0],-1))
        else:raise ValueError('Unknown conversion operation')


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('mode',choices=['plan','write','verify'])
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--progress-log',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--hold-file',type=Path)
    ap.add_argument('--priority',choices=['below-normal','idle'],default='below-normal')
    ap.add_argument('--workers',type=int,choices=range(1,17),default=1)
    ap.add_argument('--expert-bits',type=int,choices=[4,8],default=8,
        help='Independent compact layout from the existing source; MTP and PLE stay Q8')
    ap.add_argument('--source-sha256',help='Verified existing NVIDIA import full-file SHA256; required for writing')
    a=ap.parse_args();pack,meta,info,header=plan(a.source,a.expert_bits)
    import os,psutil
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if a.priority=='below-normal' else psutil.IDLE_PRIORITY_CLASS)
    else:os.nice(10 if a.priority=='below-normal' else 19)
    def note(doing,latest,next_step):
        line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
        print(line,flush=True)
        with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
    last_pause=[0.0]
    def idle():
        if not a.hold_file:return
        from slstack.lease import active_holds
        while active_holds(a.hold_file.read_text(encoding='utf-8-sig')):
            if time.monotonic()-last_pause[0]>60:
                note('conversion yields disk/CPU to active timed hold','no GPU or model process owned','resume same tensor when hold releases');last_pause[0]=time.monotonic()
            time.sleep(1)
    target=a.out.resolve();receipt=target.with_suffix('.conversion.json')
    binding=dict(source=str(a.source.resolve()),source_identity=info['source_identity'],header_sha256=info['header_sha256'],tool_sha256=sha(Path(__file__).read_bytes()),source_sha256=a.source_sha256)
    if a.mode=='plan':
        target.with_suffix('.plan.json').write_text(json.dumps(info,indent=2),encoding='utf8')
        note('Mac QFN conversion planned',f'{len(info["tensors"])} tensors / {info["total_bytes"]} bytes, full PLE/MTP/vision retained','guarded niced write');return
    state=dict(binding=binding,completed={})
    if a.resume or a.mode=='verify':
        state=json.loads(receipt.read_text())
        if state['binding']!=binding:raise ValueError('Source/tool/header changed since conversion')
    def save():receipt.write_text(json.dumps(state,indent=2),encoding='utf8')
    if a.mode=='write':
        import shutil
        if not a.source_sha256 or len(a.source_sha256)!=64:raise ValueError('Full source SHA256 is required')
        if not a.resume:
            note('verifying copied existing source','full SHA256 read; no model downloaded','conversion only after byte identity passes')
            with a.source.open('rb') as f:
                h=hashlib.sha256()
                while b:=f.read(CHUNK):idle();h.update(b)
            if h.hexdigest()!=a.source_sha256:raise ValueError('Copied source full SHA256 differs')
            note('existing NVIDIA import byte identity PASS',a.source_sha256,'niced bounded conversion')
        need=info['total_bytes']-(target.stat().st_size if target.exists() else 0)
        if shutil.disk_usage(target.parent).free<need+(2<<30):raise ValueError('Insufficient conversion disk budget')
        with target.open('r+b' if a.resume else 'xb') as out:
            if not a.resume:out.write(header);save()
            for i,t in enumerate(info['tensors']):
                if t['name'] in state['completed']:continue
                at=info['data_start']+t['offset'];out.seek(at);h=hashlib.sha256();count=0
                for b in chunks(pack,meta,t,a.workers):idle();out.write(b);h.update(b);count+=len(b)
                if count!=t['bytes']:raise ValueError('Converted tensor byte extent differs: '+t['name'])
                out.flush();state['completed'][t['name']]=dict(bytes=count,sha256=h.hexdigest());save()
                if t['operation'] in ('expert','mtp_expert') or i%64==0:note('Mac QFN weight conversion',f'{i+1}/{len(info["tensors"])} tensors; {t["name"]}','remaining tensors then disk readback')
        # Preserve every non-weight section as an authenticated companion.
        companions=target.with_suffix('.assets');companions.mkdir(exist_ok=True)
        represented={'gguf','experts','mtp/experts','mtp/dense'}
        for name in pack.names():
            if name in represented:continue
            p=companions/(name.replace('/','__')+('.gguf' if name=='vision/mmproj' else '.bin'))
            if p.exists():
                with p.open('rb') as f:existing=hashlib.file_digest(f,'sha256').hexdigest()
                if existing==pack.sections[name][2].hex():continue
                if not a.resume:raise ValueError('Incomplete companion exists; explicit resume required')
            off,n,expected=pack.sections[name];h=hashlib.sha256()
            with open(pack.path,'rb') as f,p.open('r+b' if p.exists() else 'xb') as out:
                f.seek(off)
                for left in range(0,n,CHUNK):
                    idle();b=f.read(min(CHUNK,n-left));out.write(b);h.update(b)
            if h.digest()!=expected:raise ValueError('Companion source hash differs: '+name)
        state['companions']={n:pack.sections[n][2].hex() for n in pack.names() if n not in represented};save()
        note('Mac QFN conversion writer finished',str(target.stat().st_size)+' bytes','independent readback and native admission')
    if a.mode=='verify':
        with target.open('rb') as f:
            if sha(f.read(len(header)))!=info['header_sha256']:raise ValueError('Output header differs')
            for t in info['tensors']:
                f.seek(info['data_start']+t['offset']);h=hashlib.sha256()
                for off in range(0,t['bytes'],CHUNK):idle();h.update(f.read(min(CHUNK,t['bytes']-off)))
                if state['completed'].get(t['name'])!=dict(bytes=t['bytes'],sha256=h.hexdigest()):raise ValueError('Output readback differs: '+t['name'])
        for name,expected in state.get('companions',{}).items():
            p=target.with_suffix('.assets')/(name.replace('/','__')+('.gguf' if name=='vision/mmproj' else '.bin'))
            with p.open('rb') as f:
                h=hashlib.sha256()
                while b:=f.read(CHUNK):idle();h.update(b)
            if h.hexdigest()!=expected:raise ValueError('Companion readback differs: '+name)
        note('Mac QFN tensors and companions verified',str(len(info['tensors']))+' tensors','whole-model identity receipt')
        with target.open('rb') as f:
            digest=hashlib.sha256()
            while block:=f.read(CHUNK):idle();digest.update(block)
        if target.stat().st_size!=info['total_bytes']:raise ValueError('Output extent differs from compact layout')
        state.update(verified=True,output_sha256=digest.hexdigest(),output_identity=identity(target),
            expert_bits=a.expert_bits,capability_reductions=[])
        save();note('Mac QFN full tensor readback PASS',str(len(info['tensors']))+' tensors and all companion assets','native model admission and continuation parity')
    if identity(a.source)!=info['source_identity']:raise ValueError('Conversion source changed')


if __name__=='__main__':main()
