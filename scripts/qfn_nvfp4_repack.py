"""Lossless original NVFP4 container -> native grouped Streaming Llama Pack.

Run with the existing host-memory/I/O guardian. One expert in memory, one writer.
All non-expert sections, calibration, metadata fields and scale tails survive.
The v7 expert descriptor makes older readers refuse the unfamiliar layout.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dsv41_fp4_layout import NVFP4,pack_nvfp4_blob
from scripts.slpack import MAGIC,VERSION,LEGACY_MAGIC

PROGRESS=None
ALIGN=2<<20
BLOB=2764816
LAYERS=48
EXPERTS=512
CHUNK=16<<20

def digest(raw):return hashlib.sha256(raw).hexdigest()
def sequential(path,write=False):
    flags=(os.O_RDWR if write else os.O_RDONLY)|getattr(os,'O_BINARY',0)|getattr(os,'O_SEQUENTIAL',0)
    return os.fdopen(os.open(path,flags),'r+b' if write else 'rb',buffering=0)
def encode(obj):return (json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def durable(path,obj):
    tmp=Path(str(path)+'.next')
    with tmp.open('wb') as f:f.write(encode(obj));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def progress(text):
    line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+text
    print(line,flush=True)
    with PROGRESS.open('a',encoding='utf8') as f:f.write(line+'\n')
def header(path):
    with path.open('rb') as f:raw=f.read(ALIGN)
    magic,version,count,size=struct.unpack_from('<8sIIQ',raw)
    if not ((magic==MAGIC and version==VERSION) or (magic==LEGACY_MAGIC and version in (1,2))) or size!=path.stat().st_size:raise ValueError('Invalid model header')
    sections=[];end=ALIGN;names=set()
    for i in range(count):
        name,offset,length,sha=struct.unpack_from('<64sQQ32s',raw,64+i*128)
        name=name.rstrip(b'\0').decode('ascii')
        if name in names or offset<end or offset%ALIGN or offset+length>size:raise ValueError('Invalid section '+name)
        names.add(name);end=offset+length
        sections.append(dict(name=name,offset=offset,bytes=length,sha256=sha.hex()))
    if end!=size:raise ValueError('Unexpected model tail')
    return raw,sections
def read_small(f,row):
    if row['bytes']>32<<20:raise ValueError('Auxiliary section unexpectedly large')
    f.seek(row['offset']);raw=f.read(row['bytes'])
    if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:raise ValueError('Auxiliary section hash differs: '+row['name'])
    return raw
def identity(path):
    s=path.stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,ino=s.st_ino,device=s.st_dev)
def plan(source):
    raw,rows=header(source);by={r['name']:r for r in rows}
    if by['experts']['bytes']!=LAYERS*EXPERTS*BLOB:raise ValueError('NVFP4 experts geometry differs')
    with source.open('rb') as f:
        layout=read_small(f,by['experts.layout']).decode()
        meta=json.loads(read_small(f,by['meta']))
    if not layout.splitlines()[0].startswith('# experts v6:') or [list(map(int,l.split())) for l in layout.splitlines()[1:] if l.strip()]!=[
            [i,40,40,i*EXPERTS*BLOB,BLOB] for i in range(LAYERS)]:raise ValueError('Expected original native NVFP4 v6 layout')
    if 'qfn_native_nvfp4' in meta:raise ValueError('Source already has native layout metadata')
    storage=dict(version=1,layout=NVFP4,lossless=True,expert_stride=BLOB,role_offsets=[0,921600,1843200],
        tail_offset=2764800,tail_bytes=16,weight_scale='UE4M3 per16; unchanged FP32 globals',
        values='adjacent low/high E2M1 nibbles [N,K/2]',scales='UE4M3 [K/64,N,4]',
        kernel='DSV grouped native NVFP4 derivative; all arithmetic modes remain explicit',source_experts_layout=layout)
    newmeta={**meta,'qfn_native_nvfp4':storage}
    replacements={'meta':encode(newmeta),'experts.layout':('# experts v7 '+NVFP4+'\n'+'\n'.join(layout.splitlines()[1:])+'\n').encode()}
    planned=[]
    for i,row in enumerate(rows):
        new=dict(row)
        if row['name'] in replacements:
            value=replacements[row['name']]
            if i+1<len(rows) and row['offset']+len(value)>rows[i+1]['offset']:raise ValueError('Replacement exceeds reserved alignment gap')
            new.update(bytes=len(value),sha256=digest(value))
        planned.append(new)
    return dict(source_identity=identity(source),source_header_sha256=digest(raw),target_bytes=planned[-1]['offset']+planned[-1]['bytes'],
        source_sections=rows,target_sections=planned,storage=storage,original_metadata_fields_preserved=True),raw,replacements
def verify(path,expected,source=None):
    _,rows=header(path)
    if rows!=expected:raise ValueError('Written header differs from planned section descriptors')
    full=hashlib.sha256();sections=[]
    with sequential(path) as f:
        for row in rows:
            f.seek(row['offset']);h=hashlib.sha256();left=row['bytes']
            while left:
                b=f.read(min(CHUNK,left))
                if not b:raise ValueError('Short readback: '+row['name'])
                h.update(b);left-=len(b)
            if h.hexdigest()!=row['sha256']:raise ValueError('Written section hash differs: '+row['name'])
            sections.append(dict(name=row['name'],sha256=h.hexdigest(),bytes=row['bytes']))
            progress('doing=native NVFP4 disk readback; latest='+row['name']+' SHA verified; next=remaining sections')
        f.seek(0)
        while b:=f.read(CHUNK):full.update(b)
    return dict(verified=True,sha256=full.hexdigest(),size=path.stat().st_size,sections=sections)
def main():
    global PROGRESS
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--progress-log',type=Path,required=True)
    ap.add_argument('mode',choices=('plan','write','verify'))
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();PROGRESS=args.progress_log.resolve();source=args.source.resolve();out=args.out.resolve()
    if source==out:raise ValueError('Source must remain immutable')
    if not out.is_relative_to(args.workspace.resolve()):raise ValueError('Output must remain in the explicit workspace')
    info,raw,replacements=plan(source);journal=Path(str(out)+'.repack.json');partial=Path(str(out)+'.partial')
    if args.mode=='plan':
        durable(Path(str(out)+'.plan.json'),info);progress('doing=lossless NVFP4 repack planned; latest=target '+str(info['target_bytes'])+' bytes, unchanged expert stride and all26 sections; next=guarded single-writer conversion');return
    if args.mode=='verify':
        report=json.loads(journal.read_text());durable(Path(str(out)+'.verify.json'),verify(out,report['target_sections']));return
    if out.exists():raise ValueError('Completed output already exists; use verify')
    if shutil.disk_usage(out.parent).free<info['target_bytes']+(1<<30) and not partial.exists():raise ValueError('Insufficient output-volume headroom')
    binding=dict(source=str(source),source_identity=info['source_identity'],layout=NVFP4,tool_sha256=digest(Path(__file__).read_bytes()),
                 layout_tool_sha256=digest((ROOT/'scripts/dsv41_fp4_layout.py').read_bytes()))
    state=dict(binding=binding,finished_sections={},experts_complete=0)
    if partial.exists():
        if not args.resume:raise ValueError('Partial exists; explicit --resume required')
        state=json.loads(journal.read_text())
        if state['binding']!=binding:raise ValueError('Resume source/tool binding differs')
    else:
        with partial.open('xb') as f:f.truncate(info['target_bytes'])
        durable(journal,state)
    with sequential(source) as src,sequential(partial,write=True) as dst:
        for row,new in zip(info['source_sections'],info['target_sections']):
            name=row['name'];src.seek(row['offset']);dst.seek(new['offset']);oldhash=hashlib.sha256();newhash=hashlib.sha256()
            if name=='experts':
                for i in range(LAYERS*EXPERTS):
                    blob=src.read(BLOB)
                    if len(blob)!=BLOB:raise ValueError('Short native expert')
                    oldhash.update(blob)
                    packed=pack_nvfp4_blob(blob)
                    if pack_nvfp4_blob(packed,inverse=True)!=blob:raise ValueError('Lossless expert roundtrip failed at '+str(i))
                    newhash.update(packed)
                    if i<state['experts_complete']:
                        if dst.read(BLOB)!=packed:raise ValueError('Resume packed prefix differs at '+str(i))
                    else:dst.write(packed)
                    if (i+1)%EXPERTS==0:
                        dst.flush();os.fsync(dst.fileno());state['experts_complete']=i+1;durable(journal,state)
                        progress('doing=lossless native NVFP4 repack; latest='+str((i+1)//EXPERTS)+'/48 layers, every code/scale/tail byte inverse-verified; next=remaining layers then disk SHA')
            else:
                left=row['bytes']
                while left:
                    b=src.read(min(CHUNK,left))
                    if not b:raise ValueError('Short source section '+name)
                    oldhash.update(b);left-=len(b)
                    if name not in replacements:dst.write(b);newhash.update(b)
                if name in replacements:dst.write(replacements[name]);newhash.update(replacements[name])
            if oldhash.hexdigest()!=row['sha256']:raise ValueError('Source section SHA differs: '+name)
            new['sha256']=newhash.hexdigest();state['finished_sections'][name]=new['sha256'];durable(journal,state)
            progress('doing=lossless native NVFP4 section complete; latest='+name+' sourceSHA+outputSHA verified; next=remaining sections/readback')
        updated=bytearray(raw)
        struct.pack_into('<8sI',updated,0,MAGIC,VERSION)
        struct.pack_into('<Q',updated,16,info['target_bytes'])
        for i,row in enumerate(info['target_sections']):
            struct.pack_into('<64sQQ32s',updated,64+i*128,row['name'].encode(),row['offset'],row['bytes'],bytes.fromhex(row['sha256']))
        dst.seek(0);dst.write(updated);dst.flush();os.fsync(dst.fileno())
    if identity(source)!=binding['source_identity']:raise ValueError('Immutable source changed during conversion')
    state['target_sections']=info['target_sections'];state['all_experts_bit_exact_inverse']=True;durable(journal,state)
    report=verify(partial,info['target_sections'])
    os.replace(partial,out);report.update(layout=NVFP4,model_identity=identity(out),original_metadata_fields_preserved=True,
        all_experts_bit_exact_inverse=True,source_identity=binding['source_identity'],target_sections=info['target_sections'])
    durable(Path(str(out)+'.readback.json'),report)
    progress('doing=native NVFP4 repack finished; latest='+str(report['size'])+' bytes SHA '+report['sha256']+'; next=grouped kernel numeric and exact100K timing')
if __name__=='__main__':main()
