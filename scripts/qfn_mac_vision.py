"""Losslessly widen original BF16 QFN vision tensors to native Metal F32.

Every metadata byte and tensor is retained. Existing F16/F32/Q8 tensors remain
byte-identical. Writes are bounded, exclusive, and fully read back. No download.
Run niced under qfn_dual_guard.py with an explicit physical-memory budget.
"""
import argparse,datetime,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from slstack.gguf_header import read_header

def identity(path):
 s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)
def hash_file(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--source-sha256',required=True);p.add_argument('--progress-log',type=Path,required=True);a=p.parse_args()
 def note(doing,latest,next_step):
  line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
  print(line,flush=True)
  with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
 source_id=identity(a.source)
 if hash_file(a.source)!=a.source_sha256:raise ValueError('Original vision companion full SHA differs')
 h=read_header(a.source);alignment=h['metadata'].get('general.alignment',32)
 if not isinstance(alignment,int) or alignment<32 or alignment>65536 or alignment&(alignment-1):raise ValueError('Invalid vision alignment')
 align=lambda n:(n+alignment-1)//alignment*alignment
 descriptors=sum(24+len(t['name'].encode('utf8'))+8*len(t['shape']) for t in h['tensors'])
 metadata_end=h['header_bytes']-descriptors
 with a.source.open('rb') as f:header=f.read(metadata_end)
 if len(header)!=metadata_end:raise ValueError('Truncated vision metadata')
 tensors=[];offset=0
 for t in h['tensors']:
  if t['type'] not in (0,1,8,30):raise ValueError('Unsupported original vision tensor type: '+str(t['type']))
  elements=math.prod(t['shape']);kind=0 if t['type']==30 else t['type']
  old_bytes=elements*(4 if t['type']==0 else 2) if t['type']!=8 else elements//32*34
  if t['type']==8 and elements%32:raise ValueError('Unaligned vision Q8 tensor')
  size=elements*(4 if kind==0 else 2) if kind!=8 else old_bytes
  row=dict(t,source_bytes=old_bytes,output_offset=align(offset),output_bytes=size,output_type=kind);tensors.append(row);offset=row['output_offset']+size
  name=t['name'].encode('utf8');header+=struct.pack('<Q',len(name))+name+struct.pack('<I',len(t['shape']))
  header+=struct.pack('<'+'Q'*len(t['shape']),*t['shape'])+struct.pack('<IQ',kind,row['output_offset'])
 data_start=align(len(header));header+=bytes(data_start-len(header));source_start=align(h['header_bytes'])
 receipts={}
 with a.source.open('rb') as source,a.out.open('xb') as out:
  out.write(header)
  for t in tensors:
   source.seek(source_start+t['offset']);out.seek(data_start+t['output_offset']);digest=hashlib.sha256();left=t['source_bytes']
   while left:
    b=source.read(min(left,1<<20))
    if not b:raise ValueError('Truncated original vision tensor')
    left-=len(b)
    if t['type']==30:b=(np.frombuffer(b,dtype='<u2').astype('<u4')<<16).tobytes()
    out.write(b);digest.update(b)
   receipts[t['name']]=dict(bytes=t['output_bytes'],sha256=digest.hexdigest(),source_type=t['type'],output_type=t['output_type'])
 if identity(a.source)!=source_id:raise ValueError('Vision source changed during conversion')
 with a.out.open('rb') as f:
  if f.read(len(header))!=header:raise ValueError('Vision header readback differs')
  for t in tensors:
   f.seek(data_start+t['output_offset']);digest=hashlib.sha256();left=t['output_bytes']
   while left:
    b=f.read(min(left,1<<20))
    if not b:raise ValueError('Truncated converted vision tensor')
    left-=len(b);digest.update(b)
   if digest.hexdigest()!=receipts[t['name']]['sha256']:raise ValueError('Vision tensor readback differs')
 result=dict(verified=True,source_sha256=a.source_sha256,source_identity=source_id,
  output_sha256=hash_file(a.out),output_identity=identity(a.out),tensors=receipts,
  metadata_bytes_preserved=metadata_end,numerical_conversion='exact BF16 -> F32 bit widening; other types unchanged',capability_reductions=[])
 a.out.with_suffix('.conversion.json').write_text(json.dumps(result,indent=2))
 note('native Metal vision conversion PASS',f'{len(tensors)} tensors, {a.out.stat().st_size} bytes, exact BF16 values retained','native vision admission and paired continuation')
if __name__=='__main__':main()
