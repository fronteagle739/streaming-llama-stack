"""APFS clone of the verified Metal model, with explicit Q4_0 expert A/B.

Original metadata/offsets and every non-expert tensor survive; unused padding
retains original bytes. The clone avoids another full disk copy. KV stays F16.
Run niced under the RAM guard, and qualify numerically before selecting it.
"""
import argparse,ctypes,datetime,hashlib,json,math,re,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from slstack.gguf_header import read_header

def q4(raw):
 blocks=np.frombuffer(raw,dtype=np.uint8).reshape(-1,34)
 scales=blocks[:,:2].copy().reshape(-1).view('<f2').astype(np.float32)
 values=blocks[:,2:].copy().view(np.int8).astype(np.float32)*scales[:,None]
 indices=np.argmax(np.abs(values),axis=1);peak=values[np.arange(len(values)),indices]
 delta=peak/-8.;inverse=np.divide(1.,delta,out=np.zeros_like(delta),where=delta!=0)
 codes=np.minimum(15,np.floor(values*inverse[:,None]+8.5)).astype(np.uint8)
 out=np.empty((len(values),18),np.uint8);out[:,:2]=delta.astype('<f2').view(np.uint8).reshape(-1,2)
 out[:,2:]=codes[:,:16]|(codes[:,16:]<<4)
 return out.tobytes()

def identity(path):
 s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--source-receipt',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--progress-log',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 def note(doing,latest,next_step):
  line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
  print(line,flush=True)
  with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
 receipt=json.loads(a.source_receipt.read_text());before=identity(a.source)
 if not receipt.get('verified') or receipt['output_identity']!=before:raise ValueError('Source differs from its full verified certificate')
 if a.out.exists() and not a.resume:raise FileExistsError('Comparison model must be new, or explicitly resume its verified header')
 h=read_header(a.source);align=h['metadata'].get('general.alignment',32);start=(h['header_bytes']+align-1)//align*align
 entries=sum(24+len(t['name'].encode('utf8'))+8*len(t['shape']) for t in h['tensors'])
 with a.source.open('rb') as source:header=source.read(h['header_bytes']-entries)
 old=b'Q8_0 experts; original Q8_0 PLE; F16 dense';new=b'Q4_0 experts; original Q8_0 PLE; F16 dense'
 if header.count(old)!=1:raise ValueError('Original format declaration differs')
 header=header.replace(old,new);tensors=[]
 for t in h['tensors']:
  match=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight',t['name'])
  change=bool(match and int(match[1])<48)
  if change and t['type']!=8:raise ValueError('Expected original Q8 expert blocks')
  name=t['name'].encode();kind=2 if change else t['type']
  header+=struct.pack('<Q',len(name))+name+struct.pack('<I',len(t['shape']))+struct.pack('<'+'Q'*len(t['shape']),*t['shape'])+struct.pack('<IQ',kind,t['offset'])
  if change:tensors.append(t)
 if len(tensors)!=144:raise ValueError('Complete 48x3 main expert tensors required')
 import shutil
 need=sum(math.prod(t['shape'])//32*18 for t in tensors)
 if shutil.disk_usage(a.out.parent).free<need+(8<<30):raise ValueError('Insufficient disk for bounded APFS clone changes')
 if a.resume:
  if a.out.resolve()==a.source.resolve() or a.out.stat().st_size!=before['size']:raise ValueError('Resume clone identity/extent differs')
  with a.out.open('rb') as out:
   if out.read(len(header))!=header:raise ValueError('Resume clone metadata/tensor map differs')
 else:
  lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);lib.clonefile.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];lib.clonefile.restype=ctypes.c_int
  if lib.clonefile(bytes(a.source),bytes(a.out),0):raise OSError(ctypes.get_errno(),'APFS clone refused; no full-copy fallback')
 state=dict(receipt);state['completed']=dict(receipt['completed']);state['comparison_source']=dict(sha256=receipt['output_sha256'],identity=before)
 state['format']='Q4_0 main experts, Q8 MTP/PLE, F16 dense; original offsets/padding retained'
 with a.source.open('rb') as source,a.out.open('r+b') as out:
  out.write(header)
  for i,t in enumerate(tensors):
   source.seek(start+t['offset']);out.seek(start+t['offset']);left=math.prod(t['shape'])//32*34;digest=hashlib.sha256();written=0
   while left:
    raw=source.read(min(left,34*65536))
    if not raw or len(raw)%34:raise ValueError('Truncated expert input')
    converted=q4(raw);out.write(converted);digest.update(converted);written+=len(converted);left-=len(raw)
   state['completed'][t['name']]=dict(bytes=written,sha256=digest.hexdigest())
   if i%12==11:note('native Q4 expert comparison conversion',f'{i+1}/144 expert tensors','remaining tensors then complete output readback')
 if identity(a.source)!=before:raise ValueError('Original source changed')
 with a.out.open('rb') as out:
  for t in tensors:
   out.seek(start+t['offset']);entry=state['completed'][t['name']];left=entry['bytes'];digest=hashlib.sha256()
   while left:
    raw=out.read(min(left,4<<20))
    if not raw:raise ValueError('Truncated Q4 readback')
    digest.update(raw);left-=len(raw)
   if digest.hexdigest()!=entry['sha256']:raise ValueError('Q4 tensor readback differs')
 note('Q4 transformed tensors verified','144/144; whole cloned model SHA next','native admission and unchanged continuation gate')
 with a.out.open('rb') as out:state['output_sha256']=hashlib.file_digest(out,'sha256').hexdigest()
 state.update(output_identity=identity(a.out),verified=True,capability_reductions=[])
 a.out.with_suffix('.conversion.json').write_text(json.dumps(state,indent=2))
 note('native Q4 comparison model verified',state['output_sha256'],'native layout and paired quality/performance comparison')
if __name__=='__main__':main()
