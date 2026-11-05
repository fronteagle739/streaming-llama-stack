"""Read back all converted owners/assets, then certify the complete GGUF bytes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--source-sha256',required=True);p.add_argument('--progress-log',type=Path,required=True);a=p.parse_args()
subprocess.run([sys.executable,'-B',str(ROOT/'scripts/qfn_mac_convert.py'),'verify','--source',str(a.source),
 '--out',str(a.model),'--source-sha256',a.source_sha256,'--progress-log',str(a.progress_log)],check=True)
def identity():
 s=a.model.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)
before=identity();h=hashlib.sha256()
with a.model.open('rb') as f:
 while b:=f.read(4<<20):h.update(b)
if identity()!=before:raise ValueError('Converted model changed during full hash')
path=a.model.with_suffix('.conversion.json');receipt=json.loads(path.read_text())
if not receipt['verified']:raise ValueError('Tensor/asset verification did not complete')
receipt.update(output_sha256=h.hexdigest(),output_identity=before)
path.write_text(json.dumps(receipt,indent=2))
line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing=Mac full model qualification bytes verified; latest={before["size"]}B SHA256={h.hexdigest()}; next=authenticated native continuation and recall gate'
print(line,flush=True)
with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
