"""Authenticated one-request QFN state receiver; loopback only, bounded streaming.

SSH forwards this loopback port over the same live RTX/Mac link. Payloads pass
through a bounded pipe into ds4's existing loader. No multi-GB Mac scratch file
or Mac prompt replay is used. A hash-validated commit byte gates generation.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from slstack.model_compatibility import model_description,require_compatible
from slstack.envvars import rows,apply,split_secrets


def signature(key,body):
    return hmac.new(key,json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),hashlib.sha256).hexdigest()


def validate_handshake(row,key,nonce,local,max_bytes):
    expected={'schema','nonce','producer_binary_sha256','model_compatibility','verify_state','payload_bytes','hmac'}
    if set(row)!=expected or row['schema']!='qfn-paired-stream-v1' or row['nonce']!=nonce:
        raise ValueError('Invalid QFN authenticated request shape/nonce')
    body={k:v for k,v in row.items() if k!='hmac'}
    if not hmac.compare_digest(signature(key,body),str(row['hmac'])):raise ValueError('QFN authentication failed')
    if type(row['verify_state']) is not bool:raise ValueError('Invalid verification mode')
    for field in ('producer_binary_sha256',):
        if len(row[field])!=64 or any(x not in '0123456789abcdef' for x in row[field]):raise ValueError('Invalid identity hash')
    if type(row['payload_bytes'])is not int or not 52<row['payload_bytes']<=max_bytes:raise ValueError('Payload byte budget exceeded')
    require_compatible(row['model_compatibility'],local)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--port',type=int,default=18794);p.add_argument('--pair-token-file',type=Path,required=True)
    p.add_argument('--nonce',required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--vision-receipt',type=Path,required=True)
    p.add_argument('--max-payload-bytes',type=int,default=4<<30)
    p.add_argument('--host-budget-gib',type=float,required=True)
    p.add_argument('--floor-gib',type=float,required=True)
    for option in rows():
        if 'mac-ds4' in option['scopes']:
            p.add_argument('--option-'+option['var'].lower().replace('_','-'),dest='option_'+option['var'])
    p.add_argument('decoder',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.decoder[1:] if a.decoder[:1]==['--'] else a.decoder
    if not command or not 1024<=a.port<=65535 or len(a.nonce)!=32:raise ValueError('Invalid receiver flags')
    key=a.pair_token_file.read_bytes().strip()
    if len(key)<32:raise ValueError('Pairing token must have at least 256 bits')
    receipt=json.loads(a.model_receipt.read_text())
    s=a.model.stat();ident=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)
    if not receipt.get('verified') or receipt.get('output_identity')!=ident:raise ValueError('Mac model does not match verified conversion receipt')
    local=model_description(a.model,receipt['output_sha256'],0)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',a.port));listener.listen(1)
        print('QFN receiver listening on 127.0.0.1:'+str(a.port),flush=True)
        conn,_=listener.accept()
        with conn:
            request=bytearray()
            while not request.endswith(b'\n'):
                b=conn.recv(1)
                if not b or len(request)>65536:raise ValueError('Incomplete/oversized pairing request')
                request+=b
            row=json.loads(request);validate_handshake(row,key,a.nonce,local,a.max_payload_bytes)
            def setting(name):
                if command.count(name)!=1:raise ValueError('Decoder requires exactly one '+name)
                return command[command.index(name)+1]
            if (Path(setting('--model')).resolve()!=a.model.resolve() or setting('--payload')!='-'
                or setting('--payload-framed')!='1' or int(setting('--payload-bytes'))!=row['payload_bytes']
                or Path(setting('--out')).resolve()!=a.out.resolve()):
                raise ValueError('Decoder flags differ from authenticated model/payload contract')
            vision=Path(setting('--vision'));vs=vision.stat()
            vr=json.loads(a.vision_receipt.read_text())
            vi=dict(size=vs.st_size,mtime_ns=vs.st_mtime_ns,ino=vs.st_ino,device=vs.st_dev)
            if (not vr.get('verified') or vr.get('output_identity')!=vi or
                vr.get('source_sha256')!=receipt['companions']['vision/mmproj']):
                raise ValueError('Vision differs from the verified original pack companion/conversion')
            hello=dict(event='authenticated',nonce=a.nonce,receiver=local)
            hello['hmac']=signature(key,hello);conn.sendall((json.dumps(hello)+'\n').encode())
            a.out.parent.mkdir(parents=True,exist_ok=True)
            cfg=json.loads((ROOT/'config.example.json').read_text())
            cfg.update(dict(runtime_root=str(a.out.parent),host_budget_gib=a.host_budget_gib,
                min_available_gib=a.floor_gib,runtime_min_free_gib=a.floor_gib,
                job_commit_gib=a.host_budget_gib,gpu_uuid='metal',model=str(a.model),
                env_overrides={option['var']:getattr(a,'option_'+option['var']) for option in rows()
                    if getattr(a,'option_'+option['var'],None) is not None}))
            env=apply(dict(os.environ),cfg,'mac-ds4')
            recorded,_=split_secrets({option['var']:env.get(option['var']) for option in rows()})
            working_directory=str(Path(command[0]).resolve().parent)
            settings=dict(command=command,working_directory=working_directory,receiver=local,request={k:v for k,v in row.items() if k!='hmac'},
                vision_sha256=vr['output_sha256'],vision_identity=vi,
                engine_options=recorded,
                inherited_engine_options='replaced by explicit flags, table defaults and recorded native defaults',
                payload_staging_bytes=1<<20,listener='127.0.0.1',capability_reductions=[])
            a.out.with_suffix('.settings.json').write_text(json.dumps(settings,indent=2))
            child=subprocess.Popen(command,cwd=working_directory,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,env=env)
            failures=[]
            copies=2 if '--force-tokens' in command else 1
            def pump():
                try:
                    for _ in range(copies):
                        left=row['payload_bytes'];digest=hashlib.sha256() if row['verify_state'] else None
                        while left:
                            data=conn.recv(min(left,1<<20))
                            if not data:raise ValueError('Payload stream ended early')
                            if digest:digest.update(data)
                            child.stdin.write(data);left-=len(data)
                        child.stdin.flush()
                        if digest:
                            expected=bytearray()
                            while len(expected)<64:
                                chunk=conn.recv(64-len(expected))
                                if not chunk:raise ValueError('Inline state hash trailer missing')
                                expected+=chunk
                            if not hmac.compare_digest(digest.hexdigest().encode('ascii'),expected):raise ValueError('Inline state SHA differs; test commit refused')
                        child.stdin.write(b'G');child.stdin.flush()
                except BaseException as error:failures.append(repr(error))
                finally:child.stdin.close()
            worker=threading.Thread(target=pump);worker.start()
            try:
                for line in child.stdout:conn.sendall(line)
                code=child.wait();worker.join(timeout=10)
                if worker.is_alive() or failures or code:raise RuntimeError('QFN native receive failed: '+str((code,failures)))
            finally:
                if child.poll()is None:child.terminate();child.wait(timeout=30)
                conn.shutdown(socket.SHUT_RDWR)
                worker.join(timeout=10)


if __name__=='__main__':main()
