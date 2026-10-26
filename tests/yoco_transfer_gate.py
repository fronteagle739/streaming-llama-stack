"""GPU qualification of negotiated YOCO mode3 export and incremental owners."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import sys
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.native_broker import OwnedProducer,vendor
from slstack.ds4_render import Tokenizer,render
from slstack.combo_contract import owner_layout
vendor()
from studio.dsv41_session_contract import MODEL_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();cfg=load(args.config)
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    tokenizer=Tokenizer.from_gguf(cfg['model'])
    _,tokens,_=render(dict(messages=[dict(role='user',content='Transfer the original YOCO state. '*80)],reasoning_effort='none'),tokenizer)
    raw=struct.pack('<'+'I'*len(tokens),*tokens);session=secrets.token_hex(16)
    request=dict(session=session,tier='ultra',tokens=raw,tokens_sha256=hashlib.sha256(raw).hexdigest(),yoco_keep=140,yoco_state_version=1)
    result=dict(tokens=len(tokens),yoco_keep=140);owned=None
    try:
        owned=OwnedProducer(cfg,request,threading.Event());p=owned.producer
        def command(op,**values):return dict(session=session,model_sha256=MODEL_SHA256,sequence=p.sequence,operation=op,**values)
        cp=deepcopy(p.ready['checkpoint'])
        assert p.ready['yoco_state_version']==1 and cp['mode']==3
        def export(op,checkpoint):
            meta=p.begin_export(command(op,base=checkpoint));path=Path(cfg['runtime_root'])/(op+'-'+secrets.token_hex(4)+'.kv')
            with path.open('xb') as out:
                while block:=p.read_export():out.write(block)
            p.end_export()
            if op=='export_delta':return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            sizes=[min(checkpoint['position'],128)*3*5120*4,*owner_layout(checkpoint['position']).values()]
            with path.open('rb') as file:
                assert struct.unpack('<13I',file.read(52))==(0x34565344,2,checkpoint['context'],3,128,128,checkpoint['context']+1,checkpoint['position'],40,512,128,129280,0x413431)
                assert struct.unpack('<2I',file.read(8))==(checkpoint['position'],min(checkpoint['position'],128))
                def region(size,expected):
                    digest=hashlib.sha256()
                    while size:
                        block=file.read(min(size,65536));assert block;digest.update(block);size-=len(block)
                    assert digest.hexdigest()==expected
                region(checkpoint['position']*4,checkpoint['tokens_sha256']);region(129280*4,checkpoint['logits_sha256'])
                for size,digest in zip(sizes,checkpoint['owner_sha256'],strict=True):region(size,digest)
                assert not file.read(1)
            return dict(bytes=path.stat().st_size,all_owner_hashes_verified=True,mode=3,dspark_rows=min(checkpoint['position'],128))
        result['initial']=export('export_initial',cp)
        p.request(command('acknowledge',base=cp));p.request(command('save_rewind',base=cp))
        suffix=tokenizer.rendered(' Append these new tokens now.')
        target=p.request(command('prefill',base=cp,tokens=suffix))['checkpoint']
        result['appended']=export('export_local',target)
        result['delta']=export('export_delta',target);p.request(command('acknowledge',base=target))
        restored=p.request(command('rewind',base=target,target=cp))['checkpoint'];assert restored==cp
        p.request(command('acknowledge',base=cp));result.update(rewind_hashes_equal=True,status='PASS')
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        if owned:owned.close()
        Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
