"""GPU gate: same exact-cone prompt in local mode1 and negotiated mode3/v2.

Run only in the owner's exclusive guarded GPU window. Reuse the actual WebUI
paired session's initial.tokens.le32. Complete export/import/ACK and browser
recall are a separate live gate; this compares every common cache owner and
full-vocabulary teacher-forced rows from the two producer configurations.
No payload or model copy is written.
"""
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
from slstack.combo_producer import ComboProducer
from slstack.native_broker import OwnedProducer,vendor
from slstack.lease import gpu_lease
vendor()
from studio.dsv41_session_contract import MODEL_SHA256


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def comparable(checkpoint):
    row=deepcopy(checkpoint)
    mode=row.pop('mode')
    row.pop('capture_end');row.pop('capture_length')
    if mode==3:row['owner_sha256']=row['owner_sha256'][1:]
    elif mode!=1:raise ValueError('Gate requires mode1 or complete mode3')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--tokens-le32',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--forced-ids',default='2,3,4',help='Identical evaluated token IDs; last ID is not evaluated')
    args=parser.parse_args()
    cfg=load(args.config)
    cfg.update(context=1000001,yoco_keep=2541,producer_profile='ultra',ctx_cache_precision='bf16',ctx_cache_growable=True,
               serving_mode='single_engine')
    cfg['env_overrides']={**cfg.get('env_overrides',{}),'LLAMADECODE_DS4_DECODE':'0','LLAMADECODE_APPEND_DECODE_MAX':'0'}
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    raw=Path(args.tokens_le32).read_bytes()
    if not raw or len(raw)%4:raise ValueError('Nonempty LE32 token file required')
    tokens=list(struct.unpack('<%dI'%(len(raw)//4),raw))
    forced=[int(t) for t in args.forced_ids.split(',')]
    if not forced or len(forced)>8 or any(t<0 or t>=129280 for t in forced):raise ValueError('Use1..8 native vocabulary IDs')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    result=dict(status='INCOMPLETE',scope='single vs paired RTX producer state; actual Mac import/recall requires separate WebUI proof',
                prompt_tokens=len(tokens),tokens_sha256=hashlib.sha256(raw).hexdigest(),yoco_keep=2541,
                forced_ids=forced,arms={})
    try:
        for arm in ('single','paired'):
            session=secrets.token_hex(16);owned=None
            logits=out/(arm+'-forced-logits.f32')
            try:
                if arm=='single':
                    with gpu_lease(cfg):
                        owned=ComboProducer(cfg,session,secrets.randbits(63) or 1,tokens,lambda:False)
                        initial=deepcopy(owned.checkpoint)
                        reply=owned.decode_forced(forced,logits)
                        result['arms'][arm]=dict(initial=initial,final=deepcopy(owned.checkpoint),forced=reply,
                            prefill_plan=owned.prefill_plan,ready_identity={key:owned.process.ready.get(key) for key in ('yoco_keep','yoco_state_version')})
                        result['arms'][arm]['cleanup']=owned.close();owned=None
                else:
                    request=dict(session=session,tier='ultra',tokens=raw,tokens_sha256=result['tokens_sha256'],yoco_keep=2541,yoco_state_version=2)
                    owned=OwnedProducer(cfg,request,threading.Event());p=owned.producer
                    initial=deepcopy(p.ready['checkpoint'])
                    if (p.ready.get('yoco_keep'),p.ready.get('yoco_state_version'))!=(2541,2):raise ValueError('Cone v2 negotiation failed')
                    reply=p.request(dict(session=session,model_sha256=MODEL_SHA256,sequence=p.sequence,operation='decode_forced',
                                         base=initial,tokens=forced,logits_path=str(logits)))
                    result['arms'][arm]=dict(initial=initial,final=deepcopy(reply['checkpoint']),forced=reply,
                        prefill_plan=owned.prefill_plan,ready_identity={key:p.ready.get(key) for key in ('yoco_keep','yoco_state_version')})
                    result['arms'][arm]['cleanup']=owned.close();owned=None
                if logits.stat().st_size!=len(forced)*129280*4:raise ValueError('Incomplete full-vocabulary forced logit rows')
                result['arms'][arm].update(logits_file=str(logits),logits_bytes=logits.stat().st_size,logits_sha256=digest(logits))
            finally:
                if owned is not None:owned.close()
        single,paired=(result['arms'][name] for name in ('single','paired'))
        result.update(initial_common_checkpoint_exact=comparable(single['initial'])==comparable(paired['initial']),
                      forced_common_checkpoint_exact=comparable(single['final'])==comparable(paired['final']),
                      all_forced_vocab_logits_exact=single['logits_sha256']==paired['logits_sha256'],
                      same_prefill_plan=single['prefill_plan']==paired['prefill_plan'])
        result['status']='PASS' if all(result[key] for key in ('initial_common_checkpoint_exact','forced_common_checkpoint_exact','all_forced_vocab_logits_exact','same_prefill_plan')) else 'FAIL'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc));raise
    finally:
        (out/'proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='arms'}),flush=True)
    return result['status']=='PASS'


if __name__=='__main__':raise SystemExit(0 if main() else 1)
