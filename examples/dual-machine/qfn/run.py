"""Offline QFN RTX prefill / Mac native decode benchmark, with explicit flags.

Launch each memory-heavy stage through scripts/qfn_dual_guard.py. The rtx stage
also owns the stack GPU lease. This script never changes the selected preset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from slstack.paging_limits import shared_owned_paging_limit_bytes
from slstack.config import load,environment
from slstack.envvars import rows
from slstack.journal import atomic_json


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rtx(a):
    from slstack.lease import gpu_lease
    from slstack.qfn_single import QfnEngine
    if a.option_QFN_MTP is not None and a.option_QFN_MTP!=str(a.mtp):
        raise ValueError('--option-qfn-mtp conflicts with --mtp; select the same value')
    cfg=load(overrides=dict(model=str(a.model),qfn_engine=str(a.engine),runtime_root=str(a.output),
        context=a.context,qfn_tile=a.tile,host_budget_gib=a.host_budget_gib,
        qfn_host_budget_gib=a.native_budget_gib,min_available_gib=a.floor_gib,job_commit_gib=a.commit_gib,
        task=a.task,hold_file=str(a.hold_file),gpu_uuid=a.gpu_uuid,lease_minutes=a.lease_minutes,
        gpu_idle_max_mib=a.gpu_idle_max_mib,owned_paging_limit_bytes=shared_owned_paging_limit_bytes(),yoco_keep=0,
        qfn_qsa_indexer=a.indexer,qfn_nvfp4_mode=a.nvfp4_mode,qfn_nvfp4_decode_mode=a.decode_mode,
        qfn_protect_host_working_set=a.protect_host_working_set,qfn_lock_host_resident=a.lock_host_resident,
        env_overrides={r['var']:getattr(a,'option_'+r['var']) for r in rows() if getattr(a,'option_'+r['var'],None) is not None}))
    cfg['env_overrides']['QFN_MTP']=str(a.mtp)
    from slstack.performance_contract import seal
    cfg.setdefault('qfn_engram_scan_rings', int(cfg.get('env_overrides', {}).get('LLAMADECODE_ENGRAM_SCAN_RINGS', '1')))
    cfg = seal(cfg, 'qfn')
    command=QfnEngine(cfg).command();i=command.index('--port');del command[i:i+2]
    if a.handover_prefill:command+=['--handover-prefill']
    a.output.mkdir(parents=True,exist_ok=True)
    env=environment(cfg,'qfn-engine')
    settings=dict(command=command,flags=vars(a).copy(),model_sha256=a.model_sha256,
        model_identity=dict(size=a.model.stat().st_size,mtime_ns=a.model.stat().st_mtime_ns,ino=a.model.stat().st_ino),
        engine_sha256=sha(a.engine),effective_engine_options={r['var']:env.get(r['var']) for r in rows()
            if 'qfn-engine' in r['scopes'] or 'studio' in r['scopes']},
        capability_reductions=[])
    settings['flags']={k:str(v) if isinstance(v,Path) else v for k,v in settings['flags'].items()}
    if a.model_sha256:
        # Full hashes can reuse a separately verified receipt only on exact file identity.
        proof=json.loads(a.model_receipt.read_text()) if a.model_receipt else None
        expected=proof.get('reference_certificate_binding',{}) if proof else {}
        ident=expected.get('model_identity',{});s=a.model.stat()
        if (expected.get('model_sha256')!=a.model_sha256 or
            (ident.get('bytes'),ident.get('mtime_ns'),ident.get('ino'),ident.get('device'))!=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)):
            if sha(a.model)!=a.model_sha256:raise ValueError('Model full SHA differs')
    else:raise ValueError('A validated endpoint model SHA is required')
    atomic_json(a.output/'settings.json',settings)
    raw=a.tokens.read_bytes()
    if len(raw)!=a.prompt_tokens*4:raise ValueError('Prompt is not the exact requested token count')
    tokens=list(struct.unpack('<'+'i'*a.prompt_tokens,raw))
    records=[]
    with gpu_lease(cfg), (a.output/'engine.log').open('x',encoding='utf8') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True,encoding='utf8')
        def receive():
            line=child.stdout.readline()
            if not line:raise RuntimeError('RTX engine closed before checkpoint: '+str(child.poll()))
            row=json.loads(line);records.append(row)
            if not row.get('ok'):raise RuntimeError(str(row))
            return row
        def send(row):
            child.stdin.write(json.dumps(row,separators=(',',':'))+'\n');child.stdin.flush();return receive()
        try:
            if receive().get('event')!='ready':raise ValueError('RTX engine did not become ready')
            warm=send(dict(op='tokenize',text='A short warm-up: reply with the word ready.',special=True))['tokens']
            if tokens[:len(warm)]==warm:raise ValueError('Measured prompt must differ from warm-up prefix')
            send(dict(op='prefill',tokens=warm))
            (a.output/'warmup.tokens.le32').write_bytes(struct.pack('<'+'i'*len(warm),*warm))
            start=time.perf_counter();pf=send(dict(op='prefill',tokens=tokens));prefill=time.perf_counter()-start
            if pf['reused']!=0 or pf['n_past']!=len(tokens):raise ValueError('Measured second prompt reused warm-up state')
            start=time.perf_counter();export=send(dict(op='export_state',path=str(a.output/'state.payload'),max_bytes=a.payload_budget_bytes));export_s=time.perf_counter()-start
            observation=dict(op='observe',rows=a.observe_rows,path=str(a.output/'control.f32'))
            if a.force_tokens_file:
                forced_bytes=a.force_tokens_file.read_bytes()
                if len(forced_bytes)!=a.observe_rows*4:raise ValueError('Forced reference row count differs')
                observation['tokens']=list(struct.unpack('<'+'i'*a.observe_rows,forced_bytes))
            observe=send(observation)
            forced=observe['tokens'];(a.output/'forced.tokens.le32').write_bytes(struct.pack('<'+'i'*len(forced),*forced))
            (a.output/'prompt.tokens.le32').write_bytes(raw)
            result=dict(prompt_tokens=len(tokens),warmup_tokens=len(warm),prefix_reused=pf['reused'],
                prefill_seconds=prefill,prefill_tok_s=len(tokens)/prefill,native_prefill=pf,
                export_seconds=export_s,payload=export,model_sha256=a.model_sha256,
                prompt_sha256=hashlib.sha256(raw).hexdigest(),unbiased_control=observe)
            atomic_json(a.output/'result.json',result)
            print(json.dumps(result),flush=True)
        finally:
            if child.poll() is None:
                try:send(dict(op='quit'))
                finally:child.stdin.close();child.wait(timeout=60)
            atomic_json(a.output/'protocol.json',records)
        if child.returncode:raise RuntimeError('RTX engine exit '+str(child.returncode))


def qualify(a):
    from slstack.model_compatibility import continuation_kl
    from array import array
    x,y=a.control.read_bytes()[a.skip_rows*a.vocab*4:],a.candidate.read_bytes()[a.skip_rows*a.vocab*4:]
    result=continuation_kl(x,y,a.vocab)
    left,right=array('f'),array('f');left.frombytes(x);right.frombytes(y)
    if sys.byteorder!='little':left.byteswap();right.byteswap()
    top=[]
    for i in range(0,len(left),a.vocab):
        l=left[i:i+a.vocab];r=right[i:i+a.vocab]
        top.append(max(range(a.vocab),key=l.__getitem__)==max(range(a.vocab),key=r.__getitem__))
    result.update(direction=a.reference_label,excluded_leading_rows=a.skip_rows,top1=sum(top)/len(top),
        top1_per_row=top,mean_limit_nats=a.mean_limit,max_limit_nats=a.max_limit)
    result['qualified']=result['mean_nats']<=a.mean_limit and result['max_nats']<=a.max_limit and result['top1']>=a.top1_min
    atomic_json(a.out,result);print(json.dumps(result))
    if not result['qualified']:raise ValueError('Continuation quality gate failed; no qualified pair is published')


def identity(a):
    from slstack.model_compatibility import model_description,require_compatible
    producer=model_description(a.producer_pack,a.producer_sha256,0)
    receiver=model_description(a.receiver_model,a.receiver_sha256,0)
    mixed=require_compatible(producer,receiver)
    result=dict(producer=producer,receiver=receiver,cross_quant=mixed,
        status='candidate; native continuation qualification required')
    atomic_json(a.out,result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='stage',required=True)
    r=sub.add_parser('rtx');r.add_argument('--engine',type=Path,required=True);r.add_argument('--model',type=Path,required=True)
    r.add_argument('--model-sha256',required=True);r.add_argument('--model-receipt',type=Path)
    r.add_argument('--tokens',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--prompt-tokens',type=int,default=100000);r.add_argument('--context',type=int,default=100480)
    r.add_argument('--tile',type=int,default=2048);r.add_argument('--mtp',type=int,choices=[0,1],default=1)
    r.add_argument('--host-budget-gib',type=float,required=True);r.add_argument('--native-budget-gib',type=float,required=True)
    r.add_argument('--floor-gib',type=float,required=True);r.add_argument('--commit-gib',type=float,required=True)
    r.add_argument('--hold-file',type=Path,required=True);r.add_argument('--gpu-uuid',required=True)
    r.add_argument('--indexer',default='auto');r.add_argument('--nvfp4-mode',default='dsv-native');r.add_argument('--decode-mode',default='native-q8')
    r.add_argument('--task',default='QfnPair');r.add_argument('--lease-minutes',type=float,default=19)
    r.add_argument('--gpu-idle-max-mib',type=int,default=1024);r.add_argument('--handover-prefill',action='store_true')
    r.add_argument('--protect-host-working-set',type=int,choices=[0,1],default=1)
    r.add_argument('--lock-host-resident',type=int,choices=[0,1],default=1)
    r.add_argument('--force-tokens-file',type=Path)
    r.add_argument('--observe-rows',type=int,default=257);r.add_argument('--payload-budget-bytes',type=int,default=4<<30)
    for row in rows():
        if 'qfn-engine' in row['scopes'] or 'studio' in row['scopes']:
            r.add_argument('--option-'+row['var'].lower().replace('_','-'),dest='option_'+row['var'],default=None)
    q=sub.add_parser('qualify');q.add_argument('--control',type=Path,required=True);q.add_argument('--candidate',type=Path,required=True)
    q.add_argument('--vocab',type=int,required=True);q.add_argument('--skip-rows',type=int,default=1)
    q.add_argument('--reference-label',default='KL(unbiased RTX single-machine || paired Mac)');q.add_argument('--out',type=Path,required=True)
    q.add_argument('--mean-limit',type=float,default=.001);q.add_argument('--max-limit',type=float,default=.01);q.add_argument('--top1-min',type=float,default=.99)
    i=sub.add_parser('identity');i.add_argument('--producer-pack',type=Path,required=True);i.add_argument('--producer-sha256',required=True)
    i.add_argument('--receiver-model',type=Path,required=True);i.add_argument('--receiver-sha256',required=True);i.add_argument('--out',type=Path,required=True)
    a=p.parse_args();return {'rtx':rtx,'identity':identity,'qualify':qualify}[a.stage](a)


if __name__=='__main__':main()
