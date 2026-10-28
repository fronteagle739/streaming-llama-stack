"""Opt-in, guarded same-day single-engine comparisons. No shipping presets change.

Use --root <scratch> --experiment E13-sector --candidate LLAMADECODE_ENGRAM_READ_ALIGN=512.
Each invocation runs a fresh control and candidate with identical token inputs.
Only explicit environment settings are varied, and actual child environments are
checked before admission. A successful runtime is not a correctness verdict: all
teacher-forced logits and recall tokens are retained for analysis.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import threading
import time
import urllib.request


def completed_artifact_sha256(path):
    # Windows may briefly retain a writer/AV sharing lock after the producer
    # acknowledges a completed logits dump. Retry only this read, never decode.
    deadline=time.monotonic()+1
    while True:
        try:
            with path.open('rb') as stream:
                return hashlib.file_digest(stream,'sha256').hexdigest()
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.02)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slstack.config import load, environment
from slstack.combo_producer import ComboProducer, prefill_plan
from slstack.lease import gpu_lease, AdmissionRefused
from slstack.process import OwnedProcess
from slstack.guard import admit
from slstack.gguf_header import read_header
from slstack.ds4_render import Tokenizer


def http(endpoint):
    return json.load(urllib.request.urlopen('http://127.0.0.1:8765/'+endpoint, timeout=15))


def parse_setting(raw):
    key, value = raw.split('=', 1)
    if not key or not value:raise ValueError('Setting requires NAME=value')
    return key,value


def evidence(root, experiment, cfg):
    result=Path(cfg['peer_result_file']) if cfg.get('peer_result_file') else None
    data=result.read_bytes() if result and result.exists() else b''
    row=dict(checked_at=datetime.now(timezone.utc).isoformat(),
             result_present=bool(result and result.exists()),result_sha256=hashlib.sha256(data).hexdigest(),
             ordering='E13,E2c,E6,E8,E9,E7,E10,E14,E15,E17; E12 dropped',
             decision='Owner correction takes priority. Published ledger: Engram selected rows, not whole tables; 315825603008 weight bytes/full pass.')
    (root/(experiment+'-e1-check.json')).write_text(json.dumps(row,indent=2))
    return row


def run_child(args, cfg, arm, label):
    root=args.root
    if args.trace_append:
        dump=root/(label+'.layers.bin')
        if dump.exists():raise FileExistsError('Attribution output already exists: '+str(dump))
        cfg['env_overrides'].update(DS4COMBO_V41_DUMP=str(dump),
                                   DS4COMBO_V41_DUMP_POS=str(args.count+77),LLAMADECODE_DS4_BIAS='0')
        actual=environment(cfg,'rtx-producer')
        for key in ('DS4COMBO_V41_DUMP','DS4COMBO_V41_DUMP_POS','LLAMADECODE_DS4_BIAS'):
            if actual.get(key)!=cfg['env_overrides'][key]:raise ValueError('Attribution setting not applied: '+key)
    seed_path=args.seed_file or root/'seed.tokens.le32'
    if not seed_path.resolve().is_relative_to(root):raise ValueError('Fixture must stay under configured scratch')
    seed=seed_path.read_bytes()
    # Preserve the original retrieval key and the terminal question/chat markers.
    # Only the repeated background between them is shortened or extended.
    if args.count == len(seed)//4:raw=seed
    else:
        if args.count < 384:raise ValueError('Recall fixture requires at least384 tokens')
        body=seed[256*4:-128*4]
        n=(args.count-384)*4
        raw=seed[:256*4]+(body*(n//len(body)+1))[:n]+seed[-128*4:]
    tokens=list(struct.unpack('<'+'I'*args.count,raw))
    forced=tokens[-min(16,len(tokens)):]
    model=Path(cfg['model']);identity=model.stat()
    dll=Path(cfg['llama_producer']).parent/'ds4-decode.dll'
    dll_hash=hashlib.file_digest(dll.open('rb'),'sha256').hexdigest()
    if cfg.get('producer_decode_dll_sha256') and dll_hash!=cfg['producer_decode_dll_sha256']:
        raise ValueError('Decode DLL hash differs from the pinned diagnostic build')
    kernel_assets=[]
    for asset in cfg.get('producer_kernel_assets',[]):
        path=Path(asset['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Kernel asset must stay in configured scratch')
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if digest!=asset['sha256']:raise ValueError('Kernel image hash differs from the pinned build')
        kernel_assets.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
    result=dict(experiment=args.experiment,arm=arm,day=datetime.now(timezone.utc).date().isoformat(),
                model=str(model),model_bytes=identity.st_size,model_mtime_ns=identity.st_mtime_ns,
                prompt_tokens=len(tokens),input_sha256=hashlib.sha256(raw).hexdigest(),
                seed_sha256=hashlib.sha256(seed).hexdigest(),
                recall_fixture=True,producer_sha256=cfg['producer_sha256'],
                config_sha256=cfg['_pair_config_sha256'],
                source_revision=cfg['producer_source_revision'],status='RUNNING',
                candidate_setting=args.candidate)
    result['decode_dll_sha256']=dll_hash
    result['kernel_assets']=kernel_assets
    result['trace_append']=args.trace_append
    target=root/(label+'.json')
    producer=None;owner=[];done=threading.Event();samples=[]
    if target.exists():raise FileExistsError('Retain prior evidence; use a new experiment label: '+str(target))
    target.write_text(json.dumps(result,indent=2))
    def observe():
        offset=0
        with (root/(label+'-progress.log')).open('ab') as log:
            while not done.wait(1):
                if owner:
                    process=getattr(owner[0],'process',None) or getattr(owner[0],'loading_process',None)
                    diagnostic=bytes(getattr(process,'diagnostic',b''))
                    if len(diagnostic)>offset:log.write(diagnostic[offset:]);log.flush();offset=len(diagnostic)
                try:
                    fields='timestamp,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu'
                    sample=subprocess.check_output(['nvidia-smi','--id='+cfg['gpu_uuid'],
                           '--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                    samples.append(dict(monotonic=time.monotonic(),sample=sample))
                except Exception as error:samples.append(dict(error=repr(error)))
    observer=threading.Thread(target=observe,daemon=True);observer.start()
    try:
        opened=time.monotonic()
        producer=ComboProducer(cfg,secrets.token_hex(16),1,tokens,lambda:False,on_start=owner.append)
        result.update(prefill_seconds=producer.prefill_seconds,tok_s=len(tokens)/producer.prefill_seconds,
                      plan=producer.prefill_plan,ready=producer.process.ready,
                      native_directory=str(producer.directory),prefill_finished_at=time.monotonic())
        producer.quality(0)
        logits=root/(label+'.logits.f32')
        first=producer.decode(1)
        result['ttft_seconds']=time.monotonic()-opened
        result['ttft_before_quality_probe']=True
        native_report=json.loads((producer.directory/'producer.json').read_text(encoding='utf-8'))
        prompt_seconds=float(native_report['phase_ms']['embedding_40_layers_head'])/1000
        if not 0<prompt_seconds<=producer.prefill_seconds:
            raise ValueError('Native prompt-only timing is absent or outside producer wall')
        result.update(prompt_prefill_seconds=prompt_seconds,prompt_tok_s=len(tokens)/prompt_seconds,
                      producer_startup_inclusive_seconds=producer.prefill_seconds,
                      prefill_timing_scope='prompt: seed embeddings through required frontier head/sync; process/model open and bulk export excluded; TTFT: producer initialization through first returned token, excludes lease queue')
        # The final sampled token is not evaluated. Read this same untouched
        # frontier after timing first output, so the quality probe adds no TTFT.
        result['frontier']=producer.decode_forced([0],root/(label+'.frontier.f32'))
        result['recall']=producer.decode(63)
        result['first_token']=first['tokens']
        result['recall_before_forced']=True
        result['turns']=[]
        if args.engram_turns:
            meta=read_header(model)['metadata']
            tokenizer=Tokenizer(meta['tokenizer.ggml.tokens'],meta['tokenizer.ggml.merges'])
            from slstack.combo_fixture import record
            for index in range(args.engram_turns):
                # A/B/A revisits rows beyond the immediately preceding prompt
                # cache. Fresh history affects the first three hashes only.
                offset=1 if index%2==0 else 301
                body=''.join(record(i)[0] for i in range(offset,offset+3))
                body+='\nIgnore the log details. Repeat the original archive retrieval key exactly.'
                turn_tokens=[tokenizer.eos,tokenizer.user]+tokenizer.text(body)+[tokenizer.assistant,tokenizer.think_end]
                began=time.monotonic()
                reply=producer.prefill(turn_tokens)
                turn=dict(index=index,prompt_tokens=len(turn_tokens),seconds=time.monotonic()-began,reply=reply)
                turn['frontier']=producer.decode_forced([0],root/(label+f'.turn{index}.f32'))
                turn['recall']=producer.decode(32)
                result['turns'].append(turn)
        def measure_decode():
            if args.warmup:result['warmup']=producer.decode(args.warmup)
            start=time.monotonic();result['decode']=producer.decode(args.decode)
            result['decode_seconds']=time.monotonic()-start
        result['forced_after_decode']=args.decode_before_forced
        if args.decode_before_forced:measure_decode()
        result['forced']=producer.decode_forced(forced,logits)
        result['logits_sha256']=completed_artifact_sha256(logits)
        if args.append:
            result['append_base_checkpoint']=producer.checkpoint
            result['append_quality_bias']=0
            if args.trace_append and producer.checkpoint['position']!=args.count+77:
                raise ValueError('Attribution position no longer matches the untouched append boundary')
            meta=read_header(model)['metadata']
            tokenizer=Tokenizer(meta['tokenizer.ggml.tokens'],meta['tokenizer.ggml.merges'])
            question=tokenizer.text('\nRepeat the original archive retrieval key exactly.')
            fill=args.append-len(question)-4
            if fill<0:raise ValueError('Append too short for the retrieval question and chat markers')
            body=tokenizer.text('Consider the previous conversation. ')
            turn=[tokenizer.eos,tokenizer.user]+(body*(fill//len(body)+1))[:fill]+question+[tokenizer.assistant,tokenizer.think_end]
            assert len(turn)==args.append
            result['append_input_sha256']=hashlib.sha256(struct.pack('<'+'I'*len(turn),*turn)).hexdigest()
            start=time.monotonic()
            result['append']=producer.prefill(turn)
            result['append_seconds']=time.monotonic()-start
            result['append_tokens']=args.append
            result['append_first_token']=producer.decode(1)['tokens']
            result['append_ttft_seconds']=time.monotonic()-start
            result['append_frontier']=producer.decode_forced([0],root/(label+'.append.f32'))
            result['append_recall']=producer.decode(32)
        if not args.decode_before_forced:measure_decode()
        if args.capture_kv and arm=='candidate':
            kv_path=root/(label+'.kv.f32')
            result['kv_samples']=producer.process.request(producer.command('perfexp_kv_samples',base=producer.checkpoint,path=str(kv_path)))
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=repr(error),cleanup=getattr(error,'owned_cleanup',None))
        raise
    finally:
        try:
            if producer:result['cleanup']=producer.close()
        finally:
            done.set();observer.join(6)
            result['samples']=samples
            after=model.stat()
            result['model_unchanged']=(identity.st_size,identity.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            target.write_text(json.dumps(result,indent=2,default=str))
            print(json.dumps({k:result.get(k) for k in ('experiment','arm','status','prompt_prefill_seconds','prompt_tok_s','ttft_seconds','producer_startup_inclusive_seconds','error')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--candidate',type=parse_setting,required=True)
    p.add_argument('--count',type=int,default=8192)
    p.add_argument('--decode',type=int,default=32)
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--append',type=int,default=0)
    p.add_argument('--capture-kv',action='store_true')
    p.add_argument('--trace-append',action='store_true',help='Synchronizing E2c first-append-token layer trace; never use as timing evidence')
    p.add_argument('--candidate-first',action='store_true')
    p.add_argument('--engram-turns',type=int,choices=(0,3),default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seed-file',type=Path)
    p.add_argument('--config-name',default='config.json')
    p.add_argument('--decode-before-forced',action='store_true')
    p.add_argument('--child',choices=('control','candidate'))
    p.add_argument('--lease')
    args=p.parse_args()
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,63}',args.experiment):raise ValueError('Use a bounded experiment label')
    if not 1<=args.count<=200000 or not 1<=args.decode<=128 or not 0<=args.warmup<=128:raise ValueError('Unbounded experiment')
    if not 0<=args.append<=256:raise ValueError('Unbounded append experiment')
    if args.trace_append and (not args.append or args.engram_turns or args.decode_before_forced):
        raise ValueError('Attribution requires the canonical append sequence')
    if args.decode_before_forced and (args.append or args.engram_turns):
        raise ValueError('Natural decode window cannot mix with append/turn timing')
    root=args.root.resolve()
    args.root=root
    if Path(args.config_name).name!=args.config_name or not args.config_name.endswith('.json'):
        raise ValueError('Config must be a JSON basename in the configured scratch root')
    cfg=load(root/args.config_name)
    cfg['_pair_config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True,default=str).encode()).hexdigest()
    if Path(cfg['runtime_root']).resolve().parent!=root:
        raise ValueError('Runtime and evidence must use the one configured scratch root')
    os.environ.update(TEMP=str(root),TMP=str(root),TMPDIR=str(root),PYTHONDONTWRITEBYTECODE='1',
                      CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
    arm=args.child
    key,value=args.candidate
    if arm=='candidate':cfg['env_overrides'][key]=value
    cfg['env_overrides']['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    if arm=='candidate' and environment(cfg,'rtx-producer').get(key)!=value:
        raise ValueError('Candidate setting is not applied to the producer: '+key)
    label=args.experiment+'-'+(arm or 'pair')+'-'+str(args.count)
    if arm:
        cfg['hold_watch']=dict(path=cfg['hold_file'],identity=args.lease,task='PERFEXP')
        run_child(args,cfg,arm,label)
        return
    evidence(root,args.experiment,cfg)
    for arm in (('candidate','control') if args.candidate_first else ('control','candidate')):
        prior=root/(args.experiment+'-'+arm+'-'+str(args.count)+'.json')
        if args.resume and prior.exists():
            row=json.loads(prior.read_text())
            if (row.get('status')=='PASS' and row.get('day')==datetime.now(timezone.utc).date().isoformat()
                    and row.get('producer_sha256')==cfg['producer_sha256']
                    and row.get('config_sha256')==cfg['_pair_config_sha256']
                    and row.get('seed_sha256')==hashlib.sha256((args.seed_file or root/'seed.tokens.le32').read_bytes()).hexdigest()
                    and row.get('candidate_setting')==list(args.candidate)
                    and len(row.get('turns',[]))==args.engram_turns
                    and row.get('trace_append',False)==args.trace_append
                    and row.get('append_tokens',0)==args.append):
                # A completed arm may have stopped at the external health gate.
                # Preserve its original failure receipt, and require recovery
                # before reusing its numerical evidence or admitting another arm.
                guard=json.loads((root/(args.experiment+'-'+arm+'-guard.json')).read_text())
                if not guard.get('cleanup_confirmed'):raise RuntimeError('Completed arm lacks confirmed owned cleanup')
                recovered=dict(health=http('health'),remote=http('remote-link/status'))
                remote=recovered['remote']
                if recovered['health'].get('model') is not None or not (remote.get('running') and remote.get('serving') and remote.get('end_to_end',{}).get('ok')):
                    raise RuntimeError('Completed arm cannot resume until Studio/Remote Link health recovers')
                (root/(args.experiment+'-'+arm+'-resume-health.json')).write_text(json.dumps(recovered,indent=2))
                print('RESUME_COMPLETED_ARM',arm,flush=True)
                continue
            raise ValueError('Prior arm cannot be resumed; use a new label')
        # Re-check model existence before each run; never move or modify it.
        if not Path(cfg['model']).is_file():raise FileNotFoundError('Q4 model moved; refresh config from its verified receipt')
        deadline=time.monotonic()+cfg.get('pair_wait_seconds',900)
        lease_context=None
        while lease_context is None:
            try:
                # Reserve the thin parent/guardian startup overhead as well as
                # the enforced child-tree cap; acquire no GPU hold while short.
                admit(cfg['host_budget_gib']+.25,cfg['min_available_gib'])
                # The native executable also requires one GiB beyond its own
                # planned host bank before allocating it. Respect that stricter
                # admission separately from the runtime 0.5 GiB floor.
                native_plan=prefill_plan(cfg,args.count)
                admit(native_plan['host_bytes']/2**30+.25,1)
                candidate_context=gpu_lease(cfg)
                lease=candidate_context.__enter__()
            except (AdmissionRefused,PermissionError,MemoryError) as error:
                if time.monotonic()>=deadline:raise
                print('WAIT_ADMISSION',str(error),flush=True)
                time.sleep(1)
            else:lease_context=candidate_context
        try:
            cfg['hold_watch']=dict(path=cfg['hold_file'],identity=lease,task='PERFEXP')
            before=http('health')
            # No active Studio model is the observed initial state. Refuse a
            # state change rather than silently losing another user's session.
            if before.get('model') is not None:raise RuntimeError('Studio model changed; pause/restore its exact session before benchmarking')
            (root/(args.experiment+'-'+arm+'-studio-before.json')).write_text(json.dumps(before,indent=2))
            command=[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--child',arm,'--lease',lease]
            job=OwnedProcess(cfg,command,args.experiment+'-'+arm,cwd=Path(__file__).resolve().parents[1],scope='rtx-producer',telemetry=True)
            try:
                receipt=job.wait()
                (root/(args.experiment+'-'+arm+'-guard.json')).write_text(json.dumps(receipt,indent=2))
                if receipt['exit_code'] or receipt['reason']:raise RuntimeError('Experiment child failed: '+str(job.log))
            except BaseException as error:
                # A missing/uncertain retirement receipt retains resource
                # ownership. Do not turn a guardian failure into GPU admission.
                try:
                    retired=json.loads(job.receipt.read_text())
                    confirmed=retired.get('cleanup_confirmed') is True
                except (OSError,ValueError):confirmed=False
                if not confirmed:error.owned_cleanup_uncertain=True
                raise
            finally:
                if job.log.exists():print(job.log.read_text(errors='replace')[-2500:],flush=True)
                for endpoint in ('health','remote-link/status'):
                    data=http(endpoint)
                    (root/(args.experiment+'-'+arm+'-studio-'+endpoint.replace('/','-')+'.json')).write_text(json.dumps(data,indent=2))
                    if endpoint=='remote-link/status' and not (data.get('running') and data.get('serving') and data.get('end_to_end',{}).get('ok')):
                        raise RuntimeError('Remote Link failed its post-run health gate')
        finally:
            lease_context.__exit__(*sys.exc_info())


if __name__=='__main__':main()
