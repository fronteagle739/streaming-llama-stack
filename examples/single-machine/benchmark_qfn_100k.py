"""Portable guarded 100K benchmark: same sequence, 100 warm +128 measured per mode.

Preserve natural EOG and every forwarded sampler field. The exact archived
fixture is reconstructed by scripts/build_qfn_100k_fixture.py. See
docs/QFN_100K_REPRO.md for proof scope, prerequisites and commands.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
def identity(path):
    stat=Path(path).stat()
    return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,ino=stat.st_ino,device=stat.st_dev)


def bind_model(certificate,model,expected):
    report=json.loads(Path(certificate).read_text(encoding='utf8'))
    assert report['verified'] and Path(report['model']).resolve()==Path(model).resolve()
    assert report['model_identity']==identity(model),'Model changed after complete readback'
    assert report['sha256']==expected,'Readback differs from requested model SHA256'
    return dict(model_sha256=expected,model_identity=identity(model),certificate_sha256=sha(certificate),
                binding='Explicit complete-readback certificate retained with exact current file identity')


def certify(options):
    sys.path.insert(0,options['repo_root'])
    from scripts.slpack import ModelFile
    model=Path(options['model']).resolve()
    before=identity(model)
    report=ModelFile(model,allow_legacy=True).verify_all()
    assert report['sha256']==options['model_sha256'],'Full model SHA256 differs'
    assert before==identity(model),'Model changed during complete readback'
    report.update(verified=True,model=str(model),model_identity=before,all_sections_verified=True)
    with Path(options['model_certificate']).open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)


STACK=None
COMPILERS={'nvcc.exe','cicc.exe','ptxas.exe','cl.exe','link.exe','ninja.exe','nvlink.exe','fatbinary.exe','cudafe++.exe'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def progress(message):
    line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message
    print(line,flush=True)



def child(options):
    global STACK
    STACK=Path(options['repo_root'])
    sys.path.insert(0,str(STACK))
    prefix=Path(options['output_prefix'])
    fixture=Path(options['fixture'])
    assert fixture.stat().st_size==400000 and sha(fixture)=='87fb9af861237f91724d56ca9c1a29f77c0e4aadce673ff939abe8500fa6c936'
    tokens=list(struct.unpack('<100000i',fixture.read_bytes()))
    capacity=max(100352,((100000+256+2*options.get('warmup_tokens',0)+127)//128)*128)
    command=[options.get('engine',str(STACK/'bin/qfn-engine.exe')),'--serve','--model',options['model'],
             '--vocab-gguf',options['vocab_gguf'],'--ctx',str(capacity),'--tile',str(options.get('tile',2048)),
             '--host-budget-gib',str(options.get('host_rss',3)),'--ple-row-dump',str(prefix)+'.ple.raw']
    for field,flag in (('nvfp4_mode','--nvfp4-mode'),('nvfp4_decode_mode','--nvfp4-decode-mode'),('qsa_indexer','--qsa-indexer'),
                       ('qsa_attention','--qsa-attention'),('cache_slots','--cache-slots')):
        if options[field] is not None:command += [flag,str(options[field])]
    extra=options.get('engine_arg',[])
    reserved={'--serve','--model','--vocab-gguf','--ctx','--tile','--host-budget-gib','--ple-row-dump','--host','--port'}
    assert not any(item.split('=',1)[0] in reserved for item in extra),'Conflicting benchmark/protocol argv; use the explicit benchmark options'
    command.extend(extra)
    warmup=options.get('warmup_tokens',100)
    sampler=json.loads(Path(options['sampler_json']).read_text(encoding='utf8')) if options.get('sampler_json') else {}
    assert isinstance(sampler,dict),'Sampler request must be a JSON object'
    assert not (set(sampler)&{'op','max_tokens','bias'}),'Reserved benchmark window fields conflict; no sampler field silently ignored'
    sampler={'temperature':0,'seed':1,**sampler}
    result={'command':command,'prompt_sha256':sha(fixture),'decode_design':f'quality then speed, each {warmup} warmup and128 evaluated tokens, same sequence; natural EOG active','warmup_tokens':warmup,'sampler':sampler,'exact_published_protocol':warmup==100 and sampler=={'temperature':0,'seed':1},'modes':{}}
    with Path(str(prefix)+'.engine.log').open('w',encoding='utf8') as log, Path(str(prefix)+'.events.jsonl').open('w',encoding='utf8') as events:
        engine=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                                text=True,encoding='utf8',cwd=STACK,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        def read():
            line=engine.stdout.readline()
            if not line:raise RuntimeError('Native protocol ended: '+str(engine.poll()))
            item=json.loads(line);events.write(json.dumps(item,ensure_ascii=False)+'\n');events.flush()
            if item.get('ok') is not True:raise RuntimeError(item)
            return item
        def send(item):engine.stdin.write(json.dumps(item)+'\n');engine.stdin.flush()
        try:
            result['ready']=read();assert result['ready']['event']=='ready'
            # Private attribution evidence from the actual native process,
            # outside all timed phases. Keep secrets comparable by digest.
            import psutil
            from slstack.envvars import split_secrets
            actual=psutil.Process(engine.pid)
            public,secrets=split_secrets(actual.environ())
            modules=[]
            for mapped in actual.memory_maps(grouped=True):
                path=Path(mapped.path)
                if path.suffix.lower() in ('.dll','.exe') and path.is_file():
                    modules.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
            observation=dict(pid=engine.pid,created=actual.create_time(),argv=actual.cmdline(),
                cwd=actual.cwd(),environment=public,
                secret_environment_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in secrets.items()},
                loaded_modules=modules,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            with Path(str(prefix)+'.process-observation.json').open('x',encoding='utf8') as stream:
                json.dump(observation,stream,indent=2)
            result['process_observation_sha256']=sha(str(prefix)+'.process-observation.json')
            send(dict(op='prefill',tokens=tokens));pf=read();result['prefill']=pf
            assert pf['new_tokens']==100000 and pf['n_past']==100000 and pf['route']=='layer-major',pf
            result['prefill_tok_per_s']=100000/pf['seconds']
            progress('doing='+options['label']+' exact100K prefill measured; latest='+str(result['prefill_tok_per_s'])+'tok/s seconds='+str(pf['seconds'])+'; next=quality128 then speed128 and recall')
            for name,bias in (('quality',0.0),('speed',options.get('speed_bias',1.36125))):
                warmed=None
                if warmup:
                    send(dict(sampler,op='generate',max_tokens=warmup,bias=bias));warmids=[];warmtext=''
                    while True:
                        event=read()
                        if 'token' in event:
                            warmtext+=event.get('piece','')
                            if not event.get('eog'):warmids.append(event['token'])
                        if event.get('done'):break
                    warmed=dict(ids=warmids,text=warmtext,stats=event)
                    assert len(warmids)==warmup and event['reason']=='length','Natural EOG/stop interrupted warm window; no completion extended'
                    progress('doing='+options['label']+' '+name+' warm window complete; latest='+str(warmup)+' discarded tokens; next=128 evaluated tokens')
                send(dict(sampler,op='generate',max_tokens=128,bias=bias));ids=[];text=''
                while True:
                    event=read()
                    if 'token' in event:
                        text+=event.get('piece','')
                        if not event.get('eog'):ids.append(event['token'])
                    if event.get('done'):break
                result['modes'][name]=dict(bias=bias,stats=event,ids=ids,text=text)
                if warmed is not None:result['modes'][name]['warmup']=warmed
                Path(str(prefix)+'.'+name+'.txt').write_text(text,encoding='utf8')
                assert event['decode_tokens']==len(ids)
                progress('doing='+options['label']+' '+name+'100K decode measured; latest='+str(len(ids))+'tokens '+str(event['tok_per_s'])+'tok/s reason='+event['reason']+'; next=remaining mode/recall and exactcleanup')
            result['recall']='PASS' if 'AMBER-OTTER-731' in result['modes']['quality']['text']+result['modes']['quality'].get('warmup',{}).get('text','') else 'FAIL'
            result['benchmark_complete']=all(len(m['ids'])==128 and m['stats']['reason']=='length' for m in result['modes'].values())
            send(dict(op='quit'));assert read()['event']=='bye';assert engine.wait(timeout=60)==0
        finally:
            vocab=Path(options['vocab_gguf'])
            result['vocab_sha256']=sha(vocab) if vocab.is_file() else None
            Path(str(prefix)+'.result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            original_error=sys.exc_info()[0] is not None
            if engine.poll() is None:
                try:send(dict(op='quit'));engine.wait(timeout=60)
                except (OSError,subprocess.TimeoutExpired) as error:
                    print('Owned engine cleanup: '+repr(error),file=sys.stderr,flush=True)
                    if not original_error:raise
    result['ple_sha256']=sha(Path(str(prefix)+'.ple.raw'))
    Path(str(prefix)+'.result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    assert result['recall']=='PASS','Recall failed; evidence retained'
    assert result['benchmark_complete'],'Natural EOG/stop prevented matched128+128 timing; no completion extended'


def main():
    global STACK
    if len(sys.argv)>1 and sys.argv[1]=='--child':return child(json.loads(sys.argv[2]))
    if len(sys.argv)>1 and sys.argv[1]=='--certify-child':return certify(json.loads(sys.argv[2]))
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('label')
    ap.add_argument('--model',required=True)
    ap.add_argument('--repo-root',required=True,help='Private/public reviewed source tree containing slstack and scripts')
    ap.add_argument('--runtime-root',required=True)
    ap.add_argument('--config',required=True,help='Machine-local guarded runtime configuration')
    ap.add_argument('--fixture',required=True,help='Exact 400000-byte LE32 fixture from build_qfn_100k_fixture.py')
    ap.add_argument('--output-prefix',required=True)
    ap.add_argument('--model-certificate',required=True)
    ap.add_argument('--model-sha256',required=True)
    ap.add_argument('--certify-model',action='store_true',help='Perform complete pack/section readback under CPU guardian before GPU admission')
    ap.add_argument('--engine-arg',action='append',default=[],help='Additional native engine argv token; use = for tokens beginning --')
    ap.add_argument('--sampler-json',help='Additional generation options forwarded unchanged; reserved protocol fields fail loudly')
    ap.add_argument('--commit-budget-gib',type=float,default=40,help='Explicit job commit cap including WDDM; physical RAM also protected by RSS/floor guardian')
    ap.add_argument('--min-available-gib',type=float,default=6)
    ap.add_argument('--task',default='QFN100K')
    ap.add_argument('--engine',required=True,help='Exact installed executable for integration qualification')
    ap.add_argument('--vocab-gguf',required=True,help='Model-specific complete tokenizer cache; incompatible existing caches fail loudly')
    ap.add_argument('--nvfp4-mode',choices=('native-q8','auto','w4a4','mmq','calibrated-w4a4','dsv-native'))
    ap.add_argument('--nvfp4-decode-mode',choices=('native-q8','calibrated-w4a4','dsv-native'))
    ap.add_argument('--qsa-indexer',choices=('auto', 'scalar', 'basalt-tc-refine-prefill', 'basalt-tc-refine', 'basalt-tc', 'basalt-tc-prefill', 'dsv-fp4'))
    ap.add_argument('--qsa-attention',choices=('native','basalt'))
    ap.add_argument('--cache-slots',type=int)
    ap.add_argument('--tile',type=int,default=2048)
    ap.add_argument('--speed-bias',type=float,required=True)
    ap.add_argument('--host-rss',type=int,default=3)
    ap.add_argument('--warmup-tokens',type=int,default=100,help='Discard this many evaluated tokens before each128-token decode window; preserve natural EOG/stop')
    ap.add_argument('--env',action='append',default=[])
    ap.add_argument('--parent-lease',help='Existing ancestor-owned GPU reservation; standard lease validation applies')
    args=ap.parse_args()
    if args.warmup_tokens<0:ap.error('warmup-tokens must be nonnegative; published exact protocol uses 100')
    if args.host_rss<=0 or args.commit_budget_gib<=0 or args.min_available_gib<=0:ap.error('Explicit positive memory budgets required')
    if not re.fullmatch('[a-fA-F0-9]{64}',args.model_sha256):ap.error('Model SHA256 must have 64 hex digits')
    STACK=Path(args.repo_root).resolve();args.repo_root=str(STACK)
    runtime=Path(args.runtime_root).resolve()
    args.output_prefix=str(Path(args.output_prefix).resolve())
    for name in ('model','engine','vocab_gguf','fixture','model_certificate','config'):
        setattr(args,name,str(Path(getattr(args,name)).resolve()))
    if args.sampler_json:args.sampler_json=str(Path(args.sampler_json).resolve())
    assert re.fullmatch('[a-zA-Z0-9_-]+',args.label)
    if args.cache_slots is not None and args.cache_slots<=0:ap.error('cache-slots must be positive')
    prefix=Path(args.output_prefix);assert not Path(str(prefix)+'.launch.json').exists(),'Use a fresh evidence prefix'
    sys.path.insert(0,str(STACK))
    import psutil
    from slstack.process import OwnedProcess
    def compilers():return {p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in COMPILERS}
    cfg=json.loads((STACK/'config.example.json').read_text());cfg.update(json.loads(Path(args.config).read_text()))
    cfg.update(runtime_root=str(runtime))
    assert cfg.get('host','127.0.0.1')=='127.0.0.1','Only loopback test listeners permitted'
    cfg.update(task=args.task,host_budget_gib=args.host_rss,job_commit_gib=args.commit_budget_gib,min_available_gib=args.min_available_gib,runtime_min_free_gib=6,
               paging_telemetry=True,gpu_wait_seconds=10800,lease_minutes=30,gpu_idle_max_mib=1024,gpu_min_free_mib=24000)
    if args.parent_lease:cfg['_gpu_parent_lease']=args.parent_lease
    chosen={'QFN_MTP':'0','DS4_DS_READ_PARTS':'8',**dict(x.split('=',1) for x in args.env)}
    assert chosen['QFN_MTP']=='0' and chosen['DS4_DS_READ_PARTS']=='8','Exact published benchmark requires MTP off/eight read parts'
    cfg['env_overrides']={**cfg.get('env_overrides',{}),**chosen,'TEMP':str(runtime/'tmp'),'TMP':str(runtime/'tmp'),
                          'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    from slstack.config import environment
    delivered=environment(cfg,scope='qfn-engine')
    for key,value in chosen.items():
        assert delivered.get(key)==value, f'Requested setting was not delivered: {key}={value!r}, actual={delivered.get(key)!r}'
    engine=Path(args.engine);dll=engine.parent/'qfn-basalt-nvfp4.dll'
    launch=dict(options=vars(args),env=chosen,engine_sha256=sha(engine),harness_sha256=sha(__file__),timing=True,
                host_rss_gib=args.host_rss,commit_including_WDDM_gib=args.commit_budget_gib,physical_floor_gib=args.min_available_gib)
    if args.certify_model:
        assert not Path(args.model_certificate).exists(),'Preserve the existing certificate; use a fresh certificate path'
        cpu=OwnedProcess(cfg,[sys.executable,'-B',__file__,'--certify-child',json.dumps(vars(args))],args.label+'-certify',cwd=STACK,gpu=False,unbounded=True,scope='qfn-engine')
        cert_guard=cpu.wait()
        assert cert_guard['cleanup_confirmed'] and cert_guard['exit_code']==0
        launch['certification_guard']=cert_guard
    launch['model_certificate']=bind_model(args.model_certificate,args.model,args.model_sha256)
    # Keep compiler admission immediately before GPU reservation even when a
    # complete readback was requested. Certification itself is never timed.
    quiet=time.monotonic()
    while time.monotonic()-quiet<60:
        if compilers():quiet=time.monotonic()
        time.sleep(5)
    if (args.nvfp4_mode is not None or args.nvfp4_decode_mode is not None or 'nvfp4' in args.model.lower() or args.qsa_indexer=='basalt-tc'
        or args.qsa_attention=='basalt'):launch['dll_sha256']=sha(dll)
    native_dll=engine.parent/'qfn-native-nvfp4.dll'
    if 'native-nvfp4' in args.model.lower():launch['native_dll_sha256']=sha(native_dll)
    progress('doing='+args.label+'100K timing admission; latest=60scompilerquiet RSS'+str(args.host_rss)+'GiB/WDDMcommit'+str(args.commit_budget_gib)+'GiB/floor'+str(args.min_available_gib)+'GiB; next=exactarchivedprompt recall/prefill/two128decode stages')
    job=OwnedProcess(cfg,[sys.executable,'-B',__file__,'--child',json.dumps(vars(args))],args.label,cwd=STACK,gpu=True,unbounded=True,scope='qfn-engine')
    launch.update(receipt=str(job.receipt),log=str(job.log),guardian_pid=job.proc.pid)
    Path(str(prefix)+'.launch.json').write_text(json.dumps(launch,indent=2))
    observed=set()
    while job.proc.poll() is None:
        observed.update(compilers());time.sleep(2)
    try:
        guard=job.wait()
    except Exception:
        if job.receipt.exists():
            launch.update(guard=json.loads(job.receipt.read_text()),compiler_pids_observed=sorted(observed))
            Path(str(prefix)+'.launch.json').write_text(json.dumps(launch,indent=2))
        raise
    launch.update(guard=guard,compiler_pids_observed=sorted(observed))
    Path(str(prefix)+'.launch.json').write_text(json.dumps(launch,indent=2))
    assert guard['cleanup_confirmed'] and guard['exit_code']==0
    assert sha(engine)==launch['engine_sha256'] and sha(__file__)==launch['harness_sha256']
    assert bind_model(args.model_certificate,args.model,args.model_sha256)==launch['model_certificate']
    if 'dll_sha256'in launch:assert sha(dll)==launch['dll_sha256']
    if 'native_dll_sha256'in launch:assert sha(native_dll)==launch['native_dll_sha256']
    assert not observed,'Compiler activity contaminated timing; evidence retained'
    report=json.loads(Path(str(prefix)+'.result.json').read_text())
    print(json.dumps(dict(prefill=report['prefill_tok_per_s'],recall=report['recall'],
                         modes={n:m['stats']['tok_per_s'] for n,m in report['modes'].items()},guard_cleanup=True)),flush=True)
    progress('doing='+args.label+'100K idle timing complete; latest=recallPASS/guard0cleanuptrue, artifacthashesunchanged; next=ownergatesandbaselinecomparison')


if __name__=='__main__':main()
