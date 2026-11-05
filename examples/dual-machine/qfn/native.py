"""Resident QFN native peers, with explicit settings and independent RAM guards.

The RTX listens on loopback and sends state directly from the native engine.
The Mac exposes the existing ds4-server API on loopback. SSH carries control;
an explicit private Mac listener optionally accepts outbound RTX state delivery.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import sys
import time
import threading
import datetime

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from slstack.paging_limits import shared_owned_paging_limit_bytes
from slstack.config import load
from slstack.envvars import rows
from slstack.journal import atomic_json,sha256
from slstack.model_compatibility import model_description
from slstack.process import OwnedProcess


def file_identity(path):
    s=path.stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino,device=s.st_dev)


def verified_model(path,receipt_path,expected=None):
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    current=file_identity(path)
    if receipt.get('verified') and receipt.get('output_identity')==current:
        digest=receipt['output_sha256']
    else:
        binding=receipt.get('reference_certificate_binding',{})
        identity=binding.get('model_identity',{})
        recorded=dict(size=identity.get('bytes'),mtime_ns=identity.get('mtime_ns'),
            ino=identity.get('ino'),device=identity.get('device'))
        if recorded!=current:raise ValueError('Model identity differs from its verified receipt')
        digest=binding['model_sha256']
    if expected and expected!=digest:raise ValueError('Model SHA differs from selected weights')
    return digest,receipt


def certify_no_language(output,error=None):
    """Only completed actual build guards, or no specs, can certify no model."""
    if not output.is_dir() or (output/'owner.json').exists():return False
    specs=list((output/'jobs').glob('*.spec.json'));records=[]
    for spec in specs:
        row=json.loads(spec.read_text());receipt=Path(row['receipt'])
        if Path(row['command'][0]).name not in ('make','cmake') or not receipt.is_file():return False
        proof=json.loads(receipt.read_text())
        if proof.get('release_allowed') is not True:
            certificate=Path(proof.get('shared_guard_certificate',''))
            expected=receipt.with_suffix('.shared.json')
            if (certificate!=expected or not certificate.is_file() or proof.get('exit_code')!=0 or
                proof.get('reason') is not None or proof.get('cleanup_confirmed') is not True or
                proof.get('paging_telemetry_closed') is not True or
                not 0 <= proof.get('owned_paging',{}).get('observed_paged_bytes',-1) <= shared_owned_paging_limit_bytes()):return False
            actual=json.loads(certificate.read_text())
            if actual.get('release_allowed') is not True:return False
            proof={**proof,'validated_shared_certificate':actual}
        records.append(proof)
    atomic_json(output/'cleanup.json',dict(release_allowed=True,cleanup_confirmed=True,
        engine_started=False,guardian_started=bool(specs),build_receipts=records,
        owned_paged_bytes=max([0]+[x.get('owned_paged_bytes',x.get('owned_paging',{}).get('observed_paged_bytes',0)) for x in records]),startup_error=error,
        evidence='No language guardian spec or owner exists; all actual build receipts checked'))
    return True


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('role',choices=['rtx','mac'])
    p.add_argument('--installation-profile-file',type=Path,required=True)
    p.add_argument('--source-build-manifest',type=Path)
    p.add_argument('--producer-source-sha256',required=True)
    p.add_argument('--producer-binary-sha256',required=True)
    p.add_argument('--producer-source-revision',required=True)
    p.add_argument('--installation-port',type=int,default=18796)
    p.add_argument('--cmake',type=Path)
    p.add_argument('--installation-state-root',type=Path)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-receipt',type=Path,required=True)
    p.add_argument('--model-sha256')
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair-token-file',type=Path,required=True)
    p.add_argument('--producer-model-sha256',required=True)
    p.add_argument('--compatibility-sha256',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--context',type=int,required=True)
    p.add_argument('--host-budget-gib',type=float,required=True)
    p.add_argument('--engine-host-budget-gib',type=float,required=True)
    p.add_argument('--commit-gib',type=float,required=True)
    p.add_argument('--floor-gib',type=float,required=True)
    p.add_argument('--stop-file',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--model-settings',type=Path)
    p.add_argument('--tile',type=int,default=2048)
    p.add_argument('--reference-prefill-max-tokens',type=int,default=0)
    p.add_argument('--protect-host-working-set',type=int,choices=[0,1],default=1)
    p.add_argument('--lock-host-resident',type=int,choices=[0,1],default=1)
    p.add_argument('--trim-host-heap-cache',type=int,choices=[0,1],default=0)
    p.add_argument('--populate-host-heaps',type=int,choices=[0,1],default=0)
    p.add_argument('--handover-prefill',type=int,choices=[0,1],default=1)
    p.add_argument('--ple-submit-low-water',type=int,default=0,help='RTX PLE refill batching; 0 retains the shared policy')
    p.add_argument('--ple-thread-priority',choices=['normal','above-normal'],default='normal')
    p.add_argument('--ple-slot-threads',type=int,default=1)
    p.add_argument('--ple-io-workers',type=int,default=1)
    p.add_argument('--qsa-attention',choices=['native','basalt'],default='native',help='RTX sparse-attention implementation')
    p.add_argument('--ple-poll-us',type=int,default=0)
    p.add_argument('--process-qos',choices=['inherit','high'],default='inherit',help='Owned Windows engine only; no system setting changes')
    p.add_argument('--mtp',type=int,choices=[0,1],default=1)
    p.add_argument('--mtp-depth',choices=['auto','2','3'],default='auto')
    p.add_argument('--exact-sampling',type=int,choices=[0,1],default=1)
    p.add_argument('--warm-weights',type=int,choices=[0,1],default=1)
    p.add_argument('--warmup-prompt',default='A short warm-up: reply with the word ready.')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--prefill-chunk',type=int,default=256)
    p.add_argument('--peer-host',default='127.0.0.1')
    p.add_argument('--state-listen-host',help='Mac-only private interface for outbound RTX state delivery')
    p.add_argument('--peer-port',type=int,default=8095)
    p.add_argument('--remote-prefill-min-tokens',type=int,default=11776)
    p.add_argument('--local-append-max',type=int,default=11776)
    p.add_argument('--async-catchup',type=int,choices=[0,1],default=1)
    p.add_argument('--catchup-defer',type=int,choices=[0,1],default=0)
    p.add_argument('--catchup-max-tokens',type=int,default=128)
    p.add_argument('--catchup-buffer-bytes',type=int,default=2<<30)
    p.add_argument('--stall-seconds',type=int,default=180)
    p.add_argument('--payload-budget-bytes',type=int,default=16<<30)
    p.add_argument('--verify-state',action='store_true',help='Inline test verification; never enable in speed presets')
    p.add_argument('--test-trace-rows',type=int,default=0,help='Test only: retain computed MTP logits in bounded RAM')
    p.add_argument('--test-wide-index',action='store_true',help='Test only: F32 index wire control')
    p.add_argument('--test-local-payload-roundtrip',action='store_true')
    p.add_argument('--test-owner-readback',action='store_true',help='Test only: compare restored native mutable owners against their saved bytes')
    p.add_argument('--vision',type=Path)
    p.add_argument('--vision-receipt',type=Path)
    p.add_argument('--vision-control-port',type=int,default=0)
    p.add_argument('--peer-vision-port',type=int,default=0)
    p.add_argument('--vision-backend',default='CUDA0')
    p.add_argument('--vision-min-tokens',type=int,default=64)
    p.add_argument('--vision-max-tokens',type=int,default=1024)
    p.add_argument('--vision-threads',type=int,default=8)
    p.add_argument('--vision-flash-attn',type=int,choices=[0,1],default=0)
    p.add_argument('--vision-host-budget-gib',type=float,default=3)
    p.add_argument('--hold-file',type=Path)
    p.add_argument('--gpu-uuid')
    p.add_argument('--lease-minutes',type=float,default=19)
    p.add_argument('--gpu-idle-max-mib',type=int,default=1024)
    p.add_argument('--peer-restart-limit',type=int,default=2,help='Bounded certified ordinary peer exits; guard/protection failures never restart')
    p.add_argument('--test-peer-restart-delay-ms',type=int,default=0,help='Test only: expose a controlled peer outage before restart')
    for option in rows():
        if {'qfn-engine','mac-ds4','studio'} & set(option['scopes']):
            p.add_argument('--option-'+option['var'].lower().replace('_','-'),dest='option_'+option['var'])
    a=p.parse_args()
    from slstack.qfn_installation import validate_launch_contract
    preset_sha=validate_launch_contract(a.installation_profile_file,a.role,sys.argv[1:])
    engine_sha=sha256(a.engine)
    if a.role=='rtx':
        if a.cmake or any(x.split('=',1)[0] in ('--installation-port','--installation-state-root') for x in sys.argv[1:]):
            raise ValueError('Installation receiver/build flags belong to the Mac peer')
        from slstack.qfn_build import verified_build
        if not a.source_build_manifest:raise ValueError('Actual producer build manifest is required')
        build=verified_build(a.source_build_manifest,a.engine)
        if (build['sha256'],build['source_sha256'],build['source_revision'])!=(a.producer_binary_sha256,a.producer_source_sha256,a.producer_source_revision):
            raise ValueError('Producer executable/source/revision differs from installation contract')
    elif a.source_build_manifest:raise ValueError('The source build manifest is verified on the RTX peer')
    elif not a.installation_state_root or not 0<a.installation_port<=65535 or a.installation_port==a.port:
        raise ValueError('Mac installation requires a stable state root and separate loopback control port')
    if not 0<=a.peer_restart_limit<=8 or not 0<=a.test_peer_restart_delay_ms<=10000:
        raise ValueError('Invalid explicit bounded peer restart settings')
    catchup_flags={'--async-catchup','--catchup-defer','--catchup-max-tokens','--catchup-buffer-bytes'}
    if a.role=='rtx' and any(arg.split('=',1)[0] in catchup_flags for arg in sys.argv[1:]):
        raise ValueError('Catch-up queue flags belong to the Mac peer')
    if a.role=='rtx' and (a.test_owner_readback or a.test_local_payload_roundtrip):raise ValueError('Paired rewind readback is configured on the Mac peer')
    if not 17<=a.catchup_max_tokens<=49152 or not 128<<20<=a.catchup_buffer_bytes<=a.payload_budget_bytes:
        raise ValueError('Invalid explicit QFN catch-up token/buffer bounds')
    rtx_ple_flags={'--ple-submit-low-water','--ple-thread-priority','--ple-slot-threads','--ple-io-workers','--ple-poll-us'}
    if a.role=='mac' and any(arg.split('=',1)[0] in rtx_ple_flags for arg in sys.argv[1:]):
        raise ValueError('PLE tuning flags belong to the RTX peer, including explicitly supplied defaults')
    if a.role=='mac' and any(arg.split('=',1)[0]=='--qsa-attention' for arg in sys.argv[1:]):
        raise ValueError('QSA attention selection belongs to the RTX peer')
    if a.role=='mac' and any(arg.split('=',1)[0]=='--populate-host-heaps' for arg in sys.argv[1:]):
        raise ValueError('CPU heap population belongs to the RTX peer')
    if a.role=='mac' and any(arg.split('=',1)[0]=='--trim-host-heap-cache' for arg in sys.argv[1:]):
        raise ValueError('Windows heap cache optimization belongs to the RTX peer')
    if a.role=='mac' and any(arg.split('=',1)[0]=='--process-qos' for arg in sys.argv[1:]):
        raise ValueError('Windows process QoS belongs to the RTX peer')
    if a.role=='mac' and (a.reference_prefill_max_tokens or any(x.split('=',1)[0]=='--reference-prefill-max-tokens' for x in sys.argv[1:])):
        raise ValueError('Small-prefix reference arithmetic belongs to the RTX peer')
    if a.role=='mac' and a.ple_submit_low_water:raise ValueError('PLE submission tuning belongs to the RTX peer')
    if a.role=='mac' and a.ple_thread_priority!='normal':raise ValueError('PLE thread priority belongs to the RTX peer')
    if not 1<=a.ple_slot_threads<=32:raise ValueError('PLE slot threads must be 1..32')
    if a.role=='mac' and a.ple_slot_threads!=1:raise ValueError('PLE slot mapping belongs to the RTX peer')
    if not 1<=a.ple_io_workers<=8:raise ValueError('PLE IO workers must be 1..8')
    if not 0<=a.ple_poll_us<=5000:raise ValueError('PLE completion polling must be 0..5000 microseconds')
    if a.role=='mac' and a.ple_io_workers!=1:raise ValueError('PLE IO workers belong to the RTX peer')
    if a.state_listen_host and a.role!='mac':raise ValueError('Only the Mac may listen for direct state delivery')
    if not 0<a.port<=65535 or a.context<2:raise ValueError('Invalid port/context')
    if not (0<=a.vision_control_port<=65535 and 0<=a.peer_vision_port<=65535 and
            1<=a.vision_min_tokens<=a.vision_max_tokens<=16383 and 1<=a.vision_threads<=256 and
            a.vision_host_budget_gib>0):raise ValueError('Invalid explicit image port/token/thread/host budget')
    if min(a.host_budget_gib,a.engine_host_budget_gib,a.commit_gib,a.floor_gib)<=0:
        raise ValueError('Explicit positive physical memory budgets/floor required')
    a.output.mkdir(parents=True,exist_ok=True)
    stop=a.stop_file or a.output/'stop'
    if stop.exists():raise ValueError('Stop file already exists; select a fresh run output')
    model_sha,receipt=verified_model(a.model,a.model_receipt,a.model_sha256)
    description=model_description(a.model,model_sha,0)
    if description['compatibility_sha256']!=a.compatibility_sha256:
        raise ValueError('Selected model tokenizer/shape/config is incompatible with the peer')
    scope='qfn-engine' if a.role=='rtx' else 'mac-ds4'
    allowed={r['var'] for r in rows() if scope in r['scopes'] or 'studio' in r['scopes']}
    options={r['var']:getattr(a,'option_'+r['var'],None) for r in rows()
        if getattr(a,'option_'+r['var'],None) is not None}
    if set(options)-allowed:raise ValueError('Options belong to another peer: '+str(set(options)-allowed))
    if a.role=='rtx' and options.get('QFN_MTP',str(a.mtp))!=str(a.mtp):
        raise ValueError('--option-qfn-mtp conflicts with --mtp')
    depth=options.get('DS4_QWEN4_MTP_DEPTH')
    if a.role=='mac' and depth is not None and str(depth) not in ({'0','auto'} if a.mtp_depth=='auto' else {a.mtp_depth}):
        raise ValueError('Environment-table depth flag conflicts with --mtp-depth')
    overrides=dict(model=str(a.model),runtime_root=str(a.output),context=a.context,
        host_budget_gib=a.host_budget_gib,min_available_gib=a.floor_gib,
        runtime_min_free_gib=a.floor_gib,job_commit_gib=a.commit_gib,
        owned_paging_limit_bytes=shared_owned_paging_limit_bytes(),yoco_keep=0,env_overrides=options,
        task='QfnPair',hold_file=str(a.hold_file) if a.hold_file else None,
        gpu_uuid=a.gpu_uuid or 'metal',lease_minutes=a.lease_minutes,
        gpu_idle_max_mib=a.gpu_idle_max_mib)
    if a.role=='rtx':
        if not 0<=a.ple_submit_low_water<=256:raise ValueError('PLE submit low watermark must be 0..256')
        if not a.gpu_uuid or model_sha!=a.producer_model_sha256:
            raise ValueError('RTX GPU and verified producer identity required')
        overrides.update(qfn_engine=str(a.engine),qfn_engine_port=a.port,qfn_tile=a.tile,
            qfn_host_budget_gib=a.engine_host_budget_gib,model_settings=str(a.model_settings) if a.model_settings else None)
        cfg=load(overrides=overrides)
        from slstack.model_settings import apply
        cfg=apply(cfg,'qfn')
        contract=json.loads(a.installation_profile_file.read_text())
        if sha256(a.model_settings)!=contract['model_settings_sha256']:raise ValueError('Model settings changed since preset resolution')
        cfg.setdefault('env_overrides',{})['QFN_MTP']=str(a.mtp)
        from slstack.qfn_single import QfnEngine
        cfg['qfn_protect_host_working_set']=a.protect_host_working_set
        cfg['qfn_lock_host_resident']=a.lock_host_resident
        cfg['qfn_qsa_attention']=a.qsa_attention
        from slstack.performance_contract import seal
        cfg.setdefault('qfn_engram_scan_rings', int(cfg.get('env_overrides', {}).get('LLAMADECODE_ENGRAM_SCAN_RINGS', '1')))
        cfg = seal(cfg, 'qfn')
        command=QfnEngine(cfg).command()+['--populate-host-heaps',str(a.populate_host_heaps),'--trim-host-heap-cache',str(a.trim_host_heap_cache),'--reference-prefill-max-tokens',str(a.reference_prefill_max_tokens),'--ple-submit-low-water',str(a.ple_submit_low_water),'--ple-thread-priority',a.ple_thread_priority,'--ple-slot-threads',str(a.ple_slot_threads),'--ple-io-workers',str(a.ple_io_workers),'--ple-poll-us',str(a.ple_poll_us),'--process-qos',a.process_qos,'--pair-token-file',str(a.pair_token_file),
            '--pair-model-sha256',a.producer_model_sha256,'--pair-compatibility-sha256',a.compatibility_sha256,
            '--warmup-prompt',a.warmup_prompt]
        if a.handover_prefill:command+=['--handover-prefill']
        if a.vision:
            if not a.vision_receipt or not a.vision_control_port:raise ValueError('RTX images require projector receipt and control port')
            from scripts.slpack import ModelFile
            _,vr=verified_model(a.vision,a.vision_receipt)
            if vr['source_sha256']!=ModelFile(a.model).sections['vision/mmproj'][2].hex():
                raise ValueError('RTX projector is not the selected language pack companion')
            cfg.update(model_family='qfn',vision_encoder=str(a.vision),vision_backend=a.vision_backend,
                vision_min_tokens=a.vision_min_tokens,vision_max_tokens=a.vision_max_tokens,
                vision_threads=a.vision_threads,vision_flash_attention=a.vision_flash_attn,vision_host_budget_gib=a.vision_host_budget_gib)
        elif a.vision_receipt or a.vision_control_port:raise ValueError('Incomplete RTX vision deployment')
    else:
        cfg=load(overrides=overrides)
        command=[str(a.engine),'--model',str(a.model),'--backend','metal',
            '--host','127.0.0.1','--port',str(a.port),'--ctx',str(a.context),
            '--threads',str(a.threads),'--prefill-chunk',str(a.prefill_chunk),
            '--qfn-mtp-depth',a.mtp_depth,'--mtp-draft','2' if a.mtp else '1',
            '--host-budget-gib',str(a.engine_host_budget_gib),'--host-floor-gib',str(a.floor_gib),
            '--qfn-prefill-host',a.peer_host,'--qfn-prefill-port',str(a.peer_port),
            '--qfn-pair-token-file',str(a.pair_token_file),
            '--qfn-producer-model-sha256',a.producer_model_sha256,
            '--qfn-compatibility-sha256',a.compatibility_sha256,
            '--qfn-prefill-min-tokens',str(a.remote_prefill_min_tokens),
            '--qfn-local-append-max',str(a.local_append_max),'--qfn-peer-stall-seconds',str(a.stall_seconds),
            '--qfn-state-budget-bytes',str(a.payload_budget_bytes),
            '--qfn-async-catchup',str(a.async_catchup),'--qfn-catchup-defer',str(a.catchup_defer),
            '--qfn-catchup-max-tokens',str(a.catchup_max_tokens),'--qfn-catchup-buffer-bytes',str(a.catchup_buffer_bytes)]
        if a.mtp:command+=['--mtp']
        if a.state_listen_host:command+=['--qfn-state-listen-host',a.state_listen_host]
        if a.exact_sampling:command+=['--mtp-exact-sampling']
        if a.warm_weights:command+=['--warm-weights']
        if a.verify_state:command+=['--verify-state']
        if a.test_trace_rows:command+=['--qfn-test-trace-rows',str(a.test_trace_rows)]
        if a.test_wide_index:command+=['--qfn-test-wide-index']
        if a.test_owner_readback:command+=['--qfn-test-owner-readback']
        if a.test_local_payload_roundtrip:command+=['--qfn-test-local-payload-roundtrip']
        if a.peer_vision_port:command+=['--qfn-vision-port',str(a.peer_vision_port)]
        if a.vision:
            if not a.vision_receipt:raise ValueError('Vision requires its verified receipt')
            _,vr=verified_model(a.vision,a.vision_receipt)
            if vr.get('source_sha256')!=receipt.get('companions',{}).get('vision/mmproj'):
                raise ValueError('Vision is not from the selected source pack')
            command+=['--vision',str(a.vision)]
        elif a.vision_receipt:raise ValueError('Vision receipt provided without a vision model')
    atomic_json(a.output/'settings.json',dict(flags={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        model=description,model_identity=file_identity(a.model),engine_sha256=sha256(a.engine),
        command=command,environment_overrides=cfg.get('env_overrides',{}),
        production_payload_path='native socket -> funopen -> existing ds4 session',
        no_total_lifetime_deadline=True,capability_reductions=[]))
    if a.role=='mac':
        cfg.update(model_family='qfn',producer_sha256=a.producer_binary_sha256,producer_source_revision=a.producer_source_revision,build_cmake=str(a.cmake) if a.cmake else None,mac_bind='127.0.0.1',mac_port=a.installation_port,
            mac_engine_port=a.port,pair_token=a.pair_token_file.read_text().strip(),
            pairing_state_root=str(a.installation_state_root))
        from slstack.mac import setup
        build=setup(cfg,'ds4')
        if not build['ready'] or Path(build['binary']).resolve()!=a.engine.resolve():raise ValueError('Mac installation binary is not the selected built receiver')
    if a.dry_run:
        import subprocess
        subprocess.run([*command,'--validate-options'],cwd=a.engine.resolve().parent,check=True)
        if a.role=='rtx' and a.vision:
            subprocess.run([str(a.engine),'--vision-encode','--vision-model',str(a.vision),
                '--vision-image',str(a.output/'preflight.image'),'--vision-output',str(a.output/'preflight.vision'),
                '--vision-backend',a.vision_backend,'--vision-min-tokens',str(a.vision_min_tokens),
                '--vision-max-tokens',str(a.vision_max_tokens),'--vision-threads',str(a.vision_threads),
                '--vision-flash-attn',str(a.vision_flash_attn),'--host-budget-gib',str(a.vision_host_budget_gib),
                '--protect-host-working-set','1','--validate-options'],cwd=a.engine.resolve().parent,check=True)
        print(json.dumps(dict(event='native_peer_preflight_passed',role=a.role,command=command)),flush=True)
        return
    stopping=[False]
    def request_stop(*_):stopping[0]=True
    signal.signal(signal.SIGINT,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    job=None;engine_ready=False;vision_control=None;installation_server=None;installation_thread=None;receiver=None;retired=[];restarts=0;pending_restart=None
    lifecycle_lock=threading.RLock()
    recovery_events=a.output/'recovery-events.jsonl'
    def recovery_event(event,**values):
        row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,role=a.role,**values)
        with recovery_events.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    with ExitStack() as stack:
        if a.role=='rtx':
            from slstack.lease import gpu_lease
            lease=stack.enter_context(gpu_lease(cfg))
            cfg['hold_watch']=dict(path=cfg['hold_file'],identity=lease,task='QfnPair')
        try:
            def start_job(selected_command=None):
                nonlocal job,command,engine_ready
                if sha256(a.engine)!=engine_sha:raise ValueError('Native engine changed after installation; restart refused')
                if a.role=='rtx':verified_build(a.source_build_manifest,a.engine)
                if selected_command is not None:command=list(selected_command)
                recovery_event('engine_start',engine_sha256=engine_sha,command=command,preset_sha256=preset_sha)
                engine_ready=False
                job=OwnedProcess(cfg,command,'qfn-native-'+a.role,cwd=a.engine.resolve().parent,
                    scope=scope,unbounded=True,telemetry=True)
                atomic_json(a.output/'owner.json',dict(guardian_pid=job.proc.pid,log=str(job.log),receipt=str(job.receipt)))
            def retire_job():
                nonlocal job,restarts,pending_restart
                if not job:return
                job.stop();record=json.loads(job.receipt.read_text())
                if not record.get('release_allowed'):raise RuntimeError('Native language retirement unconfirmed')
                if record.get('reason')!='stop requested':
                    if not engine_ready:raise RuntimeError('Native engine failed before readiness; startup/qualification failures never auto-restart')
                    from slstack.qfn_recovery import restart_allowed
                    diagnostic=job.log.read_text(errors='replace')[-16000:] if job.log.exists() else ''
                    if not restart_allowed(record,restarts,a.peer_restart_limit,diagnostic):
                        raise RuntimeError('Native peer exit is not eligible for recovery; guard/protection/failure evidence retained: '+str(record.get('reason')))
                    restarts+=1;pending_restart=dict(previous_child_pid=record['child_pid'],exit_code=record['exit_code'],attempt=restarts,
                        prior_receipt=str(job.receipt),prior_cleanup_confirmed=True,owned_paged_bytes=record['owned_paged_bytes'])
                    recovery_event('peer_restart_admitted',**pending_restart)
                elif not 0 <= record.get('owned_paged_bytes',-1) <= shared_owned_paging_limit_bytes() or record.get('paging_telemetry_closed') is not True:
                    raise RuntimeError('Native retirement paging/telemetry evidence failed; reload refused')
                retired.append(record);atomic_json(a.output/'retired-jobs.json',retired);job=None
            def reload_job():
                nonlocal pending_restart,engine_ready
                if job is None:start_job()
                def check_ready():
                    if stopping[0] or stop.exists():raise InterruptedError('Native language reload cancelled')
                    if job.proc.poll()is not None:job.wait();raise RuntimeError('Native language reload failed')
                if a.role=='rtx':
                    from readiness import wait_ready
                    wait_ready(a.port,a.pair_token_file.read_text().strip(),a.producer_model_sha256,
                        a.compatibility_sha256,a.stall_seconds,check_ready)
                else:
                    from slstack.engines import http_json
                    from slstack.progress import Window,activity
                    window=Window(a.stall_seconds)
                    while True:
                        check_ready()
                        try:http_json(f'http://127.0.0.1:{a.port}/v1/models',timeout=2);break
                        except OSError:
                            window.observe(activity(job.proc.pid,job.log))
                            if window.expired():raise TimeoutError('Mac peer reload stopped making progress')
                            time.sleep(.25)
                engine_ready=True
                if pending_restart:
                    recovery_event('peer_restarted',**pending_restart,new_guardian_pid=job.proc.pid)
                    pending_restart=None
            if a.role=='mac':
                from slstack.qfn_installation import QfnReceiver
                from slstack.mac import receiver_server
                base_command=list(command)
                def select_command(native,artifact,artifact_sha,peer,raw_logits):
                    return base_command+['--qfn-prefill-route','paired' if native else 'local',
                        '--qfn-producer-binary-sha256',peer['producer_sha256'],
                        '--qfn-receiver-model-sha256',model_sha,'--qfn-renderer-sha256',build['renderer_sha256'],
                        '--qfn-qualification-file',str(artifact),'--qfn-qualification-sha256',artifact_sha,
                        '--qfn-qualification-logits',str(raw_logits)]
                def start_selected(selected):
                    start_job(selected);reload_job()
                def describe():
                    actual,_=verified_model(a.model,a.model_receipt,a.model_sha256)
                    return model_description(a.model,actual,0)
                receiver=QfnReceiver(cfg,build,describe_model=describe,start_engine=start_selected,
                    retire_engine=retire_job,current_engine=lambda:job,command_for_mode=select_command,
                    lifecycle_lock=lifecycle_lock,producer_source_sha256=a.producer_source_sha256,preset_sha256=preset_sha)
                installation_server=receiver_server(receiver,cfg,'ds4',build)
                installation_thread=threading.Thread(target=installation_server.serve_forever,daemon=True)
                installation_thread.start()
            else:reload_job()
            if a.role=='rtx' and a.vision:
                from native_vision import NativeVision
                vision_control=NativeVision(cfg,a.vision_control_port,a.pair_token_file.read_text().strip(),
                    a.producer_model_sha256,a.compatibility_sha256,retire_job,reload_job,lifecycle_lock,engine_sha)
            print(json.dumps(dict(event='guarded_native_peer_started',role=a.role,log=str(job.log) if job else None,installation_port=a.installation_port if receiver else None)),flush=True)
            while not stopping[0] and not stop.exists():
                if vision_control and vision_control.busy:time.sleep(.1);continue
                with lifecycle_lock:
                    if job is None:
                        if not receiver:raise RuntimeError('Native language process is absent outside image lifecycle')
                    elif job.proc.poll() is not None:
                        retire_job()
                        until=time.monotonic()+a.test_peer_restart_delay_ms/1000
                        while time.monotonic()<until:
                            if stopping[0] or stop.exists():raise InterruptedError('Peer recovery cancelled')
                            time.sleep(min(.1,max(0,until-time.monotonic())))
                        reload_job()
                time.sleep(.25)
        finally:
            if installation_server:
                installation_server.shutdown();installation_server.server_close();installation_thread.join(timeout=5)
            if vision_control:vision_control.close()
            if job:
                job.stop()
                if not job.receipt.exists():raise RuntimeError('Native cleanup receipt missing')
                result=json.loads(job.receipt.read_text())
                if retired:
                    result['retired_jobs']=retired
                    result['release_allowed']=result.get('release_allowed') and all(r.get('release_allowed') for r in retired)
                    result['owned_paged_bytes']=max([result.get('owned_paged_bytes',0)]+[r.get('owned_paged_bytes',0) for r in retired])
                atomic_json(a.output/'cleanup.json',result)
                if not result.get('cleanup',{}).get('confirmed',result.get('cleanup_confirmed',False)):
                    raise RuntimeError('Native peer cleanup is unconfirmed')
            elif retired:
                result=dict(retired[-1],retired_jobs=retired)
                result['release_allowed']=all(r.get('release_allowed') for r in retired)
                atomic_json(a.output/'cleanup.json',result)
            elif not certify_no_language(a.output):
                raise RuntimeError('No language owner, but actual job evidence cannot certify cleanup')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        # An early validation/build refusal can precede the language lifecycle.
        # Certify only real completed build guardians, or that no job was created.
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if not (output/'cleanup.json').exists():certify_no_language(output,str(error))
        raise
