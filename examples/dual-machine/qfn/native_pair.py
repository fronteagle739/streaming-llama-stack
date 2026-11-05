"""Run the resident native QFN pair; --test-recall PHRASE adds test work.

Use benchmark.py to borrow and restore a Studio-owned Mac model. The production
path is the normal ds4-server OpenAI API and has no expected answer or extra
state transfer. All native children use slstack's independent memory guardian.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import re
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from slstack.journal import atomic_json
from slstack.model_compatibility import model_description
from pair import mac_lease


def flags(settings):
    result=[]
    for key,value in settings.items():
        if key=='options':
            for name,option in value.items():result+=['--option-'+name.lower().replace('_','-'),str(option)]
        else:result+=['--'+key.replace('_','-'),str(value)]
    return result


def stream_request(url,body,events_path):
    started=time.perf_counter();first_content=None
    request=urllib.request.Request(url+'/v1/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    text=[];timings=None;mtp=None;events=[]
    with urllib.request.urlopen(request,timeout=300) as response,events_path.open('x',encoding='utf8') as log:
        for raw in response:
            if not raw.startswith(b'data:'):continue
            payload=raw[5:].strip()
            if payload==b'[DONE]':break
            row=json.loads(payload);events.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            if 'error' in row:raise RuntimeError(str(row['error']))
            if 'timings' in row:timings=row['timings']
            if 'qfn_mtp' in row:mtp=row['qfn_mtp']
            for choice in row.get('choices',[]):
                piece=choice.get('text',choice.get('delta',{}).get('content','')) or ''
                if piece and first_content is None:first_content=time.perf_counter()-started
                text.append(piece)
    if timings is not None:
        timings['client_time_to_first_token_s']=first_content
        timings['client_request_wall_s']=time.perf_counter()-started
    return ''.join(text),timings,mtp,events


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--transport',choices=['direct','ssh'],help='Override the preset state route; control and API use SSH in both modes')
    p.add_argument('--mac-private-host',help='Mac private-link IPv4 listener address; required for direct state transport')
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--producer-build-manifest',type=Path,required=True)
    p.add_argument('--installation-port',type=int,default=18796)
    p.add_argument('--producer-pack',type=Path,required=True)
    p.add_argument('--producer-receipt',type=Path,required=True)
    p.add_argument('--producer-sha256',required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--pair-token-file',type=Path,required=True)
    p.add_argument('--ssh-host',required=True)
    p.add_argument('--mac-root',required=True)
    p.add_argument('--mac-python',required=True)
    p.add_argument('--mac-server',required=True)
    p.add_argument('--mac-model',required=True)
    p.add_argument('--mac-receipt',required=True)
    p.add_argument('--mac-vision',required=True)
    p.add_argument('--mac-vision-receipt',required=True)
    p.add_argument('--rtx-vision',type=Path)
    p.add_argument('--rtx-vision-receipt',type=Path)
    p.add_argument('--rtx-vision-control-port',type=int,default=8096)
    p.add_argument('--vision-backend',default='CUDA0')
    p.add_argument('--vision-min-tokens',type=int,default=64)
    p.add_argument('--vision-max-tokens',type=int,default=1024)
    p.add_argument('--vision-threads',type=int,default=8)
    p.add_argument('--vision-flash-attn',type=int,choices=[0,1],default=0)
    p.add_argument('--vision-host-budget-gib',type=float,default=3)
    p.add_argument('--mac-host-budget-gib',type=float,required=True)
    p.add_argument('--mac-floor-gib',type=float,required=True)
    p.add_argument('--hold-file',type=Path,required=True)
    p.add_argument('--parent-mac-lease')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--progress-log',type=Path,required=True)
    p.add_argument('--test-recall',help='Test only: expected natural answer; enables the benchmark request')
    p.add_argument('--test-tokens-file',type=Path)
    p.add_argument('--test-decode-tokens',type=int,default=1024)
    p.add_argument('--test-temperature',type=float,default=1)
    p.add_argument('--test-seed',type=int,default=1)
    p.add_argument('--test-top-k',type=int,default=20)
    p.add_argument('--test-top-p',type=float,default=.95)
    p.add_argument('--test-min-p',type=float,default=0)
    p.add_argument('--test-mtp-trace-rows',type=int,default=0)
    p.add_argument('--test-wide-index',action='store_true')
    p.add_argument('--test-cold-readiness',action='store_true',help='Test only: reproduce the old API-only readiness measurement')
    p.add_argument('--test-features',action='store_true')
    p.add_argument('--test-local-payload-roundtrip',action='store_true')
    p.add_argument('--test-installation-only',action='store_true',help='Test only: real installation gate and retirement, without a speed or API claim')
    p.add_argument('--test-peer-failure',action='store_true',help='Test only: verified owned source termination, local fallback and guarded rebuild')
    p.add_argument('--test-image-file',type=Path)
    p.add_argument('--test-feature-prefix-tokens',type=int,default=100000)
    p.add_argument('--test-proof-program',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--verify-state',action='store_true',help='Test only: inline stream verification, never a speed row')
    a=p.parse_args()
    profile=json.loads(a.profile.read_text(encoding='utf8'))
    if set(profile)!={'version','model_settings','rtx','mac','transport'} or profile['version']!=1:
        raise ValueError('Unknown or incomplete native preset')
    if a.transport:profile['transport']['mode']=a.transport
    if a.mac_private_host:profile['transport']['state_listen_host']=a.mac_private_host
    if profile['transport']['mode'] not in ('ssh','direct'):
        raise ValueError('Transport mode must be ssh or direct')
    direct=profile['transport']['mode']=='direct'
    if direct:
        import ipaddress
        address=ipaddress.IPv4Address(profile['transport'].get('state_listen_host',''))
        if not any(address in ipaddress.IPv4Network(net) for net in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16')):
            raise ValueError('Direct state listener requires the Mac private-link IPv4 address')
    elif profile['transport'].get('state_listen_host'):
        raise ValueError('SSH mode cannot configure a direct state listener')
    if profile['transport']['verify_state']:
        raise ValueError('Production presets cannot enable state hashes; use a dedicated test invocation')
    if bool(a.test_recall)!=bool(a.test_tokens_file):raise ValueError('--test-recall and its test token fixture are required together')
    if bool(a.rtx_vision)!=bool(a.rtx_vision_receipt):raise ValueError('RTX projector and its receipt are required together')
    if a.test_image_file and (not a.test_features or not a.rtx_vision):raise ValueError('Image fixture requires the explicit RTX feature qualification')
    if not a.rtx_vision and not a.test_recall and not a.dry_run:
        raise ValueError('Production deployment requires the RTX image projector and receipt; Mac-only image fallback is not a complete deployment')
    if a.verify_state and not a.test_recall:raise ValueError('State hashes are a test-only flag')
    if a.test_wide_index and not a.test_recall:raise ValueError('Wide-index control is test only')
    if a.test_cold_readiness and not a.test_recall:raise ValueError('Cold-readiness control is test only')
    if a.test_mtp_trace_rows and (not a.test_recall or not 1<=a.test_mtp_trace_rows<=min(a.test_decode_tokens,4096)):
        raise ValueError('Computed-row tracing requires test mode and enough natural decode tokens')
    if (a.test_features or a.test_peer_failure or a.test_proof_program) and not a.test_recall:
        raise ValueError('Feature/proof instrumentation requires --test-recall PHRASE')
    a.output.mkdir(parents=True,exist_ok=False)
    def note(doing,latest,next_step):
        line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
        print(line,flush=True)
        with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
    identity=uuid.uuid4().hex;remote=a.mac_root.rstrip('/');remote_run=remote+'/native-'+identity
    if not remote.startswith('/Volumes/') or '/..' in remote:raise ValueError('Explicit physical Mac scratch root required')
    from slstack.qfn_build import verified_build
    from slstack.journal import sha256
    from slstack.qfn_installation import launch_arguments,installation_digest
    from slstack.model_compatibility import digest
    build=verified_build(a.producer_build_manifest,a.engine)
    producer=model_description(a.producer_pack,a.producer_sha256,0)
    compatibility=producer['compatibility_sha256']
    def ssh(args,**kwargs):return subprocess.run(['ssh','-o','BatchMode=yes',a.ssh_host,shlex.join(args)],check=True,**kwargs)
    def copy_from(path,dest):subprocess.run(['scp','-o','BatchMode=yes',a.ssh_host+':'+path,str(dest)],check=True)
    key=a.pair_token_file.read_bytes().strip()
    if not 32<=len(key)<=512 or any(c not in b'0123456789abcdef' for c in key):raise ValueError('A private hex pairing token is required')
    remote_token=remote+'/pair-native.token'
    setup='''import os,sys
from pathlib import Path
p=Path(sys.argv[1]);key=sys.stdin.buffer.read()
if p.exists():
 if p.read_bytes()!=key:raise ValueError('Existing task token differs')
else:
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(key)
'''
    ssh([a.mac_python,'-B','-c',setup,remote_token],input=key)
    rtx_settings=dict(profile['rtx']);rtx_port=rtx_settings.pop('port')
    mac_settings=dict(profile['mac']);mac_port=mac_settings.pop('port')
    transport=dict(profile['transport']);transport.pop('mode');transport.pop('verify_state')
    transport_label='direct private TCP state; SSH control/API' if direct else 'SSH forwards; native-session socket streaming'
    rtx_run=a.output/'rtx';stop=rtx_run/'stop'
    rtx=[sys.executable,'-B',str(Path(__file__).with_name('native.py')),'rtx',
        '--model',str(a.producer_pack),'--model-receipt',str(a.producer_receipt),'--model-sha256',a.producer_sha256,
        '--engine',str(a.engine),'--output',str(rtx_run),'--pair-token-file',str(a.pair_token_file),
        '--producer-model-sha256',a.producer_sha256,'--compatibility-sha256',compatibility,
        '--gpu-uuid',a.gpu_uuid,'--hold-file',str(a.hold_file),'--port',str(rtx_port),
        '--model-settings',str(ROOT/profile['model_settings']),*flags(rtx_settings)]
    mac=[a.mac_python,'-B',remote+'/examples/dual-machine/qfn/native.py','mac',
        '--model',a.mac_model,'--model-receipt',a.mac_receipt,'--engine',a.mac_server,
        '--output',remote_run,'--pair-token-file',remote_token,
        '--producer-model-sha256',a.producer_sha256,'--compatibility-sha256',compatibility,
        '--host-budget-gib',str(a.mac_host_budget_gib),'--floor-gib',str(a.mac_floor_gib),
        '--vision',a.mac_vision,'--vision-receipt',a.mac_vision_receipt,
        '--port',str(mac_port),'--peer-port',str(rtx_port),*flags(mac_settings),*flags(transport)]
    if a.verify_state:mac+=['--verify-state']
    if a.test_mtp_trace_rows or a.test_features or a.test_peer_failure:mac+=['--test-trace-rows',str(max(a.test_mtp_trace_rows,1))]
    if a.test_peer_failure:rtx+=['--test-peer-restart-delay-ms','4000']
    if a.test_wide_index:mac+=['--test-wide-index']
    if a.test_local_payload_roundtrip and not a.test_features:raise ValueError('Local payload roundtrip requires the state-observation feature flags')
    if a.test_features:mac+=['--test-owner-readback']
    if a.test_local_payload_roundtrip:mac+=['--test-local-payload-roundtrip']
    if a.rtx_vision:
        rtx+=['--vision',str(a.rtx_vision),'--vision-receipt',str(a.rtx_vision_receipt),
            '--vision-control-port',str(a.rtx_vision_control_port),'--vision-backend',a.vision_backend,
            '--vision-min-tokens',str(a.vision_min_tokens),'--vision-max-tokens',str(a.vision_max_tokens),
            '--vision-threads',str(a.vision_threads),'--vision-flash-attn',str(a.vision_flash_attn),'--vision-host-budget-gib',str(a.vision_host_budget_gib)]
        mac+=['--peer-vision-port',str(a.rtx_vision_control_port)]
    # One stable per-model installation record; run outputs remain independent.
    installation_key=digest(dict(mac_model=a.mac_model))[:16]
    local_install=a.engine.parent.parent/'install'/installation_key
    local_install.mkdir(parents=True,exist_ok=True)
    remote_install=remote+'/install/'+installation_key
    for command in (rtx,mac):
        command+=['--producer-source-sha256',build['source_sha256'],
            '--producer-binary-sha256',build['sha256'],'--producer-source-revision',build['source_revision']]
    rtx+=['--source-build-manifest',str(a.producer_build_manifest)]
    mac+=['--installation-port',str(a.installation_port),'--installation-state-root',remote_install+'/state']
    contract=dict(profile=profile,model_settings_sha256=sha256(ROOT/profile['model_settings']),
        role_argv=dict(rtx=launch_arguments(rtx[3:]),mac=launch_arguments(mac[3:])))
    contract_path=local_install/'resolved.json';atomic_json(contract_path,contract)
    remote_contract=remote_install+'/resolved.json'
    ssh([a.mac_python,'-B','-c','from pathlib import Path;import sys;Path(sys.argv[1]).parent.mkdir(parents=True,exist_ok=True);Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())',remote_contract],input=contract_path.read_bytes())
    rtx+=['--installation-profile-file',str(contract_path)]
    mac+=['--installation-profile-file',remote_contract]
    atomic_json(a.output/'settings.json',dict(flags={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        profile=profile,rtx_command=rtx,mac_command=mac,producer=producer,transport=transport_label,
        verify_state=a.verify_state,test_mode=bool(a.test_recall),capability_reductions=[]))
    if a.dry_run:
        subprocess.run([*rtx,'--dry-run'],cwd=ROOT,check=True)
        ssh([*mac,'--dry-run'])
        note('native peer preflight passed','model/vision/compatibility/settings checked; no engine or lease started','guarded live API proof')
        return
    stopping=[False]
    def stop_request(*_):stopping[0]=True
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    windows=mac_job=tunnel=None;rtx_log=mac_log=tunnel_log=None;ready_executor=ready_future=None
    cleanup={};success=False;api_ready=False;flush_receipt=None
    with mac_lease(a.hold_file,identity,a.mac_host_budget_gib,a.mac_floor_gib,1140,a.parent_mac_lease) as lease:
        try:
            rtx_log=(a.output/'rtx-controller.log').open('xb')
            windows=subprocess.Popen(rtx,cwd=ROOT,stdout=rtx_log,stderr=subprocess.STDOUT)
            tunnel_log=(a.output/'tunnel.log').open('xb')
            tunnel_command=['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-N',
                '-L',f'127.0.0.1:{mac_port}:127.0.0.1:{mac_port}',
                '-L',f'127.0.0.1:{a.installation_port}:127.0.0.1:{a.installation_port}',
                '-R',f'127.0.0.1:{rtx_port}:127.0.0.1:{rtx_port}']
            if a.rtx_vision:tunnel_command+=['-R',f'127.0.0.1:{a.rtx_vision_control_port}:127.0.0.1:{a.rtx_vision_control_port}']
            tunnel=subprocess.Popen([*tunnel_command,a.ssh_host],stdout=tunnel_log,stderr=subprocess.STDOUT)
            mac_log=(a.output/'mac-controller.log').open('xb')
            mac_job=subprocess.Popen(['ssh','-o','BatchMode=yes',a.ssh_host,shlex.join(mac)],stdout=mac_log,stderr=subprocess.STDOUT)
            def check_ready():
                if stopping[0]:raise InterruptedError('Pair startup cancelled')
                for name,child in [('RTX',windows),('Mac',mac_job),('tunnel',tunnel)]:
                    if child.poll() is not None:raise RuntimeError(name+' exited before authenticated readiness')
            if not a.test_cold_readiness:
                from readiness import wait_ready
                ready_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='qfn-ready')
                ready_future=ready_executor.submit(wait_ready,rtx_port,key.decode(),a.producer_sha256,
                    compatibility,profile['transport']['stall_seconds'],check_ready)
            url=f'http://127.0.0.1:{mac_port}';started=time.monotonic()
            from slstack.engines import http_json
            from slstack.pairing import installation_handshake
            control=f'http://127.0.0.1:{a.installation_port}'
            def control_request(route,body=None):
                result=http_json(control+route,body,timeout=None if route in ('/pair/qualification','/decode/setup') else 30,
                    headers={'Authorization':'Bearer '+key.decode()})
                if a.test_features and route=='/pair/qualification':
                    for route_name,suffix in (('anchor','owners'),('payload','payload')):
                        with urllib.request.urlopen(url+'/v1/qfn-test/'+route_name,timeout=60) as source, (a.output/('installation-'+body['mode']+'.'+suffix)).open('xb') as target:
                            while block:=source.read(1<<20):target.write(block)
                return result
            while True:
                check_ready()
                try:health=control_request('/health');break
                except OSError:time.sleep(.25)
            if not a.test_cold_readiness:
                ready=ready_future.result()
                if ready['context']<profile['rtx']['context']:raise ValueError('RTX readiness context differs')
            if health.get('family')!='qwen4exp' or health.get('producer_source_sha256')!=build['source_sha256'] or health.get('preset_sha256')!=digest(contract) or health.get('installation_sha256')!=installation_digest():
                raise ValueError('QFN receiver installation sources/preset differ from the producer')
            peer=dict(model_compatibility=producer,producer_sha256=build['sha256'],
                producer_source_revision=build['source_revision'],producer_ipv4='127.0.0.1',broker_port=rtx_port,tier='accurate',
                **{k:health[k] for k in ('producer_source_sha256','receiver_source_sha256','renderer_sha256',
                    'preset_sha256','receiver_model_sha256','installation_sha256')})
            try:
                accepted=installation_handshake(control_request,control,peer,health,local_install/'rtx-peer.json',
                    lambda message:note('shared installation',message,'real native/paired qualification then setup'))
            except BaseException:
                # Preserve refused observations as well as accepted ones. Never
                # replace the primary qualification failure with an archive error.
                archived=[];archive_errors=[]
                for name in ('peer.json','qualification-kl.json','qualification-native.f32','qualification-paired.f32','qfn-active-contract.json'):
                    try:copy_from(remote_install+'/state/'+name,a.output/('installation-'+name));archived.append(name)
                    except BaseException as error:archive_errors.append(dict(file=name,error=repr(error)))
                atomic_json(a.output/'installation-failure-archive.json',dict(files=archived,errors=archive_errors))
                raise
            atomic_json(a.output/'installation.json',accepted)
            # Preserve each real gate observation in Windows task evidence; no Mac accumulation.
            for name in ('peer.json','qualification-kl.json','qualification-native.f32','qualification-paired.f32','qfn-active-contract.json'):
                copy_from(remote_install+'/state/'+name,a.output/('installation-'+name))
            while True:
                for name,child in [('RTX',windows),('Mac',mac_job),('tunnel',tunnel)]:
                    if child.poll() is not None:raise RuntimeError(name+' exited before API readiness; inspect controller logs')
                try:
                    with urllib.request.urlopen(url+'/v1/models',timeout=2) as response:
                        if response.status==200:break
                except OSError:pass
                if stopping[0]:raise InterruptedError('Pair startup cancelled')
                time.sleep(.5)
            mac_ready=time.monotonic()-started
            api_ready=True
            pair_ready=time.monotonic()-started
            atomic_json(a.output/'startup.json',dict(mac_api_ready_s=mac_ready,pair_ready_s=pair_ready,
                rtx_ready_wait_s=pair_ready-mac_ready,authenticated_rtx_ready=not a.test_cold_readiness))
            note('resident native pair ready',f'{url}; setup {pair_ready:.2f}s; authenticated RTX ready={not a.test_cold_readiness}; payload hashes {"on (test)" if a.verify_state else "off"}','OpenAI requests or explicit test')
            if a.test_installation_only:
                note('installation-only test accepted','real native/paired gate; no speed measurement requested','retire peers and restore production')
            elif a.test_recall:
                # The short warm-up stays on the Mac. The RTX performs its own
                # explicit different short warm-up before the measured prefill.
                stream_request(url,dict(prompt='Reply ready.',max_tokens=4,temperature=0,stream=True,seed=1),a.output/'warmup.events.jsonl')
                raw=a.test_tokens_file.read_bytes()
                if not raw or len(raw)%4:raise ValueError('Malformed token fixture')
                tokens=list(struct.unpack('<'+'i'*(len(raw)//4),raw))
                body=dict(prompt=tokens,max_tokens=a.test_decode_tokens,temperature=a.test_temperature,
                    top_k=a.test_top_k,top_p=a.test_top_p,min_p=a.test_min_p,seed=a.test_seed,stream=True,thinking=False)
                clock=time.perf_counter()
                text,timings,mtp,_=stream_request(url,body,a.output/'decode.events.jsonl')
                wall=time.perf_counter()-clock
                (a.output/'output.txt').write_text(text,encoding='utf8')
                result=dict(prompt_tokens=len(tokens),decode_tokens_requested=a.test_decode_tokens,temperature=a.test_temperature,
                    seed=a.test_seed,timings=timings,mtp=mtp,wall_seconds=wall,recall=a.test_recall in text,
                    producer_sha256=a.producer_sha256,mac_weights=a.mac_model,kv_bits=16,
                    feature_session_introspection_rows=max(a.test_mtp_trace_rows,1) if a.test_features or a.test_peer_failure else 0,
                    verify_state=a.verify_state,transport=transport_label,
                    reference_quality='Separate 256-row test against archived bias-zero RTX QFN with 16-bit KV; not 16-bit weights')
                if a.test_mtp_trace_rows:
                    trace=a.output/'mtp-computed.trace'
                    with urllib.request.urlopen(url+'/v1/qfn-test/trace',timeout=60) as response,trace.open('xb') as out:
                        while block:=response.read(1<<20):out.write(block)
                    with trace.open('rb') as f:
                        if f.read(8)!=b'QFNTF32\x01':raise ValueError('Invalid native trace frame')
                        vocab,count=struct.unpack('<II',f.read(8));ids=[];positions=[]
                        if count!=a.test_mtp_trace_rows:raise ValueError('Requested computed-row count was not captured')
                        for i in range(count):
                            position,token=struct.unpack('<Ii',f.read(8));positions.append(position);ids.append(token)
                            f.seek(vocab*4,1)
                    if positions!=list(range(len(tokens)+1,len(tokens)+count+1)):
                        raise ValueError('MTP test trace is not the natural contiguous computed prefix')
                    (a.output/'natural-force.tokens.le32').write_bytes(struct.pack('<'+'i'*(count+1),*(ids+[0])))
                    result.update(test_trace=dict(rows=count,vocab=vocab,first_position=positions[0],
                        last_position=positions[-1],imported_row0_excluded=True),instrumented=True)
                atomic_json(a.output/'result.json',result)
                note('native API measurement',json.dumps(result),'certified native retirement and production restoration')
                if not result['recall']:raise ValueError('Natural sampled recall failed')
                if a.test_features:
                    from checkpoint_tests import checkpoint_negatives
                    result['checkpoint_negative_claims']=checkpoint_negatives(url,a.output/'checkpoint-proofs',
                        a.mac_model,a.test_recall,a.test_proof_program,sys.executable)
                    from session_tests import forward_delta,rewind_anchor
                    result['persistent_forward_delta']=forward_delta(url,a.output/'session-proofs',tokens,a.test_recall,
                        a.mac_model,dict(temperature=1,top_k=20,top_p=.95,seed=1),a.test_proof_program,sys.executable)
                    result['persistent_anchor_rewind']=rewind_anchor(url,a.output/'session-proofs',tokens,a.test_recall,
                        a.mac_model,dict(temperature=1,top_k=20,top_p=.95,seed=1),a.test_proof_program,sys.executable)
                    from feature_tests import run
                    try:
                        run(url,a.output/'features',a.producer_pack,a.mac_model,a.test_recall,
                            a.test_tokens_file,a.test_proof_program,sys.executable,a.test_feature_prefix_tokens,bool(a.rtx_vision),a.test_image_file)
                    finally:
                        summary=a.output/'features/summary.json'
                        if summary.exists():result['feature_tests']=json.loads(summary.read_text(encoding='utf8'))
                        atomic_json(a.output/'result.json',result)
                if a.test_peer_failure:
                    from peer_failure_tests import peer_failure
                    try:
                        result['peer_failure_recovery']=peer_failure(url,a.output/'peer-failure-proofs',tokens,a.test_recall,
                            a.mac_model,a.test_proof_program,sys.executable)
                    finally:
                        proof=a.output/'peer-failure-proofs/peer-failure.json'
                        if proof.exists():result['peer_failure_recovery']=json.loads(proof.read_text())
                        atomic_json(a.output/'result.json',result)
            else:
                while not stopping[0]:
                    if windows.poll() is not None or mac_job.poll() is not None or tunnel.poll() is not None:
                        raise RuntimeError('Resident pair process exited')
                    time.sleep(.5)
            success=True
        finally:
            errors=[];qualification_errors=[]
            # Retire both guarded peers before closing the tunnel. Every cleanup
            # step is attempted even if another peer or SSH connection failed.
            if api_ready and windows and mac_job and windows.poll() is None and mac_job.poll() is None:
                try:
                    request=urllib.request.Request(url+'/v1/qfn/session/flush',data=b'{}',headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=profile['transport']['stall_seconds']+30) as response:
                        flush_receipt=json.load(response)
                    note('ordered catch-up drained','all queued target owners acknowledged before peer retirement','retire Mac then RTX')
                except BaseException as error:qualification_errors.append('QFN ordered catch-up flush: '+repr(error))
            if mac_job:
                try:
                    ssh([a.mac_python,'-B','-c','from pathlib import Path;import sys;Path(sys.argv[1]).touch()',remote_run+'/stop'])
                    mac_job.wait(timeout=180)
                except BaseException as error:errors.append('Mac: '+repr(error))
            if windows:
                try:stop.parent.mkdir(parents=True,exist_ok=True);stop.touch();windows.wait(timeout=180)
                except BaseException as error:errors.append('RTX: '+repr(error))
            if tunnel:
                try:tunnel.terminate();tunnel.wait(timeout=15)
                except BaseException as error:errors.append('tunnel: '+repr(error))
            if ready_executor:ready_executor.shutdown(wait=True,cancel_futures=True)
            for stream in (rtx_log,mac_log,tunnel_log):
                if stream:stream.close()
            try:
                copy_from(remote_run+'/cleanup.json',a.output/'mac.guard.json')
                copy_from(remote_run+'/settings.json',a.output/'mac.settings.json')
                mac_certificate=json.loads((a.output/'mac.guard.json').read_text())
                if mac_certificate.get('engine_started') is not False:
                    copy_from(remote_run+'/owner.json',a.output/'mac.owner.json')
                    owner=json.loads((a.output/'mac.owner.json').read_text())
                    copy_from(owner['log'],a.output/'mac-native.log')
                cleanup['mac']=json.loads((a.output/'mac.guard.json').read_text())
                cleanup['rtx']=json.loads((rtx_run/'cleanup.json').read_text())
                lease['cleanup_confirmed']=all(row.get('release_allowed') for row in cleanup.values()) and not errors
                try:
                    result_file=a.output/'result.json'
                    if result_file.exists():
                        result=json.loads(result_file.read_text())
                        imports=[]
                        for line in (a.output/'mac-native.log').read_text(errors='replace').splitlines():
                            if line.startswith('QFN_NATIVE_IMPORT '):
                                fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
                                if int(fields['rows'])==result['prompt_tokens']:imports.append(fields)
                        if not imports:raise ValueError('Missing measured native state import')
                        fields=imports[0]
                        result['matching_length_imports']=imports
                        result.update(native_import=fields,rtx_prefill_seconds=float(fields['prefill_s']),
                            rtx_prefill_tok_s=result['prompt_tokens']/float(fields['prefill_s']),
                            handover_seconds=float(fields['handover_s']),export_prepare_seconds=float(fields['export_prepare_s']),
                            wire_bytes=int(fields['wire_bytes']),canonical_bytes=int(fields['canonical_bytes']))
                        result['request_phases']=[dict(re.findall(r'(\w+)=([^ ]+)',line))
                            for line in (a.output/'mac-native.log').read_text(errors='replace').splitlines()
                            if line.startswith('QFN_REQUEST_PHASE ')]
                        result['startup']=json.loads((a.output/'startup.json').read_text())
                        if a.rtx_vision:
                            result['vision_imports']=[dict(re.findall(r'(\w+)=([^ ]+)',line))
                                for line in (a.output/'mac-native.log').read_text(errors='replace').splitlines()
                                if line.startswith('QFN_NATIVE_IMPORT ') and re.search(r' images=[1-9]',line)]
                            events=rtx_run/'vision-events.jsonl'
                            result['rtx_vision_events']=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
                            features=result.get('feature_tests',{}).get('cases',[])
                            if any(c['case']=='vision-after-handover' and c['status']=='PASS' for c in features):
                                if not result['rtx_vision_events'] or not any(int(i['rows'])>=a.test_feature_prefix_tokens for i in result['vision_imports']):
                                    qualification_errors.append('Vision API result lacks an actual long-prompt RTX image import')
                        atomic_json(result_file,result)
                except BaseException as error:qualification_errors.append('measurement receipts: '+repr(error))
            except BaseException as error:errors.append('receipts: '+repr(error))
            atomic_json(a.output/'cleanup.json',dict(peers=cleanup,ordered_catchup_flush=flush_receipt,errors=errors,qualification_errors=qualification_errors,success=success and not qualification_errors,
                release_allowed=lease['cleanup_confirmed']))
            if errors:raise RuntimeError('Native pair cleanup requires attention: '+repr(errors))
            if qualification_errors:raise ValueError('Native pair qualification failed: '+repr(qualification_errors))


if __name__=='__main__':main()
