"""Transfer a validated RTX checkpoint to guarded native Mac decode over SSH.

All listeners are loopback. Each native executable keeps one stable pathname.
The shared Mac lease must be actually released before this command can start.
"""
import argparse
import contextlib
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from slstack.lease import active_holds
from slstack.model_compatibility import model_description,require_compatible,continuation_kl
from slstack.journal import atomic_json
from slstack.envvars import rows


def signature(key,body):return hmac.new(key,json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),hashlib.sha256).hexdigest()


@contextlib.contextmanager
def mac_lease(path,identity,budget,floor,seconds,parent=None):
    if parent:
        held=active_holds(path.read_text(encoding='utf-8-sig'),mac=True)
        if len(held)!=1 or not held[0].startswith('MAC ACTIVE: QfnPair '+parent+';'):
            raise RuntimeError('Parent QfnPair benchmark lease is not the sole active Mac lease')
        yield {'cleanup_confirmed':False}
        return # the parent owns release after production restoration
    with path.open('r+',encoding='utf-8-sig') as f:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            text=f.read();held=active_holds(text,mac=True)
            if held:raise RuntimeError('Waiting for actual Mac RELEASED; no expiry bypass: '+str(held))
            expiry=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=seconds)
            f.seek(0,2);f.write(f'\nMAC ACTIVE: QfnPair {identity}; exclusive configured GPU; expires {expiry.isoformat()}; host RSS {budget} GiB; floor {floor} GiB\n');f.flush();os.fsync(f.fileno())
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    state={'cleanup_confirmed':False}
    try:yield state
    finally:
        if state['cleanup_confirmed']:
            with path.open('a',encoding='utf8') as f:f.write(f'\nMAC RELEASED: QfnPair {identity}; {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--rtx-result',type=Path,required=True);p.add_argument('--producer-pack',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pair-token-file',type=Path,required=True)
    p.add_argument('--ssh-host',required=True);p.add_argument('--mac-root',required=True);p.add_argument('--mac-python',required=True)
    p.add_argument('--mac-model',required=True);p.add_argument('--mac-receipt',required=True);p.add_argument('--mac-decoder',required=True)
    p.add_argument('--mac-vision',required=True);p.add_argument('--mac-vision-receipt',required=True);p.add_argument('--hold-file',type=Path,required=True)
    p.add_argument('--mac-host-budget-gib',type=float,required=True);p.add_argument('--mac-floor-gib',type=float,required=True)
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--port',type=int,default=18794)
    p.add_argument('--parent-mac-lease',help='Existing QfnPair benchmark controller owns ejection/restoration and final release')
    p.add_argument('--context',type=int,default=100480);p.add_argument('--decode-tokens',type=int,default=128)
    p.add_argument('--mtp-draft',type=int,choices=[1,2,3],default=3);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--prefill-chunk',type=int,default=256);p.add_argument('--warm-weights',type=int,choices=[0,1],default=1)
    p.add_argument('--temperature',type=float,default=0);p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--top-p',type=float,default=.95);p.add_argument('--min-p',type=float,default=0);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--progress-log',type=Path,required=True)
    p.add_argument('--test-recall',help='Run recall and forced-logit checks with this expected phrase; test only')
    p.add_argument('--test-force-tokens-file',type=Path,help='Explicit matched-prefix teacher-forcing fixture; requires --test-recall')
    p.add_argument('--test-reference-logits',type=Path,help='Explicit archived full-vocabulary reference; requires --test-recall')
    p.add_argument('--test-reference-skip',type=int,default=1)
    p.add_argument('--verify-state',action='store_true',help='Test only: one inline hash while streaming; off by default')
    p.add_argument('--payload-budget-bytes',type=int,default=16<<30)
    for option in rows():
        if 'mac-ds4' in option['scopes']:
            p.add_argument('--option-'+option['var'].lower().replace('_','-'),dest='option_'+option['var'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.test_force_tokens_file or a.test_reference_logits) and not a.test_recall:
        raise ValueError('Quality inputs require the explicit recall/quality test flag')
    result=json.loads(a.rtx_result.read_text());src=a.rtx_result.parent
    producer_settings=json.loads((src/'settings.json').read_text())
    if not producer_settings['flags']['mtp'] and a.mtp_draft>1:
        raise ValueError('Speculative Mac decoding needs a populated draft frontier: export with --mtp 1, or use --mtp-draft 1')
    if result['payload'].get('mtp_alignment')!='hidden-q-token-q-plus-1':raise ValueError('Corrected native MTP exporter required')
    producer=model_description(a.producer_pack,result['model_sha256'],0)
    key=a.pair_token_file.read_bytes().strip()
    if len(key)<32:raise ValueError('Use an existing private 256-bit pairing token file')
    nonce=uuid.uuid4().hex;label='mac-decode-'+nonce;remote=a.mac_root.rstrip('/');prefix=remote+'/'+label
    def ssh(args,**kw):return subprocess.run(['ssh','-o','BatchMode=yes',a.ssh_host,shlex.join(args)],check=True,**kw)
    def copy_to(local,name):subprocess.run(['scp','-o','BatchMode=yes',str(local),a.ssh_host+':'+remote+'/'+name],check=True)
    def copy_from(name,local):subprocess.run(['scp','-o','BatchMode=yes',a.ssh_host+':'+name,str(local)],check=True)
    def note(doing,latest,next_step):
        line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
        print(line,flush=True)
        with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
    command=[a.mac_decoder,'--model',a.mac_model,'--vision',a.mac_vision,'--tokens',remote+'/prompt.tokens.le32',
        '--warmup-tokens',remote+'/warmup.tokens.le32',
        '--out',prefix,'--ctx',str(a.context),'--decode-tokens',str(a.decode_tokens),'--mtp-draft',str(a.mtp_draft),
        '--threads',str(a.threads),'--prefill-chunk',str(a.prefill_chunk),'--temperature',str(a.temperature),
        '--top-k',str(a.top_k),'--top-p',str(a.top_p),'--min-p',str(a.min_p),'--seed',str(a.seed),
        '--warm-weights',str(a.warm_weights),'--payload','-','--payload-framed','1','--payload-bytes',str(result['payload']['bytes'])]
    if a.test_recall:command+=['--force-tokens',remote+'/forced.tokens.le32']
    receive=[a.mac_python,'-B',remote+'/examples/dual-machine/qfn/receive.py','--port',str(a.port),
        '--pair-token-file',remote+'/pair.token','--nonce',nonce,'--model',a.mac_model,'--model-receipt',a.mac_receipt,'--out',prefix,
        '--vision-receipt',a.mac_vision_receipt,
        '--host-budget-gib',str(a.mac_host_budget_gib),'--floor-gib',str(a.mac_floor_gib),
        '--max-payload-bytes',str(a.payload_budget_bytes)]
    for option in rows():
        value=getattr(a,'option_'+option['var'],None)
        if value is not None:receive+=['--option-'+option['var'].lower().replace('_','-'),value]
    receive+=['--',*command]
    guarded=[a.mac_python,'-B',remote+'/scripts/qfn_dual_guard.py','--root',remote,'--label',label,
        '--rss-gib',str(a.mac_host_budget_gib),'--commit-gib',str(a.mac_host_budget_gib),'--floor-gib',str(a.mac_floor_gib),
        '--seconds',str(a.seconds),'--progress-log',remote+'/progress.log','--stream-log','--',*receive]
    server=tunnel=None;remote_log=tunnel_log=None;state=None;started=time.monotonic()
    with mac_lease(a.hold_file,nonce,a.mac_host_budget_gib,a.mac_floor_gib,a.seconds,a.parent_mac_lease) as state:
        try:
            for name in ('prompt.tokens.le32','warmup.tokens.le32'):copy_to(src/name,name)
            if a.test_recall:copy_to(a.test_force_tokens_file or src/'forced.tokens.le32','forced.tokens.le32')
            copy_to(a.pair_token_file,'pair.token')
            remote_log=(a.output/'mac-guard.log').open('xb')
            server=subprocess.Popen(['ssh','-o','BatchMode=yes',a.ssh_host,shlex.join(guarded)],stdout=remote_log,stderr=subprocess.STDOUT)
            while 'QFN receiver listening on 127.0.0.1:' not in (a.output/'mac-guard.log').read_text(errors='replace'):
                if server.poll()is not None:raise RuntimeError('Mac receiver failed before listening; inspect mac-guard.log')
                if time.monotonic()-started>60:raise TimeoutError('Mac loopback receiver did not become ready')
                time.sleep(.2)
            tunnel_log=(a.output/'tunnel.log').open('xb')
            tunnel=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-N',
                '-L',f'127.0.0.1:{a.port}:127.0.0.1:{a.port}',a.ssh_host],stdout=tunnel_log,stderr=subprocess.STDOUT)
            tunnel_deadline=time.monotonic()+30
            while True:
                if server.poll()is not None or tunnel.poll()is not None:raise RuntimeError('Owned loopback transport exited before connection')
                try:conn=socket.create_connection(('127.0.0.1',a.port),timeout=2);conn.settimeout(None);break
                except OSError:
                    if time.monotonic()>tunnel_deadline:raise TimeoutError('Loopback tunnel did not become ready')
                    time.sleep(.2)
            with conn:
                body=dict(schema='qfn-paired-stream-v1',nonce=nonce,producer_binary_sha256=producer_settings['engine_sha256'],
                    model_compatibility=producer,verify_state=a.verify_state,payload_bytes=result['payload']['bytes'])
                body['hmac']=signature(key,body);conn.sendall((json.dumps(body)+'\n').encode());stream=conn.makefile('rb')
                def event():
                    raw=stream.readline()
                    if not raw:raise RuntimeError('Mac decoder closed before completion')
                    row=json.loads(raw);note('Mac paired stage',str(row.get('event',row)),'next native phase');return row
                hello=event();sig=hello.pop('hmac',None)
                if hello.get('event')!='authenticated' or hello.get('nonce')!=nonce or not hmac.compare_digest(signature(key,hello),str(sig)):
                    raise ValueError('Mac endpoint authentication failed')
                require_compatible(producer,hello['receiver']);atomic_json(a.output/'identity.json',dict(producer=producer,receiver=hello['receiver']))
                if event().get('event')!='warmup_complete':raise ValueError('Missing native warm-up event')
                def send_payload():
                    digest=hashlib.sha256() if a.verify_state else None
                    with (src/'state.payload').open('rb') as f:
                        while b:=f.read(1<<20):
                            if digest:digest.update(b)
                            conn.sendall(b)
                    if digest:conn.sendall(digest.hexdigest().encode('ascii'))
                clock=time.perf_counter();send_payload();loaded=event();transfer=time.perf_counter()-clock
                if loaded.get('event')!='imported' or loaded['rows']!=result['prompt_tokens']:raise ValueError('Native restore frontier differs')
                if event().get('event')!='decoded':raise ValueError('Native decode failed')
                if a.test_recall:send_payload()
                if event().get('event')!='complete':raise ValueError('Native observation/decode failed')
            if server.wait(timeout=60):raise RuntimeError('Mac guard reported a failure')
            for suffix in ('.json','.txt','.tokens.le32','.settings.json')+(('.f32',) if a.test_recall else ()):copy_from(prefix+suffix,a.output/('mac'+suffix))
            copy_from(remote+'/'+label+'.guard.json',a.output/'mac.guard.json')
            cert=json.loads((a.output/'mac.guard.json').read_text())
            if not cert.get('release_allowed'):raise RuntimeError('Mac cleanup not certified')
            state['cleanup_confirmed']=True
            native=json.loads((a.output/'mac.json').read_text());text=(a.output/'mac.txt').read_text(encoding='utf8')
            quality=None
            if a.test_recall:
                quality_path=a.output/'quality.json'
                compare=[sys.executable,'-B',str(ROOT/'examples/dual-machine/qfn/quality.py'),
                    '--control',str(a.test_reference_logits or src/'control.f32'),'--candidate',str(a.output/'mac.f32'),
                    '--vocab',str(native['vocab']),'--out',str(quality_path),'--control-skip',str(a.test_reference_skip),'--candidate-skip','1',
                    '--reference-label','KL(explicit unbiased single-machine reference || plain Mac)',
                    '--progress-log',str(a.progress_log)]
                guarded_compare=[sys.executable,'-B',str(ROOT/'scripts/qfn_dual_guard.py'),
                    '--root',str(a.output),'--label','quality-compare','--rss-gib','.5','--commit-gib','.5',
                    '--floor-gib','6','--seconds','600','--blas-threads','1','--progress-log',str(a.progress_log),'--',*compare]
                subprocess.run(guarded_compare,cwd=ROOT,check=True)
                quality=json.loads(quality_path.read_text());quality['excluded_imported_row0']=True
            row=dict(rtx=result,mac=native,transfer_and_import_seconds=transfer,verify_state=a.verify_state,
                handover_seconds=result['export_seconds']+transfer,
                handover_components=dict(export_seconds=result['export_seconds'],producer_verify_state=a.verify_state,
                    transport_receiver_hash_and_native_import_seconds=transfer),
                quality=quality,recall=(a.test_recall in text) if a.test_recall else None,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                native_prompt_replay=False,capability_reductions=[])
            row['quality_limits']=dict(mean_nats=.001,max_nats=.01,top1_min=.99)
            row['qualified']=(row['recall'] and quality['mean_nats']<=.001 and quality['max_nats']<=.01 and quality['top1']>=.99) if a.test_recall else None
            compact_quality={k:quality[k] for k in ('rows','vocab','mean_nats','max_nats','top1')} if quality else None
            atomic_json(a.output/'result.json',row);note('dual-machine measurement complete',json.dumps(dict(prefill=result['prefill_tok_s'],handover=row['handover_seconds'],decode=native['decode_tok_s'],quality=compact_quality,recall=row['recall'],qualified=row['qualified'])),'result table and validation')
            if a.test_recall and not row['qualified']:raise ValueError('QFN continuation/recall gate failed; pair remains unqualified')
        finally:
            if tunnel and tunnel.poll()is None:tunnel.terminate();tunnel.wait(timeout=15)
            if server:
                if server.poll()is None:
                    server.wait(timeout=max(1,a.seconds+30-(time.monotonic()-started)))
                # Obtain a failure cleanup certificate too, never invent a release.
                if not state['cleanup_confirmed']:
                    try:
                        copy_from(remote+'/'+label+'.guard.json',a.output/'mac.guard.json')
                        state['cleanup_confirmed']=bool(json.loads((a.output/'mac.guard.json').read_text()).get('release_allowed'))
                    except (OSError,subprocess.SubprocessError):pass
            else:state['cleanup_confirmed']=True
            if remote_log:remote_log.close()
            if tunnel_log:tunnel_log.close()


if __name__=='__main__':main()
