"""Normal native-server HTTP surface, relaying the real ds4-win stream.

CPU-only while idle. Every chat owns a separately bounded shared GPU lease;
the existing independent guardian watches the whole server/model process tree.
"""
from __future__ import annotations
from copy import deepcopy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import sys
import threading
import time
import urllib.request
import psutil
from .combo_measure import require_parity
from .combo_session import ComboSession
from .journal import atomic_json
from .lease import gpu_lease
from .native_broker import vendor


def conversation_key(body):
    explicit=body.get('prompt_cache_key')
    if explicit is not None and (not isinstance(explicit,str) or not 0<len(explicit)<=512):raise ValueError('Invalid conversation cache key')
    prefix=[]
    for message in body.get('messages',[]):
        prefix.append(message)
        if message.get('role')=='user':break
    # One bounded cache. Actual native token-prefix hashes still decide every
    # reuse/rewind; a key never permits stale or cross-conversation bytes.
    row=dict(explicit=explicit,prefix=prefix,tools=body.get('tools'),model=body.get('model'))
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def begins_conversation(body):
    roles=[message.get('role') for message in body.get('messages',[])]
    return roles.count('user')==1 and all(role in ('system','developer','user') for role in roles)


def request_worker_config(cfg):
    """Exclude the independent controller from the worker/state admission."""
    total=cfg['host_budget_gib']
    if type(total) not in (int,float) or not math.isfinite(total) or total<=.25:
        raise ValueError('Positive aggregate budget after controller reservation required')
    worker=cfg.get('combo_worker_host_budget_gib',total-.25)
    if (type(worker) not in (int,float) or not math.isfinite(worker) or
        worker<=0 or worker!=total-.25):
        raise ValueError('Worker budget must match the enclosing tree after controller reservation')
    result=deepcopy(cfg);result['host_budget_gib']=worker
    return result


class Proxy:
    def __init__(self,cfg,run_id,window_file):
        vendor();from studio.quality_speed_contract import NAMED
        evidence=require_parity(cfg)
        if cfg['provider_schema']['named_points']!=NAMED:raise ValueError('Current quality provider differs from qualified schema')
        self.cfg=cfg;self.run_id=run_id;self.window_file=Path(window_file)
        self.token=os.environ.get('DS4_SERVER_AUTH_TOKEN','')
        if len(self.token)<32:raise ValueError('Normal Studio native Bearer token is required')
        self.lock=threading.Lock();self.session=None;self.key=None;self.pending=0
        capability=evidence['ultra'].get('provider_initial')
        if not capability:raise ValueError('Qualified native capability receipt is required')
        self.models=evidence['ultra'].get('native_models')
        if not isinstance(self.models,dict) or not self.models.get('data'):
            raise ValueError('Qualified actual native model catalogue is required')
        self.observed={**capability,'tok_s':0,'decoder_running':False,'cached':True,'applied':False}
        self.window(False)

    def window(self,active,lease_id=None):
        row=dict(active=active,run_id=self.run_id,owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),
                 lease_id=lease_id,deadline=time.time()+self.cfg['lease_minutes']*60)
        started=time.monotonic()
        while True:
            try:atomic_json(self.window_file,row);return
            except PermissionError:
                if time.monotonic()-started>=1:raise
                time.sleep(.01)

    def quality(self,args=None):
        from studio.quality_speed_contract import setting_arguments
        if args is not None:self.pending=setting_arguments(args,self.pending)
        owner=self.session.current if self.session else None
        native_running=owner is not None and hasattr(owner,'quality')
        if native_running:
            result=self.session.quality(self.pending if args is not None else None)
            self.observed={**result,'decoder_running':True,'cached':False}
        elif self.session is not None and args is not None:self.session.quality(self.pending)
        result=deepcopy(self.observed)
        result.update(decoder_running=native_running,cached=not native_running)
        if not native_running:
            phases=deepcopy(self.pending) if isinstance(self.pending,dict) else None
            result.update(value=phases['answer'] if phases else self.pending,
                          phases=phases,per_phase=phases is not None)
        if args is not None or not result.get('decoder_running'):
            result.update(requested_value=self.pending,deferred=not native_running,
                          applied=native_running,scope='current native decoder or next decode in this cached conversation')
        return result


def relay_stream(response,write,completed,observed=None):
    """Preserve native SSE bytes; hold terminal metadata until state is saved.

    A normal client closes on finish/DONE. Stop disconnect cancellation at
    that native terminal event, then finish the checkpoint/release before
    publishing DONE. Content and tool parsing remain entirely native.
    """
    pending=b'';terminal=bytearray();finished=False
    while block:=response.read1(65536):
        pending+=block
        while b'\n' in pending:
            line,pending=pending.split(b'\n',1);line+=b'\n'
            if len(line)>1<<20:raise ValueError('Native SSE line exceeded1MiB')
            if line.startswith(b'data:'):
                data=line[5:].strip()
                if data==b'[DONE]':finished=True
                else:
                    row=json.loads(data)
                    if row.get('error'):raise RuntimeError('Native stream failed: '+str(row['error']))
                    if observed is not None:observed(row)
                    finished=finished or any(c.get('finish_reason') is not None for c in row.get('choices',[]))
                if finished:completed.set()
            if finished:
                terminal.extend(line)
                if len(terminal)>1<<20:raise ValueError('Native terminal metadata exceeded1MiB')
            else:write(line)
        if len(pending)>1<<20:raise ValueError('Native SSE pending line exceeded1MiB')
    if pending:raise ValueError('Native stream ended inside an SSE line')
    if not finished:raise ValueError('Native stream ended without terminal metadata')
    return bytes(terminal)


def handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def send(self,status,body):
            data=json.dumps(body,separators=(',',':'),allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+proxy.token)
        def read_body(self):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1<<20:raise ValueError('Native request body must fit1MiB')
            self.connection.settimeout(30)
            data=self.rfile.read(length)
            if len(data)!=length:raise ValueError('Truncated request')
            from studio.dsv41_session_contract import control
            return control(data)
        def do_GET(self):
            if self.path in ('/health','/healthz'):
                return self.send(200,dict(status='ok',model_ready=bool(proxy.session and proxy.session.current),context=proxy.cfg['context']))
            if self.path=='/v1/models':
                return self.send(200,proxy.models)
            if self.path=='/v1/quality-speed':
                if not self.authorized():return self.send(401,dict(error=dict(message='Native Bearer authentication required')))
                try:return self.send(200,proxy.quality())
                except (ValueError,RuntimeError,OSError) as exc:return self.send(409,dict(error=dict(message=str(exc))))
            return self.send(404,dict(error=dict(message='Unknown native endpoint')))
        def do_POST(self):
            if not self.authorized():return self.send(401,dict(error=dict(message='Native Bearer authentication required')))
            try:body=self.read_body()
            except (ValueError,OSError) as exc:return self.send(400,dict(error=dict(message=str(exc))))
            if self.path=='/v1/quality-speed':
                try:return self.send(200,proxy.quality(body))
                except (ValueError,RuntimeError,OSError) as exc:return self.send(409,dict(error=dict(message=str(exc))))
            if self.path!='/v1/chat/completions':return self.send(404,dict(error=dict(message='Unknown native endpoint')))
            if not proxy.lock.acquire(blocking=False):return self.send(409,dict(error=dict(message='Another request owns this decoder')))
            streaming=body.get('stream') is True;headers=False;done=threading.Event();session=None;timer=None
            metrics=None;request_started=time.monotonic()
            try:
                key=conversation_key(body)
                fresh=begins_conversation(body) and proxy.session is not None and not proxy.session.poison
                if proxy.key!=key or fresh:
                    if proxy.session:proxy.session.close()
                    proxy.session=ComboSession(request_worker_config(proxy.cfg));proxy.key=key
                session=proxy.session
                from studio.quality_speed_contract import validate
                proxy.pending=validate(body.get('quality_speed',proxy.pending))
                session.quality_setting=deepcopy(body.get('quality_speed',proxy.pending))
                if streaming:
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
                    self.close_connection=True;headers=True;self.wfile.write(b': accepted\n\n');self.wfile.flush()
                def disconnect_watch():
                    while not done.wait(.1):
                        try:
                            readable,_,_=select.select([self.connection],[],[],0)
                            if readable and not self.connection.recv(1,socket.MSG_PEEK):session.cancel.set();return
                        except OSError:session.cancel.set();return
                watcher=threading.Thread(target=disconnect_watch,daemon=True);watcher.start()
                # DS4COMBO2 liveness: no fixed lease_minutes cap on a request. The native producer's
                # own no-progress watch poisons a stalled run; this watchdog also cancels the request
                # when the producer shows no progress for the stall window, and renews the GPU window
                # while it progresses.
                from .combo_producer import stall_seconds
                stall=stall_seconds(proxy.cfg);watched=[None]
                def progress_watch():
                    while not done.wait(min(30.0,stall/10)):
                        process=getattr(watched[0],'process',None)
                        last=getattr(process,'last_progress',None)
                        if last is not None and time.monotonic()-last>=stall:session.cancel.set();return
                timer=threading.Thread(target=progress_watch,name='native-progress-watch',daemon=True);timer.start()
                with gpu_lease(proxy.cfg) as lease:
                    proxy.window(True,lease)
                    try:
                        with session.prepare(body) as (native,rendered,metrics):
                            watched[0]=native
                            api_started=time.monotonic()
                            def observed(row):
                                meaningful=any(any((choice.get('delta') or {}).get(field) for field in ('content','reasoning_content','tool_calls'))
                                               for choice in row.get('choices',[]))
                                if meaningful and 'native_api_first_token_seconds' not in metrics:
                                    metrics['native_api_first_token_seconds']=time.monotonic()-api_started
                                    metrics['total_first_token_seconds']=metrics['prepared_seconds']+metrics['native_api_first_token_seconds']
                                    metrics['coordinator_first_token_seconds']=time.monotonic()-request_started
                            request=urllib.request.Request(native.url+'/v1/chat/completions',
                                data=json.dumps(rendered,separators=(',',':'),allow_nan=False).encode(),
                                headers={'Content-Type':'application/json','Authorization':'Bearer '+native.token})
                            # No fixed socket timeout: a long silent prefill is bounded by progress_watch.
                            with urllib.request.urlopen(request,timeout=None) as response:
                                if streaming:
                                    def write(block):self.wfile.write(block);self.wfile.flush()
                                    terminal=relay_stream(response,write,done,observed)
                                else:
                                    raw=response.read((8<<20)+1)
                                    if len(raw)>8<<20:raise ValueError('Native response exceeded8MiB')
                                    result=json.loads(raw)
                                    done.set()
                            proxy.observed={**native.quality(),'decoder_running':False,'cached':True}
                        # The context has certified native/producer retirement.
                        proxy.window(False)
                    except BaseException:
                        if session.poison and session.poison.startswith('Unconfirmed GPU release'):
                            os._exit(72) # Outer kill-on-close Job retires descendants before orphan-lease recovery.
                        proxy.window(False);raise
                if streaming:self.wfile.write(terminal);self.wfile.flush()
                else:self.send(200,result)
            except Exception as exc:
                error=dict(error=dict(message=str(exc),type='handoff_refused'))
                if headers:
                    try:self.wfile.write(b'data: '+json.dumps(error,separators=(',',':')).encode()+b'\n\ndata: [DONE]\n\n');self.wfile.flush()
                    except OSError:pass
                else:self.send(503,error)
            finally:
                done.set()
                done.set()   # ends progress_watch
                if metrics is not None:atomic_json(Path(proxy.cfg['runtime_root'])/('request-'+secrets.token_hex(8)+'.json'),metrics)
                proxy.lock.release()
    return Handler


def serve(cfg,run_id,window):
    if cfg.get('serving_mode')=='single_engine':   # ULTRA producer prefill + in-process decode, no native
        from .combo_single import serve as single_engine
        return single_engine(cfg,run_id,window)
    if cfg.get('serving_mode') not in (None,'handoff'):raise ValueError('Unknown combo serving_mode')
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    proxy=Proxy(cfg,run_id,window)
    class Server(ThreadingHTTPServer):daemon_threads=True
    server=Server((cfg['host'],cfg['port']),handler(proxy))
    print('DS4COMBO router ready; CPU idle, request-scoped GPU ownership',flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        if proxy.session and not proxy.lock.locked():proxy.session.close()


def native_launch_config(cfg,argv):
    """Accept only the qualified normal Studio argv, with its assigned port."""
    expected=cfg.get('studio_expected_args')
    if not isinstance(expected,list):raise ValueError('Qualified normal Studio launch argv is required')
    def normalize(values):
        values=list(values);port=None
        if values.count('--port')!=1:raise ValueError('One normal native port is required')
        index=values.index('--port')
        if index+1>=len(values) or not str(values[index+1]).isdigit():raise ValueError('Invalid native port')
        port=int(values[index+1])
        if not 1024<=port<=65535:raise ValueError('Native port outside the qualified local range')
        values[index+1]='{port}'
        return values,port
    argv,preset=list(argv),{}
    # Per-preset streaming profile and its own N* (owner 2026-10-23): carried in
    # the preset's flags, validated against the measured inputs in combo.json.
    for flag in ('--combo-producer-profile','--combo-crossover-tokens'):
        if flag in argv:
            index=argv.index(flag)
            if index+1>=len(argv):raise ValueError(flag+' needs a value')
            preset[flag]=argv[index+1];del argv[index:index+2]
    actual,port=normalize(argv);qualified,_=normalize(expected)
    if actual!=qualified:raise ValueError('Native launch differs from the qualified combined preset; review its model/context/cache/flags')
    result=deepcopy(cfg);result.update(port=port,host='127.0.0.1')
    # A runtime that carries several measured streaming profiles takes N* (and
    # its producer) only from the launching preset: never a global or a value
    # left over from the previously active preset.
    if cfg.get('combo_crossover_profiles') and not preset:
        raise ValueError('This runtime has per-preset crossovers; the preset must pass --combo-producer-profile and --combo-crossover-tokens')
    if preset:result.update(apply_preset_crossover(cfg,preset.get('--combo-producer-profile'),preset.get('--combo-crossover-tokens')))
    return result


def apply_preset_crossover(cfg,profile,tokens):
    """Select one measured streaming preset; its N* must equal the owner crossover recomputed from
    its measured inputs (r_native_tok_s, r_stream_tok_s, switch_seconds and the optional
    native_intercept_s / stream_fixed_s, both 0 when absent)."""
    table=cfg.get('combo_crossover_profiles') or {}
    if profile not in table or tokens is None or not str(tokens).isdigit():
        raise ValueError('Preset needs --combo-producer-profile with measured inputs and --combo-crossover-tokens')
    from .combo_crossover import qualified_threshold
    row=table[profile]
    if 'combo_crossover_tokens' in row and int(row['combo_crossover_tokens'])!=int(tokens):
        raise ValueError('Preset N* differs from the measured N* of its streaming profile')
    update=dict(producer_profile=profile,combo_crossover_tokens=int(tokens),combo_crossover_inputs=dict(row,producer_profile=profile))
    # A profile measured on its own producer build (Fast: split-F32) launches that build.
    if ('llama_producer' in row)!=('producer_sha256' in row):
        raise ValueError('A profile producer needs both its path and its SHA-256 pin')
    if 'llama_producer' in row:update.update(llama_producer=row['llama_producer'],producer_sha256=row['producer_sha256'])
    # ...and is qualified by its own measured parity, never the default producer's.
    pinned=[key for key in PROFILE_QUALIFICATION if key in row]
    if pinned and not ('combo_qualification_root' in row and 'combo_native_binding' in row):
        raise ValueError('A profile qualification needs both its combo_qualification_root and combo_native_binding')
    if 'llama_producer' in row and not ('combo_qualification_root' in row and 'combo_native_binding' in row):
        raise ValueError('A profile with its own producer needs its own qualification (combo_qualification_root and combo_native_binding)')
    update.update({key:row[key] for key in pinned})
    qualified_threshold({**cfg,**update})
    return update


# Profile-pinned qualification evidence; combo_ultra_guard is optional (the shared
# guardian receipt applies when a profile does not pin its own).
PROFILE_QUALIFICATION=('combo_qualification_root','combo_native_binding','combo_ultra_guard')


def guarded(cfg,config):
    from .guard import supervise,admit
    from .config import environment
    cfg=deepcopy(cfg);cfg['combo_worker_host_budget_gib']=request_worker_config(cfg)['host_budget_gib']
    root=Path(cfg['runtime_root']);root.mkdir(parents=True,exist_ok=True)
    admit(cfg['host_budget_gib'],cfg['min_available_gib'])
    run_id=secrets.token_hex(16);window=root/('gpu-window-'+run_id+'.json')
    owner=psutil.Process(os.getppid())
    env=environment(cfg,'studio');env['PYTHONDONTWRITEBYTECODE']='1'
    token=os.environ.get('DS4_SERVER_AUTH_TOKEN','')
    if len(token)<32:raise ValueError('Normal Studio native Bearer token is required')
    env['DS4_SERVER_AUTH_TOKEN']=token
    run_config=root/('server-'+run_id+'.config.json');atomic_json(run_config,cfg)
    command=[sys.executable,'-B','-m','slstack.combo_proxy','--config',str(run_config),'--serve','--run-id',run_id,'--window',str(window)]
    spec=dict(command=command,cwd=str(Path(__file__).resolve().parents[1]),env=env,runtime_root=str(root),task='DS4COMBO',
        host_budget_gib=cfg['combo_worker_host_budget_gib'],min_available_gib=cfg['min_available_gib'],job_commit_gib=cfg['job_commit_gib'],timeout_s=None,   # a server has no lifetime cap; requests use the progress-renewed window
        
        owner_pid=owner.pid,owner_created=owner.create_time(),stop=str(root/('server-'+run_id+'.stop')),
        log=str(root/('server-'+run_id+'.log')),receipt=str(root/('server-'+run_id+'.json')),paging_telemetry=True,
        gpu_window_file=str(window),gpu_run_id=run_id,gpu_hold_file=cfg['hold_file'])
    from .envvars import split_secrets
    atomic_json(root/('server-'+run_id+'.spec.json'),dict(spec,env=split_secrets(env)[0]))   # the token stays in memory
    return supervise(spec)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--serve',action='store_true')
    parser.add_argument('--run-id');parser.add_argument('--window')
    parser.add_argument('--native-argv',nargs=argparse.REMAINDER)
    args=parser.parse_args();config=Path(args.config).resolve();cfg=json.loads(config.read_text())
    if args.serve:serve(cfg,args.run_id,args.window)
    else:
        if args.native_argv is not None:cfg=native_launch_config(cfg,args.native_argv)
        raise SystemExit(guarded(cfg,config))
