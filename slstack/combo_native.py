"""Owned native HTTP runtime; reuse Studio's suspended Job and memory watcher."""
from __future__ import annotations
from copy import deepcopy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request
from .config import path,environment
from .native_broker import vendor,model_receipt
from .journal import atomic_json,sha256


class Native:
    def __init__(self,cfg,session,epoch,sequence,cancelled,*,restore=None,runner=None):
        vendor()
        from studio.dsv41_session_process import Producer
        from studio.dsv41_windows_runner import _Native
        self.cfg=deepcopy(cfg);self.session=session;self.epoch=epoch;self.sequence=sequence
        self.control_lock=threading.Lock();self.process=None;self.closed=False
        self.leases=ExitStack()
        self.restore_lease=None
        self.token=self.cfg.setdefault('_control_token',secrets.token_urlsafe(32))
        self.url=cfg['ds4_url'];self.cancelled=cancelled
        binary=path(cfg['ds4_command'][0]);pin=sha256(binary)
        if pin!=cfg['native_sha256']:raise ValueError('Native binary pin changed')
        if path(cfg['model']).drive.upper()=='M:':raise ValueError('Timed native model weights cannot use the archival drive')
        command=[str(v).replace('{model}',str(path(cfg['model']))) for v in cfg['ds4_command']]
        command[0]=str(binary)
        calibration=cfg.get('cache_affinity_spread_file')
        if calibration and '--cache-affinity-spread-file' not in command:
            command+=['--cache-affinity-spread-file',str(path(calibration))]
        env_values=dict(DS4_COMBO_ENABLED='1',DS4_COMBO_OWNER_SESSION=session,
            DS4_COMBO_OWNER_EPOCH=str(epoch),DS4_COMBO_SEQUENCE=str(sequence),DS4_SERVER_AUTH_TOKEN=self.token)
        if restore is not None:
            ram,proof=restore
            env_values.update(DS4_COMBO_RESTORE_MAPPING=ram.name,DS4_COMBO_RESTORE_CAPACITY=str(ram.capacity),
                              DS4_COMBO_RESTORE_SHA256=proof['payload_sha256'])
        self.cfg.setdefault('env_overrides',{}).update(env_values)
        env=environment(self.cfg,'ds4-win')
        parts=cfg.get('ds_read_parts',8)
        if type(parts) is not int or not 1<=parts<=64:raise ValueError('Explicit native split reads must be in1..64')
        env['DS4_DS_READ_PARTS']=str(parts)
        libraries=[str(path(p)) for p in cfg.get('producer_library_directories',[])]
        env['PATH']=os.pathsep.join([str(binary.parent),*libraries,env.get('PATH','')])
        directory=Path(cfg['runtime_root'])/('native-'+secrets.token_hex(8));directory.mkdir(parents=True,exist_ok=False)
        self.directory=directory;self.launch_started=time.monotonic()

        def startup(process):
            self.process=process
            deadline=time.monotonic()+min(300,cfg['request_timeout_s'])
            while True:
                if cancelled():raise RuntimeError('Native startup cancelled in its owner epoch')
                if process.job.poll() is not None:raise RuntimeError('Native process exited before setup: '+process.diagnostic[-2000:].decode(errors='replace'))
                try:
                    self.http('/v1/models',timeout=2)
                    break
                except (OSError,RuntimeError):
                    if time.monotonic()>deadline:raise TimeoutError('Native readiness deadline exceeded')
                    time.sleep(.1)
            reply=self.control('setup')
            if restore is not None:
                cp=reply.get('checkpoint')
                if not cp or cp['payload_sha256']!=restore[1]['payload_sha256']:
                    raise ValueError('Native startup did not restore the exact shared-RAM checkpoint')
            self.checkpoint=reply.get('checkpoint')
            return dict(status='ready',protocol='cuda-mode1-shared-ram',checkpoint=self.checkpoint,
                        owner_epoch=epoch,next_sequence=self.sequence)

        try:
            native=runner or _Native()
            receipt=model_receipt(cfg)
            self.leases.enter_context(native.directory(directory))
            for file in (binary,path(cfg['model']),receipt):self.leases.enter_context(native.read_lease(file))
            if sha256(binary)!=pin:raise ValueError('Native binary changed before its read lease')
            if restore is not None:
                self.restore_lease=restore[0].lease();self.restore_lease.__enter__()
            self.process=Producer(session,command,env,directory,host_bytes=int(cfg['host_budget_gib']*(1<<30)),
                physical_floor=int(cfg['min_available_gib']*(1<<30)),operation_seconds=min(1140,cfg['request_timeout_s']),
                idle_seconds=min(1140,cfg['lease_minutes']*60),native=native,cancelled=cancelled,native_mode=1,startup=startup,owner_epoch=epoch)
            if self.restore_lease is not None:self.restore_lease.__exit__(None,None,None);self.restore_lease=None
            self.launch_seconds=time.monotonic()-self.launch_started
            atomic_json(directory/'launch.json',dict(argv=command,native_sha256=pin,owner_epoch=epoch,
                session=session,context=cfg['context'],split_reads=parts,model_receipt=str(receipt),
                host_budget_bytes=int(cfg['host_budget_gib']*(1<<30)),physical_floor=int(cfg['min_available_gib']*(1<<30)),
                server_auth_token_logged=False))
        except BaseException as exc:
            self.failure_cleanup=getattr(exc,'owned_cleanup',None)
            try:
                if self.process is not None:
                    self.failure_cleanup=self.process.close()
                    exc.owned_cleanup=self.failure_cleanup
                if self.restore_lease is not None and self.failure_cleanup and all(self.failure_cleanup.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
                    self.restore_lease.__exit__(None,None,None);self.restore_lease=None
            finally:self.leases.close()
            raise

    def http(self,suffix,body=None,*,timeout=None,max_bytes=4<<20):
        raw=json.dumps(body,allow_nan=False,separators=(',',':')).encode() if body is not None else None
        request=urllib.request.Request(self.url+suffix,data=raw,headers={
            'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        try:
            with urllib.request.urlopen(request,timeout=timeout or self.cfg['request_timeout_s']) as response:
                data=response.read(max_bytes+1)
                if len(data)>max_bytes:raise ValueError('Native control response exceeded its admitted metadata bound')
                return json.loads(data)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('Native HTTP '+str(exc.code)+': '+exc.read(2048).decode(errors='replace')) from exc

    def control(self,operation,**extra):
        with self.control_lock:
            if self.cancelled():raise RuntimeError('Native request cancelled in its owner epoch')
            row=dict(session=self.session,owner_epoch=self.epoch,sequence=self.sequence,**extra)
            if self.process:self.process.deadline=time.monotonic()+min(1140,self.cfg['request_timeout_s'])
            reply=self.http('/decode/'+operation,row)
            if (reply.get('status'),reply.get('session'),reply.get('owner_epoch'),reply.get('next_sequence'))!=('complete',self.session,self.epoch,self.sequence+1):
                raise ValueError('Native setup/control owner epoch or sequence differs')
            self.sequence+=1
            self.checkpoint=reply.get('checkpoint')
            if self.process:self.process.activity=time.monotonic();self.process.deadline=None
            return reply

    def render(self,body):return self.control('render',request=body)['tokens']
    def snapshot(self):return self.control('checkpoint')['checkpoint']
    def sync(self,tokens):
        cp=self.checkpoint
        return self.control('sync',tokens=tokens,checkpoint_sha256=cp['payload_sha256'] if cp else '')['checkpoint']
    def export(self,ram):
        with ram.lease():
            return self.control('export',mapping=ram.name,capacity=ram.capacity,
                                checkpoint_sha256=self.checkpoint['payload_sha256'])
    def anchor(self,ram):
        with ram.lease():
            return self.control('anchor',mapping=ram.name,capacity=ram.capacity,
                                checkpoint_sha256=self.checkpoint['payload_sha256'])
    def rewind(self,ram,checkpoint):
        with ram.lease():return self.control('rewind',mapping=ram.name,capacity=ram.capacity,
                                             checkpoint_sha256=checkpoint['payload_sha256'])['checkpoint']
    def quality(self,value=None):
        return self.http('/v1/quality-speed',{'value':value} if value is not None else None,timeout=10)
    def probe(self,ram,*,reference):
        with ram.lease():
            return self.control('reference' if reference else 'compare',mapping=ram.name,capacity=ram.capacity,
                                checkpoint_sha256=self.checkpoint['payload_sha256'])

    def close(self):
        if self.closed:return self.receipt
        if not self.process:raise RuntimeError('Native process ownership was not established')
        # All required state has already been exported. Reuse the owned
        # Job's certified shutdown directly; VRAM recovery is then checked
        # independently by release_barrier before another owner can start.
        proof=self.process.close()
        if not all(proof.get(key) for key in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
            raise RuntimeError('Native GPU owner retirement unconfirmed; handoff refused')
        self.receipt=proof;self.closed=True
        if self.restore_lease is not None:self.restore_lease.__exit__(None,None,None);self.restore_lease=None
        (self.directory/'stderr.log').write_bytes(self.process.diagnostic)
        atomic_json(self.directory/'cleanup.json',proof)
        self.leases.close()
        return proof
