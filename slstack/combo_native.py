"""Owned native HTTP runtime; reuse Studio's suspended Job and memory watcher."""
from __future__ import annotations
from copy import deepcopy
from contextlib import ExitStack
import json
import hashlib
import os
import re
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request
from .config import path,environment
from .native_broker import vendor,model_receipt
from .journal import atomic_json,sha256
from .combo_release import certified
from .combo_ram import protect_working_set
from . import combo_abort


def global_hotlist(cfg):
    """Per-model, per-cache-budget expert hotlist shared by every session (a list
    written under another expert budget would fail the engine's full-restore check)."""
    command=[str(v) for v in cfg['ds4_command']]
    def value(flag,default):return command[command.index(flag)+1] if flag in command else default
    budget=value('--ssd-streaming-cache-experts','default')
    host=value('--ram-streaming-cache-experts','')   # VRAM and host tiers both bound the list
    model=cfg['combo_identity']['model_sha256'][:16]
    key=re.sub(r'[^0-9A-Za-z]','',budget)+('-ram'+re.sub(r'[^0-9A-Za-z]','',host) if host else '')
    return Path(cfg['runtime_root'])/('expert-hotlist-'+model+'-'+key+'.txt')


def publish_hotlist(session_list,global_list):
    """After a decoder exits: a live list it wrote becomes the global hotlist (atomic)."""
    if session_list is None or not Path(session_list).exists():return False
    data=Path(session_list).read_bytes()
    if b'source=live' not in data.split(b'\n',1)[0]:return False
    tmp=Path(str(global_list)+'.tmp');tmp.write_bytes(data);os.replace(tmp,global_list);return True


def global_frequency(cfg):
    """Per-model, per-budget routed-lookup counts accumulated over decoders."""
    return global_hotlist(cfg).with_name(global_hotlist(cfg).name.replace('expert-hotlist-','expert-frequency-'))


def _read_frequency(path):
    lines=Path(path).read_text().splitlines()
    head=re.match(r'# ds4-expert-frequency v1 model_size=(\d+) vram_configured=(\d+) host_configured=(\d+)',lines[0])
    if not head:raise ValueError('bad expert frequency header in '+str(path))
    rows={}
    for line in lines[1:]:
        kind,layer,expert,count,gate,down=line.split()
        if kind!='f':raise ValueError('bad expert frequency row in '+str(path))
        rows[(int(layer),int(expert))]=[float(count),int(gate),int(down)]
    return tuple(int(v) for v in head.groups()),rows


def merge_frequency(session_freq,global_freq,decay=0.5):
    """After a decoder exits: global = decay * global + this decoder's counts (atomic)."""
    if session_freq is None or not Path(session_freq).exists():return False
    head,new=_read_frequency(session_freq)
    old={}
    if Path(global_freq).exists():
        old_head,old=_read_frequency(global_freq)
        if old_head[0]!=head[0]:old={}   # another model file: start again
    for key,(count,gate,down) in new.items():
        prior=old.get(key,[0.0,gate,down])[0]*decay
        old[key]=[prior+count,gate,down]
    for key in set(old)-set(new):old[key][0]*=decay
    text='# ds4-expert-frequency v1 model_size=%d vram_configured=%d host_configured=%d experts=%d\n'%(*head,len(old))
    text+=''.join('f %d %d %.3f %d %d\n'%(l,e,c,g,d) for (l,e),(c,g,d) in sorted(old.items()))
    tmp=Path(str(global_freq)+'.tmp');tmp.write_text(text);os.replace(tmp,global_freq);return True


def choose_restore(source,session_file,global_freq,seed):
    """(restore_source, file, seed_counts). A list a real decode of this session wrote
    ('session') wins; the frequency seed only replaces a global or missing list."""
    if source in ('global','none'):
        built=build_frequency_seed(global_freq,seed)
        if built:return 'frequency',str(seed),built
    return source,session_file,None


def build_frequency_seed(global_freq,out):
    """Restore list (engine v1 format) from the global counts: top vram_configured
    experts -> VRAM, the next host_configured -> host tier. Listed least-frequent
    first, because the engine gives later rows higher LRU priority."""
    if global_freq is None or not Path(global_freq).exists():return None
    (model_size,nv,nh),rows=_read_frequency(global_freq)
    ranked=sorted(rows.items(),key=lambda kv:(-kv[1][0],kv[0]))
    v=ranked[:nv];h=ranked[nv:nv+nh]
    if not v:return None
    text='# ds4-expert-cache-snapshot v1 model_size=%d source=frequency vram=%d host=%d\n'%(model_size,len(v),len(h))
    text+=''.join('h %d %d %d %d\n'%(l,e,g,d) for (l,e),(c,g,d) in reversed(h))
    text+=''.join('v %d %d %d %d\n'%(l,e,g,d) for (l,e),(c,g,d) in reversed(v))
    Path(out).write_text(text);return (len(v),len(h))


class LiveLog:
    """Mirror a producer's in-RAM diagnostic to stderr.log as it grows.

    The vendored Producer keeps the native's stdout/stderr only in memory and
    the owner writes it out at close, so a hard kill of this harness lost the
    whole log. This thread appends each new chunk (under the producer's own
    state lock) so a killed run keeps everything up to the last interval.
    close() still rewrites the file with the full buffer.
    """
    def __init__(self,owner,file,interval=0.25):
        self.owner,self.file,self.interval=owner,Path(file),interval
        self.written=0;self.stop_event=threading.Event()
        self.file.write_bytes(b'')
        self.thread=threading.Thread(target=self._run,name='ds4combo-native-log',daemon=True);self.thread.start()

    def flush(self):
        process=self.owner.process
        if process is None:return
        lock=getattr(process,'state_lock',None)
        with (lock or threading.Lock()):
            data=bytes(memoryview(process.diagnostic)[self.written:])
        if data:
            with open(self.file,'ab') as out:out.write(data)
            self.written+=len(data)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:self.flush()
            except Exception as exc:
                print('DS4COMBO native live log failed: '+repr(exc),flush=True);return

    def stop(self):
        self.stop_event.set();self.thread.join(5)


class Native:
    def __init__(self,cfg,session,epoch,sequence,cancelled,*,restore=None,runner=None,expert_list=None):
        combo_abort.check('native launch')
        vendor()
        from studio.dsv41_session_process import Producer
        from studio.dsv41_windows_runner import _Native
        self.cfg=deepcopy(cfg);self.session=session;self.epoch=epoch;self.sequence=sequence
        self.control_lock=threading.Lock();self.process=None;self.closed=False
        self.creation_attempted=False
        self.leases=ExitStack()
        self.restore_lease=None
        self.expert_list=None;self.global_list=None;self.restore_source=None;self.hotlist_published=False
        self.global_freq=None;self.seed_counts=None;self.launched_at=time.time()
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
        if expert_list is not None:
            # Expert-cache restore: every checkpoint/export rewrites this LRU-ordered resident
            # list; a relaunch seeds it before its first decode token (engine fails loudly if
            # it cannot be restored in full).
            env_values['DS4_COMBO_EXPERT_SNAPSHOT_FILE']=str(expert_list)
            # Cross-session hotlist: a decoder with no session list starts from the
            # last list any decoder of this model and cache budget wrote, never cold.
            self.expert_list=Path(expert_list);self.global_list=global_hotlist(cfg)
            seeded=Path(str(self.expert_list)+'.from-global')
            if self.expert_list.exists():
                # A list this session only passed on (no decode yet) is still the global one.
                data=self.expert_list.read_bytes();head=data.split(b'\n',1)[0]
                unchanged=seeded.exists() and seeded.read_text()==hashlib.sha256(data).hexdigest()
                self.restore_source='global' if seeded.exists() and (unchanged or b'source=inherited' in head) else 'session'
            elif self.global_list.exists():
                data=self.global_list.read_bytes();self.expert_list.write_bytes(data)
                seeded.write_text(hashlib.sha256(data).hexdigest());self.restore_source='global'
            else:self.restore_source='none'
            # Pre-seed from usage frequency (whole decodes, all sessions of this model and
            # budget): VRAM tier gets the top experts, the host tier the next ones.
            self.global_freq=global_frequency(cfg);restore_file=str(expert_list)
            seed=Path(str(self.expert_list)+'.seed')
            self.restore_source,restore_file,built=choose_restore(self.restore_source,restore_file,self.global_freq,seed)
            if built:self.seed_counts=built
            print('DS4COMBO_EXPERT_RESTORE_SOURCE restore_source='+self.restore_source+' list='+restore_file+
                  (' vram=%d host=%d'%built if built else ''),flush=True)
            if self.restore_source!='none':env_values['DS4_COMBO_EXPERT_RESTORE_FILE']=restore_file
        self.cfg.setdefault('env_overrides',{}).update(env_values)
        env=environment(self.cfg,'ds4-win')
        parts=cfg.get('ds_read_parts',8)
        if type(parts) is not int or not 1<=parts<=64:raise ValueError('Explicit native split reads must be in1..64')
        env['DS4_DS_READ_PARTS']=str(parts)
        libraries=[str(path(p)) for p in cfg.get('producer_library_directories',[])]
        env['PATH']=os.pathsep.join([str(binary.parent),*libraries,env.get('PATH','')])
        directory=Path(cfg['runtime_root'])/('native-'+secrets.token_hex(8));directory.mkdir(parents=True,exist_ok=False)
        self.directory=directory;self.launch_started=time.monotonic()
        self.live_log=LiveLog(self,directory/'stderr.log')

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
            self.creation_attempted=True
            self.process=Producer(session,command,env,directory,host_bytes=int(cfg['host_budget_gib']*(1<<30)),
                physical_floor=int(cfg['min_available_gib']*(1<<30)),operation_seconds=min(1140,cfg['request_timeout_s']),
                idle_seconds=min(1140,cfg['lease_minutes']*60),native=native,cancelled=cancelled,native_mode=1,startup=startup,owner_epoch=epoch,
                after_resume=lambda job:protect_working_set(job,int(cfg['host_budget_gib']*(1<<30)),
                    int(cfg['min_available_gib']*(1<<30)),int(cfg['aggregate_host_budget_gib']*(1<<30)),
                    receipt=directory/'resident-admission.json'))
            if self.restore_lease is not None:self.restore_lease.__exit__(None,None,None);self.restore_lease=None
            self.launch_seconds=time.monotonic()-self.launch_started
            atomic_json(directory/'launch.json',dict(argv=command,native_sha256=pin,owner_epoch=epoch,
                session=session,context=cfg['context'],split_reads=parts,model_receipt=str(receipt),
                host_budget_bytes=int(cfg['host_budget_gib']*(1<<30)),physical_floor=int(cfg['min_available_gib']*(1<<30)),
                resident_admission=getattr(self.process,'resume_admission',None),server_auth_token_logged=False))
        except BaseException as exc:
            self.failure_cleanup=getattr(exc,'owned_cleanup',None)
            if self.process is not None:
                try:
                    self.failure_cleanup=self.process.close()
                except BaseException as cleanup:exc.owned_cleanup_error=repr(cleanup)
            if self.failure_cleanup is not None:exc.owned_cleanup=self.failure_cleanup
            self.live_log.stop()
            try:
                full=getattr(self.process,'diagnostic',None) if self.process is not None else getattr(exc,'owned_diagnostic',None)
                if full:(directory/'stderr.log').write_bytes(bytes(full))
            except Exception as failure:print('DS4COMBO native stderr.log not saved: '+repr(failure),flush=True)
            if not self.creation_attempted or certified(self.failure_cleanup):
                if self.restore_lease is not None:self.restore_lease.__exit__(None,None,None);self.restore_lease=None
                self.leases.close()
            else:
                exc.owned_cleanup=self.failure_cleanup or dict(creation_attempted=True)
                exc.owned_runtime=self # Keep file/RAM leases until the outer Job closes.
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
        if operation!='shutdown':combo_abort.check('native '+operation)
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

    def render(self,body):
        reply=self.control('render',request=body)
        # prefix_source: memory-text = live sampled IDs kept, only the suffix tokenized.
        self.last_render=reply.get('render');return reply['tokens']
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

    def expert_restore(self):
        """Restore timing from this decoder's log (after close); count 0 when none was pending."""
        return dict(self._expert_restore(),expert_restore_source=self.restore_source,expert_seed_counts=self.seed_counts)

    def _expert_restore(self):
        raw=bytes(getattr(self.process,'diagnostic',b'') or b'')
        failed=re.search(rb'DS4COMBO_EXPERT_RESTORE_FAILED [^\r\n]*',raw)
        if failed:raise RuntimeError('Native expert-cache restore failed: '+failed.group(0).decode(errors='replace'))
        rows=re.findall(rb'DS4COMBO_EXPERT_RESTORE experts=(\d+) vram=(\d+) host=(\d+) seconds=([0-9.]+)',raw)
        pending=re.search(rb'DS4COMBO_EXPERT_RESTORE_PENDING vram=(\d+) host=(\d+)',raw)
        if len(rows)>1:raise ValueError('Multiple expert-cache restores in one native process')
        if pending and not rows and int(pending.group(1))+int(pending.group(2)):
            return dict(expert_restore_count=0,expert_restore_seconds=0.0,expert_restore_applied=False)
        if not rows:return dict(expert_restore_count=0,expert_restore_seconds=0.0,expert_restore_applied=None)
        n,v,h,sec=rows[0]
        return dict(expert_restore_count=int(n),expert_restore_vram=int(v),expert_restore_host=int(h),
                    expert_restore_seconds=float(sec),expert_restore_applied=True)

    def close(self):
        if self.closed:return self.receipt
        if not self.process:raise RuntimeError('Native process ownership was not established')
        # All required state has already been exported. Reuse the owned
        # Job's certified shutdown directly; VRAM recovery is then checked
        # independently by release_barrier before another owner can start.
        # Studio-style exit (always): ask the engine to drain DirectStorage,
        # free its GPU/host memory and return from main. The Job terminate in
        # process.close() is only a logged fallback (the kill path leaked RAM).
        self.shutdown_path='terminate: decoder already exited'
        exited=None
        if self.process.job is not None and self.process.job.poll() is None:
            started=time.monotonic()
            try:
                self.control('shutdown')
                # The budget starts once the reply has arrived.
                deadline=time.monotonic()+float(self.cfg.get('native_graceful_shutdown_seconds',30))
                while (exited:=self.process.job.poll()) is None and time.monotonic()<deadline:time.sleep(0.05)
                self.shutdown_path='graceful-pending' if exited is not None else 'graceful-timeout-terminate'
            except BaseException as exc:
                self.shutdown_path='graceful-failed-terminate: '+repr(exc)[:200]
            self.shutdown_seconds=time.monotonic()-started
        # KILL PATH: the decoder is still alive here, so process.close() terminates it
        # (the leak-prone retire). Counted in the receipt and logged loudly; behaviour unchanged.
        kill_path=bool(self.process.job is not None and self.process.job.poll() is None)
        if kill_path:print('DS4COMBO KILL PATH native: terminating a live decoder ('+self.shutdown_path+')',flush=True)
        proof=self.process.close()
        if self.shutdown_path=='graceful-pending':
            # Exiting before the deadline is not enough: a crash in teardown
            # (DirectStorage drain abort, cudaFree fault, ...) also exits. Only a
            # zero exit code with the completion line on stderr (read after
            # close, so it has flushed) counts as graceful.
            done=re.search(rb'DS4COMBO_NATIVE_GRACEFUL_SHUTDOWN_COMPLETE [^\r\n]*',bytes(self.process.diagnostic or b''))
            self.shutdown_path='graceful' if exited==0 and done else 'graceful-crashed exit='+str(exited)
            if done:self.shutdown_detail=done.group(0).decode(errors='replace')
        if self.shutdown_path!='graceful':
            print('DS4COMBO native shutdown fell back: '+self.shutdown_path,flush=True)
        proof=dict(proof,kill_path=kill_path,native_shutdown_path=self.shutdown_path,native_shutdown_seconds=getattr(self,"shutdown_seconds",0.0),native_shutdown_detail=getattr(self,"shutdown_detail",None))
        if not all(proof.get(key) for key in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
            raise RuntimeError('Native GPU owner retirement unconfirmed; handoff refused')
        self.receipt=proof;self.closed=True
        if self.restore_lease is not None:self.restore_lease.__exit__(None,None,None);self.restore_lease=None
        if getattr(self,'global_list',None) is not None:
            self.hotlist_published=publish_hotlist(self.expert_list,self.global_list)
            proof['expert_hotlist_published']=self.hotlist_published
            # Only counts this process wrote (a non-decoding process leaves an older file).
            freq=Path(str(self.expert_list)+'.freq')
            fresh=freq.exists() and freq.stat().st_mtime>=getattr(self,'launched_at',0)
            proof['expert_frequency_merged']=bool(fresh and merge_frequency(freq,self.global_freq))
        try:
            live=getattr(self,'live_log',None)
            if live is not None:live.stop()
            (self.directory/'stderr.log').write_bytes(self.process.diagnostic)
            atomic_json(self.directory/'cleanup.json',proof)
        finally:self.leases.close()
        return proof
