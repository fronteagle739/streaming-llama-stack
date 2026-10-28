"""Cancel an owned single-engine tree without terminating a GPU process.

The memory/paging verdict remains a failure. A stalled retirement retains its
ownership certificate instead of falling back to terminate or kill-on-close.
"""
import dataclasses
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import psutil

from .progress import Window, activity
from .native_broker import vendor
vendor()
from studio.job_guard import OwnedProcesses


def validate(policy):
    url=urllib.parse.urlsplit(policy['url'])
    if (url.scheme!='http' or url.hostname not in ('127.0.0.1','::1','localhost') or
            url.username or url.password or url.path!='/v1/stack/shutdown' or url.query or url.fragment):
        raise ValueError('Cooperative stop must use the local stack shutdown endpoint')


class CooperativeOwnedProcesses(OwnedProcesses):
    def __init__(self,child,spec):
        super().__init__(child)
        self.spec=spec

    def stop(self,grace=30,timeout=None):
        errors=[];request_errors=[];done=threading.Event()
        try:self.refresh()
        except Exception as error:errors.append(type(error).__name__)
        root=Path(self.spec['runtime_root']).resolve()
        # These directories belong to this guardian. Never touch another
        # runtime or follow a link outside the owned root.
        for pattern in ('ultra-*','single-engine/ultra-*'):
            for directory in root.glob(pattern):
                if directory.is_dir() and directory.resolve().is_relative_to(root):
                    for name in ('guard.abort','decode.cancel'):
                        try:(directory/name).touch(exist_ok=True)
                        except OSError as error:request_errors.append(type(error).__name__)
        explicit=self.spec['cooperative_stop'].get('abort_file')
        if explicit:
            try:Path(explicit).touch(exist_ok=True)
            except OSError as error:request_errors.append(type(error).__name__)
        policy=self.spec['cooperative_stop']
        token=self.spec.get('env',{}).get('DS4_SERVER_AUTH_TOKEN','')
        def request():
            # Startup can race binding the local endpoint. Bounded retries
            # do not generate artificial progress in the retirement window.
            for _ in range(3):
                if done.is_set():return
                try:
                    headers={'Content-Type':'application/json'}
                    if token:headers['Authorization']='Bearer '+token
                    req=urllib.request.Request(policy['url'],data=b'{}',headers=headers)
                    with urllib.request.urlopen(req,timeout=3) as response:response.read(4096)
                    return
                except OSError as error:request_errors.append(type(error).__name__)
                done.wait(.2)
        worker=threading.Thread(target=request,daemon=True);worker.start()
        window=Window(float(policy.get('stall_seconds',grace)))
        try:
            while True:
                try:live=self.refresh()
                except Exception as error:errors.append(type(error).__name__);break
                if not live:break
                # Controller HTTP retries/heartbeats are not model progress.
                signal=[]
                for p in live:
                    try:
                        model=p.name().lower()=='llama-dsv41-prefill.exe'
                        signal.append((p.pid,p.create_time(),activity(p.pid,self.spec['log']) if model else None))
                    except psutil.NoSuchProcess:pass
                    except Exception as error:errors.append(type(error).__name__)
                window.observe(tuple(signal))
                if window.expired():break
                time.sleep(.05)
        finally:done.set();worker.join(timeout=.1)
        try:survivors=[p.pid for p in self.live()]
        except Exception as error:errors.append(type(error).__name__);survivors=list(self.identities)
        return dict(confirmed=not survivors and not errors,surviving_owned_pids=survivors,errors=errors,
                    identities=[dataclasses.asdict(x) for x in self.identities.values()],
                    retired_identities=[dataclasses.asdict(x) for x in self.retired_identities],
                    forced_kill=False,cooperative=True,cooperative_request_errors=request_errors)
