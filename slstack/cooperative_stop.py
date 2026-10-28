"""Single-engine abort files and HTTP shutdown before the common retirement wait."""
from pathlib import Path
import threading
import urllib.parse
import urllib.request

from .shutdown import ProgressOwnedProcesses


def validate(policy):
    url=urllib.parse.urlsplit(policy['url'])
    if (url.scheme!='http' or url.hostname not in ('127.0.0.1','::1','localhost') or
            url.username or url.password or url.path!='/v1/stack/shutdown' or url.query or url.fragment):
        raise ValueError('Cooperative stop must use the local stack shutdown endpoint')


class CooperativeOwnedProcesses(ProgressOwnedProcesses):
    def stop(self,grace=180,timeout=10):
        # Inner producers have two retirement windows (30 + 90 seconds).
        # Give their controllers time to finish those before an outer fallback.
        return super().stop(grace=grace,timeout=timeout)

    def request(self,members,done,request_errors):
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
        threading.Thread(target=request,daemon=True).start()
