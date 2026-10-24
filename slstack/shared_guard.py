"""Stack adapter around the pinned Studio ownership and paging implementation."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import psutil
from .journal import atomic_json
from .lease import HoldWatcher
from .native_broker import vendor
from .config import ROOT

vendor()
from studio.job_guard import GuardedJob
from studio.job_paging import GLOBAL_PAGING_LIMIT

GiB = 2**30


class CleanupUnconfirmed(RuntimeError):
    owned_cleanup_uncertain = True


def supervise(spec):
    for name, pin in {'job_guard.py':'56d96b556f6b95ba72054100bc8c27a70937cba83a15577a476a1fd1cff87648',
                      'job_paging.py':'205f347b92be699b4f72863942568cc08d534ac19e8b7b154cad2521d6d2c120'}.items():
        source = (ROOT/'vendor/studio/studio'/name).read_bytes().split(b'\n',2)[2].replace(b'\r\n',b'\n')
        if hashlib.sha256(source).hexdigest() != pin:
            raise RuntimeError('Pinned shared guard source changed: ' + name)
    watch = spec.get('hold_watch')
    holds = HoldWatcher(watch['path'],watch['identity'],mac=watch.get('mac',False),task=watch.get('task','DS4STACK')) if watch else None
    parent = psutil.Process(spec['owner_pid'])
    if abs(parent.create_time()-spec['owner_created']) > .01:
        raise RuntimeError('Owning process identity changed before launch')
    class Adapter(GuardedJob):
        def _priority(self):
            super()._priority()
            if Path(spec['stop']).exists():
                raise RuntimeError('stop requested')
            if not parent.is_running() or abs(parent.create_time()-spec['owner_created']) > .01:
                raise RuntimeError('owning process exited')
            if holds and (reason := holds.reason()):
                raise RuntimeError('Yielding to shared hold: ' + reason)
    receipt = Path(spec['receipt'])
    certificate = receipt.with_suffix('.shared.json')
    host = spec['host_budget_gib'] * GiB
    commitment = spec['job_commit_gib'] * GiB if os.name == 'nt' else host
    job = Adapter(rss_limit=host,commit_limit=host,floor=spec['min_available_gib']*GiB,
                  device_commit_allowance=max(0,commitment-host) if os.name=='nt' else 0,
                  seconds=spec['timeout_s'],receipt=certificate)
    started = time.monotonic()
    code, reason = 1, None
    try:
        with open(spec['log'],'ab',buffering=0) as log:
            job.start([sys.executable,'-B','-m','slstack.guard_child',spec['_spec_path']],
                      cwd=spec['cwd'],env=spec['env'],stdout=log)
            code = job.wait()
    except BaseException as error:
        reason = str(error)
    finally:
        if 'ended' not in job.record:
            job.close()
        certain = job.child is None or bool(job.record.get('release_allowed'))
        record = {'exit_code':code,'reason':reason,'peak_tree_rss':job.record['peak_rss'],
                  'min_available':job.record['minimum_available'],'elapsed_s':time.monotonic()-started,
                  'host_budget_gib':spec['host_budget_gib'],'floor_gib':spec['min_available_gib'],
                  'job_commit_gib':commitment/GiB,
                  'child_pid':job.child.pid if job.child else None,
                  'shared_guard_commit':'2774636281ab32bf871497f0a8caae23c6b6cdc7',
                  'shared_guard_certificate':str(certificate),'cleanup_confirmed':certain,
                  'paging_policy':job.watcher.details(),
                  'owned_paging':{'attribution':job.watcher.observer.attribution,
                    'observed_paged_bytes':job.watcher.job_paged,
                    'observed_processes':job.watcher.observer.last_processes,
                    'scanned_pages':job.watcher.observer.scanned_pages,
                    'completed_scans':job.watcher.observer.completed_scans},
                  'system_pagefile_pressure':{'maximum_growth_bytes':job.watcher.max_global_growth,
                    'kill_above_bytes':GLOBAL_PAGING_LIMIT,'scope':'global; not process attribution'},
                  'paging_interpretation':'Owned observation is bounded and incremental. Zero observed pages is not an exhaustive zero-I/O proof. Darwin disk-only process attribution is unavailable.'}
        atomic_json(receipt,record)
        if not certain:
            identities = {(item['pid'],item['created']) for item in job.record.get('cleanup',{}).get('identities',[])}
            for file in (Path(spec['runtime_root'])/'state/leases').glob('*.json'):
                row = json.loads(file.read_text())
                if (row.get('guardian_pid'),row.get('guardian_created')) in identities and not row.get('released'):
                    atomic_json(file,{**row,'cleanup_uncertain':True,'cleanup_certificate':str(certificate)})
            raise CleanupUnconfirmed('Owned exit unconfirmed; resource lease must remain held: ' + str(certificate))
    return 0 if code==0 and reason is None else 1
