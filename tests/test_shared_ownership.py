# Shared provider 91986e7dcd61a67d50d2dc32f1fb5373319635f6 ownership tests; production-service-only tests omitted.
from slstack.native_broker import vendor
vendor()
"""CPU fault injection: no real pressure, models, production holds or services."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from studio.job_guard import GuardedJob, OwnedProcesses, ProcessIdentity, priority_holds
from studio.job_paging import GLOBAL_PAGING_LIMIT, PagingWatch


def wait_for(predicate, seconds=8):
    deadline=time.monotonic()+seconds
    while not predicate():
        if time.monotonic()>deadline:
            raise AssertionError('bounded fixture wait expired')
        time.sleep(.02)


def watcher(*, read=None, observe=None, warnings=None):
    observer=SimpleNamespace(sample=observe or (lambda proc:0),attribution='injected CPU fixture')
    return PagingWatch(read or (lambda:0),observer=observer,owned_paging_limit=0,warning=(warnings.append if warnings is not None else None))


def launch(tmp_path, *, name='guard', watch=None, **limits):
    options=dict(rss_limit=300_000_000,commit_limit=300_000_000,floor=2_000_000_000,
        seconds=10,receipt=tmp_path/(name+'.json'),watcher=watch or watcher())
    options.update(limits)
    job=GuardedJob(**options)
    job.start([sys.executable,'-B','-c','import time;time.sleep(60)'])
    wait_for(lambda:len(job.owned.identities)>=2)
    return job


@pytest.mark.parametrize('fault',['floor','sensor','deadline','rss','commit','own_paging','global_paging'])
def test_every_failure_stops_verified_owned_tree_and_certifies_exit(tmp_path,monkeypatch,fault):
    control={'global':0,'own':0,'error':False}
    def read():
        if control['error']:raise OSError('TEST sensor unavailable')
        return control['global']
    job=launch(tmp_path,watch=watcher(read=read,observe=lambda proc:control['own']),
        seconds=.5 if fault=='deadline' else 10)
    try:
        started=time.monotonic()
        if fault=='floor':
            vm=psutil.virtual_memory()
            monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1,total=vm.total))
        elif fault=='sensor':control['error']=True
        elif fault=='rss':job.rss_limit=1
        elif fault=='commit':job.commit_limit=1
        elif fault=='own_paging':control['own']=4096
        elif fault=='global_paging':control['global']=GLOBAL_PAGING_LIMIT+1
        wait_for(job.cancelled.is_set)
        with pytest.raises(RuntimeError):job.wait()
        receipt=json.loads(job.receipt.read_text())
        assert receipt['release_allowed'] and receipt['cleanup']['confirmed']
        assert not receipt['cleanup']['surviving_owned_pids']
        assert all(ProcessIdentity(**row).process() is None for row in receipt['cleanup']['identities'])
        assert time.monotonic()-started<6
    finally:job.close()


def test_small_global_paging_warns_live_without_claiming_attribution(tmp_path):
    value={'bytes':0};warnings=[]
    job=launch(tmp_path,watch=watcher(read=lambda:value['bytes'],warnings=warnings))
    try:
        value['bytes']=15_000_000
        wait_for(lambda:any('global' in text for text in warnings))
        assert not job.cancelled.is_set()
        assert job.watcher.job_paged==0
        assert any('global' in text for text in warnings)
    finally:job.close()


def test_mid_lease_priority_interrupts_even_while_api_caller_is_stalled(tmp_path):
    ledger=tmp_path/'holds.md';ledger.write_text('')
    job=launch(tmp_path,ledger=ledger,priority_owners=['KIT11FIX','DS4COMBO'])
    try:
        started=time.monotonic()
        ledger.write_text('ACTIVE: KIT11FIX TEST manual build\n')
        # A stalled API caller does not run any guard callbacks.
        time.sleep(.5)
        assert job.cancelled.is_set()
        with pytest.raises(RuntimeError,match='HIGHER PRIORITY'):job.wait()
        assert time.monotonic()-started<6
        assert json.loads(job.receipt.read_text())['release_allowed']
    finally:job.close()


def test_foreign_later_server_like_process_is_never_adopted_or_signalled(tmp_path):
    job=launch(tmp_path)
    foreign=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)',
        'TEST foreign ds4-server'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        time.sleep(.25)
        job.close()
        assert foreign.poll() is None
        receipt=json.loads(job.receipt.read_text())
        assert foreign.pid not in [row['pid'] for row in receipt['cleanup']['identities']]
        assert receipt['release_allowed']
    finally:
        foreign.terminate();foreign.wait(timeout=5)
        job.close()


def test_reused_pid_never_signals_replacement(tmp_path,monkeypatch):
    job=launch(tmp_path)
    try:
        identity=next(iter(job.owned.identities.values()))
        replacement=dataclasses.replace(identity,created=identity.created+100)
        signals=[]
        real=psutil.Process
        fake=SimpleNamespace(pid=identity.pid,create_time=lambda:replacement.created,
            exe=lambda:replacement.executable,is_running=lambda:True,status=lambda:'running',
            terminate=lambda:signals.append('TERM'),kill=lambda:signals.append('KILL'))
        with monkeypatch.context() as patch:
            patch.setattr(psutil,'Process',lambda pid:fake if pid==identity.pid else real(pid))
            assert identity.process() is None
            owned=object.__new__(OwnedProcesses);owned.identities={identity.pid:identity}
            assert owned.stop()['confirmed']
        assert signals==[]
    finally:job.close()


def test_unreadable_ledger_cancels_and_cleans_without_releasing_early(tmp_path):
    ledger=tmp_path/'holds.md';ledger.write_text('')
    job=launch(tmp_path,ledger=ledger,priority_owners=['KIT11FIX'])
    try:
        # Move the exact fixture aside; do not delete it.
        ledger.rename(tmp_path/'holds-preserved.md')
        wait_for(job.cancelled.is_set)
        with pytest.raises(RuntimeError,match='FileNotFoundError'):job.wait()
        assert json.loads(job.receipt.read_text())['cleanup']['confirmed']
    finally:job.close()


def test_cleanup_uncertainty_keeps_lease_held(tmp_path,monkeypatch):
    job=launch(tmp_path)
    job.close()
    monkeypatch.setattr(job.owned,'stop',lambda:{'confirmed':False,'errors':['TEST denied exit certificate'],
        'surviving_owned_pids':[999999],'identities':[]})
    job.close()
    assert not json.loads(job.receipt.read_text())['release_allowed']


def test_success_also_retires_background_descendants_between_polls(tmp_path):
    pid_file=tmp_path/'background-pid.json'
    code=('import json,pathlib,subprocess,sys; '
          'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
          'pathlib.Path(sys.argv[1]).write_text(str(p.pid))')
    job=GuardedJob(rss_limit=300_000_000,commit_limit=300_000_000,floor=2_000_000_000,
        seconds=10,receipt=tmp_path/'background.json',watcher=watcher())
    job.start([sys.executable,'-B','-c',code,str(pid_file)])
    try:
        assert job.wait()==0
        pid=int(pid_file.read_text())
        try:
            process=psutil.Process(pid)
            assert not process.is_running() or process.status()==psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:pass
        assert json.loads(job.receipt.read_text())['release_allowed']
    finally:job.close()


def test_lease_parser_preserves_manual_hold_and_separates_machines():
    lines=['ACTIVE: KIT11FIX manual build; 2026-10-23T00:00:00Z',
           'MAC ACTIVE: DS4COMBO '+'a'*32+'; expires 2020-01-01T00:00:00Z',
           'ACTIVE: DS4COMBO '+'b'*32+'; window',
           'RELEASED: DS4COMBO '+'b'*32+'; done']
    assert priority_holds('\n'.join(lines),['KIT11FIX','DS4COMBO'])==lines[:1]
    assert priority_holds('\n'.join(lines),['DS4COMBO'],resource='MAC')==lines[1:2]
    assert priority_holds('\n'.join(lines+['RELEASED: KIT11FIX done']),['KIT11FIX'])==[]


def test_unavailable_mac_disk_attribution_is_explicit(monkeypatch):
    from studio.job_paging import JobPaging
    with monkeypatch.context() as patch:
        patch.setattr(sys,'platform','darwin')
        observer=JobPaging()
        assert observer.attribution.startswith('UNAVAILABLE:')
        assert 'compression' in observer.attribution
        assert observer.sample(psutil.Process())==0


def test_canonical_cleared_record_releases_only_its_lease():
    first='a'*32
    second='b'*32
    active='ACTIVE: KIT11FIX '+second+'; still owns GPU'
    text='\n'.join(['ACTIVE: KIT11FIX '+first+'; earlier window', active,
                    '  CLEARED: KIT11FIX '+first+'; done'])
    assert priority_holds(text,['KIT11FIX'])==[active]






@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,0])
def test_invalid_budget_fails_before_launch(tmp_path,value):
    with pytest.raises(ValueError):GuardedJob(rss_limit=value,commit_limit=1,floor=1,
        seconds=1,receipt=tmp_path/'not-created.json')
    assert not (tmp_path/'not-created.json').exists()


@pytest.mark.skipif(os.name!='nt',reason='WDDM backing is Windows-only')
def test_device_backing_does_not_raise_physical_host_budget_or_floor(tmp_path):
    job=launch(tmp_path,device_commit_allowance=32_000_000_000)
    try:
        assert job.record['host_commit_limit']==300_000_000
        assert job.record['commit_limit']==32_300_000_000
        job.rss_limit=1
        wait_for(job.cancelled.is_set)
        with pytest.raises(RuntimeError,match='RAM budget'):job.wait()
        assert json.loads(job.receipt.read_text())['release_allowed']
    finally:job.close()


def test_service_handoff_requires_owned_identity_and_leaves_healthy_service_alive(tmp_path):
    job=launch(tmp_path)
    foreign=psutil.Process()
    try:
        with pytest.raises(RuntimeError,match='exact owned'):
            job.handoff(foreign.pid)
        service=next(identity for pid,identity in job.owned.identities.items() if pid!=job.child.pid)
        receipt=job.handoff(service.pid)
        assert receipt['status']=='HEALTHY_SERVICE_HANDOFF'
        assert receipt['release_allowed'] and service.process() is not None
        assert not job.monitor.is_alive()
        if os.name=='nt':
            assert receipt['windows_startup_limits_released']
            assert job.kernel_job is None
    finally:
        job.close() # exact fixture launch, not a production service


def test_failed_startup_cannot_be_handed_off(tmp_path):
    job=launch(tmp_path)
    try:
        job.failure='TEST startup failed'
        with pytest.raises(RuntimeError,match='failed startup'):
            job.handoff(job.child.pid)
    finally:job.close()


def test_exec_image_change_keeps_owned_birth_identity(monkeypatch):
    identity=ProcessIdentity(12345,123.5,'launcher')
    process=SimpleNamespace(pid=12345,create_time=lambda:123.5,exe=lambda:'running-image',
        is_running=lambda:True,status=lambda:'running')
    monkeypatch.setattr(psutil,'Process',lambda pid:process)
    assert identity.process() is process
    assert identity.same_process(ProcessIdentity.read(process))
    assert not identity.same_process(dataclasses.replace(identity,created=124.5))


@pytest.mark.skipif(sys.platform!='darwin',reason='Apple framework launcher re-exec')
def test_framework_python_bootstrap_keeps_ownership(tmp_path):
    job=GuardedJob(rss_limit=300_000_000,commit_limit=300_000_000,floor=32_000_000_000,
        seconds=10,receipt=tmp_path/'framework.json')
    try:
        job.start([sys.executable,'-B','-c','import time;time.sleep(.3)'])
        assert job.wait()==0
        assert job.record['peak_rss']>0
        assert job.record['cleanup']['confirmed'] and job.record['release_allowed']
    finally:job.close()
