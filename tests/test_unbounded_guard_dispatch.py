from slstack import guard, shared_guard


def test_mac_host_budget_does_not_inherit_windows_commit_cap():
    from slstack.process import commit_budget_gib
    cfg=dict(host_budget_gib=400,job_commit_gib=48)
    assert commit_budget_gib(cfg,windows=False)==400
    assert commit_budget_gib(cfg,windows=True)==48


def test_graceful_cleanup_retains_live_owner_without_killing():
    from types import SimpleNamespace
    survivor=SimpleNamespace(pid=123,terminate=lambda:None,kill=lambda:(_ for _ in ()).throw(AssertionError('SIGKILL forbidden')))
    owned=object.__new__(guard.GracefulOwnedProcesses)
    owned.identities={};owned.retired_identities=[]
    owned.refresh=lambda:None;owned.live=lambda:[survivor]
    receipt=owned.stop(grace=0)
    assert receipt['surviving_owned_pids']==[123] and not receipt['confirmed']
    assert receipt['forced_kill'] is False


def test_persistent_server_retains_owned_memory_and_paging_guard(monkeypatch):
    calls=[]
    class Watch:
        def __init__(self,*a,**k): calls.append('paging-start')
        def close(self): calls.append('paging-close')
    monkeypatch.setattr(guard,'PagingWatch',Watch)
    monkeypatch.setattr(guard,'admit',lambda *a: calls.append('admitted'))
    monkeypatch.setattr(guard,'_supervise_owned',lambda spec,*a: calls.append(('owned',spec['timeout_s'])) or 0)
    monkeypatch.setattr(shared_guard,'supervise',lambda spec: (_ for _ in ()).throw(AssertionError('fixed lifetime adapter selected')))
    assert guard._supervise(dict(paging_telemetry=True, timeout_s=None, host_budget_gib=400,
                                 min_available_gib=4,receipt='nonexistent-receipt'))==0
    assert calls==['admitted','paging-start',('owned',None),'paging-close']
