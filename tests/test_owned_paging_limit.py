"""Explicit zero must reach both native guardians and retain the common default."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from slstack import guard, shared_guard
from slstack.config import load
from slstack.paging_limits import owned_paging_limit_bytes
from slstack.process import OwnedProcess
from studio.job_paging import OWNED_PAGING_LIMIT, PagingWatch


@pytest.mark.parametrize('value', [0, 4096, OWNED_PAGING_LIMIT, None])
def test_supported_limits(value):
    assert owned_paging_limit_bytes(dict(owned_paging_limit_bytes=value)) == value


@pytest.mark.parametrize('value', [-1, True, False, 0.0, '0', float('nan'), OWNED_PAGING_LIMIT+1])
def test_invalid_limits_refused(value):
    with pytest.raises(ValueError, match='owned_paging_limit_bytes'):
        owned_paging_limit_bytes(dict(owned_paging_limit_bytes=value))


def test_config_refuses_invalid_limit_before_model_resolution():
    with pytest.raises(ValueError, match='owned_paging_limit_bytes'):
        load(overrides={'owned_paging_limit_bytes': -1})


@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('limit', [None, 0, 4096])
def test_both_dispatch_paths_receive_exact_limit(monkeypatch, persistent, limit):
    spec = dict(paging_telemetry=True, timeout_s=None if persistent else 10,
                host_budget_gib=1, min_available_gib=1, receipt='no-such-receipt',
                owned_paging_limit_bytes=limit)
    if not persistent:
        def shared(received):
            assert received['owned_paging_limit_bytes'] == limit
            return 0
        monkeypatch.setattr(shared_guard, 'supervise', shared)
    else:
        monkeypatch.setattr(guard, 'admit', lambda *a: None)
        class Watch:
            def __init__(self, *a, **kw):
                assert kw['owned_paging_limit'] == limit
            def close(self): pass
        monkeypatch.setattr(guard, 'PagingWatch', Watch)
        monkeypatch.setattr(guard, '_supervise_owned', lambda *a: 0)
    assert guard._supervise(spec) == 0


def test_shared_adapter_passes_zero_to_pinned_guard_before_any_child():
    spec = dict(paging_telemetry=True, owned_paging_limit_bytes=0,
                owner_pid=__import__('os').getpid(), owner_created=__import__('psutil').Process().create_time(),
                receipt='unused', host_budget_gib=1, job_commit_gib=1,
                min_available_gib=1, timeout_s=10)
    class ReachedGuard(Exception): pass
    with patch.object(shared_guard.GuardedJob, '__init__', side_effect=ReachedGuard) as init:
        with pytest.raises(ReachedGuard): shared_guard.supervise(spec)
    assert init.call_args.kwargs['owned_paging_limit'] == 0


@pytest.mark.parametrize('supervisor', [guard._supervise, shared_guard.supervise])
def test_cannot_silently_disable_explicit_limit(supervisor):
    with pytest.raises(ValueError, match='requires paging telemetry'):
        supervisor(dict(owned_paging_limit_bytes=0, paging_telemetry=False))


def test_zero_policy_fails_on_first_observed_owned_page():
    observed = {'bytes': 0}
    observer = SimpleNamespace(sample=lambda proc: observed['bytes'])
    watch = PagingWatch(lambda: 0, observer=observer, owned_paging_limit=0)
    limits = dict(rss_limit=10**6, commit_limit=10**6, floor=1)
    assert watch.check(None, 0, 0, **limits) is None
    observed['bytes'] = 4096
    assert 'exceeds 0 MiB allowance' in watch.check(None, 0, 0, **limits)
    assert watch.details()['owned_paging_limit_bytes'] == 0


def test_default_policy_keeps_shared_warning_allowance():
    observer = SimpleNamespace(sample=lambda proc: 4096)
    watch = PagingWatch(lambda: 0, observer=observer)
    assert watch.check(None, 0, 0, rss_limit=10**6, commit_limit=10**6, floor=1) is None
    assert watch.owned_paging_limit == OWNED_PAGING_LIMIT


def test_launcher_forwards_zero_and_enables_telemetry(tmp_path):
    cfg = dict(runtime_root=str(tmp_path), gpu_uuid='GPU-test',
               host_budget_gib=.25, min_available_gib=1, job_commit_gib=.25,
               request_timeout_s=10, owned_paging_limit_bytes=0,
               env_overrides={})
    with patch('slstack.process.admit'), patch('slstack.process.environment', return_value={}), \
         patch('slstack.process.subprocess.Popen', return_value=SimpleNamespace()):
        job = OwnedProcess(cfg, ['unused'], 'limit-spec', scope='studio')
    try:
        spec = json.loads(job.spec.read_text())
        assert spec['owned_paging_limit_bytes'] == 0
        assert spec['paging_telemetry'] is True
    finally:
        job.guard_log.close()


def test_launcher_refuses_explicit_telemetry_off_before_admission():
    with patch('slstack.process.admit', side_effect=AssertionError('must refuse before admission')):
        with pytest.raises(ValueError, match='requires paging telemetry'):
            OwnedProcess({'owned_paging_limit_bytes': 0}, ['unused'], 'refused', telemetry=False)


def test_installation_preserves_explicit_zero():
    from slstack.single_runtime import serving_config
    cfg = dict(host_budget_gib=1, owned_paging_limit_bytes=0)
    installed = serving_config(cfg, runtime_root='runtime', studio_root='studio',
                               llama_producer='producer', library_directories=[], revision='source')
    assert installed['owned_paging_limit_bytes'] == 0


@pytest.mark.parametrize('limit', [0, 4096, None])
def test_normal_studio_proxy_forwards_limit_to_actual_guardian(monkeypatch, tmp_path, limit):
    from slstack import combo_proxy, config
    for name in ('inherit_launch_lease','transient_runtime_config','decode_vram_margin_config'):
        monkeypatch.setattr(combo_proxy, name, lambda cfg: cfg)
    monkeypatch.setattr(guard, 'admit', lambda *a: None)
    monkeypatch.setattr(config, 'environment', lambda *a: {})
    monkeypatch.setenv('DS4_SERVER_AUTH_TOKEN', 'test-secret-'*4)
    received=[]
    monkeypatch.setattr(guard, 'supervise', lambda spec: received.append(spec) or 0)
    cfg=dict(runtime_root=str(tmp_path), host_budget_gib=1.25, job_commit_gib=1.25,
             min_available_gib=1, hold_file=str(tmp_path/'holds'), port=12345,
             owned_paging_limit_bytes=limit, serving_mode='single_engine')
    assert combo_proxy.guarded(cfg, 'unused') == 0
    assert received[0].get('owned_paging_limit_bytes') == limit
    persisted=json.loads(next(tmp_path.glob('server-*.spec.json')).read_text())
    assert persisted.get('owned_paging_limit_bytes') == limit
    assert persisted['paging_telemetry'] is True
    assert persisted['cooperative_stop']['url']=='http://127.0.0.1:12345/v1/stack/shutdown'
