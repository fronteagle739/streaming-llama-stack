"""A released coordinator reservation must permit the exact leased proxy writer."""
from types import SimpleNamespace

import pytest

from scripts.qfn_fp4_proxy import check_runtime
from slstack import lease


@pytest.fixture
def proxy_args(tmp_path, monkeypatch):
    monkeypatch.setattr(lease, 'read_hold_text', lambda path: 'fixture')
    monkeypatch.setattr(lease, 'active_holds', lambda text: ['ACTIVE: QFNFP4A fixture'])
    return SimpleNamespace(runtime=tmp_path, timing_marker=tmp_path/'root-timing.pending',
                           hold_file=tmp_path/'hold.md')


@pytest.mark.parametrize('text', ['RELEASED done\n', '\ufeffRELEASED completed\n'])
def test_released_marker_permits_exact_exclusive_lease(proxy_args, text):
    proxy_args.timing_marker.write_text(text, encoding='utf8')
    check_runtime(proxy_args, heavy=True)


@pytest.mark.parametrize('text', ['', 'ACTIVE measurement\n', 'UNKNOWN\n', 'RELEASED_LATER\n'])
def test_unreleased_or_unknown_marker_fails_closed(proxy_args, text):
    proxy_args.timing_marker.write_text(text, encoding='utf8')
    with pytest.raises(InterruptedError, match='Coordinator timing marker active'):
        check_runtime(proxy_args, heavy=True)


def test_release_does_not_bypass_foreign_hold(proxy_args, monkeypatch):
    proxy_args.timing_marker.write_text('RELEASED\n', encoding='utf8')
    monkeypatch.setattr(lease, 'active_holds', lambda text: ['ACTIVE: FOREIGN fixture'])
    with pytest.raises(RuntimeError, match='Missing exclusive QFNFP4A'):
        check_runtime(proxy_args, heavy=True)


def test_no_marker_still_requires_single_exact_lease(proxy_args, monkeypatch):
    monkeypatch.setattr(lease, 'active_holds', lambda text: [])
    with pytest.raises(RuntimeError, match='Missing exclusive QFNFP4A'):
        check_runtime(proxy_args, heavy=True)
