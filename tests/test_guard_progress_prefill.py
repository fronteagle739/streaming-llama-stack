"""Real CPU-only fake engines exercise the shipping guard adapter, without models."""
import os
import json
import sys
import uuid

from slstack.config import load
from slstack.process import OwnedProcess


def run_fake(source):
    cfg = load(overrides=dict(host_budget_gib=.5, min_available_gib=2,
                              job_commit_gib=.5, paging_telemetry=True))
    cfg['runtime_root'] = os.environ['STUDIO_RUNTIME_ROOT']
    job = OwnedProcess(cfg, [sys.executable, '-B', '-c', source],
                       'fake-prefill-' + uuid.uuid4().hex, timeout=.8)
    try:
        try:
            return job.wait()
        except RuntimeError:
            if not job.receipt.exists():
                raise
            return json.loads(job.receipt.read_text())
    finally:
        if job.proc.poll() is None:
            job.stop()


def test_slow_progressing_prefill_outlives_old_total_deadline():
    result = run_fake('import time\nfor n in range(40):\n print(n,flush=True);time.sleep(.05)')
    assert result['exit_code'] == 0 and result['reason'] is None
    assert result['elapsed_s'] > 2 > .8
    assert result['cleanup_confirmed']


def test_stalled_prefill_fails_via_no_progress_rule():
    result = run_fake('import time;print("prefill started",flush=True);time.sleep(10)')
    assert 'no progress' in result['reason']
    assert result['cleanup_confirmed']
