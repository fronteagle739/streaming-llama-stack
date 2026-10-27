"""Isolate the grouped GEMM Mac tests, including implicit NumPy readbacks."""
import os
from pathlib import Path
import sys

import pytest

# Load the stdlib harness without importing mlx_vlm (Windows CPU tests need it).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_gemm_harness import LaunchHung, Variant, evaluate, run_variants


def _grouped_test(item):
    return Path(str(item.path)).name == "test_grouped_gemm.py"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # Intercept BEFORE fixture setup: projection fixtures also submit GPU work.
    if not _grouped_test(item) or os.environ.get("DSV41_GROUPED_TEST_WORKER") == "1":
        return None
    job = Variant("pytest", item.nodeid)
    env = dict(os.environ, DSV41_GROUPED_TEST_WORKER="1", PYTHONUNBUFFERED="1")
    import subprocess
    from _pytest.runner import CallInfo
    outcome = []

    def run():
        code, _ = run_variants([job],
            lambda _: [sys.executable, "-u", "-m", "pytest", "-q", "-s", item.nodeid],
            timeout_for=lambda _: 600.0, shapes=lambda _: dict(test=item.nodeid),
            popen=lambda cmd, **kwargs: subprocess.Popen(cmd, env=env, **kwargs))
        outcome.append(code)
        assert code == 0, f"isolated grouped GEMM test exited {code}"

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    call = CallInfo.from_call(run, when="call")
    result = pytest.TestReport.from_item_and_call(item, call)
    item.ihook.pytest_runtest_logreport(report=result)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    if outcome == [124]:
        pytest.exit("HUNG: remaining GPU tests skipped; Metal queue may still be blocked", returncode=124)
    return True


@pytest.fixture(autouse=True)
def grouped_test_launch_watchdog(request, monkeypatch):
    if not _grouped_test(request.node) or os.environ.get("DSV41_GROUPED_TEST_WORKER") != "1":
        return
    import mlx.core as mx
    import mlx_vlm.custom_kernels.deepseek_v41.grouped_gemm as kernel
    import mlx_vlm.models.deepseek_v41.moe_prefill as adapter
    original = kernel._grouped_mxfp4

    def guarded(x, q, scales, offsets, **kwargs):
        # Pytest never enables unsafe attribution. It is only available through
        # the controller's explicitly opted-in, 120-second probe subprocesses.
        if kwargs.get("mode", "full") != "full":
            raise ValueError("MODE=1/2 require the --attribution-unsafe controller")
        try:
            return evaluate(mx, request.node.nodeid,
                dict(X=x.shape, W=q.shape, scales=scales.shape, config=str(kwargs.get("config"))),
                lambda: original(x, q, scales, offsets, **kwargs))[0]
        except LaunchHung:
            os._exit(124)

    monkeypatch.setattr(kernel, "_grouped_mxfp4", guarded)
    monkeypatch.setattr(adapter, "_grouped_mxfp4", guarded)
    monkeypatch.setattr(request.module, "_grouped_mxfp4", guarded)
