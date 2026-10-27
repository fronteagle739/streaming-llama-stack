"""CPU-only controller and host watchdogs for experimental Metal benchmarks.

A timeout bounds the host, not the GPU command buffer. After HUNG, report the
remaining jobs as skipped: another process cannot repair a poisoned GPU queue.
Run this file directly to keep even MLX/package imports out of the controller.
"""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Thread
from time import perf_counter

if __package__:
    from .grouped_gemm_tuning import CONFIGS
else:
    from grouped_gemm_tuning import CONFIGS

HUNG = 124
UNSAFE_TIMEOUT = 120.0
ROOT = Path(__file__).resolve().parents[3]
MODULE = "mlx_vlm.custom_kernels.deepseek_v41.test_grouped_gemm"


def report(**record):
    print(json.dumps(record), flush=True)


class LaunchHung(RuntimeError):
    pass


def bounded_call(label, shapes, fn, timeout=120.0):
    """Include construction, synchronization and readback in the deadline.

    Daemon threads are deliberate: executors join hung threads on shutdown.
    The caller MUST exit the worker on LaunchHung, without MLX cleanup.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    report(launch=label, shapes=shapes, timeout_s=timeout)
    done = Queue(maxsize=1)

    def invoke():
        try:
            start = perf_counter()
            value = fn()
            done.put((True, value, (perf_counter() - start) * 1000))
        except BaseException as exc:
            done.put((False, exc, None))

    Thread(target=invoke, daemon=True, name="grouped-gemm-watchdog").start()
    try:
        ok, value, elapsed = done.get(timeout=timeout)
    except Empty:
        report(variant=label, status="HUNG", timeout_s=timeout)
        raise LaunchHung(label) from None
    if not ok:
        raise value
    return value, elapsed


def evaluate(mx, label, shapes, fn, timeout=120.0):
    def execute():
        mx.synchronize()
        value = fn()
        mx.eval(value)
        mx.synchronize()
        return value
    return bounded_call(label, shapes, execute, timeout)


@dataclass(frozen=True)
class Variant:
    projection: str
    config: str
    mode: str = "full"

    @property
    def label(self):
        return f"{self.projection}/{self.config}/{self.mode}"


def plan_variants(configs, *, attribution=False, unsafe=False, probe_mode=None):
    names = list(dict.fromkeys((["legacy"] if attribution or unsafe else []) + list(configs)))
    for name in names:
        if name not in CONFIGS:
            raise ValueError(f"unknown config: {name}")
    jobs = [Variant(p, c) for p in ("gate", "up", "down", "routed_layer")
            for c in ("gather_qmm", "production", *names)]
    if unsafe:
        # All full-kernel/layer measurements finish before ANY MODE=1/2 launch.
        jobs += [Variant(p, c, m) for m in ((probe_mode,) if probe_mode else ("decode", "mma"))
                 for p in ("gate", "up", "down") for c in names]
    return jobs


def shapes_for(job, tokens, release):
    d, inter = (5120, 2304) if release else (128, 96)
    if job.projection == "routed_layer":
        return dict(X=[tokens, d], routes=[tokens, 6], gate_up=[384, inter, d],
                    down=[384, d, inter])
    n, k = (d, inter) if job.projection == "down" else (inter, d)
    return dict(X=[tokens * 6, k], W=[384, n, k // 8], scales=[384, n, k // 32])


def run_variants(jobs, command, *, timeout_for, shapes, popen=subprocess.Popen):
    """Inherit unbuffered child output; never wait indefinitely, even after kill."""
    records, poisoned, exit_code = [], False, 0
    for job in jobs:
        if poisoned:
            record = dict(variant=job.label, status="SKIPPED_QUEUE_HUNG", exit_code=None)
        else:
            timeout = timeout_for(job)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("process timeout must be finite and positive")
            report(process_launch=job.label, shapes=shapes(job), timeout_s=timeout)
            proc = None
            try:
                proc = popen(command(job), cwd=ROOT)
                code = proc.wait(timeout=timeout)
                status = "HUNG" if code == HUNG else ("OK" if code == 0 else "FAILED")
            except subprocess.TimeoutExpired:
                status, code = "HUNG", None
                try:
                    proc.kill()
                except OSError as exc:
                    # The process may have exited between wait and kill. A
                    # failed kill still must not bypass reporting/skipping.
                    report(variant=job.label, kill_error=str(exc))
                try:
                    code = proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass  # An uninterruptible driver wait must not block the controller.
            except OSError as exc:
                status, code = "FAILED", 1
                report(variant=job.label, error=str(exc))
            poisoned = status == "HUNG"
            record = dict(variant=job.label, status=status, exit_code=code)
            if poisoned:
                exit_code = HUNG
            elif code and not exit_code:
                exit_code = code if code > 0 else 128 - code
        records.append(record)
        report(**record)
    return exit_code, records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing", action="store_true", required=True)
    p.add_argument("--release", action="store_true")
    p.add_argument("--tokens", type=int, choices=(2048, 8192), default=2048)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--config", nargs="+", choices=list(CONFIGS), default=["fast"])
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--attribution", action="store_true", help="safe full-kernel comparison including legacy")
    p.add_argument("--attribution-unsafe", action="store_true",
                   help="opt in to MODE=1/2 LAST, each in a separate process limited to 120 s")
    p.add_argument("--probe-mode", choices=("decode", "mma"), help="limit unsafe probes to one mode")
    p.add_argument("--case", choices=("random", "balanced32", "skewed"), default="random")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--capture", help="unique prefix; each variant gets its own gputrace")
    p.add_argument("--launch-timeout", type=float, default=120.0)
    p.add_argument("--process-timeout", type=float, default=600.0,
                   help="full-kernel process deadline including setup; unsafe always 120 s")
    p.add_argument("--worker", nargs=3, metavar=("PROJECTION", "CONFIG", "MODE"), help=argparse.SUPPRESS)
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.repetitions < 20:
        p.error("at least 20 repetitions are required")
    if args.case != "random" and args.tokens != 2048:
        p.error("balanced32/skewed require --tokens 2048")
    if any(not math.isfinite(t) or t <= 0 for t in (args.launch_timeout, args.process_timeout)):
        p.error("timeouts must be finite and positive")
    if args.probe_mode and not args.attribution_unsafe:
        p.error("--probe-mode requires --attribution-unsafe")
    if args.worker:
        job = Variant(*args.worker)
        if job not in plan_variants(list(CONFIGS), unsafe=args.attribution_unsafe):
            p.error("invalid worker or MODE=1/2 without --attribution-unsafe")
    return args


def controller(args):
    jobs = plan_variants(list(CONFIGS) if args.sweep else args.config,
                         attribution=args.attribution, unsafe=args.attribution_unsafe,
                         probe_mode=args.probe_mode)
    report(attribution="unsafe probes enabled LAST" if args.attribution_unsafe else "full kernels only",
           queue_policy="HUNG skips remaining jobs; process termination cannot cancel Metal work")

    def command(job):
        cmd = [sys.executable, "-u", "-m", MODULE, "--timing", "--tokens", str(args.tokens),
               "--repetitions", str(args.repetitions), "--case", args.case,
               "--launch-timeout", str(min(args.launch_timeout, UNSAFE_TIMEOUT)),
               "--worker", job.projection, job.config, job.mode]
        for flag, enabled in (("--release", args.release), ("--verbose", args.verbose),
                              ("--attribution-unsafe", args.attribution_unsafe)):
            if enabled:
                cmd.append(flag)
        if args.capture:
            cmd += ["--capture", args.capture]
        return cmd

    code, records = run_variants(jobs, command,
        timeout_for=lambda job: UNSAFE_TIMEOUT if job.mode != "full" else args.process_timeout,
        shapes=lambda job: shapes_for(job, args.tokens, args.release))
    report(summary=records, exit_code=code)
    return code


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        parser().error("workers must use the test_grouped_gemm module")
    sys.exit(controller(args))
