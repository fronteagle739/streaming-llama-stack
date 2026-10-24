"""GPU gate: the single-engine serving path end to end (no HTTP, no native ds4-win).

Turn 1 renders the case in process, launches the ULTRA producer on it, greedy-decodes
to EOS and checks the case's expected substring. Turn 2 appends the assistant reply
and a follow-up user message: the producer must stay resident and prefill only the
new suffix (prefix_source memory-token or memory-text). Quality-speed is set through
the same path /v1/quality-speed uses before turn 2.

usage (cwd = the stack repo):
  python -B tests/combo_single_gate.py --cfg <cfg.json> \
      --case <case-rn-2000-qa.json> --out <runs>/single-gate.json \
      [--max-tokens 512] [--quality 1.36125] [--no-lease]
--no-lease skips the RTX hold ledger admission (use it when the run already sits
inside a hold you took yourself). LLAMADECODE_DS4_DECODE=1 comes from env_overrides
or the env table default; the reply records the producer environment that was used.
"""
import argparse, contextlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack.combo_single import SingleEngine
from slstack.combo_proxy import request_worker_config


def turn(engine, body, name):
    rows = [];final = {};metrics = dict(turn=name)
    def emit(kind, *values):
        if kind == 'delta':rows.append(values)
        else:final.update(values[0])
    started = time.monotonic();engine.run(body, emit, metrics);metrics['wall_s'] = time.monotonic() - started
    parsed = final['parsed']
    streamed = {f: ''.join(t for k, t in rows if k == f) for f in ('reasoning_content', 'content')}
    metrics.update(content=parsed['content'], reasoning=parsed['reasoning'], finish=parsed['finish'],
                   tool_calls=parsed['calls'], streamed_prefix_ok=all((parsed[k] or '').startswith(streamed[f])
                   for k, f in (('reasoning', 'reasoning_content'), ('content', 'content'))))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True);parser.add_argument('--case', required=True);parser.add_argument('--out', required=True)
    parser.add_argument('--max-tokens', type=int, default=512);parser.add_argument('--quality', type=float, default=1.36125)
    parser.add_argument('--no-lease', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.cfg).read_text())
    cfg.setdefault('env_overrides', {})['LLAMADECODE_DS4_DECODE'] = '1'
    cfg.setdefault('combo_worker_host_budget_gib', cfg['host_budget_gib'] - .25)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ.setdefault('LLAMADECODE_ABORT_FILE', str(Path(args.out).with_suffix('.abort')))
    case = json.loads(Path(args.case).read_text(encoding='utf-8'))
    expect = (case.get('content_check') or {}).get('expect_substring')
    body = dict(case['body'], max_tokens=args.max_tokens, stream=False);body.pop('quality_speed', None)
    lease = (lambda: contextlib.nullcontext()) if args.no_lease else None
    engine = SingleEngine(request_worker_config(cfg), lease_factory=lease)
    result = dict(cfg=args.cfg, case=args.case, expect=expect)
    try:
        engine.set_quality(0.0)
        one = turn(engine, body, 'first');result['first'] = one
        result['content_check'] = bool(expect) and expect in (one['content'] or '')
        engine.set_quality(args.quality)
        follow = dict(body, messages=list(body['messages']) + [
            dict(role='assistant', content=one['content'], reasoning_content=one['reasoning']),
            dict(role='user', content='Repeat only the file name, nothing else.')])
        two = turn(engine, follow, 'follow-up');result['follow_up'] = two
        result['resident_reuse'] = two.get('prefix_source') in ('memory-token', 'memory-text') and two.get('reused_prefix_tokens', 0) > 0
        result['producer_directory'] = str(engine.producer.directory) if engine.producer else None
    except BaseException as exc:
        result['error'] = repr(exc)[:2000]
    finally:
        try:engine.close()
        except BaseException as exc:result['close_error'] = repr(exc)[:500]
        result['status'] = 'PASS' if (result.get('content_check') and result.get('resident_reuse') and 'error' not in result and
            all(result.get(k, {}).get('streamed_prefix_ok') for k in ('first', 'follow_up'))) else 'FAIL'
        Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
        print('SINGLE_GATE ' + result['status'] + ' ' + json.dumps({k: result.get(k) for k in ('content_check', 'resident_reuse', 'error')}), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':main()
