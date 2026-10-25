"""GPU gate: small appends on the resident single-engine producer, and the cut-turn resume.

One producer for the whole run. Turn 1 is the case (greedy, short). Then, for each
target N in --sizes (default 26 52 96), a follow-up whose rendered suffix appends
exactly N new tokens; the metric is append_prefill_seconds (the producer's prefill op
for those N tokens), and the first --probe greedy tokens after the append are recorded
so a per-token run (--multirow 0) and a multi-row run (--multirow 1) can be compared
token for token. Then the cut-turn resume (item 1): a turn ended by a stop sequence and
a turn ended by a client disconnect, each followed by a turn that must append only the
template closing + new message (prefix_source memory-cut-turn, no relaunch).

usage (cwd = the stack repo; GPU must be free; the run takes the RTX hold unless --no-lease):
  python -B tests/combo_append_gate.py --cfg <single-engine cfg.json> \
      --case <case.json> --out <runs>/append-mr1.json --multirow 1
  python -B tests/combo_append_gate.py ... --out <runs>/append-mr0.json --multirow 0
  python -B tests/combo_append_gate.py --compare <runs>/append-mr0.json <runs>/append-mr1.json
"""
import argparse, contextlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import ds4_render
from slstack.combo_single import SingleEngine
from slstack.combo_proxy import request_worker_config

WORDS = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa '
         'quebec romeo sierra tango uniform victor whiskey xray yankee zulu').split()


def turn(engine, body, name, client_gone=None):
    rows = [];final = {};metrics = dict(turn=name)
    def emit(kind, *values):
        if kind == 'delta':rows.append(values)
        else:final.update(values[0])
    started = time.monotonic();engine.run(body, emit, metrics, client_gone=client_gone);metrics['wall_s'] = time.monotonic() - started
    parsed = final['parsed']
    metrics.update(content=parsed['content'], reasoning=parsed['reasoning'], finish=parsed['finish'],
                   streamed_reasoning=''.join(t for k, t in rows if k == 'reasoning_content'),
                   streamed_content=''.join(t for k, t in rows if k == 'content'))
    return metrics


def sized_message(engine, messages, body, target):
    """A user message whose follow-up render appends exactly `target` new tokens."""
    tok = engine.tokenize()
    def appended(text):
        follow = dict(body, messages=messages + [dict(role='user', content=text)])
        rendered, canonical, _ = ds4_render.render(follow, tok, tool_memory=engine.tool_memory)
        tokens, _ = ds4_render.continue_live(engine.live, rendered, canonical, tok)
        return None if tokens is None else len(tokens) - len(engine.live)
    words = []
    while True:
        words.append(WORDS[len(words) % len(WORDS)])
        n = appended('Reply with one word. ' + ' '.join(words))
        if n is None:raise RuntimeError('follow-up render does not continue the live frontier')
        if n >= target:break
    text = 'Reply with one word. ' + ' '.join(words)
    for cut in range(len(words[-1]) + 1):   # trim the last word until the count is exact
        candidate = text[:len(text) - cut]
        if appended(candidate) == target:return candidate
    return text   # exact count unreachable at a word boundary: the record states the real count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg');parser.add_argument('--case');parser.add_argument('--out')
    parser.add_argument('--multirow', choices=('0', '1'), default='1');parser.add_argument('--sizes', type=int, nargs='*', default=[26, 52, 96])
    parser.add_argument('--probe', type=int, default=24);parser.add_argument('--no-lease', action='store_true')
    parser.add_argument('--compare', nargs=2)
    args = parser.parse_args()
    if args.compare:
        a, b = (json.loads(Path(p).read_text(encoding='utf-8')) for p in args.compare)
        rows = []
        for x, y in zip(a['appends'], b['appends']):
            rows.append(dict(target=x['target'], seconds=(x.get('append_prefill_seconds'), y.get('append_prefill_seconds')),
                             same_probe=x.get('probe_content') == y.get('probe_content')))
        print(json.dumps(rows, indent=1));raise SystemExit(0 if all(r['same_probe'] for r in rows) else 1)
    cfg = json.loads(Path(args.cfg).read_text())
    env = cfg.setdefault('env_overrides', {});env['LLAMADECODE_DS4_DECODE'] = '1';env['LLAMADECODE_APPEND_MULTIROW'] = args.multirow
    cfg.setdefault('combo_worker_host_budget_gib', cfg['host_budget_gib'] - .25)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ.setdefault('LLAMADECODE_ABORT_FILE', str(Path(args.out).with_suffix('.abort')))
    case = json.loads(Path(args.case).read_text(encoding='utf-8'))
    body = dict(case['body'], max_tokens=256, stream=False);body.pop('quality_speed', None);body.pop('stop', None)
    lease = (lambda: contextlib.nullcontext()) if args.no_lease else None
    engine = SingleEngine(request_worker_config(cfg), lease_factory=lease)
    launches = [];factory = engine.producer_factory
    def counted(tokens):launches.append(len(tokens));return factory(tokens)
    engine.producer_factory = counted
    result = dict(cfg=args.cfg, case=args.case, multirow=args.multirow, appends=[], cut_turns=[])
    try:
        engine.set_quality(0.0)
        one = turn(engine, body, 'first');result['first'] = {k: one.get(k) for k in ('prompt_tokens', 'producer_launch_prefill_seconds', 'completion_tokens', 'finish')}
        messages = list(body['messages']) + [dict(role='assistant', content=one['content'], reasoning_content=one['reasoning'])]
        for target in args.sizes:
            text = sized_message(engine, messages, body, target)
            follow = dict(body, messages=messages + [dict(role='user', content=text)], max_tokens=args.probe)
            row = turn(engine, follow, 'append-%d' % target)
            result['appends'].append(dict(target=target, new_tokens=row.get('new_tokens'), prefix_source=row.get('prefix_source'),
                append_prefill_seconds=row.get('append_prefill_seconds'), first_token_seconds=row.get('first_token_seconds'),
                probe_content=row['content'], probe_reasoning=row['reasoning'], live_tokens=row.get('live_tokens')))
            messages = follow['messages'] + [dict(role='assistant', content=row['content'], reasoning_content=row['reasoning'])]
        # Item 1: a stop-sequence cut, then a disconnect cut, each resumed without a relaunch.
        stop_body = dict(body, messages=messages + [dict(role='user', content='Count from one to forty in words, separated by commas.')],
                         max_tokens=400, stop=['twenty'])
        cut = turn(engine, stop_body, 'stop-sequence')
        messages = stop_body['messages'] + [dict(role='assistant', content=cut['content'], reasoning_content=cut['reasoning'])]
        after = turn(engine, dict(body, messages=messages + [dict(role='user', content='Thanks.')], max_tokens=16), 'after-stop')
        result['cut_turns'].append(dict(kind='stop sequence', ended=cut.get('decode_ended'), prefix_source=after.get('prefix_source'),
            new_tokens=after.get('new_tokens'), append_prefill_seconds=after.get('append_prefill_seconds'), fresh_reason=after.get('fresh_reason')))
        messages = messages + [dict(role='user', content='Thanks.'), dict(role='assistant', content=after['content'], reasoning_content=after['reasoning'])]
        shown = [0]
        def gone():shown[0] += 1;return shown[0] > 40
        dis_body = dict(body, messages=messages + [dict(role='user', content='Write a long poem about the sea.')], max_tokens=400)
        cut = turn(engine, dis_body, 'disconnect', client_gone=gone)
        messages = dis_body['messages'] + [dict(role='assistant', content=cut['streamed_content'], reasoning_content=cut['streamed_reasoning'])]
        after = turn(engine, dict(body, messages=messages + [dict(role='user', content='Stop there.')], max_tokens=16), 'after-disconnect')
        result['cut_turns'].append(dict(kind='client disconnected', ended=cut.get('decode_ended'), prefix_source=after.get('prefix_source'),
            new_tokens=after.get('new_tokens'), append_prefill_seconds=after.get('append_prefill_seconds'), fresh_reason=after.get('fresh_reason')))
        result['producer_launches'] = launches
        result['producer_directory'] = str(engine.producer.directory) if engine.producer else None
    except BaseException as exc:
        result['error'] = repr(exc)[:2000];result['producer_launches'] = launches
    finally:
        try:engine.close()
        except BaseException as exc:result['close_error'] = repr(exc)[:500]
        ok_appends = len(result['appends']) == len(args.sizes) and all(r['prefix_source'] in ('memory-token', 'memory-text') for r in result['appends'])
        ok_cuts = len(result['cut_turns']) == 2 and all(r['prefix_source'] == 'memory-cut-turn' for r in result['cut_turns'])
        result['status'] = 'PASS' if ok_appends and ok_cuts and len(launches) == 1 and 'error' not in result else 'FAIL'
        Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
        print('APPEND_GATE ' + result['status'] + ' ' + json.dumps(dict(
            appends=[(r['target'], r['new_tokens'], r['append_prefill_seconds']) for r in result['appends']],
            cuts=[(r['kind'], r['prefix_source'], r['new_tokens']) for r in result['cut_turns']],
            launches=len(launches), error=result.get('error'))), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':main()
