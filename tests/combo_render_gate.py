"""GPU gate: native ds4-win render vs the in-process renderer, byte-identical token ids.

Starts ONE native ds4-win (the qualified cfg's ds4_command, combo mode) and asks it
to render each request body (Native.render, the /decode/render control op on a
fresh native, i.e. canonical ids); renders the same bodies with slstack.ds4_render;
asserts identical id lists. No decode, no producer.

usage (cwd = the stack repo):
  python -B tests/combo_render_gate.py --cfg runtime/cfg-r.json \
      --out runtime/render-gate.json [--case FILE ...]
Default cases: runtime/case-rn-2000-qa.json, runtime/case-rn-16000.json
and every body in tests/fixtures/render_gate_cases.json (multi-turn with tool calls).
Exit 0 only when every case is identical.
"""
import argparse, json, os, secrets, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack.combo_native import Native
from slstack.combo_release import release_barrier
from slstack.combo_session import render_body
from slstack.native_broker import vendor
from slstack import ds4_render

DEFAULT_CASES = ['runtime/case-rn-2000-qa.json', 'runtime/case-rn-16000.json']


def bodies(paths):
    for path in paths:
        row = json.loads(Path(path).read_text(encoding='utf-8'))
        if 'cases' in row:
            for case in row['cases']:yield path + '#' + case['name'], case['body']
        else:yield path, row['body']


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True);parser.add_argument('--out', required=True)
    parser.add_argument('--case', nargs='*', default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.cfg).read_text())
    cases = args.case or DEFAULT_CASES + [str(ROOT / 'tests/fixtures/render_gate_cases.json')]
    vendor()
    from studio.dsv41_windows_runner import _Native
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ.setdefault('LLAMADECODE_ABORT_FILE', str(Path(args.out).with_suffix('.abort')))
    tokenizer = ds4_render.Tokenizer.from_gguf(cfg['model'])
    result = dict(cfg=args.cfg, native_sha256=cfg.get('native_sha256'), cases=[], started=time.time())
    device = _Native();baseline = device.gpu(0, cfg['gpu_uuid']);owner = None
    try:
        childcfg = dict(cfg, host_budget_gib=cfg['native_host_budget_gib'], aggregate_host_budget_gib=cfg['host_budget_gib'])
        owner = Native(childcfg, secrets.token_hex(16), secrets.randbits(63) or 1, 1, lambda: False)
        for name, body in bodies(cases):
            row = dict(case=name)
            try:
                request = render_body(body)
                native = owner.render(request);row['native_render'] = owner.last_render
                started = time.monotonic();text, mine, _ = ds4_render.render(request, tokenizer)
                row.update(native_tokens=len(native), local_tokens=len(mine), local_render_s=time.monotonic() - started)
                at = first_difference(native, mine);row['identical'] = at is None
                if at is not None:
                    row['first_difference'] = at
                    row['native_at'] = [[t, tokenizer.token_bytes(t).decode('utf-8', 'replace')] for t in native[at:at + 8]]
                    row['local_at'] = [[t, tokenizer.token_bytes(t).decode('utf-8', 'replace')] for t in mine[at:at + 8]]
                    row['context_before'] = tokenizer.detokenize(native[max(0, at - 16):at]).decode('utf-8', 'replace')
            except Exception as exc:
                row.update(identical=False, error=repr(exc)[:2000])
            result['cases'].append(row)
            print(json.dumps({k: row.get(k) for k in ('case', 'identical', 'native_tokens', 'local_tokens', 'first_difference', 'error')}), flush=True)
    except BaseException as exc:
        result['error'] = repr(exc)[:2000]
    finally:
        if owner is not None:
            try:result['native_release'] = release_barrier(owner, device, cfg, baseline, lambda: False)
            except BaseException as exc:result['native_release_error'] = repr(exc)[:500]
        result['status'] = 'PASS' if result['cases'] and all(r['identical'] for r in result['cases']) and 'error' not in result else 'FAIL'
        Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
        print('RENDER_GATE ' + result['status'] + ' ' + args.out, flush=True)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':main()
