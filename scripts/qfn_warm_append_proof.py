"""Guarded real-model regression for short warm appends and CUDA graph reuse.

The config supplies the model, vocabulary, engine and explicit memory budgets.
The token file is little-endian int32. No listener or download is created.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STACK))
from slstack.config import load
from slstack.model_settings import apply
from slstack.process import OwnedProcess
from slstack.qfn_single import QfnEngine


def child(cfg, tokens_file, prefix):
    raw = Path(tokens_file).read_bytes()
    assert raw and len(raw) % 4 == 0, 'Expected complete int32 prompt tokens'
    tokens = list(struct.unpack('<' + str(len(raw) // 4) + 'i', raw))
    command = QfnEngine(cfg).command()
    # This proof explicitly uses the engine's stdio protocol.
    port = command.index('--port')
    del command[port:port + 2]
    command += ['--host-budget-gib', str(cfg['host_budget_gib'])]
    records = []
    with Path(str(prefix) + '.engine.log').open('x', encoding='utf8') as log:
        engine = subprocess.Popen(command, cwd=STACK, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf8')

        def send(row):
            engine.stdin.write(json.dumps(row) + '\n')
            engine.stdin.flush()

        def read():
            line = engine.stdout.readline()
            assert line, ('engine exited', engine.poll())
            row = json.loads(line)
            records.append(row)
            assert row.get('ok') is True, row
            return row

        def generate(token, count):
            send(dict(op='generate', max_tokens=count, temperature=0, logit_bias={str(token): 1000}))
            emitted = []
            while True:
                row = read()
                if row.get('done'):
                    assert row['reason'] == 'length' and emitted == [token] * count, row
                    return
                assert 'token' in row and not row.get('eog'), row
                emitted.append(row['token'])
                tokens.append(row['token'])

        try:
            assert read()['event'] == 'ready'
            send(dict(op='prefill', tokens=tokens))
            assert read()['n_past'] == len(tokens)
            generate(42, 8)
            for count in (4, 1, 2, 3, 4):
                previous = len(tokens)
                tokens.extend([42] * count)
                send(dict(op='prefill', tokens=tokens))
                row = read()
                assert row['route'] == 'decode' and row['new_tokens'] == count, row
                assert row['reused'] == previous and row['n_past'] == len(tokens), row
            generate(43, 4)
        finally:
            if engine.poll() is None:
                send(dict(op='quit'))
                assert read()['event'] == 'bye'
                engine.wait(timeout=60)
            Path(str(prefix) + '.events.json').write_text(json.dumps(records, indent=2))
        assert engine.returncode == 0
    print('PASS: warm batches 4,1,2,3,4 reused the full prefix and generation continued.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tokens', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = apply(load(args.config), 'qfn')
    prefix = Path(args.output_prefix).resolve()
    if args.child:
        child(cfg, args.tokens, prefix)
        return
    assert not Path(str(prefix) + '.events.json').exists(), 'Preserve existing evidence'
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--config', args.config,
               '--tokens', args.tokens, '--output-prefix', str(prefix), '--child']
    job = OwnedProcess(cfg, command, 'qfn-warm-append-proof', gpu=True,
                       scope='qfn-engine', unbounded=True)
    try:
        guard = job.wait()
    finally:
        print(job.log.read_text(errors='replace')[-3000:], flush=True)
    assert guard['exit_code'] == 0 and guard['cleanup_confirmed'] and guard['paging_telemetry_closed']
    Path(str(prefix) + '.guard.json').write_text(json.dumps(dict(
        receipt=str(job.receipt), log=str(job.log), guard=guard), indent=2))


if __name__ == '__main__':
    main()
