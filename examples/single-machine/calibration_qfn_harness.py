"""Portable adapter for the archived QFN measurement mechanisms.

The stdio exchange is recall_100k.py's existing prefill/generate protocol;
the feed command is native_nvfp4_forced_feed.py's --feed-from/--feed-chunk
mechanism. The rowwise comparison below is the existing compare_qfn_logits.py
calculation, with vocabulary supplied by the caller. Model inference and chat
rendering remain entirely in the same production qfn-engine executable.
Run through calibrate-router.py's guardian and GPU lease, never standalone.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

STACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STACK))
from slstack.config import environment
from slstack.model_settings import apply
from slstack.qfn_single import QfnEngine


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def tokens(path):
    raw = Path(path).read_bytes()
    require(len(raw) % 4 == 0, 'Incomplete LE32 token fixture')
    return list(struct.unpack('<' + str(len(raw) // 4) + 'i', raw))


@contextmanager
def serve(command, env, prefix):
    """Same stdio lifecycle and graceful quit used by recall_100k.child."""
    with Path(str(prefix) + '.engine.log').open('x', encoding='utf8') as log, \
            Path(str(prefix) + '.events.jsonl').open('x', encoding='utf8') as events:
        engine = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, text=True, encoding='utf8', cwd=STACK, env=env)
        def send(value):
            engine.stdin.write(json.dumps(value) + '\n'); engine.stdin.flush()
        def receive():
            line = engine.stdout.readline()
            require(bool(line), 'Engine protocol ended: ' + str(engine.poll()))
            item = json.loads(line)
            events.write(json.dumps(item, ensure_ascii=False) + '\n'); events.flush()
            require(item.get('ok') is True, 'Engine rejected request: ' + str(item))
            return item
        try:
            ready = receive(); require(ready['event'] == 'ready', 'Missing ready event')
            yield send, receive, ready
            send(dict(op='quit')); require(receive()['event'] == 'bye', 'Missing quit acknowledgement')
            require(engine.wait(timeout=60) == 0, 'Engine exit was unsuccessful')
        finally:
            if engine.poll() is None:
                # The enclosing existing guardian owns retirement if quit fails.
                send(dict(op='cancel')); send(dict(op='quit')); engine.wait(timeout=60)


def generate(send, receive, body):
    send(dict(op='generate', **body)); ids, text = [], ''
    while True:
        event = receive()
        if 'token' in event and not event.get('eog'):
            ids.append(event['token']); text += event.get('piece', '')
        if event.get('done'):
            return dict(ids=ids, text=text, stats=event)


def compare(left_path, right_path, vocab):
    """Existing full-row F64 KL/top-1 calculation; no selected columns."""
    import numpy as np
    a, b = Path(left_path), Path(right_path)
    require(a.stat().st_size == b.stat().st_size and a.stat().st_size % (vocab * 4) == 0,
            'Full vocabulary dumps differ in size')
    rows = a.stat().st_size // (vocab * 4)
    kl, details, matches, maximum = [], [], 0, 0.0
    with a.open('rb') as left, b.open('rb') as right:
        for index in range(rows):
            x = np.frombuffer(left.read(vocab * 4), dtype='<f4').astype(np.float64)
            y = np.frombuffer(right.read(vocab * 4), dtype='<f4').astype(np.float64)
            require(x.size == y.size == vocab and np.isfinite(x).all() and np.isfinite(y).all(),
                    'Incomplete or nonfinite vocabulary row')
            top_left, top_right = int(np.argmax(x)), int(np.argmax(y))
            matches += int(top_left == top_right)
            maxabs = float(np.max(np.abs(x - y))); maximum = max(maximum, maxabs)
            x -= x.max(); y -= y.max()
            logp = x - np.log(np.exp(x).sum()); logq = y - np.log(np.exp(y).sum())
            kl.append(float(np.dot(np.exp(logp), logp - logq)))
            details.append(dict(row=index, kl=kl[-1], maxabs=maxabs, top_left=top_left, top_right=top_right))
        require(left.read(1) == right.read(1) == b'', 'Trailing logit data')
    return dict(rows=rows, vocab=vocab, kl_mean=float(np.mean(kl)), kl_max=float(np.max(kl)),
                top1_agreement=matches / rows, maxabs=maximum, reference_sha256=sha(a),
                candidate_sha256=sha(b), row_details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('quality', 'timing', 'generation'))
    for name in ('config', 'settings', 'reference-settings', 'model', 'model-sha256', 'engine', 'vocab-gguf',
                 'prompt-tokens', 'forced-ids', 'output-prefix', 'result'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--vocab', type=int, required=True)
    parser.add_argument('--bias', type=float, required=True)
    parser.add_argument('--host-budget-gib', type=float, required=True)
    parser.add_argument('--warmup-tokens', type=int, default=100)
    parser.add_argument('--evaluated-tokens', type=int, default=128)
    parser.add_argument('--recall-text', required=True, help='Expected answer present in the unbiased recall continuation')
    parser.add_argument('--request-file')
    args = parser.parse_args()
    require(os.environ.get('SLSTACK_CALIBRATION_GUARDED') == '1', 'Use calibrate-router.py guardian/lease')
    cfg = read(args.config)
    cfg.update(model=str(Path(args.model).resolve()), model_settings=args.settings,
               qfn_engine=str(Path(args.engine).resolve()), qfn_vocab_gguf=str(Path(args.vocab_gguf).resolve()),
               # --context denotes prompt tokens; reserve full forced/warm/evaluation continuation.
               context=((args.context + 1024 + args.warmup_tokens + args.evaluated_tokens + 127) // 128) * 128)
    if args.stage == 'generation':
        require(args.request_file, 'Generation requires an actual request JSON')
        request = read(args.request_file)
        cfg['context'] = max(cfg['context'], int(request['max_tokens']) + 4096)
    base_cfg = dict(cfg)
    def configured(settings_file):
        local = dict(base_cfg, model_settings=settings_file)
        local['qfn_host_budget_gib'] = args.host_budget_gib
        local = apply(local, 'qfn')
        expected = read(settings_file)
        # Explicit local overrides cannot secretly defeat a requested setting sweep.
        for key, field in (('tile', 'qfn_tile'), ('prefill', 'qfn_nvfp4_mode'),
                           ('decode', 'qfn_nvfp4_decode_mode'), ('indexer', 'qfn_qsa_indexer'),
                           ('attention', 'qfn_qsa_attention')):
            if key in expected['engine']:
                require(local[field] == expected['engine'][key], 'Local override conflicts with calibration profile: ' + field)
        env = environment(local, scope='qfn-engine')
        for name, value in (('QFN_MTP', expected['engine']['mtp']),
                            ('DS4_DS_READ_PARTS', expected['engine']['read_parts']),
                            ('QFN_LFU_HALF_LIFE', expected['cache']['half_life']),
                            ('QFN_SEED_LFU_MAX', expected['cache']['seed_lfu_max'])):
            require(float(env[name]) == float(value), 'Local environment conflicts with calibration profile: ' + name)
        from slstack.performance_contract import seal
        local.setdefault('qfn_engram_scan_rings', int(local.get('env_overrides', {}).get('LLAMADECODE_ENGRAM_SCAN_RINGS', '1')))
        local = seal(local, 'qfn')
        command = QfnEngine(local).command()
        port_index = command.index('--port'); del command[port_index:port_index + 2]
        return local, command, env
    cfg, command, env = configured(args.settings)
    prefix = Path(args.output_prefix)
    prompt = tokens(args.prompt_tokens); forced = tokens(args.forced_ids)
    require(len(prompt) == args.context, 'Prompt fixture must have exactly chosen-context tokens')
    require(128 <= len(forced) <= 1024, 'Supply 128..1024 pinned teacher-forced reference IDs')
    require(all(0 <= token < args.vocab for token in prompt + forced), 'Token fixture outside model vocabulary')
    binding = dict(context=args.context, bias=args.bias, model_sha256=args.model_sha256,
                   settings_sha256=sha(args.settings), prompt_sha256=sha(args.prompt_tokens),
                   forced_ids_sha256=sha(args.forced_ids), engine_sha256=sha(args.engine),
                   vocab_sha256=sha(args.vocab_gguf), effective_config=cfg, command=command)
    if args.stage == 'quality':
        combined = Path(str(prefix) + '.forced.le32')
        with combined.open('xb') as stream:
            stream.write(Path(args.prompt_tokens).read_bytes()); stream.write(Path(args.forced_ids).read_bytes())
        def feed(label, bias, production_command, production_env):
            output = Path(str(prefix) + '.' + label)
            argv = [item for item in production_command if item != '--serve']
            argv += ['--tokens', str(combined), '--out', str(output), '--feed-from', str(args.context),
                     '--feed-chunk', '1', '--temp', '0', '--seed', '1', '--bias', str(bias)]
            with Path(str(output) + '.engine.log').open('x', encoding='utf8') as log:
                subprocess.run(argv, cwd=STACK, env=production_env, stdout=log, stderr=log, check=True)
            positions = Path(str(output) + '.feed.pos')
            require(positions.read_bytes() == struct.pack('<' + str(len(forced)) + 'i', *range(args.context, args.context + len(forced))),
                    'Forced-feed positions do not span the complete continuation')
            return str(output) + '.feed.f32', str(positions)
        # Every candidate is compared to the original unbiased profile, including
        # engine/cache setting swaps. No sparse/top-k reference is used.
        _, reference_command, reference_env = configured(args.reference_settings)
        reference, reference_pos = feed('reference', 0, reference_command, reference_env)
        candidate, candidate_pos = feed('candidate', args.bias, command, env)
        result = compare(reference, candidate, args.vocab)
        result.update(reference_logits=reference, candidate_logits=candidate,
                      reference_positions=reference_pos, candidate_positions=candidate_pos)
    elif args.stage == 'timing':
        require(args.warmup_tokens > 0 and args.evaluated_tokens >= 128, 'Warm timing window required')
        with serve(command, env, prefix) as (send, receive, ready):
            require(ready['n_vocab'] == args.vocab, 'Vocabulary mismatch')
            send(dict(op='prefill', tokens=prompt)); pf = receive()
            require(pf['new_tokens'] == args.context and pf['n_past'] == args.context and pf['route'] == 'layer-major', 'Exact-context prefill missing')
            # Recall is tested unbiased exactly as in recall_100k; the candidate
            # continuation follows on the same sequence after its warm window.
            recall = generate(send, receive, dict(max_tokens=128, temperature=0, seed=1, bias=0))
            warm = generate(send, receive, dict(max_tokens=args.warmup_tokens, temperature=0, seed=1, bias=args.bias))
            measured = generate(send, receive, dict(max_tokens=args.evaluated_tokens, temperature=0, seed=1, bias=args.bias))
            require(len(warm['ids']) == args.warmup_tokens and warm['stats']['reason'] == 'length', 'Natural stop interrupted warm window; not extended')
            require(len(measured['ids']) == args.evaluated_tokens and measured['stats']['reason'] == 'length', 'Natural stop interrupted matched timing; not extended')
            stats = measured['stats']
            result = dict(prefill_tok_per_s=args.context / pf['seconds'], decode_tok_per_s=stats['tok_per_s'],
                          cache_hits=stats['expert_hits'], cache_misses=stats['expert_misses'],
                          evaluated_tokens=len(measured['ids']), warmup_tokens=len(warm['ids']),
                          recall='PASS' if args.recall_text in recall['text'] else 'FAIL',
                          prefill=pf, recall_generation=recall, warmup=warm, measured=measured)
    else:
        require('bias' not in request and 'quality_speed' not in request,
                'Calibration supplies bias separately; request must not override it')
        with serve(command, env, prefix) as (send, receive, ready):
            require(ready['n_vocab'] == args.vocab, 'Vocabulary mismatch')
            send(dict(op='chat', body={**request, 'bias': args.bias}))
            while True:
                event = receive()
                if event.get('done'):
                    break
            result = dict(request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                          reason=event['finish_reason'], text=event['message'].get('content', ''),
                          generated_tokens=event['usage']['completion_tokens'], max_tokens=request['max_tokens'],
                          message=event['message'], stats=event['stats'], final_event=event)
    save(args.result, {**binding, **result})


if __name__ == '__main__':
    main()
