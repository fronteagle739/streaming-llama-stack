"""Guarded, opt-in Q4/MXFP4 comparison using an already verified model receipt.

Supply an experiment config, exact little-endian uint32 prompt tokens, and an
optional fixed continuation. Each invocation takes the shared benchmark lease.
The proxy is lossy; this example never changes a production preset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slstack.config import load
from slstack.process import OwnedProcess

LABEL = 'Q4\u2192MXFP4 requant (lossy), speed proxy for native FP4'


def token_ids(path):
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError('Token file must contain nonempty little-endian uint32 IDs')
    return raw, list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))


def run(args, cfg):
    from slstack.combo_producer import ComboProducer
    from slstack.native_broker import vendor
    from slstack.gguf_header import read_header
    vendor()
    from studio.dsv41_session_process import Producer

    trace = args.output.with_suffix('.trace.log')

    def drain(process):
        try:
            with trace.open('ab', buffering=0) as stream:
                while block := process.errors.read(65536):
                    stream.write(block)
                    with process.state_lock:
                        process.diagnostic.extend(block)
                        if len(process.diagnostic) > 1 << 20:
                            del process.diagnostic[:len(process.diagnostic) - (1 << 20)]
                    process.last_progress = time.monotonic()
        except BaseException as error:
            process._poison(error)

    Producer._drain = drain
    raw, tokens = token_ids(args.tokens)
    record = dict(status='RUNNING', arm=args.arm,
                  label=LABEL if args.arm == 'proxy' else 'Q4_K/Q8_0 control',
                  prompt_tokens=len(tokens), input_sha256=hashlib.sha256(raw).hexdigest(),
                  profile=cfg['producer_profile'], yoco_keep=cfg['yoco_keep'], bias=args.bias,
                  producer_sha256=cfg['producer_sha256'],
                  decode_library_sha256=hashlib.file_digest(
                      (Path(cfg['llama_producer']).parent / 'ds4-decode.dll').open('rb'), 'sha256').hexdigest(),
                  profiled=args.profile_stages,
                  affinity_warmup_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_WARMUP_TOKENS','100')),
                  affinity_ramp_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_RAMP_TOKENS','8')),
                  memory_limits={key: cfg[key] for key in (
                      'host_budget_gib', 'producer_host_budget_gib', 'min_available_gib', 'runtime_min_free_gib')})
    if args.bias:
        record['affinity_calibration_sha256'] = hashlib.file_digest(
            Path(cfg['cache_affinity_spread_file']).open('rb'), 'sha256').hexdigest()
    producer = None
    try:
        if args.arm == 'proxy':
            header = read_header(cfg['model'])
            if header['metadata'].get('deepseek41.speed_proxy') != LABEL:
                raise ValueError('Proxy model marker is absent or incorrect')
            if sum(t['type'] == 39 for t in header['tensors']) != 120:
                raise ValueError('Expected exactly 120 stored MXFP4 expert tensors')
        producer = ComboProducer(cfg, secrets.token_hex(16), 1, tokens,
                                 lambda: args.output.with_suffix('.stop').exists())
        record.update(prefill_seconds=producer.prefill_seconds,
                      prefill_tok_s=len(tokens) / producer.prefill_seconds,
                      plan=producer.prefill_plan, model_sha256=producer.model_sha256,
                      ready=producer.process.ready)
        record['quality_reply'] = producer.quality(args.bias)
        if args.continuation:
            _, continuation = token_ids(args.continuation)
            logits = args.output.with_name(args.output.stem + '-' + secrets.token_hex(6) + '-logits.f32')
            record.update(forced_tokens=continuation,
                          forced_reply=producer.decode_forced(continuation, logits),
                          logits_path=str(logits),
                          logits_sha256=hashlib.file_digest(logits.open('rb'), 'sha256').hexdigest())
        if args.decode:
            record['decode'] = producer.decode(args.decode)
            steps = record['decode']['step_ms'][100:]
            record['warm_decode'] = dict(discarded_steps=min(100, args.decode), steps=len(steps),
                                         seconds=sum(steps) / 1000,
                                         tok_s=len(steps) * 1000 / sum(steps) if steps else None)
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAIL', error=repr(error))
        raise
    finally:
        if producer is not None:
            record['cleanup'] = producer.close()
        args.output.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--tokens', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', required=True, choices=('q4', 'proxy'))
    p.add_argument('--bias', type=float, default=0)
    p.add_argument('--decode', type=int, default=0)
    p.add_argument('--continuation', type=Path)
    p.add_argument('--calibrate', type=Path)
    p.add_argument('--profile-stages', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 <= args.bias <= 1.36125 or args.decode < 0:
        p.error('Bias must be 0..1.36125 and decode count nonnegative')
    if args.output.exists():
        p.error('Refusing to overwrite an existing measurement')
    cfg = load(str(args.config))
    if args.calibrate:
        if args.bias or args.decode < 64 or args.calibrate.exists():
            p.error('Calibration needs bias zero, at least 64 decode tokens and a new output path')
        cfg['cache_affinity_spread_file'] = ''
        cfg['env_overrides']['DS4_CACHE_AFFINITY_CALIBRATE'] = str(args.calibrate.resolve())
    if args.arm == 'proxy':
        cfg['env_overrides'].update(LLAMA_DSV41_STORED_MXFP4='1', GGML_CUDA_DISABLE_GRAPHS='1')
    if args.profile_stages:
        cfg['env_overrides'].update(LLAMA_CUDA_OP_PROFILE='1', LLAMA_CUDA_OP_PROFILE_NODES='1',
                                    LLAMA_CUDA_OP_PROFILE_DUMP_EACH='1', LLAMADECODE_TILE_TIMING='1')
    if args.worker:
        run(args, cfg)
    else:
        # OwnedProcess enforces the config's explicit host budget, physical floor,
        # paging stop, single-engine admission and shared GPU reservation.
        job = OwnedProcess(cfg, [sys.executable, '-B', str(Path(__file__).resolve()),
                                *sys.argv[1:], '--worker'], 'native-fp4-speed-proxy',
                           cwd=args.output.resolve().parent, gpu=True)
        receipt = job.wait()
        args.output.with_suffix('.guard.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        if receipt.get('exit_code') or receipt.get('reason'):
            raise RuntimeError('Guarded measurement failed; inspect the guard receipt and trace')


if __name__ == '__main__':
    main()
