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
import re
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


def completed_logits_sha256(path):
    # The native forced-decode reply can precede its exclusive Windows file
    # handle's RAII close. Retry only the completed artifact read, never decode.
    deadline = time.monotonic() + 1.0
    while True:
        try:
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


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
                  flashinfer_requested=cfg['env_overrides'].get('LLAMA_DSV41_FLASHINFER_MXFP4') == '1',
                  decode_mma_requested=cfg['env_overrides'].get('DS4_CUDA_MXFP4_DECODE_MMA') == '1',
                  route_histogram_requested=cfg['env_overrides'].get('LLAMA_DSV41_ROUTE_HISTOGRAM') == '1',
                  affinity_warmup_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_WARMUP_TOKENS','100')),
                  affinity_ramp_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_RAMP_TOKENS','8')),
                  memory_limits={key: cfg[key] for key in (
                      'host_budget_gib', 'producer_host_budget_gib', 'min_available_gib', 'runtime_min_free_gib')})
    if args.bias:
        record['affinity_calibration_sha256'] = hashlib.file_digest(
            Path(cfg['cache_affinity_spread_file']).open('rb'), 'sha256').hexdigest()
    producer = None
    try:
        if record['flashinfer_requested']:
            if args.arm != 'proxy':
                raise ValueError('FlashInfer stored-weight experiment requires the proxy arm')
            bridge = Path(cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_MXFP4_DLL'])
            record['flashinfer_bridge_sha256'] = hashlib.file_digest(bridge.open('rb'), 'sha256').hexdigest()
        header = read_header(cfg['model'])
        experts = [t for t in header['tensors'] if t['name'].endswith(
            ('_gate_exps.weight', '_up_exps.weight', '_down_exps.weight'))]
        if args.arm == 'q4' and (len(experts) != 120 or any(t['type'] != 12 for t in experts)):
            raise ValueError('Q4 control requires exactly 120 original Q4_K expert tensors')
        if args.arm == 'proxy':
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
        from timing_report import from_receipt
        record.update(from_receipt(producer.directory / 'producer.json', record))
        record['quality_reply'] = producer.quality(args.bias)
        if args.continuation:
            _, continuation = token_ids(args.continuation)
            logits = args.output.with_name(args.output.stem + '-' + secrets.token_hex(6) + '-logits.f32')
            record.update(forced_tokens=continuation,
                          forced_reply=producer.decode_forced(continuation, logits),
                          logits_path=str(logits),
                          logits_sha256=completed_logits_sha256(logits))
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
        if record['status'] == 'PASS' and record['flashinfer_requested']:
            text = trace.read_text(encoding='utf-8', errors='replace')
            actual = re.findall(r'NATIVEFP4_FLASHINFER_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=(\d+) ', text)
            stored = re.findall(r'NATIVEFP4_STORED_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=(\d+) ', text)
            record['flashinfer_completed_graphs'] = len(actual)
            record['flashinfer_completed_products'] = sum(int(row[3]) for row in actual)
            if not actual or actual != stored or any(int(row[3]) != (3 if int(row[2]) > 4 else 0) for row in actual):
                record.update(status='FAIL', error='Missing or inconsistent completed FlashInfer W4A8 graph witnesses')
        if record['status'] == 'PASS' and record['decode_mma_requested'] and args.decode:
            layers = sorted(int(layer) for layer in re.findall(
                r'NATIVEFP4_DECODE_QUEUED layer=(\d+) calls=3 rows=1 offline_weights=1 runtime_weight_conversion=0 activation=fp4',
                trace.read_text(encoding='utf-8', errors='replace')))
            record['decode_mma_queued_layers'] = layers
            if layers != list(range(40)):
                record.update(status='FAIL', error='Missing native FP4 decode launch witnesses for all40 layers')
        args.output.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
        if record['status'] == 'FAIL' and (record.get('flashinfer_requested') or (
                record.get('decode_mma_requested') and 'decode_mma_queued_layers' in record)):
            raise RuntimeError(record['error'])


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
