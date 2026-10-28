"""Measure one exact-token prefill under the shared GPU and memory guardian.

The config selects producer_chunks_per_pass and producer_carry_registration.
Use the same binary and token file for paired controls. CUDA profiling changes
scheduling; profile runs are diagnostic evidence, not throughput measurements.
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


def run(args, cfg):
    from slstack.combo_producer import ComboProducer
    from slstack.ds4_render import Tokenizer
    from slstack.native_broker import vendor
    vendor()
    from studio.dsv41_session_process import Producer

    trace = args.output.with_suffix('.trace.log')

    def drain(process):
        try:
            with trace.open('xb', buffering=0) as stream:
                while block := process.errors.read(65536):
                    stream.write(block)
                    with process.state_lock:
                        process.diagnostic.extend(block)
                        if len(process.diagnostic) > 1 << 20:
                            del process.diagnostic[:len(process.diagnostic)-(1 << 20)]
                    process.last_progress = time.monotonic()
        except BaseException as error:
            process._poison(error)

    Producer._drain = drain
    raw = args.tokens.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError('Expected nonempty little-endian uint32 tokens')
    tokens = list(struct.unpack('<'+'I'*(len(raw)//4), raw))
    record = dict(status='RUNNING', tokens=len(tokens), input_sha256=hashlib.sha256(raw).hexdigest(),
        producer_sha256=cfg['producer_sha256'], context=cfg['context'],
        chunks_per_pass=cfg.get('producer_chunks_per_pass', 1),
        carry_registration=cfg.get('producer_carry_registration', 'tile'),
        profiled=args.profile, limits={k:cfg.get(k) for k in
            ('host_budget_gib', 'producer_host_max_gib', 'min_available_gib', 'runtime_min_free_gib')})
    producer = None
    try:
        producer = ComboProducer(cfg, secrets.token_hex(16), 1, tokens,
            lambda:args.output.with_suffix('.stop').exists())
        record.update(prefill_seconds=producer.prefill_seconds,
            tokens_per_second=len(tokens)/producer.prefill_seconds,
            plan=producer.prefill_plan, checkpoint=producer.checkpoint)
        if args.recall:
            producer.quality(0)
            tokenizer = Tokenizer.from_gguf(cfg['model'])
            record['decode'] = producer.decode(args.decode_tokens)
            record['text'] = tokenizer.detokenize(record['decode']['tokens']).decode('utf-8', 'replace')
            record['recall'] = all(value in record['text'] for value in args.recall)
            if not record['recall']:
                raise ValueError('Recall failed')
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAIL', error=repr(error))
        raise
    finally:
        if producer is not None:
            record['cleanup'] = producer.close()
        args.output.write_text(json.dumps(record, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--recall', nargs='+', help='Required case-sensitive answer substrings')
    p.add_argument('--decode-tokens', type=int, default=48)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.trace.log').exists():
        p.error('Measurement and trace paths must be new')
    if not 0 < args.decode_tokens <= 256:
        p.error('Decode count must be 1..256')
    cfg = load(args.config)
    if args.profile:
        cfg['env_overrides'].update(LLAMA_CUDA_OP_PROFILE='1', LLAMA_CUDA_OP_PROFILE_NODES='1',
            LLAMA_CUDA_OP_PROFILE_DUMP_EACH='1', LLAMADECODE_TILE_TIMING='1')
    if args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid']
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        run(args, cfg)
    else:
        job = OwnedProcess(cfg, [sys.executable, '-B', str(Path(__file__).resolve()),
            *sys.argv[1:], '--worker'], 'singlepass-measure', cwd=args.output.resolve().parent, gpu=True)
        receipt = job.wait()
        args.output.with_suffix('.guard.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        if receipt.get('exit_code') or receipt.get('reason'):
            raise RuntimeError('Measurement failed; inspect the guard receipt')


if __name__ == '__main__':
    main()
