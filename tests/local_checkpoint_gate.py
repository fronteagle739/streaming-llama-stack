"""GPU qualification: compare continued KV/logits with save/relaunch/restore.

Run under the stack guardian with an idle GPU, --config and --out. The prompt
exceeds the YOCO replay window. Both branches append identical exact token IDs.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.combo_producer import ComboProducer
from slstack.ds4_render import Tokenizer, render
from slstack.lease import gpu_lease
from slstack.local_cache import Cache, options


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--config', required=True);parser.add_argument('--out', required=True)
    args = parser.parse_args();cfg = load(args.config);cfg['serving_mode'] = 'single_engine'
    cfg.setdefault('env_overrides', {})['LLAMADECODE_DS4_DECODE'] = '1'
    # Compare the same streaming prefill arithmetic on both branches. The
    # production short-append decode optimization uses a different kernel path.
    cfg['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    Path(cfg['runtime_root']).mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_gguf(cfg['model'])
    body = dict(messages=[dict(role='user', content=('Checkpoint continuation uses the original cache. ' * 60)+'What is 7+5?')], reasoning_effort='none')
    _, tokens, _ = render(body, tok);suffix = tok.rendered('The answer is')
    result = dict(prompt_tokens=len(tokens), appended_tokens=len(suffix), yoco_keep=cfg.get('yoco_keep'), context=cfg['context'])
    producer = None
    with gpu_lease(cfg):
        try:
            producer = ComboProducer(cfg, secrets.token_hex(16), secrets.randbits(63) or 1, tokens, lambda:False)
            cache = Cache(cfg);saved = cache.save(producer, tokens, {}, None)
            producer.prefill(suffix);expected = deepcopy(producer.checkpoint)
            producer.close();producer = None
            root, _ = options(cfg);reader = cache.verify(saved, root/saved['payload'])
            contract = deepcopy(saved['contract']);session, epoch = secrets.token_hex(16), secrets.randbits(63) or 1
            contract.update(request_id=session, owner_epoch=epoch)
            producer = ComboProducer(cfg, session, epoch, suffix, lambda:False, base=(reader, contract))
            actual = deepcopy(producer.checkpoint)
            result.update(bytes=saved['bytes'], save_seconds=saved['save_seconds'], expected=expected, actual=actual,
                          token_hash_equal=expected['tokens_sha256']==actual['tokens_sha256'],
                          logits_hash_equal=expected['logits_sha256']==actual['logits_sha256'],
                          owner_hashes_equal=expected['owner_sha256']==actual['owner_sha256'],
                          appended_only=producer.prefill_plan['prompt_tokens']==len(suffix))
            result['status'] = 'PASS' if expected==actual and result['appended_only'] else 'FAIL'
        except BaseException as exc:
            result.update(status='FAIL', error=repr(exc));raise
        finally:
            if producer is not None:producer.close()
            Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k not in ('expected','actual')}), flush=True)
    return result['status']=='PASS'


if __name__=='__main__':raise SystemExit(0 if main() else 1)
