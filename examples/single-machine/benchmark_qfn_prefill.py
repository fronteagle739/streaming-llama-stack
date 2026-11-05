"""QFN one-change RAM/queue arm: short warm-up, different exact 100K prompt, same native process.

Requires a sealed explicit per-model configuration. QfnEngine owns the normal
RAM/paging guard and GPU lease. No decoding or sampler settings are altered.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from slstack.performance_contract import validate,record
from slstack.qfn_single import QfnEngine
from slstack.benchmark_idle import admit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));validate(cfg,'qfn')
    raw=a.fixture.read_bytes()
    if len(raw)!=400000:raise ValueError('QFN comparison requires exactly 100,000 native tokens')
    tokens=list(struct.unpack('<100000i',raw))
    # Use real tokens from a different passage and force a different first ID,
    # so the native prefix cache must report zero reused tokens.
    at=next(i for i,t in enumerate(tokens[1:],1) if t!=tokens[0])
    warm=tokens[at:at+32]
    engine=QfnEngine(cfg);result=dict(status='RUNNING',prompt_tokens=100000,
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        protocol='warm-short-then-different-prompt-same-process',warmup_tokens=warm)
    job=None
    try:
        result['idle']=admit(cfg['gpu_uuid'])
        os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        engine.start();job=engine.job
        from slstack.config import environment
        env=environment(cfg,'qfn-engine')
        env['LLAMADECODE_ENGRAM_SCAN_RINGS']=str(cfg['qfn_engram_scan_rings'])
        result['settings_receipt']=record(a.output.with_suffix('.settings.json'),cfg,'qfn',
            argv=engine.command(),environment=env,fixture_sha256=result['fixture_sha256'])
        def prefill(ids):
            events=list(engine.run(dict(op='prefill',tokens=ids),lambda e:'n_past' in e))
            row=events[-1]
            if not row.get('ok'):raise RuntimeError(row)
            return row
        result['warmup']=prefill(warm)
        result['prefill']=pf=prefill(tokens)
        if pf['reused']!=0 or pf['new_tokens']!=100000 or pf['n_past']!=100000 or pf['route']!='layer-major':
            raise RuntimeError('Warm-protocol fresh-prompt witness differs: '+str(pf))
        result.update(prefill_tok_s=100000/pf['seconds'],same_native_connection=True,status='MEASURED')
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        engine.retire('prefill comparison completed')
        if job is not None:
            receipt=json.loads(Path(job.receipt).read_text())
            result['guard']=receipt
            if not receipt.get('cleanup_confirmed') or receipt.get('exit_code') or receipt.get('reason'):
                result.update(status='FAIL',error='QFN guard or cleanup failed')
            elif result['status']=='MEASURED':result['status']='PASS'
        with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    if result['status']!='PASS':raise RuntimeError(result.get('error'))
    print(json.dumps({k:result[k] for k in ('status','prompt_tokens','prefill_tok_s')}))


if __name__=='__main__':main()
