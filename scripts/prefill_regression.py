"""Rerun an exact DSV fixture with a native short warm-up and gate against control spread.

Run through the normal stack memory guard in your assigned idle GPU window.
All engine settings are read from the sealed configuration, never ambient env.
The baseline must contain matched warm-protocol controls, not cold historical rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.performance_contract import regression, validate, baseline_from_controls


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--baseline-from',nargs='+',type=Path,help='Create a baseline from at least two completed warm control receipts')
    p.add_argument('--record-control',action='store_true',help='Record a warm control for a new matched baseline')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.baseline_from:
        if a.config or a.fixture or a.baseline or a.record_control:
            p.error('--baseline-from is a separate receipt-validation operation')
        baseline=baseline_from_controls([json.loads(x.read_text()) for x in a.baseline_from])
        with a.output.open('x',encoding='utf-8') as stream:json.dump(baseline,stream,indent=2)
        return
    if not a.config or not a.fixture:p.error('--config and --fixture are required for native measurements')
    if bool(a.baseline)==a.record_control:
        p.error('Select --baseline or --record-control')
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    validate(cfg,'dsv41')
    if not cfg.get('producer_warmup_one_token') or cfg.get('producer_fixed_direct_open',True):
        raise ValueError('Regression gate requires --producer-warmup-one-token 1 in the recorded configuration')
    if cfg.get('producer_warmup_defer_prompt_data') is not True:
        raise ValueError('Regression gate requires measured-prompt reads to start after warm-up')
    raw=a.fixture.read_bytes()
    if len(raw)%4:raise ValueError('Expected uint32 LE native fixture')
    tokens=list(struct.unpack('<'+'I'*(len(raw)//4),raw))
    baseline=json.loads(a.baseline.read_text()) if a.baseline else None
    identity=dict(fixture_sha256=hashlib.sha256(raw).hexdigest(),prompt_tokens=len(tokens),
        settings_sha256=cfg['performance_manifest']['sha256'],pack_sha256=cfg['model_sha256'],
        clock='native-second-prompt-open-and-run',prompt_data_policy='after-warm-up',
        protocol='warm-short-then-different-prompt-same-process')
    # Refuse mismatched controls before touching a model or taking a lease.
    if baseline:
        regression(baseline,{**identity,'prefill_tok_s':max(baseline['control_tok_s'])})
    from slstack.lease import gpu_lease
    from slstack.combo_producer import ComboProducer
    from slstack.combo_release import certified
    from slstack.benchmark_idle import admit
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    producer=None;result=dict(identity,status='RUNNING')
    try:
        result['idle']=admit(cfg['gpu_uuid'])
        with gpu_lease(cfg):
            producer=ComboProducer(cfg,uuid.uuid4().hex,1,tokens,lambda:False)
            native=json.loads((producer.directory/'producer.json').read_text())
            warm=native['warm_measurement']
            if not warm['enabled'] or not warm['same_process_and_CUDA_backend'] or not warm.get('next_prompt_data_deferred'):
                raise RuntimeError('Native warm protocol witness missing')
            phase=native['phase_ms']
            prompt_ms=sum(phase.get(k,0) for k in ('inspect_admit_allocate','embedding_40_layers_head',
                'append_owner_admission','chunk_token_preparation','deferred_prompt_engram_admission'))
            if producer.checkpoint['position']!=len(tokens):raise RuntimeError('Warm token leaked into the measured frontier')
            result.update(prefill_tok_s=len(tokens)*1000/prompt_ms,prompt_ms=prompt_ms,
                legacy_phase_tok_s=len(tokens)*1000/phase['embedding_40_layers_head'],
                warm_measurement=warm,phase_ms=native['phase_ms'],
                settings_receipt=json.loads((producer.directory/'performance-settings.json').read_text()))
            cleanup=producer.close();producer=None
            if not certified(cleanup):raise RuntimeError('Native cleanup not certified')
            # Keep the certificate even when the comparison raises: a failed
            # performance gate still has a completed, certified native close.
            result['cleanup']=cleanup
            result.update(gate=regression(baseline,result) if baseline else 'CONTROL_ONLY',status='PASS')
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        if producer:result['cleanup']=producer.close()
        with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)


if __name__=='__main__':main()
