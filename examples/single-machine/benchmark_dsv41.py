"""Reproduce the DSV4.1 packed FP4 table at exact 100K or 199,952 tokens.

Use scripts/qfn_dual_guard.py for the controller; native children also retain
their own memory guards. Provide explicit local model/build paths in --config.
"""
import argparse, hashlib, json, os, struct, sys, time, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slstack.combo_producer import ComboProducer
from slstack.combo_release import certified
from slstack.lease import gpu_lease, HoldWatcher
from slstack.ds4_render import Tokenizer
from slstack.generation import defaults,publish,vocabulary,validate
from slstack.performance_contract import seal
from scripts.dsv41_100k_ram_profile import load_performance_config,add_performance_flags,apply_performance_flags
VOCAB=129280
DISCARD=100
MEASURE=128

def settings(producer, cfg, tokenizer, bias):
    row = defaults()
    row.update(per_phase=False, value=bias)
    row['sampler']['temperature'] = 0
    publish(cfg['generation_control_file'], validate(row))
    quality = producer.quality(bias)
    assert quality['supported'] and quality['value'] == bias
    generation = producer.generation(dict(control_file=cfg['generation_control_file'],
        vocabulary_file=vocabulary(tokenizer, producer.directory), think_open=tokenizer.think_start,
        think_close=tokenizer.think_end, dsml=tokenizer.dsml, reasoning=False, seed=1,
        overrides={'temperature': 0}))
    assert generation['generation_control_version'] == 1
    return dict(control=row, quality=quality, generation=generation)

def compare_window(left_path, right_path):
    """Validate every float; score exactly 128 full vocab predictive distributions."""
    import numpy as np
    size = (DISCARD + MEASURE + 1)*VOCAB*4
    assert left_path.stat().st_size == right_path.stat().st_size == size
    kl, details, matches = [], [], 0
    maximum = 0.0
    with left_path.open('rb') as left, right_path.open('rb') as right:
        for index in range(DISCARD+MEASURE+1):
            x = np.frombuffer(left.read(VOCAB*4), dtype='<f4').astype(np.float64)
            y = np.frombuffer(right.read(VOCAB*4), dtype='<f4').astype(np.float64)
            assert x.size == y.size == VOCAB and np.isfinite(x).all() and np.isfinite(y).all()
            if not DISCARD <= index < DISCARD+MEASURE:
                continue
            top_left, top_right = int(np.argmax(x)), int(np.argmax(y))
            matches += top_left == top_right
            maximum = max(maximum, float(np.max(np.abs(x-y))))
            x -= x.max(); y -= y.max()
            logp = x - np.log(np.exp(x).sum()); logq = y - np.log(np.exp(y).sum())
            value = float(np.dot(np.exp(logp), logp-logq))
            kl.append(value)
            details.append(dict(index=index, position=100000+index, kl=value, top_left=top_left, top_right=top_right))
        assert left.read(1) == right.read(1) == b''
    return dict(rows=MEASURE, vocab=VOCAB, validated_full_rows=DISCARD+MEASURE+1,
        scored_row_range=[DISCARD, DISCARD+MEASURE], kl_mean=float(np.mean(kl)), kl_max=float(np.max(kl)),
        top1_agreement=matches/MEASURE, top1_matches=matches, maxabs=maximum,
        reference_sha256=hashlib.file_digest(left_path.open('rb'), 'sha256').hexdigest(),
        candidate_sha256=hashlib.file_digest(right_path.open('rb'), 'sha256').hexdigest(), row_details=details)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--fixture-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bias',type=float,required=True,choices=(0,.35,1.36125))
    p.add_argument('--mode',required=True,choices=('timing','quality'))
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-logits',type=Path)
    p.add_argument('--validate-only',action='store_true')
    add_performance_flags(p)
    a=p.parse_args()
    cfg=load_performance_config(a.config,experiment=a.performance_experiment)
    cfg=apply_performance_flags(cfg,a)
    cfg.update(task='DsvBenchmark',generation_control_file=str(a.output.with_suffix('.generation.json')),
        stack_preferences_file=str(a.output.with_suffix('.preferences.json')))
    cfg.setdefault('env_overrides',{}).update(DS4_CACHE_AFFINITY_WARMUP_TOKENS='0',DS4_CACHE_AFFINITY_RAMP_TOKENS='0')
    cfg=seal(cfg,'dsv41')
    raw=a.fixture.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.fixture_sha256:raise ValueError('Fixture SHA256 differs')
    if len(raw) not in (400000,799808):raise ValueError('Fixture must contain exactly 100000 or 199952 LE32 tokens')
    ids=list(struct.unpack('<'+'i'*(len(raw)//4),raw))
    if not all(0<=t<VOCAB for t in ids):raise ValueError('Fixture token outside full vocabulary')
    if a.mode=='quality' and not a.reference:p.error('--reference is required for quality')
    if a.validate_only:
        print(json.dumps(dict(validated=True,prompt_tokens=len(ids),manifest=cfg['performance_manifest'])))
        return
    if a.output.exists():raise FileExistsError(a.output)
    from slstack.benchmark_idle import admit
    row=dict(status='RUNNING',mode=a.mode,bias=a.bias,prompt_tokens=len(ids),fixture_sha256=a.fixture_sha256,
        protocol=dict(temperature=0,seed=1,discarded_evaluated=DISCARD,measured_evaluated=MEASURE,
            full_vocabulary=VOCAB,bias_warmup=0,bias_ramp=0),configuration=cfg)
    producer=None
    try:
        row['idle']=admit(cfg['gpu_uuid'])
        with gpu_lease(cfg) as lease:
            watcher=HoldWatcher(cfg['hold_file'],lease,task=cfg['task'])
            deadline=time.monotonic()+600
            stop=lambda:bool(watcher.reason()) or time.monotonic()>=deadline
            tokenizer=Tokenizer.from_gguf(cfg['model'])
            if len(tokenizer.tokens)!=VOCAB:raise ValueError('Model vocabulary differs')
            producer=ComboProducer(cfg,uuid.uuid4().hex,1,ids,stop)
            native=json.loads((producer.directory/'producer.json').read_text())
            row.update(directory=str(producer.directory),phase_ms=native['phase_ms'],plan=producer.prefill_plan,
                prefill_tok_s=len(ids)*1000/native['phase_ms']['embedding_40_layers_head'])
            assert producer.checkpoint['position']==len(ids)
            row['settings']=settings(producer,cfg,tokenizer,a.bias)
            if a.mode=='timing':
                reply=producer.decode(DISCARD+MEASURE+1,stop_tokens=[tokenizer.eos])
                assert not reply['stopped'] and not reply['cancelled'],'Natural stop interrupted the matched evaluation window'
                assert reply['evaluated_tokens']==len(reply['step_ms'])==DISCARD+MEASURE
                assert len(reply['tokens'])==DISCARD+MEASURE+1
                row.update(native_decode=reply,decode_tok_s=MEASURE*1000/sum(reply['step_ms'][DISCARD:]))
                if a.bias==0:
                    ref=a.output.with_suffix('.reference.le32')
                    with ref.open('xb') as f:f.write(struct.pack('<'+'i'*len(reply['tokens']),*reply['tokens']))
                    row['reference']=str(ref)
            else:
                ref=a.reference.read_bytes()
                if len(ref)!=(DISCARD+MEASURE+1)*4:raise ValueError('Reference requires exactly 229 sampled IDs')
                reference_ids=list(struct.unpack('<229i',ref))
                logits=a.output.with_suffix('.f32')
                if logits.exists():raise FileExistsError(logits)
                row['forced_reply']=producer.decode_forced(reference_ids,logits)
                assert row['forced_reply']['evaluated_tokens']==DISCARD+MEASURE
                assert row['forced_reply']['vectors']==DISCARD+MEASURE+1
                row.update(logits=str(logits),quality=compare_window(a.reference_logits or logits,logits))
                for item in row['quality']['row_details']:item['position']+=len(ids)-100000
            assert producer.checkpoint['position']==len(ids)+DISCARD+MEASURE
            row['cleanup']=producer.close();producer=None
            assert certified(row['cleanup']),row['cleanup']
            row['status']='PASS'
    except BaseException as error:
        row.update(status='FAILED',error=repr(error));raise
    finally:
        if producer:row['final_cleanup']=producer.close()
        with a.output.open('x',encoding='utf8') as f:json.dump(row,f,indent=2)
    print(json.dumps({k:row.get(k) for k in ('status','prompt_tokens','bias','prefill_tok_s','decode_tok_s','quality')}))

if __name__=='__main__':main()
