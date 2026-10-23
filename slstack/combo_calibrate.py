"""Empirical native/ULTRA bracket, using Kimi's existing crossover equation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from .combo_measure import measure,require_parity
from .combo_crossover import crossover
from .journal import atomic_json

WORDS='light water screen sample motion colour timing control stable clear record measure careful test result review distance width angle source observation camera training repeat signal geometry method change check compare'.split()


class CalibrationPending(Exception):
    def __init__(self,words,repeat=0):self.words=words;self.repeat=repeat


def case(words):
    text='TEST DS4COMBO crossover measurement.\n'
    if words:text+='Context: '+' '.join(WORDS[(i*7+i//len(WORDS))%len(WORDS)] for i in range(words))+'.\n'
    text+='Write forty numbered measurement-review notes with concrete explanations. Continue until all forty notes are complete.'
    return dict(model='deepseek-chat',messages=[dict(role='user',content=text)],max_tokens=128,temperature=0,quality_speed=0)


def calibrate(cfg,output,*,pair_budget=1):
    require_parity(cfg);root=Path(cfg['runtime_root']);pairs={};new_pairs=0
    def sample(words,repeat=0):
        nonlocal new_pairs
        key=(words,repeat)
        if key in pairs:return pairs[key]
        suffix=f'-r{repeat}' if repeat else ''
        targets=[root/f'crossover-{route}-w{words:04d}{suffix}.json' for route in ('native','ultra')]
        if not all(target.exists() for target in targets):
            if new_pairs>=pair_budget:raise CalibrationPending(words,repeat)
            new_pairs+=1
        values={}
        for route in (('native','ultra') if not repeat else ('ultra','native')):
            target=root/f'crossover-{route}-w{words:04d}{suffix}.json'
            existing=json.loads(target.read_text()) if target.exists() else None
            if existing is not None:
                request_sha=hashlib.sha256(json.dumps(case(words),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if (existing.get('status')!='PASS' or existing.get('request_sha256')!=request_sha or existing.get('identity')!=cfg['combo_identity'] or
                    any(existing.get(k)!=cfg[k] for k in ('native_sha256','producer_sha256','provider_schema','native_prefill_chunk'))):
                    raise ValueError('Existing crossover case is failed or from a different profile; preserve it and choose an explicit new campaign')
                result=existing
            else:result=measure(cfg,case(words),target,route)
            turn=result['turns'][0]
            if '100' not in turn.get('engine_milestones',{}):
                raise ValueError('Crossover requires100 real decode tokens to charge cold-cache refill')
            values[route]=dict(first_token_seconds=result['turns'][0]['total_first_token_seconds'],
                prompt_tokens=result['turns'][0]['metrics']['prompt_tokens'],receipt=str(target),
                prefill_tok_s=result['turns'][0].get('ultra_prefill_tok_s'),
                decision_seconds=turn['metrics']['prepared_seconds']+turn['engine_milestones']['100']['seconds'])
        if values['native']['prompt_tokens']!=values['ultra']['prompt_tokens']:
            raise ValueError('Crossover paths rendered different native prompt IDs/counts')
        row=dict(words=words,repeat=repeat,tokens=values['native']['prompt_tokens'],native=values['native'],ultra=values['ultra'])
        row['native_minus_ultra_seconds']=values['native']['decision_seconds']-values['ultra']['decision_seconds']
        pairs[key]=row;atomic_json(root/'crossover-samples.json',dict(status='RUNNING',samples=list(pairs.values())))
        print('CROSSOVER_SAMPLE '+json.dumps(row),flush=True);return row
    for words in (0,64,256):sample(words)
    # Find an actual bracket rather than extrapolating a missing ULTRA win.
    # Each CLI admission adds at most one pair and preserves completed pairs,
    # so the campaign never needs a lease longer than19minutes.
    for words in (512,1024):
        ordered=sorted(pairs.values(),key=lambda v:v['tokens'])
        if any(a['native_minus_ultra_seconds']<0<b['native_minus_ultra_seconds'] for a,b in zip(ordered,ordered[1:])):break
        sample(words)
    for _ in range(4):
        ordered=sorted(pairs.values(),key=lambda v:v['tokens'])
        brackets=[(a,b) for a,b in zip(ordered,ordered[1:]) if a['native_minus_ultra_seconds']<0<b['native_minus_ultra_seconds']]
        if not brackets:raise ValueError('No measured native-win/ULTRA-win crossover bracket; automatic routing remains refused')
        low,high=min(brackets,key=lambda p:p[1]['tokens']-p[0]['tokens'])
        if high['tokens']-low['tokens']<=32:break
        middle=(low['words']+high['words'])//2
        if (middle,0) in pairs:break
        sample(middle)
    ordered=sorted(pairs.values(),key=lambda v:v['tokens'])
    brackets=[(a,b) for a,b in zip(ordered,ordered[1:]) if a['native_minus_ultra_seconds']<0<b['native_minus_ultra_seconds']]
    low,high=min(brackets,key=lambda p:p[1]['tokens']-p[0]['tokens'])
    repeated_low=sample(low['words'],1);repeated_high=sample(high['words'],1)
    if not repeated_low['native_minus_ultra_seconds']<0<repeated_high['native_minus_ultra_seconds']:
        raise ValueError('Reversed-order repeated crossover endpoints disagree; no automatic route qualification')
    all_samples=sorted(pairs.values(),key=lambda v:(v['tokens'],v['repeat']))
    native_fit=statistics.linear_regression([r['tokens'] for r in ordered],[r['native']['decision_seconds'] for r in ordered])
    if native_fit.slope<=0:raise ValueError('Native timing slope is not positive; no crossover claim')
    native_rate=1/native_fit.slope
    stream_rate=statistics.median(r['ultra']['prefill_tok_s'] for r in ordered if r['ultra']['prefill_tok_s'])
    if not stream_rate>native_rate:raise ValueError('Measured ULTRA prefill is not faster than native')
    # Route conservatively at the observed ULTRA-winning upper bracket. The
    # floor parameter is calibrated to that measured bracket, not presented
    # as an independently timed engine-load duration.
    threshold=high['tokens'];denominator=1/native_rate-1/stream_rate
    fitted_floor_ms=(threshold-.25)*denominator*1000
    if crossover(fitted_floor_ms,stream_rate,native_rate,1)!=threshold:raise ValueError('Kimi crossover round/clamp inconsistency')
    proof=dict(status='PASS',native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
        identity=cfg['combo_identity'],provider_schema=cfg['provider_schema'],context=cfg['context'],native_prefill_chunk=cfg['native_prefill_chunk'],
        stream_floor_ms=fitted_floor_ms,stream_tok_s=stream_rate,native_tok_s=native_rate,minimum_tokens=1,
        crossover_tokens=threshold,bracket_verified=True,repeated_bracket_verified=True,bracket=[low['tokens'],high['tokens']],
        decision_metric='Actual preparation plus first100 native decode tokens, including incremental cold-cache refill; TTFT recorded separately',
        floor_method='Kimi floor parameter fitted conservatively to the observed upper crossover bracket; not a separate load-time measurement',
        native_timing_fit=dict(slope=native_fit.slope,intercept=native_fit.intercept),samples=all_samples)
    atomic_json(output,proof);return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    try:calibrate(cfg,args.output)
    except CalibrationPending as exc:
        pending=dict(status='PENDING',next_words=exc.words,next_repeat=exc.repeat,reason='One measured pair per bounded GPU admission; no automatic routing qualification yet')
        atomic_json(args.output,pending);print(json.dumps(pending),flush=True)
