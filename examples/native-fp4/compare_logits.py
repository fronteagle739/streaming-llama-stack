"""Bounded-row full-vocabulary KL against the frozen Q4 bias-zero control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
VOCAB=129280

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()

def metrics(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any():
        raise ValueError('NaN/+Inf logits')
    if not np.isfinite(a).any() or not np.isfinite(b).any():raise ValueError('No finite logit')
    a=a-np.max(a);b=b-np.max(b)
    lp=a-np.log(np.exp(a).sum());lq=b-np.log(np.exp(b).sum())
    p=np.exp(lp);mask=p>0
    kl=float(np.sum(p[mask]*(lp[mask]-lq[mask])))
    return dict(kl_control_to_candidate=kl,top1_control=int(np.argmax(a)),top1_candidate=int(np.argmax(b)))

def validate_pair(r,c,skip_first):
    if any(row['status']!='PASS' for row in (r,c)):raise ValueError('Incomplete input measurement')
    if r['bias']!=0:raise ValueError('Reference must be bias zero')
    if r['arm']==c['arm'] and r['model_sha256']!=c['model_sha256']:raise ValueError('Different models within an arm')
    if r['arm']!=c['arm'] and r['arm']!='q4':raise ValueError('Cross-model reference must be Q4')
    for key in ('input_sha256','prompt_tokens','forced_tokens','producer_sha256','decode_library_sha256','profile','yoco_keep'):
        if r[key]!=c[key]:raise ValueError('Unmatched '+key)
    count=len(r['forced_tokens'])
    if skip_first<1 or count-skip_first<3:raise ValueError('Insufficient forced continuation after excluded rows')
    warmup=c.get('affinity_warmup_tokens',100)+c.get('affinity_ramp_tokens',8)
    if r['bias']!=c['bias'] and skip_first<=warmup:
        raise ValueError('Dial comparison must exclude warmup/ramp; collect a longer forced continuation')
    return count

def compare(reference,candidate,out,skip_first=1):
    r=json.loads(reference.read_text());c=json.loads(candidate.read_text())
    count=validate_pair(r,c,skip_first)
    arrays=[]
    for record in (r,c):
        path=Path(record['logits_path'])
        if path.stat().st_size!=count*VOCAB*4:raise ValueError('Unexpected raw logit extent')
        if digest(path)!=record['logits_sha256']:raise ValueError('Changed logit file')
        arrays.append(np.memmap(path,dtype='<f4',mode='r',shape=(count,VOCAB)))
    # The frontier row was computed by prefill before quality_speed changed.
    # Exclude it consistently, so every compared row uses the selected dial.
    rows=[dict(row=i,**metrics(arrays[0][i],arrays[1][i])) for i in range(skip_first,count)]
    kls=[row['kl_control_to_candidate'] for row in rows]
    report=dict(status='PASS',reference=str(reference),candidate=str(candidate),
        label=c['label'],reference_sha256=digest(reference),candidate_sha256=digest(candidate),
        input_sha256=r['input_sha256'],producer_sha256=r['producer_sha256'],
        vocabulary=VOCAB,forced_rows=count,compared_rows=len(rows),excluded_rows=list(range(skip_first)),
        exclusion_reason='Frontier precedes the live setter; requested warmup/ramp prefix also excluded',
        kl_direction=f"{r['arm']} bias zero to candidate; natural logarithm",
        mean_kl=float(np.mean(kls)),max_kl=max(kls),
        top1_agreement=sum(row['top1_control']==row['top1_candidate'] for row in rows)/len(rows),rows=rows)
    for row in rows:
        if not math.isfinite(row['kl_control_to_candidate']):row['kl_control_to_candidate']='infinite'
    for key in ('mean_kl','max_kl'):
        if not math.isfinite(report[key]):report[key]='infinite'
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:report[key] for key in ('status','label','compared_rows','mean_kl','max_kl','top1_agreement')}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--skip-first',type=int,default=1)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:
        a=np.log([.75,.25]);b=np.log([.5,.5]);m=metrics(a,b)
        expected=.75*math.log(1.5)+.25*math.log(.5)
        assert abs(m['kl_control_to_candidate']-expected)<1e-14
        assert metrics(a,a)['kl_control_to_candidate']==0
        assert abs(metrics(a+100,b-100)['kl_control_to_candidate']-expected)<1e-13
        record=dict(status='PASS',arm='q4',bias=0,model_sha256='model',input_sha256='input',
                    prompt_tokens=49555,forced_tokens=list(range(140)),producer_sha256='producer',
                    decode_library_sha256='decode',profile='ultra',yoco_keep=140)
        biased=dict(record,bias=1.36125)
        try:validate_pair(record,biased,1)
        except ValueError as error:assert 'warmup/ramp' in str(error)
        else:raise AssertionError('Accepted an unmeasured dial')
        assert validate_pair(record,biased,109)==140
        for key in ('input_sha256','forced_tokens','decode_library_sha256'):
            try:validate_pair(record,dict(record,**{key:'different'}),1)
            except ValueError:pass
            else:raise AssertionError('Accepted unmatched '+key)
        print('PASS: analytic KL, shift invariance, warmup/ramp gate, and matched-input/library gates')
    else:
        if not all((args.reference,args.candidate,args.out)):p.error('reference, candidate and out are required')
        compare(args.reference,args.candidate,args.out,args.skip_first)
