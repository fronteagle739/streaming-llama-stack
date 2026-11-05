"""Real persistent-session proofs, enabled by --test-recall plus --test-features."""
import json
from pathlib import Path
import subprocess
import urllib.request


def forward_delta(url,output,tokens,phrase,model,settings,proof_program,python):
    from native_pair import stream_request
    output=Path(output);output.mkdir(exist_ok=False)
    def frontier():
        with urllib.request.urlopen(url+'/v1/qfn-test/frontier',timeout=30) as response:
            return json.load(response)
    before=frontier()
    # Use the actual sampled frontier, not re-tokenized rendered text. The
    # suffix is different work and exceeds the production local-append bound.
    suffix=tokens[-12000:]
    prompt=before['tokens']+suffix
    record=dict(case='paired-forward-delta',model=model,temperature=1,
        settings=settings,previous='native handover benchmark',status='FAIL',actual='',
        source_prefix_tokens=len(tokens),image_prefill='none')
    try:
        text,timing,mtp,_=stream_request(url,dict(prompt=prompt,max_tokens=128,temperature=1,
            top_k=20,top_p=.95,min_p=0,seed=1,stream=True),output/'forward.events.jsonl')
        after=frontier()
        record.update(actual=text,before={k:v for k,v in before.items() if k!='tokens'},
            after={k:v for k,v in after.items() if k!='tokens'},timings=timing,mtp=mtp,
            old_prefix_rows=len(before['tokens']),new_prefill_rows=len(suffix))
        assert phrase in text,'Sampled recall failed after the real owner round trip'
        assert after['initial_bytes']==before['initial_bytes'],'Follow-up resent a full state'
        assert 0<after['forward_bytes']-before['forward_bytes']<before['initial_bytes'],'No bounded forward delta'
        assert after['reverse_bytes']>before['reverse_bytes'],'Decode was not synchronized back to CUDA'
        assert after['tokens'][:len(prompt)]==prompt,'Imported token prefix changed'
        record['status']='PASS'
    except Exception as error:
        record['error']=str(error)
    path=output/'forward-delta.json';path.write_text(json.dumps(record,indent=2),encoding='utf8')
    if proof_program:
        subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(path),
            '--name','feature-'+output.parent.name+'-paired-forward-delta'],check=True)
    if record['status']!='PASS':raise AssertionError(record['error'])
    return record


def rewind_anchor(url,output,tokens,phrase,model,settings,proof_program,python):
    from native_pair import stream_request
    output=Path(output)
    def frontier():
        with urllib.request.urlopen(url+'/v1/qfn-test/frontier',timeout=30) as response:return json.load(response)
    before=frontier();anchor=before['anchor_position']
    assert 0<anchor<before['position'],'No paired pre-decode anchor was saved'
    prompt=before['tokens'][:anchor]+tokens[-12000:]
    record=dict(case='paired-anchor-rewind',model=model,temperature=1,settings=settings,
        previous='paired forward delta PASS',status='FAIL',actual='',source_prefix_tokens=len(tokens),image_prefill='none')
    try:
        text,timing,mtp,_=stream_request(url,dict(prompt=prompt,max_tokens=128,temperature=1,
            top_k=20,top_p=.95,min_p=0,seed=1,stream=True),output/'rewind.events.jsonl')
        after=frontier();record.update(actual=text,before={k:v for k,v in before.items() if k!='tokens'},
            after={k:v for k,v in after.items() if k!='tokens'},timings=timing,mtp=mtp,
            preserved_prompt_rows=anchor,new_prefill_rows=12000)
        assert phrase in text,'Sampled recall failed after paired anchor rewind'
        assert after['rewinds']>before['rewinds'],'No paired rewind occurred'
        assert after['initial_bytes']==before['initial_bytes'],'Rewind resent the full state'
        assert 0<after['forward_bytes']-before['forward_bytes']<before['initial_bytes'],'No forward delta after rewind'
        assert after['tokens'][:len(prompt)]==prompt,'Restored prefix differs'
        record['status']='PASS'
    except Exception as error:record['error']=str(error)
    path=output/'paired-rewind.json';path.write_text(json.dumps(record,indent=2),encoding='utf8')
    if proof_program:subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(path),
        '--name','feature-'+output.parent.name+'-paired-anchor-rewind'],check=True)
    if record['status']!='PASS':raise AssertionError(record['error'])
    return record
