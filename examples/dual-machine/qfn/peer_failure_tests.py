"""Real owned-source termination, visible local fallback and guarded epoch rebuild."""
import json,time,subprocess,datetime
from pathlib import Path
import psutil
import urllib.request


def peer_failure(url,output,tokens,phrase,model,proof_program,python):
    from native_pair import stream_request
    output=Path(output);output.mkdir(exist_ok=False)
    root=output.parent;cfg=json.loads((root/'settings.json').read_text())
    def note(doing,latest,next_step):
        with Path(cfg['flags']['progress_log']).open('a',encoding='utf8') as f:
            f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' doing='+doing+'; latest='+latest+'; next='+next_step+'\n')
    def frontier():
        with urllib.request.urlopen(url+'/v1/qfn-test/frontier',timeout=60) as f:return json.load(f)
    def request(prompt,path):
        return stream_request(url,dict(prompt=prompt,max_tokens=128,temperature=1,top_k=20,top_p=.95,min_p=0,seed=1,stream=True),path)
    record=dict(case='peer-failure-local-fallback-and-rtx-rebuild',model=model,temperature=1,
        settings=dict(temperature=1,top_k=20,top_p=.95,seed=1),previous='fresh guarded native handover PASS',
        status='FAIL',actual='',source_prefix_tokens=100000,image_prefill='none')
    try:
        text,timings,mtp,_=request(tokens,output/'seed.events.jsonl');assert phrase in text,'Fault seed recall failed'
        before=frontier();owner=json.loads((root/'rtx/owner.json').read_text())
        guardian=psutil.Process(owner['guardian_pid']);engine=Path(cfg['flags']['engine']).resolve()
        children=[p for p in guardian.children(recursive=True) if p.is_running() and Path(p.exe()).resolve()==engine]
        assert len(children)==1,'The owned guardian does not have exactly one selected stable-path engine'
        target=children[0];pid,birth=target.pid,target.create_time()
        assert target.parent() and guardian.pid in [p.pid for p in target.parents()],'Source process ancestry changed'
        assert target.exe().casefold()==str(engine).casefold() and target.create_time()==birth,'Source process birth/executable changed'
        (output/'injected-source.json').write_text(json.dumps(dict(pid=pid,created=birth,guardian_pid=guardian.pid,engine=str(engine),action='terminate only this verified owned engine')),encoding='utf8')
        target.terminate();target.wait(timeout=15)
        note('owned source fault injected','verified engine PID'+str(pid)+' terminated, guard/limits unchanged','visible local fallback and certified guarded restart')
        prompt=before['tokens']+tokens[-12000:]
        local,local_timing,local_mtp,local_events=request(prompt,output/'fallback.events.jsonl')
        assert phrase in local,'Sampled recall failed after local fallback'
        notices=[r['prefill_progress'].get('notice','') for r in local_events if 'prefill_progress' in r]
        assert any('prefilling on Mac' in n for n in notices),'Peer failure was not exposed as a local fallback'
        fallback=frontier();assert fallback['tokens'][:len(prompt)]==prompt,'Fallback lost its valid prefix'
        assert not fallback['session_id'],'Failed remote session was retained as valid'
        assert local_timing['cache_n']==len(before['tokens']) and local_timing['prompt_n']==12000,'Fallback replayed or mischarged the valid long prefix'
        note('peer failure local fallback measured','T1 recallPASS, cached'+str(local_timing['cache_n'])+' new12000 rows; visible notice present','wait for fresh guarded RTX readiness')
        if proof_program:
            staged=output/'local-fallback.json'
            staged.write_text(json.dumps(dict(record,case='peer-failure-local-fallback',status='PASS',
                actual='The verified owned RTX engine exited; the API announced Mac fallback, retained its'+str(local_timing['cache_n'])+'-row prefix, evaluated only12000 new rows and recalled '+phrase+' at temperature1.',
                timings=local_timing,notices=notices),indent=2))
            subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(staged),
                '--name','feature-'+output.parent.name+'-local-fallback'],check=True)
        from slstack.progress import Window,activity
        from slstack.qfn_recovery import restart_allowed
        events=root/'rtx/recovery-events.jsonl';window=Window(cfg['profile']['transport']['stall_seconds'])
        while True:
            rows=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
            ready=[row for row in rows if row.get('event')=='peer_restarted' and row.get('previous_child_pid')==pid]
            if ready:break
            if (root/'rtx/cleanup.json').exists():raise RuntimeError('The source controller stopped before a certified restart completed')
            latest=json.loads((root/'rtx/owner.json').read_text())
            if psutil.pid_exists(latest['guardian_pid']):window.observe(activity(latest['guardian_pid'],Path(latest['log'])))
            elif latest['guardian_pid']!=guardian.pid and Path(latest['receipt']).exists():
                receipt=json.loads(Path(latest['receipt']).read_text())
                diagnostic=Path(latest['log']).read_text(errors='replace')[-16000:]
                if not restart_allowed(receipt,0,1,diagnostic):raise RuntimeError('Replacement source failed its guard/protection gate: '+str(receipt.get('reason')))
            if window.expired():raise TimeoutError('Guarded source restart stopped making progress')
            time.sleep(.25)
        newer=json.loads((root/'rtx/owner.json').read_text());assert newer['guardian_pid']!=guardian.pid,'Source guardian was reused'
        next_prompt=fallback['tokens']+tokens[-12000:]
        rebuilt,rebuild_timing,rebuild_mtp,rebuild_events=request(next_prompt,output/'rebuild.events.jsonl')
        assert phrase in rebuilt,'Sampled recall failed after guarded source rebuild'
        after=frontier();assert after['session_id'] and after['session_id']!=before['session_id'],'Producer identity was not rebuilt'
        assert after['checkpoint']['epoch']!=before['checkpoint']['epoch'],'Source owner epoch was reused across process death'
        assert after['tokens'][:len(next_prompt)]==next_prompt,'Rebuilt producer token prefix differs'
        assert any('Rebuilding' in r.get('prefill_progress',{}).get('notice','') or 'rebuilding' in r.get('prefill_progress',{}).get('notice','') for r in rebuild_events),'Full remote rebuild was not announced'
        assert rebuild_timing['cache_n']==0,'Lost source state was falsely counted as cached'
        note('guarded peer rebuild measured','T1 recallPASS; new source guardian'+str(newer['guardian_pid'])+' and owner epoch; full rebuilt rows'+str(rebuild_timing['prompt_n']),'drain catch-up and restore production')
        record.update(status='PASS',actual='Verified owned RTX process termination produced a visible Mac fallback over only12000 new rows, then a newly guarded RTX epoch rebuilt the complete request; both128-token temperature1 replies recalled '+phrase+'.',
            source_pid=pid,source_created=birth,old_guardian=guardian.pid,new_guardian=newer['guardian_pid'],
            local_fallback=dict(timings=local_timing,mtp=local_mtp,notices=notices,old_prefix_rows=len(before['tokens']),new_rows=12000),
            rebuilt=dict(timings=rebuild_timing,mtp=rebuild_mtp,new_epoch=after['checkpoint']['epoch'],position=after['position']),restart=ready[-1])
    except Exception as error:record['error']=str(error)
    path=output/'peer-failure.json';path.write_text(json.dumps(record,indent=2),encoding='utf8')
    if proof_program:subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(path),
        '--name','feature-'+output.parent.name+'-peer-failure'],check=True)
    if record['status']!='PASS':raise AssertionError(record['error'])
    return record
