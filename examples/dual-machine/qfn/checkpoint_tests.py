"""Authenticated malformed command probes; real owner state must stay unchanged."""
import copy,json,socket,subprocess
from pathlib import Path
import urllib.request


def checkpoint_negatives(url,output,model,phrase,proof_program,python):
    output=Path(output);output.mkdir(exist_ok=False);cfg=json.loads((output.parent/'settings.json').read_text())
    with urllib.request.urlopen(url+'/v1/qfn-test/frontier',timeout=60) as f:before=json.load(f)
    key=Path(cfg['flags']['pair_token_file']).read_text().strip()
    identity=cfg['producer'];port=cfg['profile']['rtx']['port']
    base=before['checkpoint'];seq=before['next_sequence'];sid=before['session_id']
    def request(body):
        with socket.create_connection(('127.0.0.1',port),timeout=30) as sock:
            stream=sock.makefile('rwb',buffering=0)
            def send(row):sock.sendall(json.dumps(row,separators=(',',':')).encode()+b'\n')
            send(dict(op='pair',token=key,model_sha256=identity['model_sha256'],compatibility_sha256=identity['compatibility_sha256']))
            paired=json.loads(stream.readline())
            if paired.get('event')=='ready':paired=json.loads(stream.readline())
            if not paired.get('ok') or paired.get('event')!='paired':raise ValueError('Probe pairing failed')
            send(body);row=json.loads(stream.readline())
            if row.get('ok') is not False or not row.get('error'):raise ValueError('Malformed command was accepted')
            return row['error']
    common=dict(session=sid,sequence=seq)
    probes=[('wrong-session',dict(common,op='checkpoint',session='0'*32)),
        ('stale-sequence',dict(common,op='checkpoint',sequence=seq-1)),
        ('future-sequence',dict(common,op='checkpoint',sequence=seq+1)),
        ('ack-without-pending-import',dict(common,op='acknowledge',checkpoint=base)),
        ('replayed-acknowledgement',dict(common,op='acknowledge',sequence=seq-2,checkpoint=base)),
        ('unknown-anchor',dict(common,op='rewind',base=base,anchor=base,anchor_id='0'*32))]
    for field,value in [('epoch','0'*32),('revision',base['revision']+1),('mtp_rows',base['mtp_rows']-1),('tokens_sha256','0'*64)]:
        bad=copy.deepcopy(base);bad[field]=value
        probes.append(('wrong-base-'+field,dict(common,op='apply_decode_delta',base=bad,
            target=base['position']+1,target_mtp_rows=base['mtp_rows']+1,
            tokens_sha256=base['tokens_sha256'],bytes=40,max_bytes=1<<30,stall_seconds=30,verify_state=False)))
    probes.append(('replayed-payload-command',dict(common,op='handover',sequence=seq-2,
        wire_encoding='f16-index-v1',verify_state=False,max_bytes=16<<30,base=base,tokens=before['tokens'])))
    rows=[]
    record=dict(case='checkpoint-negative-claims',model=model,temperature=1,settings=dict(temperature=1),
        previous='1024-token sampled recall PASS',status='FAIL',actual='',source_prefix_tokens=100000,image_prefill='none')
    try:
        for name,body in probes:
            error=request(body)
            expected=('reverse delta base has not been acknowledged' if name.startswith('wrong-base-') else
                'paired rewind anchor or saved checkpoint differs' if name=='unknown-anchor' else
                'paired checkpoint acknowledgement differs from actual frontier' if name=='ack-without-pending-import' else
                'paired session/ordered sequence differs')
            rows.append(dict(case=name,error=error,expected=expected))
            assert expected in error,name+' failed for an unexpected reason: '+error
        with urllib.request.urlopen(url+'/v1/qfn-test/frontier',timeout=60) as f:after=json.load(f)
        assert after['checkpoint']==base,'Rejected command changed actual checkpoint'
        assert after['tokens']==before['tokens'],'Rejected command changed native tokens'
        assert after['next_sequence']==seq+1,'Rejected command consumed an ordered command sequence'
        assert after['session_id']==sid and after['buffered_bytes']==0,'Rejected claim replaced or left queued state'
        record.update(status='PASS',actual=str(len(rows))+' authenticated malformed, stale and repeated claims refused; actual paired checkpoint and tokens unchanged.',
            probes=rows,position=base['position'],commands_refused=len(rows),sequence_before=seq,sequence_after_probe_and_one_checkpoint=after['next_sequence'])
    except Exception as error:record.update(error=str(error),probes=rows)
    path=output/'checkpoint-negatives.json';path.write_text(json.dumps(record,indent=2),encoding='utf8')
    if proof_program:subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(path),
        '--name','feature-'+output.parent.name+'-checkpoint-negative-claims'],check=True)
    if record['status']!='PASS':raise AssertionError(record['error'])
    return record
