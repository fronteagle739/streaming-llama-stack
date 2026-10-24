"""Six-turn routing acceptance; real native replay and full-vector probes.

Run under the shared dynamic-window guardian. Native replay receives the
same actual conversation transcript as the automatic router, including its
earlier answers; no alternate answer silently changes the compared prompt.
All checkpoint and reference-vector bytes remain in locked RAM.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from .combo_crossover import qualified_threshold
from .combo_measure import require_parity,stream_completion,milestones,transfer_summary
from .combo_ram import LockedRam
from .combo_session import ComboSession
from .journal import atomic_json
from .lease import gpu_lease

ROUTES=('native','ultra','native','native','ultra','native')


def prompts(threshold):
    # Unique facts let the later short question check retained document data.
    filler=' '.join(f'Observation {i}: review timing, calibration and maintenance before authorising the pilot.'
                    for i in range(max(8,(threshold+11)//12+4)))
    instruction=(' Answer the question first with concise key=value fields. Then list the integers1 through100 '
                 'separated by commas, with no other commentary. This explicit timing tail supplies enough decode tokens. Think briefly.')
    return [
        'TEST DS4COMBO. What should a careful pilot review check?'+instruction,
        'TEST document A. Project CERULEAN-47 is owned by Juniper, has capacity 217 units, and requires a calibration review.\n'+filler+
            '\nQuestion: state the project owner and capacity, then discuss its review.'+instruction,
        'From document A, which team owns CERULEAN-47 and what is its capacity?'+instruction,
        'What review must CERULEAN-47 pass before its pilot?'+instruction,
        'TEST document B. Project COPPER-82 depends on CERULEAN-47 and has a budget of 63 credits.\n'+filler+
            '\nQuestion: identify the dependency and budget, then discuss the two projects.'+instruction,
        'Recall both documents: give CERULEAN-47 owner and capacity, and COPPER-82 dependency and budget.'+instruction]


def facts(index,message):
    text=(message.get('content','')+' '+message.get('reasoning_content','')).lower()
    required={1:('juniper','217'),2:('juniper','217'),3:('calibration',),
              4:('cerulean','63'),5:('juniper','217','63','cerulean')}.get(index,())
    return dict(required=list(required),missing=[word for word in required if word not in text])


@contextmanager
def bounded_window(cfg,window,sessions):
    with gpu_lease(cfg) as lease:
        window(True,lease)
        try:yield
        finally:
            if any(s.poison and s.poison.startswith('Unconfirmed GPU release') for s in sessions):os._exit(72)
            window(False)


def run(cfg,output,reasoning_history,window):
    require_parity(cfg);threshold=qualified_threshold(cfg)
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    auto=ComboSession(cfg);reference=ComboSession(cfg,qualification_route='native')
    # Reference banks intentionally bind one conversation owner epoch. Both
    # GPU owners are sequential and certified retired before the other opens.
    reference.session=auto.session;reference.epoch=auto.epoch
    bank=None;transcript=[]
    proof=dict(status='RUNNING',reasoning_history=reasoning_history,session=auto.session,owner_epoch=auto.epoch,
               native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
               identity=cfg['combo_identity'],provider_schema=cfg['provider_schema'],threshold=threshold,
               prompts=prompts(threshold),max_tokens=384,reasoning_effort='low',turns=[])
    proof['decode_fixture']='Document-fact recall plus an explicit integer timing tail; the separate50K campaign measures review prose'
    atomic_json(output,proof)
    try:
        bank=LockedRam(32+384*4+384*129280*4,session=auto.session,
                       host_budget_gib=cfg['host_budget_gib'],min_available_gib=cfg['min_available_gib'])
        for index,text in enumerate(proof['prompts']):
            body=dict(model='deepseek-reasoner',messages=[*transcript,dict(role='user',content=text)],
                      include_previous_reasoning=reasoning_history,max_tokens=384,reasoning_effort='low',temperature=0,quality_speed=0)
            row=dict(turn=index+1,expected_route=ROUTES[index],status='RUNNING')
            proof['turns'].append(row);atomic_json(output,proof)
            with bounded_window(cfg,window,(reference,)):
                with reference.prepare(body,qualification_probe=lambda owner:owner.probe(bank,reference=True)) as (owner,rendered,metrics):
                    expected=stream_completion(owner,rendered);reference_dir=owner.directory
                row['native_replay']=dict(result=expected,metrics=metrics,
                    engine_milestones=milestones((reference_dir/'stderr.log').read_bytes()))
                atomic_json(output,proof)
            with bounded_window(cfg,window,(auto,)):
                try:
                    with auto.prepare(body,qualification_probe=lambda owner:owner.probe(bank,reference=False)) as (owner,rendered,metrics):
                        actual=stream_completion(owner,rendered);native_dir=owner.directory
                        actual['provider_final']=owner.quality()
                        actual['total_first_token_seconds_including_qualification']=metrics['prepared_seconds']+actual['native_api_first_token_seconds']
                    row.update(result=actual,metrics=metrics,engine_milestones=milestones((native_dir/'stderr.log').read_bytes()),
                               greedy_message_byte_exact=(json.dumps(actual['message'],ensure_ascii=False,separators=(',',':')).encode()==
                                                          json.dumps(expected['message'],ensure_ascii=False,separators=(',',':')).encode()),
                               content_check=facts(index,actual['message']))
                    row.update(transfer_summary(metrics))
                    if metrics['route']!=ROUTES[index]:raise ValueError('Alternating route did not match measured new-token crossover')
                    if index and metrics['reused_prefix_tokens']<=0:raise ValueError('Conversation prefix was not reused')
                    if ROUTES[index]=='native' and not row['greedy_message_byte_exact']:
                        raise ValueError('Native-routed turn differs from native-only replay of the same transcript')
                    if metrics['probe']['vectors']!=384:raise ValueError('Incomplete384-vector turn comparison')
                    if row['content_check']['missing']:raise ValueError('Earlier document facts were not recalled')
                    transcript=[*body['messages'],actual['message']]
                    row['status']='PASS';atomic_json(output,proof)
                finally:atomic_json(output,proof)
            # Replay and router each acquire a fresh independently bounded lease.
        proof['status']='PASS'
    except BaseException as exc:
        proof['status']='FAIL';proof['error']=repr(exc);raise
    finally:
        auto.close();reference.close()
        if bank is not None:bank.close()
        atomic_json(output,proof)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--reasoning-history',choices=['on','off'],required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--window',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    # Share the exact marker implementation used by the normal Studio router.
    from .combo_proxy import Proxy
    marker=object.__new__(Proxy);marker.cfg=cfg;marker.run_id=args.run_id;marker.window_file=Path(args.window)
    marker.window(False)
    run(cfg,args.output,args.reasoning_history=='on',marker.window)
