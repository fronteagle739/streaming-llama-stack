"""Hardware gate; launched only by the independent19-minute GPU guardian.

Full vectors and canonical checkpoints stay in explicitly locked RAM. No
model judge, quantized replacement or disk checkpoint is used.
"""
import json,os,sys,threading,time,secrets
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.combo_native import Native
from slstack.combo_producer import ComboProducer
from slstack.combo_ram import LockedRam
from slstack.combo_state import contract_from_checkpoint,verify_payload,payload_bytes
from slstack.combo_release import release_barrier
from slstack.journal import atomic_json
from slstack.native_broker import vendor


def main(config_path,output,resume_full=None):
    cfg=json.loads(Path(config_path).read_text());vendor()
    from studio.dsv41_windows_runner import _Native
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    device=_Native();baseline=device.gpu(0,cfg['gpu_uuid'])
    session=secrets.token_hex(16);epoch=secrets.randbits(63) or 1;cancel=threading.Event()
    childcfg=dict(cfg,host_budget_gib=cfg['native_host_budget_gib'])
    proof=dict(status='RUNNING',session=session,owner_epoch=epoch,releases=[],
               native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
               native_prefill_chunk=cfg['native_prefill_chunk'],
               decoder_baseline='035c04d2b17024132a9ee37932f7e95b72b07822',
               producer_source_revision=cfg['producer_source_revision'],identity=cfg['combo_identity'])
    owner=None;state=None;vectors=None
    try:
        proof['phase']='native_startup';atomic_json(output,proof)
        owner=Native(childcfg,session,epoch,1,cancel.is_set)
        capability=owner.quality()
        proof['native_models']=owner.http('/v1/models')
        if capability.get('supported') is not True or capability.get('phases_supported') is not True:
            raise ValueError('Current native quality/speed provider schema is unavailable')
        proof['provider_schema']=cfg['provider_schema']
        proof['provider_initial']=capability
        proof['provider_points']=[]
        for name,value in cfg['provider_schema']['named_points'].items():
            actual=owner.quality(value)
            if abs(actual.get('value',-1)-value)>1e-5:raise ValueError('Native quality/speed named point differs')
            proof['provider_points'].append(dict(name=name,value=value,reply=actual))
        phases=dict(reasoning=0,tool_call=.35,answer=.65,code=.9)
        actual=owner.quality(phases)
        if actual.get('per_phase') is not True or any(abs(actual.get('phases',{}).get(k,-1)-v)>1e-5 for k,v in phases.items()):
            raise ValueError('Native four-phase quality/speed setter differs from the bound provider')
        proof['provider_phase_validation']=actual
        owner.quality(0)
        prompt=(Path(__file__).parent/'fixtures'/'combo-parity-prompt.txt').read_text(encoding='utf-8')
        body=dict(model='deepseek-chat',messages=[dict(role='user',content=prompt)],max_tokens=384,temperature=0)
        tokens=owner.render(body)
        if len(tokens)<128:raise ValueError('Qualification prompt must cross raw-ring and64KiB owner-page boundaries')
        proof['phase']='native_prefill';atomic_json(output,proof)
        native_cp=owner.sync(tokens);position=len(tokens)
        state=LockedRam(payload_bytes(position),session=session,host_budget_gib=cfg['host_budget_gib'],min_available_gib=cfg['min_available_gib'])
        owner.export(state)
        native_contract=contract_from_checkpoint(native_cp,cfg['combo_identity'],session,epoch,1,'unbiased')
        proof['native_state']=verify_payload(state,native_contract,native_cp['payload_sha256'])
        vectors=LockedRam(32+384*4+384*129280*4,session=session,host_budget_gib=cfg['host_budget_gib'],min_available_gib=cfg['min_available_gib'])
        proof['phase']='native_reference384';atomic_json(output,proof)
        started=time.monotonic();proof['reference']=owner.probe(vectors,reference=True);proof['reference_seconds']=time.monotonic()-started
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        # Reuse a completed full comparison after an interrupted later gate.
        # The in-RAM reference is regenerated; its complete bank must match.
        if resume_full:
            from slstack.journal import sha256
            prior=json.loads(Path(resume_full).read_text())
            binding=json.loads(Path(cfg['combo_core_binding']).read_text())
            if (binding['status']!='PASS' or binding['old_native_sha256']!=prior['native_sha256'] or
                binding['native_sha256']!=cfg['native_sha256'] or
                prior['producer_sha256']!=cfg['producer_sha256'] or prior['identity']!=cfg['combo_identity'] or
                prior['reference']['reference_sha256']!=proof['reference']['reference_sha256'] or
                prior['comparison']['vectors']!=384 or len(prior['releases'])<3):
                raise ValueError('Interrupted full comparison cannot be bound to this unchanged core/reference')
            if any(sha256(path)!=value for path,value in binding['unchanged'].items()):
                raise ValueError('Native core/fixture changed after retained comparison')
            for released in prior['releases'][:3]:
                if not all(released['cleanup'].get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
                    raise ValueError('Retained completed comparison lacks certified ownership release')
            proof['reused_full']=dict(path=str(resume_full),sha256=sha256(resume_full),
                core_binding_sha256=sha256(cfg['combo_core_binding']),releases=prior['releases'][:3],
                original_native_sha256=prior['native_sha256'],original_guardian_receipt_missing=True)
            for key in ('comparison','comparison_seconds','ultra_prefill_wall_seconds','ultra_export','ultra_state'):
                proof[key]=prior[key]
        else:
            full_comparison(owner,device,cfg,childcfg,session,epoch,tokens,cancel,baseline,state,vectors,proof,output)
        # Independently exercise native -> producer INITIAL BASE import.
        base_position=max(2,(position//2)&~1)
        if base_position>=position:raise ValueError('Qualification prompt is too short for a nonempty imported base')
        proof['phase']='native_base_prefill';atomic_json(output,proof)
        owner=Native(childcfg,session,epoch,1,cancel.is_set,runner=device)
        base_cp=owner.sync(tokens[:base_position]);owner.export(state)
        base_contract=contract_from_checkpoint(base_cp,cfg['combo_identity'],session,epoch,3,'unbiased')
        proof['base_state']=verify_payload(state,base_contract,base_cp['payload_sha256'])
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        proof['phase']='ultra_base_import_append';atomic_json(output,proof)
        owner=ComboProducer(cfg,session,epoch,tokens[base_position:],cancel.is_set,base=(state,base_contract),runner=device)
        proof['base_import_prefill_wall_seconds']=owner.prefill_seconds
        appended_cp=owner.checkpoint;proof['appended_export']=owner.export(state)
        appended_contract=contract_from_checkpoint(appended_cp,cfg['combo_identity'],session,epoch,4,'unbiased')
        appended_payload=verify_payload(state,appended_contract)
        proof['appended_state']=appended_payload
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        proof['phase']='native_appended_import';atomic_json(output,proof)
        owner=Native(childcfg,session,epoch,1,cancel.is_set,restore=(state,appended_payload),runner=device)
        proof['phase']='appended_comparison384';atomic_json(output,proof)
        proof['appended_comparison']=owner.probe(vectors,reference=False)
        if proof['appended_comparison']['reference_sha256']!=proof['reference']['reference_sha256']:
            raise ValueError('Reference bank changed during native-base append qualification')
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        proof['status']='PASS';proof['quality_scope']='Completed384-vector KL/top1 comparison; native byte equality is a separate mandatory gate. ULTRA arithmetic is approximate.'
    except BaseException as exc:
        proof['status']='FAIL';proof['error']=repr(exc);raise
    finally:
        if owner is not None:
            try:proof['failure_cleanup']=release_barrier(owner,device,cfg,baseline,lambda:False)
            except BaseException as exc:proof['cleanup_error']=repr(exc)
        if vectors is not None:vectors.close()
        if state is not None:state.close()
        atomic_json(output,proof)


def full_comparison(owner,device,cfg,childcfg,session,epoch,tokens,cancel,baseline,state,vectors,proof,output):
    try:
        proof['phase']='ultra_full_prefill';atomic_json(output,proof)
        owner=ComboProducer(cfg,session,epoch,tokens,cancel.is_set,runner=device)
        proof['ultra_prefill_wall_seconds']=owner.prefill_seconds
        ultra_cp=owner.checkpoint
        proof['ultra_export']=owner.export(state)
        ultra_contract=contract_from_checkpoint(ultra_cp,cfg['combo_identity'],session,epoch,2,'unbiased')
        ultra_payload=verify_payload(state,ultra_contract)
        proof['ultra_state']=ultra_payload
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        proof['phase']='native_full_import';atomic_json(output,proof)
        owner=Native(childcfg,session,epoch,1,cancel.is_set,restore=(state,ultra_payload),runner=device)
        proof['phase']='full_ultra_comparison384';atomic_json(output,proof)
        started=time.monotonic();proof['comparison']=owner.probe(vectors,reference=False);proof['comparison_seconds']=time.monotonic()-started
        if proof['comparison']['reference_sha256']!=proof['reference']['reference_sha256']:
            raise ValueError('Reference bank changed across GPU ownership transitions')
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
    finally:
        if owner is not None:
            try:proof['failure_cleanup']=release_barrier(owner,device,cfg,baseline,lambda:False)
            except BaseException as exc:proof['cleanup_error']=repr(exc)
        atomic_json(output,proof)


if __name__=='__main__':main(*sys.argv[1:])
