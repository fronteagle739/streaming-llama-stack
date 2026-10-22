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


def main(config_path,output):
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
        owner=Native(childcfg,session,epoch,1,cancel.is_set)
        capability=owner.quality()
        if capability.get('supported') is not True or capability.get('phases_supported') is not True:
            raise ValueError('Current native quality/speed provider schema is unavailable')
        proof['provider_schema']=cfg['provider_schema']
        proof['provider_points']=[]
        for name,value in cfg['provider_schema']['named_points'].items():
            actual=owner.quality(value)
            if abs(actual.get('value',-1)-value)>1e-5:raise ValueError('Native quality/speed named point differs')
            proof['provider_points'].append(dict(name=name,value=value,reply=actual))
        owner.quality(0)
        prompt=(Path(__file__).parent/'fixtures'/'combo-parity-prompt.txt').read_text(encoding='utf-8')
        body=dict(model='deepseek-chat',messages=[dict(role='user',content=prompt)],max_tokens=384,temperature=0)
        tokens=owner.render(body)
        if len(tokens)<128:raise ValueError('Qualification prompt must cross raw-ring and64KiB owner-page boundaries')
        native_cp=owner.sync(tokens);position=len(tokens)
        state=LockedRam(payload_bytes(position),session=session,host_budget_gib=cfg['host_budget_gib'],min_available_gib=cfg['min_available_gib'])
        owner.export(state)
        native_contract=contract_from_checkpoint(native_cp,cfg['combo_identity'],session,epoch,1,'unbiased')
        proof['native_state']=verify_payload(state,native_contract,native_cp['payload_sha256'])
        vectors=LockedRam(32+384*4+384*129280*4,session=session,host_budget_gib=cfg['host_budget_gib'],min_available_gib=cfg['min_available_gib'])
        started=time.monotonic();proof['reference']=owner.probe(vectors,reference=True);proof['reference_seconds']=time.monotonic()-started
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        # First test the full ULTRA prefill on the identical rendered IDs.
        owner=ComboProducer(cfg,session,epoch,tokens,cancel.is_set,runner=device)
        proof['ultra_prefill_wall_seconds']=owner.prefill_seconds
        ultra_cp=owner.checkpoint
        proof['ultra_export']=owner.export(state)
        ultra_contract=contract_from_checkpoint(ultra_cp,cfg['combo_identity'],session,epoch,2,'unbiased')
        ultra_payload=verify_payload(state,ultra_contract)
        proof['ultra_state']=ultra_payload
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        owner=Native(childcfg,session,epoch,1,cancel.is_set,restore=(state,ultra_payload),runner=device)
        started=time.monotonic();proof['comparison']=owner.probe(vectors,reference=False);proof['comparison_seconds']=time.monotonic()-started
        if proof['comparison']['reference_sha256']!=proof['reference']['reference_sha256']:
            raise ValueError('Reference bank changed across GPU ownership transitions')
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        # Independently exercise native -> producer INITIAL BASE import, not
        # merely an empty-session ULTRA full prefill. Both compare to the same
        # native reference. The even base also covers an odd target if present.
        base_position=max(2,(position//2)&~1)
        if base_position>=position:raise ValueError('Qualification prompt is too short for a nonempty imported base')
        owner=Native(childcfg,session,epoch,1,cancel.is_set,runner=device)
        base_cp=owner.sync(tokens[:base_position]);owner.export(state)
        base_contract=contract_from_checkpoint(base_cp,cfg['combo_identity'],session,epoch,3,'unbiased')
        proof['base_state']=verify_payload(state,base_contract,base_cp['payload_sha256'])
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        owner=ComboProducer(cfg,session,epoch,tokens[base_position:],cancel.is_set,base=(state,base_contract),runner=device)
        proof['base_import_prefill_wall_seconds']=owner.prefill_seconds
        appended_cp=owner.checkpoint;proof['appended_export']=owner.export(state)
        appended_contract=contract_from_checkpoint(appended_cp,cfg['combo_identity'],session,epoch,4,'unbiased')
        appended_payload=verify_payload(state,appended_contract)
        proof['appended_state']=appended_payload
        proof['releases'].append(release_barrier(owner,device,cfg,baseline,cancel.is_set));owner=None
        owner=Native(childcfg,session,epoch,1,cancel.is_set,restore=(state,appended_payload),runner=device)
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


if __name__=='__main__':main(*sys.argv[1:])
