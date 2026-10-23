"""Real request queue and independent Answer/Code acceptance under a GPU lease."""
import json,os,re,secrets,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.combo_native import Native
from slstack.combo_release import release_barrier
from slstack.native_broker import vendor
from slstack.journal import atomic_json


def main(config_path,output):
    cfg=json.loads(Path(config_path).read_text());vendor()
    from studio.dsv41_windows_runner import _Native
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    device=_Native();baseline=device.gpu(0,cfg['gpu_uuid']);owner=None
    proof=dict(status='RUNNING',native_sha256=cfg['native_sha256'],provider_schema=cfg['provider_schema'],requests=[])
    try:
        owner=Native(dict(cfg,host_budget_gib=cfg['native_host_budget_gib']),secrets.token_hex(16),secrets.randbits(63) or 1,1,lambda:False,runner=device)
        proof['invalid_requests']=[]
        for value in ({'cod':0},{'code':'invalid'},{'code':False},{'code\x00extra':0}):
            for endpoint,body in (('/v1/quality-speed',{'value':value}),('/v1/chat/completions',{'quality_speed':value,'messages':[]})):
                try:owner.http(endpoint,body)
                except RuntimeError as error:
                    if not str(error).startswith('Native HTTP 400:'):raise
                else:raise ValueError('Invalid request was silently accepted')
            proof['invalid_requests'].append(value)
        phases=dict(reasoning='fast',tool_call='balanced',answer='max_speed',code='max_quality')
        body=dict(model='deepseek-chat',reasoning_effort='none',quality_speed=phases,
            messages=[dict(role='user',content='TEST: Output only a fenced C++ code block. Inside it write the lines int value01 = 1; through int value40 = 40;, one declaration per line. Do not explain.')],
            temperature=0,max_tokens=160,stream=False)
        for index in range(2):
            proof['phase']='request_'+str(index+1);atomic_json(output,proof)
            response=owner.http('/v1/chat/completions',body)
            state=owner.quality();text=response['choices'][0]['message'].get('content','')
            if ('```' not in text or state['phases']['code']!=0 or
                abs(state['phases']['answer']-1.36125)>1e-5 or state['phase_decode']['code']['tokens']<=8):
                raise ValueError('Independent Code/Answer request did not reach real decode: '+repr(state))
            if state['phase']=='code' and state['applied_value']!=0:raise ValueError('Code inherited Answer during decode')
            proof['requests'].append(dict(response=response,native=state))
            if index==0:
                live=owner.quality(dict(answer='max_speed',code='fast'))
                if abs(live['phases']['code']-.65)>1e-5:raise ValueError('Live Code update failed')
                proof['live_update']=live
        proof['release']=release_barrier(owner,device,cfg,baseline,lambda:False)
        proof['native_directory']=str(owner.directory);owner=None;proof['status']='PASS'
    except BaseException as error:
        proof.update(status='FAIL',error=repr(error));raise
    finally:
        if owner is not None:
            try:proof['failure_cleanup']=release_barrier(owner,device,cfg,baseline,lambda:False)
            except BaseException as error:proof['cleanup_error']=repr(error)
        atomic_json(output,proof)


if __name__=='__main__':main(*sys.argv[1:])
