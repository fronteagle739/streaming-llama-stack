"""A packaged launcher borrows a real ancestor lease without changing its config."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock,patch
import uuid

from slstack.config import load
from slstack.lease import active_holds,gpu_lease
from slstack.combo_proxy import inherit_launch_lease


def test_real_managed_child_inherits_and_invalid_reference_refuses():
    root=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('managed-parent-'+uuid.uuid4().hex)
    root.mkdir();ledger=root/'holds';ledger.write_text('')
    cfg=load(overrides=dict(gpu_uuid='GPU-test',host_budget_gib=.5,min_available_gib=2,lease_minutes=1))
    cfg.update(runtime_root=str(root),hold_file=str(ledger))
    original_run=subprocess.run
    def gpu_only(command,*args,**kwargs):
        if command[0]=='nvidia-smi':return Mock(returncode=0,stdout='0, 32000, 0')
        return original_run(command,*args,**kwargs)
    with patch('slstack.lease.subprocess.run',side_effect=gpu_only),gpu_lease(cfg) as identity:
        record=root/'state/leases'/(identity+'.json')
        row=json.loads(record.read_text())
        transient=record.parent.parent/'managed-runtime'
        record.write_text(json.dumps(dict(row,child_runtime_root=str(transient))))
        source=root/'config.json';source.write_text(json.dumps(cfg));original=source.read_bytes()
        code='''import json,sys
from slstack.combo_proxy import inherit_launch_lease
from slstack.lease import gpu_lease
cfg=inherit_launch_lease(json.load(open(sys.argv[1])))
with gpu_lease(cfg) as token:print(json.dumps(dict(token=token,parent=cfg['_gpu_parent_lease'],runtime=cfg['runtime_root'],temp=cfg['env_overrides']['TEMP'])))
'''
        env=dict(os.environ,SLSTACK_PARENT_GPU_LEASE=str(record))
        result=original_run([sys.executable,'-B','-c',code,str(source)],env=env,capture_output=True,text=True,timeout=15)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)==dict(token=identity,parent=str(record),runtime=str(transient),temp=str(transient/'temp'))
        assert source.read_bytes()==original and len(active_holds(ledger.read_text()))==1
        row=json.loads(record.read_text());record.write_text(json.dumps(dict(row,released=True)))
        refused=original_run([sys.executable,'-B','-c',code,str(source)],env=env,capture_output=True,text=True,timeout=15)
        assert refused.returncode!=0 and 'Inherited GPU reservation' in refused.stderr
        record.write_text(json.dumps(row))
        record.write_text(json.dumps(dict(row,child_runtime_root=str(root.parent/'outside-task'))))
        refused=original_run([sys.executable,'-B','-c',code,str(source)],env=env,capture_output=True,text=True,timeout=15)
        assert refused.returncode!=0 and 'scratch root' in refused.stderr
        record.write_text(json.dumps(row))
    assert active_holds(ledger.read_text())==[]


def test_normal_launch_keeps_config_and_conflicting_parent_is_refused():
    cfg=dict(task='fixture')
    assert inherit_launch_lease(cfg,{})==cfg and '_gpu_parent_lease' not in cfg
    try:inherit_launch_lease(dict(cfg,_gpu_parent_lease='one'),{'SLSTACK_PARENT_GPU_LEASE':'two'})
    except ValueError:pass
    else:raise AssertionError('Conflicting parent silently replaced')
