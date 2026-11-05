"""An explicit scratch location must not borrow or disable the native GPU guard."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import uuid

import pytest
from slstack.config import load
from slstack.lease import AdmissionRefused, active_holds, gpu_lease


def configured_root():
    root=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('managed-transient-'+uuid.uuid4().hex)
    root.mkdir()
    cfg=load(overrides=dict(gpu_uuid='GPU-test',host_budget_gib=.5,min_available_gib=2,lease_minutes=1))
    cfg.update(runtime_root=str(root/'installed/runtime'),hold_file=str(root/'holds'),task='DIRECTIONS')
    (root/'holds').write_text('')
    return root,cfg


def test_transient_root_keeps_independent_admission_and_original_config():
    from slstack.combo_proxy import transient_runtime_config
    root,cfg=configured_root();before=copy.deepcopy(cfg)
    target=root/'scratch';original_env=dict(cfg.get('env_overrides',{}))
    actual=transient_runtime_config(cfg,{'SLSTACK_TRANSIENT_RUNTIME_ROOT':str(target)})
    assert cfg==before and '_gpu_parent_lease' not in actual
    assert actual['runtime_root']==str(target)
    changed=copy.deepcopy(actual);changed['runtime_root']=cfg['runtime_root'];changed['env_overrides']=original_env
    assert changed==cfg
    for name,directory in (('TEMP','temp'),('TMP','temp'),('TMPDIR','temp'),('CUDA_CACHE_PATH','cuda-cache')):
        assert actual['env_overrides'][name]==str(target/directory)
    with patch('slstack.lease.subprocess.run',return_value=Mock(returncode=0,stdout='0, 32000, 0')):
        with gpu_lease(actual) as identifier:
            assert len(active_holds((root/'holds').read_text()))==1
            record=json.loads((target/'state/leases'/(identifier+'.json')).read_text())
            assert record['task']==actual['task'] and record['released'] is False
        assert active_holds((root/'holds').read_text())==[]
    assert cfg==before


@pytest.mark.parametrize('path',['relative/runtime','', 'bad\x00path'])
def test_transient_root_refuses_bad_explicit_paths(path):
    from slstack.combo_proxy import transient_runtime_config
    _,cfg=configured_root()
    with pytest.raises(ValueError):transient_runtime_config(cfg,{'SLSTACK_TRANSIENT_RUNTIME_ROOT':path})


def test_default_is_unchanged_and_file_target_refuses():
    from slstack.combo_proxy import transient_runtime_config
    root,cfg=configured_root()
    assert transient_runtime_config(cfg,{})==cfg
    target=root/'file';target.write_text('retain')
    with pytest.raises(ValueError):transient_runtime_config(cfg,{'SLSTACK_TRANSIENT_RUNTIME_ROOT':str(target)})
    assert target.read_text()=='retain'


def test_foreign_hold_still_blocks_and_conflicting_parent_root_refuses():
    from slstack.combo_proxy import transient_runtime_config
    root,cfg=configured_root()
    actual=transient_runtime_config(cfg,{'SLSTACK_TRANSIENT_RUNTIME_ROOT':str(root/'scratch')})
    (root/'holds').write_text('ACTIVE: QfnRuntime abc123; preserve foreign work\n')
    with patch('slstack.lease.subprocess.run') as command,pytest.raises(AdmissionRefused):
        with gpu_lease(actual):raise AssertionError('Foreign hold was ignored')
    command.assert_not_called()
    with pytest.raises(ValueError,match='conflicts'):
        transient_runtime_config(dict(cfg,_gpu_parent_lease='reference'),{'SLSTACK_TRANSIENT_RUNTIME_ROOT':str(root/'other')})
