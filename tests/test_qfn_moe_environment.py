"""Public QFN controls must survive the production adapter and its real guardian."""
import json
import sys

import pytest

from slstack.config import load
from slstack.envvars import apply, rows
from slstack.qfn_single import QfnEngine


@pytest.mark.parametrize('parts', ['1', '8', '64'])
def test_explicit_expert_read_parts_reaches_qfn_without_changing_other_scopes(parts, tmp_path):
    cfg = load(overrides={'runtime_root': str(tmp_path), 'env_overrides': {'DS4_DS_READ_PARTS': parts}})
    for scope in ('qfn-engine', 'ds4-win', 'rtx-llama'):
        assert apply({}, cfg, scope)['DS4_DS_READ_PARTS'] == parts


CONTROLS = {'QFN_MOE_PASS': '1', 'QFN_MOE_CHUNK': '16384',
            'QFN_PREFILL_RELEASE_CACHE': '1', 'QFN_MMQ_STREAM_K': '0'}


@pytest.mark.parametrize('scope', ['qfn-engine', 'rtx-llama'])
def test_cuda_profile_controls_reach_engine_scope(scope, tmp_path):
    chosen = {'LLAMA_CUDA_OP_PROFILE': '1', 'LLAMA_CUDA_OP_PROFILE_NODES': '1'}
    cfg = load(overrides={'runtime_root': str(tmp_path), 'env_overrides': chosen})
    delivered = apply({}, cfg, scope)
    assert all(delivered.get(key) == value for key, value in chosen.items())


def test_qfn_controls_preserve_studio_envelope_and_unset_defaults(tmp_path):
    cfg = load(overrides={'gpu_uuid': 'GPU-test', 'model': str(tmp_path / 'absent-probe.basalt')})
    studio, qfn = apply({}, cfg, 'studio'), apply({}, cfg, 'qfn-engine')
    assert all(qfn.get(key) == value for key, value in studio.items())
    assert not set(CONTROLS) & qfn.keys()
    catalog = {r['var']: r for r in rows()}
    for key in CONTROLS:
        assert catalog[key]['scopes'] == ['qfn-engine']
        assert catalog[key]['flag'] and catalog[key]['doc'] and catalog[key]['reads']


@pytest.mark.parametrize('key', ['QFN_MOE_PASS', 'QFN_PREFILL_RELEASE_CACHE', 'QFN_MMQ_STREAM_K'])
def test_invalid_boolean_fails_in_environment_adapter(key):
    with pytest.raises(ValueError):
        apply({}, {'env_overrides': {key: '2'}}, 'qfn-engine')


@pytest.mark.parametrize('chunk', ['0', '-1', 'nan', 'unlimited'])
def test_invalid_chunk_fails_in_environment_adapter(chunk):
    with pytest.raises(ValueError):
        apply({}, {'env_overrides': {'QFN_MOE_CHUNK': chunk}}, 'qfn-engine')


@pytest.mark.parametrize('chunk', ['16384', ''])
def test_production_qfn_launch_delivers_controls_to_real_guarded_child(tmp_path, chunk):
    chosen = {**CONTROLS, 'QFN_MOE_CHUNK': chunk, 'QFN_MTP': '1', 'QFN_CARRY_BF16': '1', 'QFN_FUSED_HC': '1', 'DS4_DS_READ_PARTS': '8'}
    output = tmp_path / 'observed.json'
    cfg = load(overrides={'runtime_root': str(tmp_path), 'gpu_uuid': 'GPU-test', 'model': str(tmp_path / 'probe.basalt'),
        'host_budget_gib': .25, 'min_available_gib': 6, 'job_commit_gib': 1,
        'env_overrides': chosen})
    engine = QfnEngine(cfg)
    code = 'import json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({k:os.environ.get(k) for k in json.loads(sys.argv[2])}))'
    engine.command = lambda: [sys.executable, '-B', '-c', code, str(output), json.dumps(list(chosen))]
    job = engine.launch()
    receipt = job.wait()
    assert receipt['exit_code'] == 0 and receipt['cleanup_confirmed']
    assert json.loads(output.read_text()) == chosen
    spec = json.loads(job.spec.read_text())
    assert spec['paging_telemetry'] is True
    assert spec['host_budget_gib'] == .25 and spec['min_available_gib'] == 6
    assert spec['job_commit_gib'] == 1


@pytest.mark.parametrize('key', ['QFN_NATIVE_NVFP4_FUSED_MID', 'QFN_V3_GRAPH_TRACE'])
@pytest.mark.parametrize('value', ['0', '1'])
def test_native_decode_diagnostics_and_reference_schedule_reach_guard(key, value, tmp_path):
    cfg=load(overrides={'runtime_root':str(tmp_path),'env_overrides':{key:value}})
    assert apply({},cfg,'qfn-engine')[key]==value
