from copy import deepcopy
import json
import pytest
from slstack.performance_contract import seal, validate, record, regression, digest
from scripts.dsv41_single import overlay


def test_both_dsv_presets_record_the_full_measured_caps_and_queue_policy():
    for name in ('262k', '1m'):
        cfg=overlay(name)
        manifest=validate(cfg,'dsv41')
        assert (cfg['host_budget_gib'],cfg['combo_worker_host_budget_gib'],
                cfg['producer_host_max_gib'],cfg['producer_engram_scan_rings'])==(10.25,10,7.75,'auto')
        assert manifest['settings']['env_overrides']['OPENBLAS_NUM_THREADS']=='1'


@pytest.mark.parametrize('field,value', [('producer_host_max_gib',6.25),
    ('host_budget_gib',7), ('combo_worker_host_budget_gib',6.75),
    ('producer_engram_scan_rings',1),('producer_prefill_sweep_tokens',49152)])
def test_preset_edits_fail_before_native_launch(field,value):
    cfg=overlay('1m');cfg[field]=value
    with pytest.raises(ValueError,match='drift|aggregate'):
        validate(cfg,'dsv41')


def test_queue_environment_drift_and_manifest_corruption_fail_loudly():
    cfg=overlay('1m');cfg['env_overrides']['LLAMADECODE_ENGRAM_SCAN_DEPTH']='1'
    with pytest.raises(ValueError,match='drift'):validate(cfg,'dsv41')
    cfg=overlay('1m');cfg['performance_manifest']['settings']['producer_host_max_gib']=6.25
    with pytest.raises(ValueError,match='digest'):validate(cfg,'dsv41')


@pytest.mark.parametrize('model', ['dsv41','qfn'])
def test_no_model_can_inherit_missing_memory_and_queue_settings(model):
    with pytest.raises(ValueError,match='missing explicit'):seal({},model)


def test_qfn_has_separate_measured_values_and_rejects_native_queue_conflicts():
    cfg=json.load(open('examples/single-machine/qfn.example.json'))
    cfg=seal(cfg,'qfn');validate(cfg,'qfn')
    assert (cfg['host_budget_gib'],cfg['qfn_host_budget_gib'],cfg['qfn_engram_scan_rings'])==(4.25,6,1)
    assert cfg['performance_manifest']['settings']!=overlay('1m')['performance_manifest']['settings']
    cfg['env_overrides']={'LLAMADECODE_ENGRAM_SCAN_RINGS':'4'}
    with pytest.raises(ValueError,match='conflicts'):validate(cfg,'qfn')


def test_fractional_qfn_tile_is_rejected_before_native_conversion():
    cfg=json.load(open('examples/single-machine/qfn.example.json'))
    cfg['qfn_tile']=2048.5
    with pytest.raises(ValueError,match='fractional'):seal(cfg,'qfn')


def test_per_number_receipt_is_complete_and_cannot_overwrite(tmp_path):
    cfg=overlay('1m');p=tmp_path/'settings.json'
    receipt=record(p,cfg,'dsv41',argv=['engine','--warmup-one-token','1'],
        environment={'OPENBLAS_NUM_THREADS':'1','API_TOKEN':'private'},fixture_sha256='abc',plan={'sweeps':[100000]})
    assert receipt['environment']=={'OPENBLAS_NUM_THREADS':'1'}
    assert receipt['fixture_sha256']=='abc' and receipt['settings_manifest']==cfg['performance_manifest']
    with pytest.raises(FileExistsError):record(p,cfg,'dsv41',argv=[],environment={},fixture_sha256='abc')


def test_regression_gate_requires_matched_warm_fixtures_and_fails_beyond_spread():
    base=dict(fixture_sha256='a',prompt_tokens=100000,settings_sha256='b',pack_sha256='c',
              clock='native-second-prompt-open-and-run',prompt_data_policy='after-warm-up',
              protocol='warm-short-then-different-prompt-same-process',control_tok_s=[3800.,3900.])
    assert regression(base,{**base,'prefill_tok_s':3850})['status']=='PASS'
    with pytest.raises(RuntimeError,match='regression'):regression(base,{**base,'prefill_tok_s':3799})
    with pytest.raises(ValueError,match='matched'):regression(base,{**base,'fixture_sha256':'other','prefill_tok_s':9999})
    with pytest.raises(ValueError,match='matched'):regression(base,{**base,'protocol':'cold','prefill_tok_s':9999})


def test_worker_budget_reservation_preserves_the_recorded_contract():
    cfg=overlay('1m');cfg['host_budget_gib']=10
    assert validate(cfg,'dsv41')


def test_old_unrecorded_low_caps_refuse_unless_explicitly_experimental():
    with pytest.raises(ValueError,match='lower DSV'):
        overlay('1m',dict(host_budget_gib=7,producer_host_max_gib=6.25))
    cfg=overlay('1m',dict(host_budget_gib=7,producer_host_max_gib=6.25,performance_experiment=True))
    assert validate(cfg,'dsv41')['settings']['performance_experiment'] is True


def test_native_transport_reset_is_reported_to_the_active_request():
    import queue
    from slstack.qfn_single import QfnEngine
    class Closed:
        def __iter__(self):raise ConnectionResetError('closed')
    events=queue.Queue();QfnEngine._pump(Closed(),events)
    failure=events.get_nowait()
    assert failure['ok'] is False and 'transport closed' in failure['error']
    assert events.get_nowait() is None


def test_warm_request_has_an_explicit_separate_transient_bill_without_changing_sweeps():
    from slstack.combo_producer import prefill_arguments
    cfg=overlay('1m');cfg.pop('performance_manifest')
    _,cold=prefill_arguments(cfg,100000,16<<30)
    cfg.update(producer_warmup_one_token=True,producer_fixed_direct_open=False)
    cfg.pop('producer_warmup_host_reserve_mib',None)
    with pytest.raises(ValueError,match='explicit producer_warmup_host'):
        prefill_arguments(cfg,100000,16<<30)
    cfg['producer_warmup_host_reserve_mib']=512
    args,warm=prefill_arguments(cfg,100000,16<<30)
    assert warm['sweeps']==cold['sweeps']==[100000]
    assert warm['host_bytes']==cold['host_bytes']+(512<<20)
    assert args['--job-host-bytes']==str(warm['host_bytes'])
    assert args['--warmup-one-token']=='1' and args['--fixed-direct-open']=='0'
    cfg['combo_worker_host_budget_gib']=8
    with pytest.raises(MemoryError,match='Warm transient bill'):
        prefill_arguments(cfg,100000,16<<30)


def test_service_port_and_engine_environment_port_remain_distinct_controls():
    import argparse
    from slstack.envvars import add_flags
    p=argparse.ArgumentParser();p.add_argument('--port',type=int);add_flags(p)
    a=p.parse_args(['--port','8088','--engine-env-port','11234'])
    assert (a.port,a.env_PORT)==(8088,'11234')
    legacy=argparse.ArgumentParser();add_flags(legacy)
    assert legacy.parse_args(['--port','11234']).env_PORT=='11234'


def test_owner_selected_qfn_speed_bias_is_a_preset_control_not_new_tuning():
    from slstack.qfn_single import bias_of
    assert bias_of('max_speed',.68)==.68
    assert bias_of('max_quality',.68)==0
    assert bias_of(.35,.68)==.35
    with pytest.raises(ValueError):bias_of('max_speed',9)


def test_receipts_keep_token_count_flags_but_fingerprint_credentials():
    from slstack.performance_contract import engine_environment
    result=engine_environment(dict(DS4_SERVER_AUTH_TOKEN='private',LLAMADECODE_PREFILL_TOKENS='100000'))
    assert result['LLAMADECODE_PREFILL_TOKENS']=='100000'
    assert result['DS4_SERVER_AUTH_TOKEN'].startswith('sha256:')
    assert 'private' not in json.dumps(result)


def test_baseline_builder_excludes_failed_cold_and_different_binary_controls():
    from slstack.performance_contract import baseline_from_controls
    row=dict(status='PASS',fixture_sha256='f',prompt_tokens=100000,settings_sha256='s',pack_sha256='p',
        clock='native-second-prompt-open-and-run',prompt_data_policy='after-warm-up',
        protocol='warm-short-then-different-prompt-same-process',prefill_tok_s=3800,
        warm_measurement=dict(enabled=True,same_process_and_CUDA_backend=True,next_prompt_data_deferred=True),
        settings_receipt=dict(producer_sha256='binary'),
        cleanup=dict(owned_job_empty=True,owned_process_signaled=True,owned_pid_gone=True))
    other=deepcopy(row);other['prefill_tok_s']=3900
    assert baseline_from_controls([row,other])['control_tok_s']==[3800,3900]
    for change in (dict(status='FAIL'),dict(warm_measurement={}),dict(settings_receipt=dict(producer_sha256='other'))):
        bad={**other,**change}
        with pytest.raises(ValueError):baseline_from_controls([row,bad])
    # paging_growth is a global observer on the native close receipt; only
    # attributed owned pages can disqualify an otherwise successful control.
    assert baseline_from_controls([row,{**other,'cleanup':{**other['cleanup'],'paging_growth':28868608}}])
    for change in (dict(failure='paging'),dict(exit_code=137),dict(owned_paged_bytes=4096),dict(budget_breaches=1)):
        bad={**other,'cleanup':{**other['cleanup'],**change}}
        with pytest.raises(ValueError,match='guard failure'):baseline_from_controls([row,bad])
