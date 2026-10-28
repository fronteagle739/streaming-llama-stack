"""The paired exporter must use the actual single-engine prefill configuration."""
from slstack.combo_producer import prefill_arguments


CFG = dict(context=1000001, producer_profile='ultra', yoco_keep=140,
           producer_host_budget_gib=6.25, producer_host_max_gib=7.75,
           producer_prefill_sweep_tokens=49152, producer_prefill_fold_cap_tokens=61440,
           producer_desktop_reserve_gib=.5, host_budget_gib=9.25,
           ctx_cache_precision='f32', ctx_cache_growable=True)


def test_paired_and_single_share_every_prefill_flag_except_required_capture():
    for count in (49787, 198287):
        single, plan = prefill_arguments(CFG, count, 24 << 30)
        paired, peer_plan = prefill_arguments(CFG, count, 24 << 30, export_state=True)
        assert plan == peer_plan
        differences = {key for key in single.keys() | paired.keys() if single.get(key) != paired.get(key)}
        assert differences == {'--capture-dspark', '--yoco-remote-state-version'}
        assert paired['--capture-dspark'] == '1' and paired['--yoco-remote-state-version'] == '1'
        assert paired['--warmup-one-token'] == '0' and paired['--fixed-direct-open'] == '1'
        assert paired['--ctx-cache-growable'] == '1' and paired['--ctx-cache-precision'] == 'f32'
        assert paired['--job-host-bytes'] == str(plan['host_bytes'])
        assert plan['host_bytes'] <= 7.75 * 2**30
    assert peer_plan['sweeps'] == [49152, 49152, 49152, 50831]


def test_50k_uses_one_folded_sweep_and_no_warmup():
    arguments, plan = prefill_arguments(CFG, 49787, 24 << 30, export_state=True)
    assert plan['sweeps'] == [49787]
    assert arguments['--prefill-chunk-tokens'] == '49787'
    assert arguments['--prefill-sweep-tokens'] == '49152'
    assert arguments['--prefill-fold-cap-tokens'] == '61440'


def test_desktop_reserve_is_512mib_and_runtime_floor_is_half():
    from unittest.mock import patch
    from slstack.config import load
    from slstack.mac import budget
    cfg = load(overrides={'gpu_uuid': 'GPU-test'})
    assert cfg['producer_desktop_reserve_gib'] == .5 and cfg['min_available_gib'] == .5
    assert cfg['runtime_min_free_gib'] == 0.5
    with patch('slstack.memory_counters.available_physical_bytes', return_value=int(400.5*2**30)):
        mac = budget({**cfg, 'mac_host_budget_gib': 400})
    assert mac['host_budget_gib'] == 400 and mac['min_available_gib'] == .5
    assert mac['runtime_min_free_gib'] == 0.5

def test_broker_reads_actual_progress_before_cold_producer_ready():
    from slstack.native_broker import OwnedProducer
    from types import SimpleNamespace
    import threading
    import time
    owner=OwnedProducer.__new__(OwnedProducer)
    owner.producer=None
    owner.loading_process=SimpleNamespace(diagnostic=b'[dsv41-sweep] layer=0 begin tokens=49152\n[dsv41-sweep] layer=20 complete\n',ready=None)
    owner.prefill_plan={'sweeps':[49152,49152,49152,50831]}
    owner.logical_tokens=198287;owner.progress_started=time.monotonic()-5
    owner.progress_state={};owner.progress_lock=threading.Lock()
    p=owner.prefill_progress()
    assert p['tokens_total']==198287 and p['tokens_done']==int(49152*21/40)
    assert p['elapsed_s']>=5
    owner.loading_process.diagnostic+=b'[dsv41-sweep] layer=39 complete\n[dsv41-sweep] layer=0 begin tokens=49152\n[dsv41-sweep] layer=20 complete\n'
    assert owner.prefill_progress()['tokens_done']>p['tokens_done']


def test_exact_mode3_bf16_window_includes_vmm_padding_without_raising_device_cap():
    from slstack.native_profiles import profile
    cfg={**CFG,'ctx_cache_precision':'bf16'}
    args,_=prefill_arguments(cfg,49787,24<<30,export_state=True)
    old=profile('ultra')['arguments']
    assert int(args['--cache-bytes'])==3225432064
    assert int(args['--cache-bytes'])-int(old['--cache-bytes'])==12288
    assert int(old['--graph-bytes'])-int(args['--graph-bytes'])==12288
    assert args['--job-device-bytes']==old['--job-device-bytes']


def test_custom_prefill_techniques_survive_shared_builder():
    overrides={'dense_f16':False}
    single,_=prefill_arguments({**CFG,'native_techniques':overrides},49787,24<<30)
    paired,_=prefill_arguments(CFG,49787,24<<30,export_state=True,techniques=overrides)
    assert single['--dense-inputs-f16']==paired['--dense-inputs-f16']=='0'
    assert single['--expert-workspace-bytes']==paired['--expert-workspace-bytes']
