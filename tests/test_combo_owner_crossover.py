import json,pytest
from slstack.combo_crossover import owner_crossover,qualified_threshold
from slstack.combo_proxy import apply_preset_crossover

def test_owner_formula_smallest_n():
    n=owner_crossover(3.130923989483746,1084.8874623126364,58.26613879992129)
    assert n==183 and n/3.1309>58.266+n/1084.887 and not (n-1)/3.1309>58.266+(n-1)/1084.887

def test_preset_flags_validate_against_inputs():
    cfg=dict(combo_crossover_profiles=dict(ultra=dict(r_native_tok_s=3.13,r_stream_tok_s=1150.0,switch_seconds=58.0)))
    n=owner_crossover(3.13,1150.0,58.0)
    assert apply_preset_crossover(cfg,'ultra',str(n))['combo_crossover_tokens']==n
    with pytest.raises(ValueError):apply_preset_crossover(cfg,'ultra',str(n+1))
    with pytest.raises(ValueError):apply_preset_crossover(cfg,'fast',str(n))

def test_intercepts_hand_worked():
    # Rn=4, Rs=1000, S=60, native intercept 2 s, stream fixed 10 s:
    # 2+N/4 > 70+N/1000  <=>  N*(0.25-0.001) > 68  <=>  N > 273.09...  so N*=274.
    assert owner_crossover(4.0,1000.0,60.0,native_intercept_s=2.0,stream_fixed_s=10.0)==274
    assert not (2+273/4>70+273/1000) and 2+274/4>70+274/1000

def test_intercepts_default_to_original_formula():
    assert owner_crossover(3.13,1150.0,58.0)==owner_crossover(3.13,1150.0,58.0,0.0,0.0)

def test_native_intercept_above_switch_gives_one():
    # T_native(1)=100.25 already exceeds 60+0.001: the smallest N is 1.
    assert owner_crossover(4.0,1000.0,60.0,native_intercept_s=100.0)==1

def test_negative_intercept_refused():
    with pytest.raises(ValueError):owner_crossover(4.0,1000.0,60.0,native_intercept_s=-1.0)

def test_preset_self_check_uses_intercepts():
    row=dict(r_native_tok_s=4.0,r_stream_tok_s=1000.0,switch_seconds=60.0,native_intercept_s=2.0,stream_fixed_s=10.0)
    cfg=dict(combo_crossover_profiles=dict(ultra=row))
    assert apply_preset_crossover(cfg,'ultra','274')['combo_crossover_tokens']==274
    with pytest.raises(ValueError):apply_preset_crossover(cfg,'ultra','273')
    assert qualified_threshold(dict(producer_profile='ultra',combo_crossover_tokens=274,
        combo_crossover_inputs=dict(row,producer_profile='ultra')))==274

def _runtime():
    rows=dict(ultra=dict(r_native_tok_s=266.157,r_stream_tok_s=1151.0,switch_seconds=13.21,native_intercept_s=14.502,stream_fixed_s=20.0,
                         llama_producer='direct3.exe',producer_sha256='a'*64),
              fast=dict(r_native_tok_s=266.157,r_stream_tok_s=800.0,switch_seconds=13.21,native_intercept_s=14.502,stream_fixed_s=20.0,
                        llama_producer='fast.exe',producer_sha256='b'*64))
    for row in rows.values():row['combo_crossover_tokens']=owner_crossover(row['r_native_tok_s'],row['r_stream_tok_s'],
        row['switch_seconds'],row['native_intercept_s'],row['stream_fixed_s'])
    base=['-m','m.gguf','--port','8080','--ctx','262144','--cuda']
    # A stale global N*/producer from an earlier preset must never survive a launch.
    return dict(studio_expected_args=base,combo_crossover_profiles=rows,combo_crossover_tokens=1,
                producer_profile='ultra',llama_producer='stale.exe',producer_sha256='c'*64),base,rows

def test_active_preset_selects_its_own_n_star_and_producer():
    from slstack.combo_proxy import native_launch_config
    cfg,base,rows=_runtime()
    seen={}
    for profile in ('ultra','fast','ultra'):
        n=rows[profile]['combo_crossover_tokens']
        argv=[*base,'--combo-producer-profile',profile,'--combo-crossover-tokens',str(n)];argv[3]='18808'
        launched=native_launch_config(cfg,argv)
        assert qualified_threshold(launched)==n==launched['combo_crossover_tokens']
        assert launched['llama_producer']==rows[profile]['llama_producer']
        assert launched['producer_sha256']==rows[profile]['producer_sha256']
        assert launched['producer_profile']==profile
        seen[profile]=n
    assert seen['ultra']!=seen['fast']
    assert cfg['combo_crossover_tokens']==1 and cfg['llama_producer']=='stale.exe'   # launch config is a copy

def test_profile_runtime_refuses_launch_without_preset_crossover():
    from slstack.combo_proxy import native_launch_config
    cfg,base,rows=_runtime()
    argv=list(base);argv[3]='18808'
    with pytest.raises(ValueError,match='per-preset'):native_launch_config(cfg,argv)
    n=rows['fast']['combo_crossover_tokens']
    with pytest.raises(ValueError):native_launch_config(cfg,[*argv,'--combo-producer-profile','fast','--combo-crossover-tokens',str(n+1)])
    with pytest.raises(ValueError):native_launch_config(cfg,[*argv,'--combo-producer-profile','ultra','--combo-crossover-tokens',str(n)])

def test_session_threshold_comes_from_the_launch_config(monkeypatch):
    import slstack.combo_session as combo_session
    from slstack.combo_proxy import native_launch_config
    cfg,base,rows=_runtime()
    monkeypatch.setattr(combo_session,'vendor',lambda:None)
    class Native:pass
    for profile in ('fast','ultra'):
        n=rows[profile]['combo_crossover_tokens']
        launched=native_launch_config(cfg,[*base,'--combo-producer-profile',profile,'--combo-crossover-tokens',str(n)])
        launched['runtime_root']='.'
        session=combo_session.ComboSession(launched,device=Native())
        assert session.threshold==n
