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
