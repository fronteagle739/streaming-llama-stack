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
