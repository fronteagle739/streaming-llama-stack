"""Named profiles preserve explicit settings and survive the DSV installed-config boundary."""
import json
from copy import deepcopy
import pytest
from slstack import model_settings


def profile(model):
    return model_settings.load('examples/single-machine/'+model+'.settings.json')


def test_named_defaults_preserve_user_engine_and_sampler_options():
    original = dict(model_settings=profile('qfn'), model_mode='max_speed', qfn_tile=1024,
                    env_overrides={'QFN_MTP':'1','DS4_DS_READ_PARTS':'4'},
                    temperature=.8, typical_p=.9, stop=['finish'])
    before = deepcopy(original)
    result = model_settings.apply(original,'qfn')
    assert original == before
    assert result['qfn_quality_speed'] == 'max_speed'
    assert result['qfn_tile'] == 1024
    assert result['env_overrides']['QFN_MTP'] == '1'
    assert result['env_overrides']['DS4_DS_READ_PARTS'] == '4'
    assert result['stop'] == ['finish'] and result['typical_p'] == .9
    assert result['qfn_nvfp4_decode_mode'] == 'dsv-native'


def test_dsv_mode_is_numeric_and_retained_for_installed_proxy():
    from slstack.single_runtime import KEPT
    row = profile('dsv41')
    row['bias']['balanced'] = .2
    cfg = model_settings.apply(dict(model_settings=row,model_mode='balanced'),'dsv41')
    assert cfg['generation_initial_bias'] == .2
    assert 'generation_initial_bias' in KEPT
    assert cfg['env_overrides']['DS4_DS_READ_PARTS'] == '8'


@pytest.mark.parametrize('value', [float('nan'),float('inf'),True,-1,9])
def test_invalid_named_bias_fails_before_engine_launch(value):
    row = profile('qfn'); row['bias']['max_speed'] = value
    with pytest.raises(ValueError):model_settings.load(row)


def test_wrong_model_profile_fails_and_numeric_request_range_remains_available():
    from slstack.qfn_single import bias_of
    with pytest.raises(ValueError):model_settings.apply(dict(model_settings=profile('dsv41')),'qfn')
    assert bias_of(8,profile('qfn')) == 8
    with pytest.raises(ValueError):bias_of(float('inf'),profile('qfn'))


def test_owner_max_speed_setting_keeps_continuous_numeric_dial():
    from slstack.qfn_single import bias_of
    row = profile('qfn')
    assert row['bias']['max_speed'] == .68
    assert row['calibration']['selection']['max_speed']['bias'] == .68
    assert bias_of('max_speed') == .68
    assert bias_of('max_speed', row) == .68
    assert bias_of('max_quality', row) == 0
    assert bias_of('balanced', row) == 0
    for value in (0, .175, .35, .5, .68, 1.36125, 8):
        assert bias_of(value, row) == value
