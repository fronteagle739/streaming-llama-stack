"""Explicit cache headroom must preserve model options and safety floors."""
import copy

import pytest

from slstack.combo_proxy import decode_vram_margin_config


def config():
    return dict(host_budget_gib=6.75,min_available_gib=4,context=1000001,
                gpu_idle_max_mib=4096,env_overrides=dict(LLAMADECODE_VRAM_RESERVE_MIB='128',
                LLAMADECODE_VRAM_MARGIN_MIB='64',DS4_CUDA_SINGLE_GPU_HC_GRAPHS='1'))


def test_explicit_headroom_changes_only_cache_growth_margin():
    cfg=config();before=copy.deepcopy(cfg)
    result=decode_vram_margin_config(cfg,{'SLSTACK_DECODE_VRAM_MARGIN_MIB':'512'})
    assert cfg==before
    assert result['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB']=='512'
    result['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB']='64'
    assert result==before


def test_default_does_not_change_package_or_share_nested_state():
    cfg=config();result=decode_vram_margin_config(cfg,{})
    assert result==cfg
    result['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB']='512'
    assert cfg['env_overrides']['LLAMADECODE_VRAM_MARGIN_MIB']=='64'


@pytest.mark.parametrize('value',['','-1','4097','512.0',' 512','512 ','1e3','NaN','\u0665\u0661\u0662',None,512])
def test_bad_explicit_budget_is_refused(value):
    with pytest.raises(ValueError):
        decode_vram_margin_config(config(),{'SLSTACK_DECODE_VRAM_MARGIN_MIB':value})


@pytest.mark.parametrize('value',['0','64','512','4096'])
def test_supported_budget_bounds(value):
    assert decode_vram_margin_config({}, {'SLSTACK_DECODE_VRAM_MARGIN_MIB':value})=={
        'env_overrides':{'LLAMADECODE_VRAM_MARGIN_MIB':value}}
