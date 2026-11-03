"""The explicit score-only candidate reaches the production engine without changing feature options."""
import pytest
from slstack.qfn_single import QfnEngine

@pytest.mark.parametrize('mode', ['auto', 'scalar', 'basalt-tc-refine-prefill', 'basalt-tc-refine', 'basalt-tc', 'basalt-tc-prefill', 'dsv-fp4'])
def test_qsa_policy_propagates_to_real_engine(tmp_path, mode):
    base = dict(runtime_root=str(tmp_path), model='model.basalt', qfn_tile=2048,
                qfn_cache_slots=48, qfn_experts_file='same-format.basalt')
    original = QfnEngine(base).command()
    assert QfnEngine(dict(base, qfn_qsa_indexer=mode)).command() == original + ['--qsa-indexer', mode]

@pytest.mark.parametrize('mode', ['', 'tc', True, 0, 'warp-oracle'])
def test_invalid_qsa_policy_fails_loudly(tmp_path, mode):
    with pytest.raises(ValueError, match='qfn_qsa_indexer'):
        QfnEngine(dict(runtime_root=str(tmp_path), model='model.basalt', qfn_qsa_indexer=mode)).command()

@pytest.mark.parametrize('indexer', ['auto', 'scalar', 'basalt-tc-refine-prefill', 'basalt-tc-refine', 'basalt-tc', 'basalt-tc-prefill', 'dsv-fp4'])
@pytest.mark.parametrize('attention', ['native', 'basalt'])
@pytest.mark.parametrize('nvfp4', ['auto', 'w4a4', 'native-q8', 'mmq', 'calibrated-w4a4', 'dsv-native'])
def test_complete_qsa_and_nvfp4_command_preserves_every_original_option(tmp_path,indexer,attention,nvfp4):
    base=dict(runtime_root=str(tmp_path),model='exact-model.basalt',context=100352,
              qfn_tile=2048,qfn_cache_slots=14497,qfn_experts_file='same-format-copy.basalt',
              qfn_engine='stable-qfn-engine.exe',qfn_engine_port=8095,qfn_nvfp4_mode=nvfp4)
    original=QfnEngine(base).command()
    actual=QfnEngine(dict(base,qfn_qsa_indexer=indexer,qfn_qsa_attention=attention)).command()
    for flag,value in (('--qsa-indexer',indexer),('--qsa-attention',attention)):
        at=actual.index(flag);assert actual[at+1]==value;actual[at:at+2]=[]
    assert actual==original

@pytest.mark.parametrize('policy', ['', 'auto', 'scalar', 'Basalt', True, 0, None])
def test_invalid_explicit_attention_is_not_ignored(tmp_path,policy):
    with pytest.raises(ValueError,match='qfn_qsa_attention'):
        QfnEngine(dict(runtime_root=str(tmp_path),model='model.basalt',qfn_qsa_attention=policy)).command()
