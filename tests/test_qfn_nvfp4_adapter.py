"""The explicit prefill diagnostic mode must reach the real engine command unchanged."""
import pytest
from slstack.qfn_single import QfnEngine
from slstack.envvars import apply
from slstack.config import load


@pytest.mark.parametrize('mode', ['auto', 'w4a4', 'native-q8', 'mmq', 'calibrated-w4a4'])
def test_mode_propagates_without_changing_other_engine_options(tmp_path, mode):
    base = dict(runtime_root=str(tmp_path), model='model.basalt', context=100352,
                qfn_tile=2048, qfn_cache_slots=48, qfn_experts_file='same-format-copy.basalt')
    original = QfnEngine(base).command()
    actual = QfnEngine(dict(base, qfn_nvfp4_mode=mode)).command()
    assert actual == original + ['--nvfp4-mode', mode]
    assert '--mtp' not in actual  # no experiment silently switches the independent copied-Q2 drafter off


@pytest.mark.parametrize('mode', ['fp8', '', 'none', True, 0])
def test_unsupported_mode_is_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match='qfn_nvfp4_mode'):
        QfnEngine(dict(runtime_root=str(tmp_path), model='model.basalt', qfn_nvfp4_mode=mode)).command()


@pytest.mark.parametrize('decode', ['native-q8', 'calibrated-w4a4', 'dsv-native'])
@pytest.mark.parametrize('prefill', ['auto', 'w4a4', 'native-q8', 'mmq', 'calibrated-w4a4'])
def test_decode_mode_is_independent_and_preserves_existing_command(tmp_path, decode, prefill):
    base = dict(runtime_root=str(tmp_path), model='model.basalt', context=100352,
                qfn_tile=2048, qfn_cache_slots=48, qfn_experts_file='same-format-copy.basalt',
                qfn_nvfp4_mode=prefill, qfn_qsa_indexer='scalar', qfn_qsa_attention='native')
    original = QfnEngine(base).command()
    actual = QfnEngine(dict(base, qfn_nvfp4_decode_mode=decode)).command()
    assert actual == original + ['--nvfp4-decode-mode', decode]
    assert '--mtp' not in actual


@pytest.mark.parametrize('mode', ['auto', 'w4a4', 'mmq', 'fp8', '', True, 0])
def test_unsupported_decode_mode_is_refused(tmp_path, mode):
    with pytest.raises(ValueError, match='qfn_nvfp4_decode_mode'):
        QfnEngine(dict(runtime_root=str(tmp_path), model='model.basalt', qfn_nvfp4_decode_mode=mode)).command()


def test_explicit_model_vocab_cache_path_preserves_existing_default_and_files(tmp_path):
    base = dict(runtime_root=str(tmp_path), model='model.basalt')
    original = QfnEngine(base).command()
    old = tmp_path / 'qfn-vocab.gguf'
    old.write_bytes(b'owner-existing-cache')
    chosen = tmp_path / 'nvidia-nfc-vocab.gguf'
    actual = QfnEngine(dict(base, qfn_vocab_gguf=str(chosen))).command()
    expected = list(original)
    expected[expected.index('--vocab-gguf') + 1] = str(chosen)
    assert actual == expected
    assert old.read_bytes() == b'owner-existing-cache' and not chosen.exists()


@pytest.mark.parametrize('path', ['', True, 0, []])
def test_invalid_explicit_vocab_path_is_refused(tmp_path, path):
    with pytest.raises(ValueError, match='qfn_vocab_gguf'):
        QfnEngine(dict(runtime_root=str(tmp_path), model='model.basalt', qfn_vocab_gguf=path)).command()


def test_documented_host_thread_settings_keep_existing_limits_and_guard_overrides(tmp_path):
    cfg = load(overrides={'runtime_root': str(tmp_path), 'model': str(tmp_path / 'absent.basalt'), 'gpu_uuid': 'GPU-test'})
    for scope in ('qfn-engine', 'studio', 'build', 'image'):
        inherited = {'OMP_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '2'}
        assert all(apply(dict(inherited), cfg, scope)[key] == value for key, value in inherited.items())
        explicit = apply(dict(inherited), {**cfg, 'env_overrides': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}}, scope)
        assert explicit['OMP_NUM_THREADS'] == explicit['OPENBLAS_NUM_THREADS'] == '1'
    with pytest.raises(ValueError, match='positive integer'):
        apply({}, {**cfg, 'env_overrides': {'OMP_NUM_THREADS': 'unlimited'}}, 'qfn-engine')
