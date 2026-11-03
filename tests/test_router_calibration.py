"""CPU evidence/selection boundary proofs; never starts an engine or GPU job."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples/single-machine' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


router = module('router_calibration_test_subject', 'calibrate-router.py')
adapter = module('qfn_calibration_test_subject', 'calibration_qfn_harness.py')


def dumps(tmp_path):
    rows, vocab, context = 128, 8, 16
    left, right = tmp_path / 'reference.f32', tmp_path / 'candidate.f32'
    original = np.arange(rows * vocab, dtype='<f4').reshape(rows, vocab) / 1000
    left.write_bytes(original.tobytes()); right.write_bytes(original.tobytes())
    positions = struct.pack('<128i', *range(context, context + rows))
    a, b = tmp_path / 'reference.pos', tmp_path / 'candidate.pos'
    a.write_bytes(positions); b.write_bytes(positions)
    evidence = dict(rows=rows, vocab=vocab, context=context, kl_mean=0.0, top1_agreement=1.0,
                    reference_logits=str(left), candidate_logits=str(right),
                    reference_positions=str(a), candidate_positions=str(b),
                    reference_sha256=router.digest(left), candidate_sha256=router.digest(right))
    return left, right, a, b, evidence


def test_complete_vocabulary_evidence_and_identical_rows_pass(tmp_path):
    left, right, _, _, evidence = dumps(tmp_path)
    router.quality(evidence, 16, 8)
    compared = adapter.compare(left, right, 8)
    assert compared['rows'] == 128 and compared['vocab'] == 8
    assert compared['top1_agreement'] == 1 and compared['kl_mean'] == pytest.approx(0)
    assert len(compared['row_details']) == 128


def test_comparison_detects_non_top1_changes_in_full_vocabulary(tmp_path):
    left, right, _, _, _ = dumps(tmp_path)
    logits = np.frombuffer(right.read_bytes(), dtype='<f4').copy().reshape(128, 8)
    logits[:, 0] -= 2  # Lowest-ranked column changes; the top-1 winner does not.
    right.write_bytes(logits.tobytes())
    compared = adapter.compare(left, right, 8)
    assert compared['top1_agreement'] == 1
    assert compared['kl_mean'] > .01


@pytest.mark.parametrize('corruption', ['truncated', 'positions', 'hash', 'context'])
def test_quality_rejects_incomplete_or_differently_bound_evidence(tmp_path, corruption):
    _, right, _, pos, evidence = dumps(tmp_path)
    if corruption == 'truncated':
        right.write_bytes(right.read_bytes()[:-4])
    elif corruption == 'positions':
        pos.write_bytes(struct.pack('<128i', *range(17, 145)))
    elif corruption == 'hash':
        evidence['candidate_sha256'] = '0' * 64
    else:
        evidence['context'] = 15
    with pytest.raises(ValueError):
        router.quality(evidence, 16, 8)


@pytest.mark.parametrize('corruption', ['nan', 'truncated'])
def test_comparison_rejects_nonfinite_or_truncated_rows(tmp_path, corruption):
    left, right, _, _, _ = dumps(tmp_path)
    if corruption == 'nan':
        raw = bytearray(right.read_bytes()); raw[:4] = struct.pack('<f', float('nan'))
        right.write_bytes(raw)
    else:
        right.write_bytes(right.read_bytes()[:-4])
    with pytest.raises(ValueError):
        adapter.compare(left, right, 8)


def natural_evidence():
    request = dict(messages=[dict(role='user', content='What is two plus two?')],
                   temperature=.7, max_tokens=128, seed=17, top_p=.9)
    evidence = dict(request_sha256=hashlib.sha256(router.canonical(request)).hexdigest(),
                    bias=.35, reason='stop', text='Four.', generated_tokens=12, max_tokens=128)
    return request, evidence


def test_sampled_natural_stop_passes_with_exact_request_binding():
    request, evidence = natural_evidence()
    assert router.generation(evidence, request, .35)


@pytest.mark.parametrize('change', [dict(reason='length'), dict(text=''), dict(generated_tokens=128), dict(generated_tokens=0)])
def test_length_or_empty_generations_cannot_qualify(change):
    request, evidence = natural_evidence(); evidence.update(change)
    assert not router.generation(evidence, request, .35)


@pytest.mark.parametrize('change', [dict(request_sha256='0' * 64), dict(bias=.68), dict(max_tokens=256)])
def test_generation_rejects_mismatched_request_bias_or_cap(change):
    request, evidence = natural_evidence(); evidence.update(change)
    with pytest.raises(ValueError):
        router.generation(evidence, request, .35)


def test_settings_publish_preserves_backup_and_unrelated_fields(tmp_path):
    path = tmp_path / 'settings.json'
    before = b'{"unrelated":{"retain":true},"bias":{"max":8}}\n'
    path.write_bytes(before)
    updated = json.loads(before); updated['bias']['max_speed'] = .35
    backup = router.publish_settings(path, before, updated, 'unit')
    assert Path(backup).read_bytes() == before
    assert json.loads(path.read_bytes()) == updated
    assert json.loads(path.read_bytes())['bias']['max'] == 8
    assert json.loads(path.read_bytes())['unrelated'] == {'retain': True}


def test_concurrent_settings_edit_prevents_replacement(tmp_path):
    path = tmp_path / 'settings.json'; path.write_bytes(b'{"edited":true}')
    with pytest.raises(ValueError, match='changed during calibration'):
        router.publish_settings(path, b'{}', {}, 'unit')
    assert path.read_bytes() == b'{"edited":true}'


@pytest.mark.parametrize('field', ['unsupported_attention', 'unsupported_feature'])
def test_unsupported_profile_fields_fail_loudly(field):
    with pytest.raises(ValueError, match='Unsupported engine setting'):
        router.validate_profile(dict(name='candidate', engine={field: 1}, cache={}))


def test_known_tuning_profile_fields_are_explicitly_supported():
    router.validate_profile(dict(name='candidate', engine=dict(tile=1024, read_parts=8, attention='native'),
                                 cache=dict(half_life=16, seed_lfu_max=.6)))
    with pytest.raises(ValueError, match='Unsupported cache setting'):
        router.validate_profile(dict(name='candidate', engine={}, cache=dict(no_op_tier=1)))


def test_production_command_retains_profile_fields(tmp_path):
    from slstack.model_settings import apply
    from slstack.qfn_single import QfnEngine
    settings = dict(version=1, model_key='qfn-native-nvfp4',
                    bias=dict(min=0, max=8, max_quality=0, balanced=0, max_speed=.35),
                    engine=dict(tile=1024, read_parts=8, mtp=1, prefill='dsv-native', decode='dsv-native',
                                indexer='basalt-tc-refine-prefill', attention='native'),
                    cache=dict(seed_lfu_max=.6, half_life=16), router_spreads=dict(reference_only=True))
    cfg = apply(dict(runtime_root=str(tmp_path), model='model.basalt', qfn_engine='stable-engine.exe',
                     qfn_vocab_gguf='vocab.gguf', context=100480, model_settings=settings), 'qfn')
    argv = QfnEngine(cfg).command()
    for flag, value in (('--tile', '1024'), ('--nvfp4-mode', 'dsv-native'), ('--nvfp4-decode-mode', 'dsv-native'),
                        ('--qsa-indexer', 'basalt-tc-refine-prefill'), ('--qsa-attention', 'native')):
        assert argv[argv.index(flag) + 1] == value
    for field, value in (('QFN_MTP', '1'), ('DS4_DS_READ_PARTS', '8'), ('QFN_LFU_HALF_LIFE', '16'), ('QFN_SEED_LFU_MAX', '0.6')):
        assert cfg['env_overrides'][field] == value
