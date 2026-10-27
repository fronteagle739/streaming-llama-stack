import pytest
from slstack.combo_crossover import paired_threshold, owner_crossover


def test_studio_paired_default_and_explicit_zero():
    assert paired_threshold({}) == 11776
    assert paired_threshold({'paired_local_prefill_max_tokens': 0}) == 0


def test_existing_measured_crossover_is_used_when_supplied():
    row = dict(producer_profile='ultra', r_native_tok_s=4., r_stream_tok_s=1000., switch_seconds=60.)
    n = owner_crossover(4., 1000., 60.)
    assert paired_threshold(dict(combo_crossover_tokens=n, combo_crossover_inputs=row)) == n
    with pytest.raises(ValueError):
        paired_threshold(dict(combo_crossover_tokens=n+1, combo_crossover_inputs=row))


@pytest.mark.parametrize('value', [-1, True, 49153, 1.5])
def test_invalid_native_bounds_refused(value):
    with pytest.raises(ValueError):
        paired_threshold({'paired_local_prefill_max_tokens': value})
