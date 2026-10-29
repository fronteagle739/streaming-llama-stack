"""Group legacy prefill chunks into one existing layer-major weight pass.

Only the carry/Engram capacity grows. The admitted CUDA graph, attention tile,
hidden bank and causal tile order are unchanged. The native executor stages rows
beyond its resident prefix through its existing exact BF16 RAM spill slot.
"""
import math

GiB = 1 << 30
PAGE = 65536


def selection(value):
    if value == 'auto':
        return value
    if type(value) is not int or not 1 <= value <= 4:
        raise ValueError('producer_chunks_per_pass must be auto or an integer from 1 to 4')
    return value


def plan(cfg, prompt_tokens, available):
    from .combo_producer import LEGACY_SWEEP, carry_bank_bytes, engram_prompt_bytes, fold_sweeps
    requested = selection(cfg['producer_chunks_per_pass'])
    if type(prompt_tokens) is not int or not 0 < prompt_tokens < int(cfg['context']):
        raise ValueError('Multichunk prompt must fit the configured context')
    if cfg.get('serving_mode', cfg.get('serving', 'single_engine')) != 'single_engine':
        raise ValueError('Multichunk requires single_engine')
    primary = cfg.get('producer_prefill_sweep_tokens', LEGACY_SWEEP)
    if primary == 'auto':primary = LEGACY_SWEEP   # the single-stream default; multichunk keeps its 49152 primary
    fold_cap = cfg.get('producer_prefill_fold_cap_tokens', 61440)
    if primary != LEGACY_SWEEP or type(fold_cap) is not int or not primary <= fold_cap < 2 * primary:
        raise ValueError('Multichunk requires the 49152-token primary and a fold cap below two chunks')
    base = int(cfg.get('producer_host_budget_gib', 6.25) * GiB)
    maximum = float(cfg.get('producer_host_max_gib', cfg.get('producer_host_budget_gib', 6.25)))
    # Include the enclosing worker/controller in admission. An explicit chunk
    # count never silently exceeds the user's process budget or gets reduced.
    outer = float(cfg.get('host_budget_gib', maximum + .25)) - .25
    if not all(math.isfinite(v) and v > 0 for v in (maximum, outer)):
        raise ValueError('Finite positive multichunk host budgets required')
    top = int(min(maximum, outer) * GiB)
    reserve = max(.5, float(cfg.get('producer_host_reserve_gib', .5)),
                  float(cfg.get('producer_desktop_reserve_gib', .5)),
                  float(cfg.get('runtime_min_free_gib', .5)), float(cfg.get('min_available_gib', .5)))
    if available is not None:
        top = min(top, int(available) - math.ceil(reserve * GiB))
    top = top // PAGE * PAGE
    if top < base:
        raise MemoryError('Multichunk base carry cannot fit the physical RAM/process budget with its reserve')
    limit = min((1 << 20), int(cfg['context']) - 1)
    # Whole chunk groups only. Fold at most the legacy tail allowance onto a
    # group, rather than multiplying the old fold cap by N.
    for chunks in (range(4, 0, -1) if requested == 'auto' else (requested,)):
        if chunks == 1:
            from .combo_producer import prefill_plan
            result = prefill_plan(dict(cfg, producer_chunks_per_pass=1, producer_host_max_gib=top/GiB), prompt_tokens, available)
            if result['host_bytes'] > top:
                raise MemoryError('Legacy fallback exceeds the physical multichunk ceiling')
            result.update(chunks_requested=requested, chunks_per_pass=1,
                          carry_bank_bytes=carry_bank_bytes(result['capacity']),
                          engram_prompt_bytes=engram_prompt_bytes(result['capacity']),
                          reserve_bytes=math.ceil(reserve*GiB), device_capacity_changed=False)
            return result
        sweep = chunks * primary
        cap = min(limit, sweep + fold_cap - primary)
        sweeps, tail, rebalance = fold_sweeps(prompt_tokens, sweep, cap, cap)
        capacity = max(primary, max(sweeps))
        carry = carry_bank_bytes(capacity) - carry_bank_bytes(primary)
        engram = engram_prompt_bytes(capacity) - engram_prompt_bytes(primary)
        host = -(-(base + carry + engram) // PAGE) * PAGE
        if capacity <= limit and host <= top:
            return dict(capacity=capacity, host_bytes=host, carry_delta=carry, engram_delta=engram,
                        base_host_bytes=base, max_host_bytes=top, available_bytes=available,
                        capped_by='multichunk_physical_and_process_budget', prompt_tokens=prompt_tokens,
                        single_sweep=len(sweeps) == 1, policy='fold', sweep_tokens=sweep, sweeps=sweeps,
                        fold_tail=tail, fold_cap=cap, rebalance=rebalance, reason=None,
                        chunks_requested=requested, chunks_per_pass=chunks,
                        carry_bank_bytes=carry_bank_bytes(capacity), engram_prompt_bytes=engram_prompt_bytes(capacity),
                        reserve_bytes=math.ceil(reserve * GiB), device_capacity_changed=False)
    raise MemoryError('Requested multichunk pass needs %d host bytes; admitted ceiling is %d; '
                      'reduce producer_chunks_per_pass or explicitly raise the host budget within physical RAM' % (host, top))
