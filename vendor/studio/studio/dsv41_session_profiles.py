# Derived from Studio backend/studio/dsv41_session_profiles.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from .dsv41_session_contract import CONTEXT, MAX_PREFILL, require
GIB = 1 << 30
TECHNIQUES = {'expert_q8_1': ('8-bit activation rounding in expert matmuls', None), 'dense_f16': ('FP16 inputs in dense GEMMs; FP32 accumulation', '--dense-inputs-f16'), 'dense_q8_1': ('8-bit activation rounding with original Q8 dense weights', '--dense-activation-q8-1'), 'hc_f16': ('FP16 inputs in HC projections; FP32 accumulation', '--hc-projection-f16-gemm'), 'attention_f16': ('FP16 attention probabilities in token-tile attention', '--token-tile-attention-f16-probabilities'), 'index_bf16': ('BF16 indexer operands; FP32 accumulation', '--index-bf16'), 'split_f32_qk': ('Three-term split-F32 tensor-core attention scores', '--split-f32-tensor-core-qk'), 'attention_kv_f16': ('FP16 attention key/value table', '--attention-kv-f16-table'), 'dense_fp4': ('FP4 dense weight conversion (HYPER only)', '--dense-mxfp4'), 'expert_fp4': ('FP4 expert weight and activation conversion (HYPER only)', '--expert-weights-activations-fp4')}
DEFAULTS = {tier: {key: False for key in TECHNIQUES} for tier in ['accurate', 'fast', 'ultra', 'hyper']}
DEFAULTS['fast'].update(index_bf16=True, split_f32_qk=True)
for tier in ['ultra', 'hyper']:
    DEFAULTS[tier].update({k: True for k in ['expert_q8_1', 'dense_f16', 'hc_f16', 'attention_f16', 'index_bf16', 'attention_kv_f16']})
DEFAULTS['ultra']['dense_q8_1'] = True
DEFAULTS['hyper'].update(dense_fp4=True, expert_fp4=True)

def techniques(tier, overrides=None):
    require(tier in DEFAULTS, 'Unknown native preset tier')
    overrides = {} if overrides is None else overrides
    require(type(overrides) is dict and set(overrides) <= set(TECHNIQUES) and all((type(v) is bool for v in overrides.values())), 'Invalid per-technique settings')
    result = {**DEFAULTS[tier], **overrides}
    require(tier == 'hyper' or not (result['dense_fp4'] or result['expert_fp4']), 'Weight conversion is HYPER only')
    require(not result['expert_fp4'] or (result['dense_fp4'] and result['expert_q8_1']), 'FP4 experts require the existing FP4 dense and rounded-expert path')
    require(not result['dense_fp4'] or (result['dense_f16'] and (not result['dense_q8_1'])), 'FP4 dense requires FP16 input and excludes Q8 dense rounding')
    require(not result['split_f32_qk'] or not result['attention_f16'], 'Split-F32 scores require native attention')
    require(not result['attention_kv_f16'] or result['attention_f16'], 'FP16 KV table requires token-tile attention')
    return result

def profile(tier, overrides=None):
    t = techniques(tier, overrides)
    custom = t != DEFAULTS[tier]
    native = not t['expert_q8_1']
    native_attention = native and (not t['attention_f16'])
    pool = (384 if native or custom else 320) << 20
    casts = (512 if not t['index_bf16'] or custom else 448) << 20
    graph = 7516192768 - (8 << 20) - (pool - (320 << 20)) - (casts - (448 << 20))
    bill = dict(provider_device=17180917760, cache=3225419776, hidden_device=2521649916, expert_runtime=2097152, pinned_device=67108864, index_device=67108864, graph=graph, expert_pool=pool, weight_cast_device=casts)
    require(sum(bill.values()) == 31377412860, '1M device bill changed')
    paired = not t['dense_fp4']
    args = {'--context': CONTEXT, '--cuda-device': 0, '--carry-storage': 'ram', '--provider-device-bytes': bill['provider_device'], '--provider-host-bytes': 2783969280, '--cache-bytes': bill['cache'], '--graph-bytes': graph, '--job-device-bytes': 31377412860, '--job-host-bytes': 8 * GIB, '--gpu-reserve-bytes': GIB, '--carry-file-bytes': 0, '--carry-host-bytes': 4545576960, '--carry-outstanding-bytes': 0, '--output-outstanding-bytes': 0, '--hidden-device-bytes': bill['hidden_device'], '--direct-carry-budget': 1, '--compressed-cache-bf16': 1, '--resident-hidden-final-readback': 1, '--engram-prefetch-bytes': 636485632, '--batch-seed-ram': 1, '--pair-ffn-tiles': int(paired), '--fixed-resident-plan': int(not paired), '--tile-host-registration': int(tier != 'ultra'), '--resident-pageable-copies': int(tier == 'ultra'), '--native-rhs-pack': int(native), '--reuse-child-raw-metadata': int(paired), '--pv-strided-probabilities': int(native_attention), '--fuse-kv-gather-half': int(native_attention), '--pv-view-f32': int(native_attention), '--compact-kv-gather': int(native_attention), '--masked-half-inplace': int(native_attention), '--index-cublas-device-bytes': 67108864, '--index-f32-cublas': int(not t['index_bf16']), '--expert-profile': 'native_half_q4k_perf_v1' if native else 'q8_1_activation_rounding', '--expert-workspace-bytes': pool, '--expert-runtime-bytes': 2097152, '--pinned-device-bytes': 67108864, '--cast-device-bytes': casts, '--cast-audit': 0, '--reuse-graph': 1, '--fetch-profile': 'two-buffer-direct-auto-async', '--compute-tile': 4096, '--compute-step-tokens': 0, '--warmup-one-token': 1, '--engram-tile-lookahead': 1, '--resident-hidden-bf16': 1, '--async-input-uploads': 1, '--resident-compact-ids': 1, '--resident-metadata': 1, '--prefill-chunk-tokens': MAX_PREFILL, '--backend-temporary-reserve-bytes': 805306368 if paired and t['attention_kv_f16'] else GIB, '--capture-dspark': 1, '--unaligned-append-step-tokens': 2048 if paired else 0, '--export-hash-only': 1, '--session-stdio': 1, '--session-control-host-bytes': 64 << 20, '--session-idle-timeout-ms': 3600000, '--index-key-windows': 1, '--merge-attention-selected-rows': 0, '--hc-bf16-epilogue': int(tier in ['ultra', 'hyper'])}
    for key, (_, arg) in TECHNIQUES.items():
        if arg:
            args[arg] = int(t[key])
    return dict(tier=tier, techniques=t, custom=custom, device_bill=bill, arguments={k: str(v) for k, v in args.items()}, numerical_evidence='Custom combination: default-tier envelope does not apply' if custom else tier + ' measured default set', admission='Static 1M default plan plus mandatory physical/graph admission' if not custom else 'Mandatory runtime graph admission; no static qualification of this custom combination')
