from __future__ import annotations
import os, re, struct
from pathlib import Path
from typing import Any, Optional
from . import preset_samplers

def gguf_arch(model):
    try:
        from .gguf_meta import _read_kv_header
        return str((_read_kv_header(Path(_resolve_gguf_model_path(model))) or {}).get('general.architecture') or '')
    except (OSError, ValueError, TypeError, KeyError, struct.error):
        return ''

def write_template(value):
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_file():
        raise ValueError('Set jinja to the generated chat-template file')
    return candidate

def _rtx_kv_cache_argv(rtx):
    pc = rtx.get('prompt_cache') or {}
    if not pc.get('enabled') or not pc.get('rtx_dir'):
        return []
    return ['--kv-cache-dir', str(pc['rtx_dir']), '--kv-cache-max-gb', str(float(pc.get('max_gb', 16)))]
UPSTREAM_DECODE_RECEIVER_BACKENDS = {'llama-decode-upstream', 'upstream-llama-decode'}

def _is_mmproj_path(path: Path) -> bool:
    return 'mmproj' in path.name.lower()

def _resolve_gguf_model_path(value: Any) -> str:
    raw = str(value or '').strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_dir():
        first = sorted((x for x in p.glob('*00001-of-*.gguf') if not _is_mmproj_path(x)))
        if first:
            return str(first[0])
        ggufs = sorted((x for x in p.glob('*.gguf') if not _is_mmproj_path(x)))
        if ggufs:
            return str(ggufs[0])
    return raw
ARCH_STREAM_DEFAULTS: dict = {'minimax-m3': {'attn': 'gqa', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'minimax-m2': {'attn': 'gqa', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'kimi': {'attn': 'mla', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'deepseek2': {'attn': 'mla', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'deepseek_v4': {'attn': 'mla', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'glm-dsa': {'attn': 'mla', 'fa': 'off', 'ctk': 'f16', 'ctv': 'f16'}, 'glm5-next': {'attn': 'mla', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'qwen4exp': {'attn': 'gqa', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}, 'deepseek_v41': {'attn': 'mla', 'fa': 'on', 'ctk': 'f16', 'ctv': 'f16'}}
ARCH_STREAM_TOGGLES: dict[str, dict[str, set[str]]] = {'glm-dsa': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'deepseek_v4': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'deepseek2': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'kimi': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'minimax-m3': {'fa': {'on', 'off'}, 'ctv': {'f16', 'q8_0'}}, 'minimax-m2': {'fa': {'on', 'off'}, 'ctv': {'f16', 'q8_0'}}, 'glm5-next': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'qwen4exp': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}, 'deepseek_v41': {'fa': {'on', 'off'}, 'ctk': {'f16', 'q8_0'}, 'ctv': {'f16', 'q8_0'}}}
ARCH_STREAM_ALIASES: dict[str, str] = {'glm5-next': 'glm5-next', 'glm5next': 'glm5-next', 'glm-5.3-flash': 'glm5-next', 'qwen4exp': 'qwen4exp', 'qwen4-exp': 'qwen4exp', 'qwen3.8-flash-next': 'qwen4exp', 'deepseek41': 'deepseek_v41', 'deepseek-v41': 'deepseek_v41', 'deepseek-v4.1': 'deepseek_v41', 'minimax-m3': 'minimax-m3', 'minimaxm3': 'minimax-m3', 'minimax-m2': 'minimax-m2', 'minimaxm2': 'minimax-m2', 'kimi': 'kimi', 'kimi-k2': 'kimi', 'moonshot': 'kimi', 'deepseek2': 'deepseek2', 'deepseek-v2': 'deepseek2', 'deepseek-v3': 'deepseek2', 'deepseek-v4': 'deepseek_v4', 'deepseek4': 'deepseek_v4', 'deepseek4-mtp-support': 'deepseek_v4', 'deepseek-v32': 'deepseek_v4', 'deepseek-v3.2': 'deepseek_v4', 'deepseek-v3-2': 'deepseek_v4', 'glm-dsa': 'glm-dsa', 'glm': 'glm-dsa', 'glm4': 'glm-dsa', 'glm4-moe': 'glm-dsa', 'glm-moe': 'glm-dsa', 'glm-moe-dsa': 'glm-dsa', 'glm-5.1': 'glm-dsa', 'glm-5.2': 'glm-dsa', 'zai': 'glm-dsa'}

ARCH_STREAM_ALIASES = {**{
    'glm-5.1': 'glm-dsa', 'glm-5.2': 'glm-dsa', 'glm-5.3': 'glm-dsa',
    'glm5.1': 'glm-dsa', 'glm5.2': 'glm-dsa', 'glm5.3': 'glm-dsa',
    'glm5.3-flash': 'glm5-next',
}, **ARCH_STREAM_ALIASES}

def _stream_arch_key(value: Any) -> str:
    return re.sub('[\\s_]+', '-', str(value or '').strip().lower())

def stream_arch_id(value: Any, model_path: str='') -> str:
    key = _stream_arch_key(value)
    if key in ARCH_STREAM_ALIASES:
        return ARCH_STREAM_ALIASES[key]
    haystack = _stream_arch_key(f"{value or ''} {model_path or ''}")
    for needle, arch in sorted(ARCH_STREAM_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if needle and needle in haystack:
            return arch
    return ''

def arch_stream_defaults(arch) -> dict:
    return dict(ARCH_STREAM_DEFAULTS.get(stream_arch_id(arch) or str(arch or '').lower(), {'attn': 'gqa', 'fa': 'off', 'ctk': 'f16', 'ctv': 'f16'}))

def canonical_streaming_rtx(rtx: dict) -> dict:
    r = dict(rtx or {})
    model_path = str(r.get('model_path') or '')
    arch = stream_arch_id(r.get('arch') or r.get('shape'), model_path)
    if not arch and model_path:
        arch = stream_arch_id(gguf_arch(model_path), model_path)
    if arch in ARCH_STREAM_DEFAULTS:
        defaults = ARCH_STREAM_DEFAULTS[arch]
        toggles = ARCH_STREAM_TOGGLES.get(arch, {})
        r['arch'] = arch
        for key, value in defaults.items():
            allowed = toggles.get(key)
            current = str(r.get(key) or '').strip().lower()
            r[key] = current if allowed and current in allowed else value
    if arch == 'glm-dsa':
        _default_mode = 'dense' if r.get('dsa_sparse') is False else 'full'
        mode = str(r.get('dsa_mode') or os.environ.get('STUDIO_GLM_DSA_MODE') or _default_mode).strip().lower()
        if mode == 'semi':
            mode = 'full'
        if mode not in ('dense', 'full'):
            mode = 'full'
        r['dsa_mode'] = mode
        r['dsa_sparse'] = mode != 'dense'
        if str(r.get('fa') or '').strip().lower() == 'on' and mode == 'dense':
            if os.environ.get('STUDIO_GLM_DSA_ALLOW_FA_DENSE') == '1':
                print('studio: glm-dsa dense: KEEPING -fa on because STUDIO_GLM_DSA_ALLOW_FA_DENSE=1 -- this configuration garbles on SM120 (measured 2026-08-27)', flush=True)
            else:
                print('studio: glm-dsa dense: forcing -fa off; dense+fa=on garbles on SM120 (measured 2026-08-27). Set STUDIO_GLM_DSA_ALLOW_FA_DENSE=1 to override.', flush=True)
                r['fa'] = 'off'
        if mode != 'dense':
            r['fa'] = 'off'
        try:
            if int(r.get('wf_gpu_bufs', 2)) >= 2 and int(r.get('ubatch', 512)) > 512:
                r['ubatch'] = 512
        except (TypeError, ValueError):
            pass
    return r

def streaming_prefill_env(rtx: dict, mac: Optional[dict]=None) -> dict:
    rtx = canonical_streaming_rtx(rtx)
    d = arch_stream_defaults(rtx.get('arch'))
    attn = str(rtx.get('attn') or d['attn']).lower()
    kv_ram = bool(rtx.get('kv_ram', True))
    env = {'CUDA_VISIBLE_DEVICES': '0', 'LLAMA_PREFILL_WF': '1', 'LLAMA_PREFILL_LAYER_MAJOR': '1', 'LLAMA_PREFILL_HIDDEN_MODE': 'gpu', 'LLAMA_PREFILL_KV_RAM': str(int(kv_ram)), 'LLAMA_NO_HOST': '1', 'LLAMA_PREFILL_TB_ENABLE': '1', 'LLAMA_SERVER_RELEASE_FREES_SLOT_KV': '1', 'LLAMA_WF_GPU_BUFS': str(int(rtx.get('wf_gpu_bufs', 2))), 'GGML_OP_OFFLOAD_MIN_BATCH': '1', 'GGML_CUDA_NO_VMM': '1'}
    if str(rtx.get('decode_mode') or 'split_thunderbolt').strip().lower() != 'stream_local':
        env['LLAMA_SERVER_SKIP_LOGITS_MARK'] = '1'
    if str(rtx.get('arch') or '').lower() == 'kimi':
        env.pop('LLAMA_SERVER_RELEASE_FREES_SLOT_KV', None)
        env.update({'LLAMA_PREFILL_KV_RAM_STORAGE': 'auto', 'LLAMA_PREFILL_KV_RAM_PINNED_POOL': '4', 'LLAMA_PREFILL_ALLOW_GPU_MASK_MLA': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'LLAMA_PREFILL_KV_DOWNLOAD_ZERO_SCAN_ALL': '1', 'LLAMA_PREFILL_KV_DL_HARD_SYNC': '1', 'LLAMA_PREFILL_KV_DOWNLOAD_ZERO_SCAN': '1', 'LLAMA_PREFILL_UB_SYNC_MODE': 'first_layer_only', 'LLAMA_PREFILL_FA_MASK_CACHE_MAX_MB': '4096', 'LLAMA_PREFILL_K_UBATCH_PROBE': '0', 'LLAMA_FA_MLA_DISABLE_NKV_PATCH': '1', 'LLAMA_FA_SM120_NCOLS32': '1', 'LLAMA_PREFILL_F16_MASK_REFRESH': '1', 'LLAMA_PREFILL_NO_FA_DISABLE': '1', 'LLAMA_PREFILL_VIABILITY_AS_PREFIX': '1', 'LLAMA_PREFILL_CACHED_PREFIX_PROMPT_DIRECT': '0', 'LLAMA_PREFILL_RETAIN_SLOT_AFTER_HANDOFF': '1', 'LLAMA_CUDA_OP_PROFILE': '1', 'LLAMA_DS_STAGING_MB': '500'})
    if attn == 'mla':
        env['LLAMA_PREFILL_MLA_MATCH_LAYOUT'] = '1'
        env['LLAMA_PREFILL_GPU_MASK_FIX'] = '1'
        env['LLAMA_PREFILL_FORCE_SAFE_MLA_MULTIUBATCH'] = '1'
        _dsa = rtx.get('dsa_sparse')
        if _dsa is True:
            env['LLAMA_MLA_SPARSE'] = '1'
            env['LLAMA_DSA_SCORE_CUBLAS'] = '1'
            env['LLAMA_DSA_SCORE_CUBLAS_K_SLAB'] = str(int(rtx.get('dsa_k_slab', 16384)))
            _sparse_fa = str(rtx.get('sparse_fa') or 'on').strip().lower() != 'off'
            env['LLAMA_MLA_SPARSE_FA_V6'] = '1' if _sparse_fa else '0'
            env['LLAMA_MLA_SPARSE_FUSED_KERNEL'] = '1'
            env['LLAMA_MLA_SPARSE_FA_DECOUPLE'] = '1' if _sparse_fa else '0'
            env['LLAMA_DSA_TOPK_CAUSAL'] = '1'
            env['LLAMA_DSA_POS_CAUSAL_MASK'] = '1'
            env['LLAMA_MLA_SPARSE_FA_V6_NCOLS64'] = '1' if _sparse_fa else '0'
            env['LLAMA_MLA_SPARSE_FA_V6_TILE_ALIGNED_GRID'] = '1' if _sparse_fa else '0'
            _dsa_top_k = rtx.get('dsa_top_k')
            if _dsa_top_k:
                try:
                    _tk = int(_dsa_top_k)
                    if _tk > 0 and _tk != 2048:
                        env['LLAMA_DSA_TOPK_OVERRIDE'] = str(_tk)
                except (TypeError, ValueError):
                    pass
            if str(rtx.get('dsa_mode') or 'full').strip().lower() == 'full':
                env['LLAMA_DSA_SHARED_SPARSE_REUSE'] = '1'
                _cov_bands = rtx.get('dsa_coverage_bands')
                if _cov_bands:
                    env['LLAMA_DSA_TOPK_COVERAGE_BANDS'] = str(int(_cov_bands))
            env['LLAMA_DSA_SHARED_CADENCE'] = '1'
    mac_backend = str((mac or {}).get('backend') or '').strip().lower()
    if str(rtx.get('arch') or '').strip().lower() == 'glm-dsa' and mac_backend in UPSTREAM_DECODE_RECEIVER_BACKENDS:
        env['LLAMA_PREFILL_STATE_WIRE'] = 'upstream-dsa'
    if rtx.get('ds_staging_mb') is not None:
        env['LLAMA_DS_STAGING_MB'] = str(int(rtx['ds_staging_mb']))
    without_thinking_caps([], rtx.get('env') or {})
    extra = rtx.get('extra_env') or {}
    return without_thinking_caps([], {**env, **{str(k): str(v) for k, v in extra.items()}})[1]

def streaming_prefill_argv(rtx: dict, mac: dict, server: str, *, host: str='0.0.0.0', port: int=8080) -> list:
    rtx = canonical_streaming_rtx(rtx)
    arch = str(rtx.get('arch', 'studio'))
    d = arch_stream_defaults(arch)
    ctx = int(rtx.get('ctx', 8192))
    batch = int(rtx.get('batch', ctx))
    ub = int(rtx.get('ubatch', 512))
    crossover = int(rtx.get('crossover', 3000))
    rlayers = int(rtx.get('remote_layers', 0))
    fa = str(rtx.get('fa') or d['fa']).lower()
    ctk = str(rtx.get('ctk') or d['ctk'])
    ctv = str(rtx.get('ctv') or d['ctv'])
    kv_host = mac.get('ssh_host') or mac.get('host') or ''
    kv_port = int(mac.get('kv_port', 9100))
    tok_port = int(mac.get('token_port', 9101))
    stream_local = str(rtx.get('decode_mode') or '').strip().lower() == 'stream_local'
    if rlayers <= 0 and (not stream_local):
        raise RuntimeError('streaming_prefill_argv: rtx.remote_layers (model n_layers) is required')
    argv = [server, '-m', _resolve_gguf_model_path(rtx.get('model_path', '')), '-ngl', '0', '-ps', '-fa', fa, '-ctk', ctk, '-ctv', ctv, '-b', str(batch), '-ub', str(ub), '-c', str(ctx), '-np', str(int(rtx.get('np', 1))), '--port', str(port), '--host', host, '-a', arch, '--jinja', '--reasoning-format', 'none', '--prefill-crossover', str(crossover)]
    _template_path = write_template(rtx.get('jinja'))
    if _template_path:
        argv += ['--chat-template-file', str(_template_path)]
    if stream_local:
        argv += ['--decode-mode', 'stream_local']
    else:
        argv += ['--decode-mode', 'split_thunderbolt', '--decode-remote-layers', str(rlayers), '--decode-strict-partition', '--decode-require-no-fallback', '--prefill-decode-transport-required', '--decode-remote-failover', 'fail', '--prefill-transport-mode', 'bulk', '--kv-transport', 'tcp', '--kv-host', kv_host, '--kv-port', str(kv_port), '--token-stream-host', kv_host, '--token-stream-port', str(tok_port)]
    if rtx.get('yarn_beta_fast') is not None:
        argv += ['--yarn-beta-fast', str(float(rtx['yarn_beta_fast']))]
    if rtx.get('prefill_overlap'):
        argv += ['--prefill-min-stream-batch-tokens', '1', '--prefill-overlap', '--prefill-prefetch-distance', str(int(rtx.get('prefetch_distance', 3))), '--prefill-layer-buffers', str(int(rtx.get('layer_buffers', 3)))]
    return argv + preset_samplers.launch_args(rtx) + _rtx_kv_cache_argv(rtx)

def without_thinking_caps(argv: list, env: dict) -> tuple[list, dict]:
    import logging
    logger = logging.getLogger(__name__)
    clean_env = dict(env)
    for key in ('LLAMA_ARG_THINK_BUDGET', 'LLAMA_ARG_REASONING_BUDGET'):
        if key in clean_env:
            clean_env.pop(key)
            logger.warning('Ignored obsolete %s; reasoning effort is the thinking control', key)
    clean_argv = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if str(arg).split('=', 1)[0] in {'--reasoning-budget', '--thinking-budget'}:
            logger.warning('Ignored obsolete thinking budget flag; use reasoning effort')
            skip = '=' not in str(arg)
            continue
        clean_argv.append(arg)
    return (clean_argv, clean_env)
