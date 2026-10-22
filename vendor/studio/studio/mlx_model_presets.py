# Derived from Studio backend/studio/mlx_model_presets.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import os
from typing import Optional
from . import paths, prism_mlx

def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or '').strip() or default
MAC_REMOTE_HOST = _env('STUDIO_MAC_REMOTE_HOST', '<mac-ip>')
MLX_MODELS: dict[str, dict] = {'glm-5.1': {'label': 'GLM-5.1', 'mac_model': _env('STUDIO_GLM51_MAC_MLX', ''), 'mlx_server': 'mlx-vlm', 'rtx_gguf': _env('STUDIO_GLM51_RTX_GGUF', ''), 'ctx': 131072, 'total_layers': 78, 'pipeline_layers': _env('STUDIO_GLM51_PIPELINE_LAYERS', '66,12'), 'modes': ('remote', 'streaming', 'distributed')}, 'glm-5.2': {'label': 'GLM-5.2', 'mac_model': _env('STUDIO_GLM52_MAC_MLX', ''), 'mlx_server': _env('STUDIO_GLM52_MLX_SERVER', 'mtp-shim'), 'rtx_gguf': _env('STUDIO_GLM52_RTX_GGUF', ''), 'stream_arch': 'glm-dsa', 'stream_backend': 'kimi-decode', 'ctx': 131072, 'total_layers': 78, 'pipeline_layers': _env('STUDIO_GLM52_PIPELINE_LAYERS', '66,12'), 'modes': ('remote', 'streaming'), 'mtp': {'mtp_mode': _env('STUDIO_GLM52_MTP_MODE', 'speculative'), 'mtp_max_draft': int(_env('STUDIO_GLM52_MTP_MAX_DRAFT', '7')), 'mtp_sidecar': _env('STUDIO_GLM52_MTP_SIDECAR', ''), 'mtp_token_budget': int(_env('STUDIO_GLM52_MTP_TOKEN_BUDGET', '0')), 'mtp_adaptive_depth': _env('STUDIO_GLM52_MTP_ADAPTIVE', 'streak')}}, 'kimi-k2.7-code': {'label': 'Kimi-K2.7-Code', 'mac_model': _env('STUDIO_K27_MAC_MLX', ''), 'mlx_server': 'mlx-vlm', 'rtx_gguf': _env('STUDIO_K27_RTX_GGUF', ''), 'stream_arch': 'kimi', 'stream_backend': 'kimi-decode', 'ctx': 131072, 'total_layers': 61, 'pipeline_layers': _env('STUDIO_K27_PIPELINE_LAYERS', '51,10'), 'modes': ('remote', 'streaming')}, 'mimo-v2.5-pro': {'label': 'MiMo-V2.5-Pro', 'mac_model': _env('STUDIO_MIMO25_MAC_MLX', ''), 'mlx_server': 'mlx-vlm', 'rtx_gguf': _env('STUDIO_MIMO25_RTX_GGUF', ''), 'ctx': 65536, 'total_layers': 70, 'pipeline_layers': _env('STUDIO_MIMO25_PIPELINE_LAYERS', '56,14'), 'rank0_budget_gb': float(_env('STUDIO_MIMO25_RANK0_BUDGET_GB', '494')), 'rank1_budget_gb': float(_env('STUDIO_MIMO25_RANK1_BUDGET_GB', '137')), 'modes': ('distributed',)}, 'kimi-k3': {'label': 'Kimi-K3 (UD-IQ1_S)', 'mac_model': _env('STUDIO_K3_MAC_GGUF', ''), 'mlx_server': '', 'rtx_gguf': _env('STUDIO_K3_RTX_GGUF', ''), 'ctx': 2048, 'total_layers': 93, 'pipeline_layers': '', 'rpc_peers': _env('STUDIO_K3_RPC_PEERS', '<mac-ip>:50052,<mac-ip>:50052'), 'rpc_devices': 'MTL0,RPC0,RPC1', 'tensor_split': _env('STUDIO_K3_TS', '81,14,5'), 'ngl': 99, 'mmap': True, 'binary': _env('STUDIO_K3_BIN', '~/build/llama-k3/build/bin/llama-cli'), 'requires_patch': 'DSV4_HC_PRE Metal block-size fix (K3 uses attn_res.block_size 12; the Metal backend hardcoded DeepSeek-V4\'s 4 in ggml-metal-device.m:1354 and ggml-metal-ops.cpp:1436). Without it every Mac aborts with "unsupported op \'DSV4_HC_PRE\'".', 'measured': {'prefill_tok_s': 7.4, 'decode_tok_s': 4.0, 'prompt_tokens': 7, 'date': '2026-08-11', 'note': 'prefill measured on a 7-token prompt; not a meaningful prefill number, sweep still owed'}, 'modes': ('rpc3',)}, 'dsv4-pro': {'label': 'DSV4-Pro', 'mac_model': _env('STUDIO_DSV4_RANK0_MODEL', ''), 'mlx_server': 'mlx-vlm', 'rtx_gguf': '', 'ctx': 32768, 'total_layers': 61, 'pipeline_layers': _env('STUDIO_DSV4_PIPELINE_LAYERS', '51,10'), 'modes': ('distributed',)}, 'h3-vlm': {'label': 'H3 Qwen3-VL-32B (own VLM)', 'mac_model': _env('STUDIO_H3VLM_MAC_MLX', ''), 'mlx_server': 'mlx-vlm', 'rtx_gguf': '', 'ctx': 32768, 'modes': ('remote',)}}

def build_preset(model_key: str, mode: str) -> dict:
    m = MLX_MODELS[model_key]
    if mode not in m['modes']:
        raise ValueError(f"{model_key} does not support mode {mode!r} (has {m['modes']})")
    label = m['label']
    if mode == 'remote':
        return {'name': f'{label} (Mac remote)', 'mode': 'remote', 'rtx': {'model_path': ''}, 'mac': {'model_path': m['mac_model'], 'mlx_server': m['mlx_server'], 'backend': m['mlx_server'], 'host': MAC_REMOTE_HOST, 'ctx': m['ctx'], **m.get('mtp', {})}}
    if mode == 'streaming':
        if not m['rtx_gguf']:
            raise ValueError(f'{model_key} has no rtx_gguf -> cannot build a streaming preset')
        rtx = {'model_path': m['rtx_gguf'], 'ctx': m['ctx'], 'ngl': 99, 'batch': min(int(m['ctx']), 24576), 'ubatch': 1024, 'weight_stream': True, 'prefetch': True}
        mac_backend = m['mlx_server']
        if m.get('stream_arch'):
            rtx.update({'arch': m['stream_arch'], 'stream_engine': 'general', 'remote_layers': m['total_layers'], 'crossover': 1})
            mac_backend = m.get('stream_backend', m['mlx_server'])
        return {'name': f'{label} (RTX stream + Mac decode)', 'mode': 'streaming', 'rtx': rtx, 'mac': {'model_path': m['mac_model'], 'mlx_server': mac_backend, 'backend': mac_backend, 'host': MAC_REMOTE_HOST, 'ctx': m['ctx'], **m.get('mtp', {})}}
    if mode == 'rpc3':
        return {'name': f'{label} (3-machine RPC baseline)', 'mode': 'rpc3', 'rtx': {'model_path': ''}, 'mac': {'model_path': m['mac_model'], 'backend': 'llama.cpp-rpc', 'host': MAC_REMOTE_HOST, 'ctx': m['ctx'], 'total_layers': m['total_layers'], 'rpc_peers': m['rpc_peers'], 'rpc_devices': m['rpc_devices'], 'tensor_split': m['tensor_split'], 'ngl': m['ngl'], 'mmap': m['mmap'], 'binary': m['binary'], 'requires_patch': m['requires_patch'], 'measured': m['measured']}}
    if mode == 'distributed':
        return {'name': f'{label} (2-Mac layer-split)', 'mode': 'remote', 'rtx': {'model_path': ''}, 'mac': {'model_path': m['mac_model'], 'mlx_server': m['mlx_server'], 'backend': m['mlx_server'], 'host': MAC_REMOTE_HOST, 'ctx': m['ctx'], 'layer_split': True, 'distributed_backend': 'rpc', 'weight_source': 'staged', 'total_layers': m['total_layers'], 'pipeline_layers': m['pipeline_layers'], **{k: m[k] for k in ('rank0_budget_gb', 'rank1_budget_gb', 'mem_reserve_gb') if k in m}}}
    raise ValueError(f'unknown mode {mode!r}')

def gguf_streaming_preset(label, *, rtx_gguf, mac_gguf, arch, n_layers, mac_host='<mac-ip>', ctx=8192, batch=15000, ubatch=512, crossover=3000, fa='off', ctk='f16', ctv='f16', wf_gpu_bufs=2, attn='gqa', kv_port=9100, token_port=9101, max_tokens=256) -> dict:
    return {'name': f'{label} (RTX stream + Mac decode)', 'mode': 'streaming', 'rtx': {'model_path': rtx_gguf, 'arch': arch, 'attn': attn, 'ctx': ctx, 'batch': batch, 'ubatch': ubatch, 'remote_layers': int(n_layers), 'crossover': crossover, 'fa': fa, 'ctk': ctk, 'ctv': ctv, 'weight_stream': True, 'stream_engine': 'general', 'wf_gpu_bufs': wf_gpu_bufs}, 'mac': {'host': mac_host, 'kv_port': kv_port, 'token_port': token_port, 'backend': 'llama-decode', 'model_path': mac_gguf, 'ctx': ctx, 'ngl': 99, 'max_tokens': max_tokens}}
GGUF_STREAM_MODELS = {'minimax-m3': dict(label='MiniMax-M3', rtx_gguf=_env('STUDIO_M3_RTX_GGUF', ''), mac_gguf=_env('STUDIO_M3_MAC_GGUF', ''), arch='minimax-m3', n_layers=60, ctx=32768, batch=15000, ubatch=512, crossover=3000, fa='off', ctk='f16', ctv='f16', wf_gpu_bufs=1, attn='gqa')}

def build_prism_bonsai_local_preset(model_path: str) -> dict:
    status = prism_mlx.inspect_bonsai_model(model_path)
    if not status.compatible:
        raise ValueError(f'incompatible Ternary Bonsai checkpoint: {status.reason}')
    python = prism_mlx.runtime_python()
    cache_dir = paths.app_data_root() / 'mlx-kv' / 'prism-bonsai-27b'
    return {'name': 'Ternary Bonsai 27B (Mac Prism MLX)', 'mode': 'local', 'rtx': {'backend': 'mlx-vlm', 'mlx_runtime': 'prism', 'model_path': str(status.path), 'ctx': 262144, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 20}, 'mac': {'mlx_runtime': 'prism', 'python': str(python), 'prism_python': str(python), 'prompt_cache': {'enabled': True, 'mac_dir': str(cache_dir), 'max_gb': 10.0}}}
QWEN38_MAC_MLX = _env('STUDIO_QWEN38_MAC_MLX', 'models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit')
GLM46_MAC_GGUF = _env('STUDIO_GLM46_MAC_GGUF', os.path.expanduser('~/.omlx/models/dawncr0w/bartowski/ArliAI_GLM-4.6-Derestricted-GGUF/ArliAI_GLM-4.6-Derestricted-IQ2_M-00001-of-00003.gguf'))
DSV4F_0731_MAC_GGUF = _env('STUDIO_DSV4F_0731_MAC_GGUF', 'models/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf')

def build_qwen38_mlx_serve_preset() -> dict:
    return {'name': 'Qwen3.8-Flash-Next (Mac mlx-serve)', 'mode': 'remote', 'rtx': {'model_path': ''}, 'mac': {'backend': 'mlx-serve', 'host': MAC_REMOTE_HOST, 'model_path': QWEN38_MAC_MLX, 'ctx': 65536, 'max_tokens': 4096, 'mlx_serve_mtp': 1, 'mlx_serve_skip_mem_preflight': 1, 'mlx_serve_pld': 1, 'mlx_serve_bind_address': '0.0.0.0'}}
QWEN38_UNCENSORED_MAC_MLX = _env('STUDIO_QWEN38_UNCENSORED_MAC_MLX', 'models/Qwen3.8-Flash-Next-Uncensored-MLX-Serve-mixed-4-8bit')

def build_qwen38_uncensored_preset(*, ctx: int=512000) -> dict:
    return {'name': f'Qwen3.8-Flash-Next Uncensored (Mac mlx-serve) {ctx // 1000}k context', 'mode': 'remote', 'rtx': {'model_path': ''}, 'mac': {'backend': 'mlx-serve', 'host': MAC_REMOTE_HOST, 'model_path': QWEN38_UNCENSORED_MAC_MLX, 'ctx': int(ctx), 'max_tokens': 4096, 'mlx_serve_mtp': 1, 'mlx_serve_pld': 1, 'mlx_serve_skip_mem_preflight': 1, 'mlx_serve_bind_address': '0.0.0.0'}}

def build_glm46_llama_preset() -> dict:
    return {'name': 'GLM-4.6 Derestricted IQ2_M (Mac llama-server)', 'mode': 'local', 'rtx': {'model_path': GLM46_MAC_GGUF, 'ctx': 8192, 'batch': 2048, 'ubatch': 512, 'ngl': 99, 'max_tokens': 2048}, 'mac': {}}

def build_same_mac_local_presets() -> list[dict]:
    return [build_qwen38_mlx_serve_preset(), build_qwen38_uncensored_preset(), build_glm46_llama_preset()]

def build_optional_local_presets() -> list[dict]:
    model = prism_mlx.discover_bonsai_model()
    return [build_prism_bonsai_local_preset(str(model))] if model is not None else []

def build_gguf_stream_presets() -> list[dict]:
    return [gguf_streaming_preset(**spec) for spec in GGUF_STREAM_MODELS.values()]

def build_all_presets() -> list[dict]:
    out: list[dict] = []
    for key, m in MLX_MODELS.items():
        for mode in m['modes']:
            out.append(build_preset(key, mode))
    out.extend(build_gguf_stream_presets())
    out.extend(build_same_mac_local_presets())
    out.extend(build_optional_local_presets())
    return out

def seed_mlx_presets(*, preset_store=None, overwrite: bool=False) -> dict:
    if preset_store is None:
        from . import preset_store as preset_store
    existing = {str(p.get('name')): p for p in preset_store.list_all()}
    created, skipped = ([], [])
    for spec in build_all_presets():
        name = spec['name']
        if name in existing and (not overwrite):
            skipped.append(name)
            continue
        preset_store.create(name, spec['rtx'], spec['mac'], mode=spec['mode'])
        created.append(name)
    return {'created': created, 'skipped': skipped}
