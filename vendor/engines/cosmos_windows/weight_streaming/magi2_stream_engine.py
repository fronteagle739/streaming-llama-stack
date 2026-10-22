# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_stream_engine.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from pathlib import Path
from typing import Optional
from cosmos_windows.weight_streaming.magi2_backend import Magi2GenerationRequest
MAGI2_LOAD_MECHANISMS = {'text_encode': 'Qwen3.5-27B: meta-build + per-layer NVMe stream via build_streamed_text_encoder (RssProbe-guarded); each decoder layer touched once, ~1-2 layers resident.', 'ref_image_encode': 'Wan2.2 VAE: meta-build + safetensors byte-index bind; first frame encoded, evicted.', 'preview_denoise': 'MoE preview DiT: whole-model double-buffered layer ring, tensors streamed NVMe->bounded pinned (256MB)->GPU slot in chunks (never a whole-layer host stage).', 'refiner_denoise': 'Refiner: 14GB dense streamed per-layer through the ring (or resident after preview evict); windowed local attention bounds 1080p activations.', 'vae_decode': 'Turbo/Wan VAE: meta-build + byte-index bind; TILED spatial + temporal-chunk decode (magi2_vae_tiling) so peak VRAM is one tile, not the whole 1080p clip.', 'mux': 'ffmpeg subprocess; negligible host/VRAM.'}

def magi2_load_mechanisms() -> dict:
    return dict(MAGI2_LOAD_MECHANISMS)

def _setup_stream_sdpa(request: Magi2GenerationRequest, params: dict, emit):
    import torch
    from cosmos_windows.weight_streaming.magi2_attention_adapter import install_dependency_shims, install_magi2_attention_intercept
    from cosmos_windows.weight_streaming.magi2_shard_index import build_magi2_shard_index
    from cosmos_windows.weight_streaming.magi2_layer_ring import assert_ring_invariant, directstorage_available, dry_run_pass, plan_layer_ring
    from cosmos_windows.weight_streaming.magi2_runtime import assert_budgets_fit, stage_budget_table
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    try:
        from cosmos_windows.weight_streaming.ram_guard import preflight_host_ram
        preflight_host_ram(4.0, 'MAGI-2 stream-sdpa (bounded pinned staging)', skip_env='MAGI2_SKIP_RAM_PREFLIGHT')
    except ImportError:
        pass
    emit.emit('stage_log', stage='stream_sdpa_setup', message='installing Hopper-dep shims + importing reference model', load_mechanisms=MAGI2_LOAD_MECHANISMS)
    import sys as _sys
    from cosmos_windows.weight_streaming.magi2_backend import resolve_magi2_repo_root
    _repo = resolve_magi2_repo_root(getattr(request, 'repo_root', None))
    if _repo is not None and str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))
    shimmed = install_dependency_shims()
    try:
        import inference.model.magi2_preview
        import inference.model.magi2_refiner
        import inference.common.arch
    except Exception as exc:
        raise RuntimeError(f'stream-sdpa could not import the reference model modules (inference.model.magi2_preview/refiner): {type(exc).__name__}: {exc}. Ensure the MAGI-2-preview repo is on PYTHONPATH and Triton is installed.') from exc
    intercept = install_magi2_attention_intercept()
    emit.emit('attention_backend_selected', backend='cudnn-sdpa', intercept=intercept.as_dict(), message='SM120 attention intercept installed (FA3 -> tier-A SDPA)')
    from cosmos_windows.weight_streaming.magi2_moe_binding import install_moe_kernel_sm120_patch
    moe_patch = install_moe_kernel_sm120_patch(emit)
    if moe_patch.get('patched'):
        emit.emit('stage_log', stage='moe_kernel_patch', message='MoE Triton kernel SM120-safe', **{k: moe_patch[k] for k in ('reason', 'config') if k in moe_patch})
    if intercept.would_take_hopper_path:
        emit.emit('stage_log', stage='attention', message='confirmed: reference would take Hopper FA3 path on this arch; intercepted')
    preview_dir = Path(request.model_dir) / 'preview'
    emit.emit('stage_log', stage='index', message=f'building shard index over {preview_dir}')
    shard_index = build_magi2_shard_index(preview_dir)
    full_index = build_index(preview_dir)
    ring_budget = int(request.ring_budget_gb * 1000000000.0)
    plan = plan_layer_ring(full_index, vram_budget_bytes=ring_budget)
    assert_ring_invariant(plan)
    sched = dry_run_pass(plan, steps=params['steps'])
    emit.emit('stage_log', stage='layer_ring', message='whole-model layer-ring plan ready', plan=plan.summary(), modeled_stream_seconds=round(sched.modeled_seconds, 1), bytes_per_pass=plan.bytes_per_pass(), shard_summary=shard_index.summary())
    import sys as _sys
    is_windows = _sys.platform == 'win32'
    ds_ok = is_windows and directstorage_available()
    transport = 'directstorage' if ds_ok else 'pinned'
    if ds_ok:
        emit.emit('transport_selected', transport='directstorage', level='info', message='weight path = TRUE DirectStorage (D3D12 heap -> CUDA ext-mem; ~0 pageable host RAM)')
    elif not is_windows:
        emit.emit('transport_selected', transport='pinned', level='info', reason='portable (no DirectStorage on this platform)', message='weight path = bounded-pinned portable tier (macOS/Metal / Linux, RSS-guarded)')
    else:
        emit.fallback('layer-ring transport', 'DirectStorage DLL/runtime not available on Windows -> bounded-pinned host staging (RSS-guarded)')
    budget = stage_budget_table(request.mode, request.resolution, ring_budget_gb=request.ring_budget_gb, transport=transport)
    assert_budgets_fit(budget)
    emit.emit('stage_log', stage='budget', message='per-stage VRAM/RSS budget verified (free-before-alloc, DS weight path ~0 RSS)', transport=transport, budget=[{'stage': b.stage, 'peak_vram_gb': b.peak_vram_gb, 'peak_rss_gb': b.peak_rss_gb, 'mechanism': b.mechanism} for b in budget])
    return {'torch': torch, 'shimmed': shimmed, 'intercept': intercept, 'shard_index': shard_index, 'full_index': full_index, 'plan': plan, 'preview_dir': preview_dir}

def run_real_stream_sdpa(request: Magi2GenerationRequest, params: dict, emit) -> dict:
    import time as _time
    from cosmos_windows.weight_streaming.magi2_generate import generate
    ctx = _setup_stream_sdpa(request, params, emit)
    plan = ctx['plan']
    emit.emit('stage_log', stage='engine', message=f'stream-sdpa ready: {plan.num_layers}-layer ring (depth {plan.ring_depth}, ~{plan.bytes_per_pass() / 1000000000.0:.0f}GB/step); starting real generation')
    out_path = Path(request.out)
    out_prefix = str(out_path.with_suffix('')) if out_path.suffix else str(out_path)
    steps = params.get('steps') or 100
    negative = params.get('negative_prompt')
    import os as _os
    transport = _os.environ.get('MAGI2_TRANSPORT', 'auto')

    def _emit_log(**kw):
        ev = kw.pop('event', 'stage_log')
        emit.emit(ev, **kw)
    t0 = _time.time()
    rep = generate(model_dir=Path(request.model_dir), prompt=request.prompt, negative_prompt=negative, out_prefix=out_prefix, resolution=request.resolution, steps=steps, transport=transport, ring_budget_gb=request.ring_budget_gb, seconds=request.seconds, image=str(request.image) if request.image else None, compute_dtype=params.get('compute_dtype', 'bf16'), enable_refiner=params.get('enable_refiner'), refiner_steps=params.get('refiner_steps') or 5, refiner_width=request.refiner_width, refiner_height=request.refiner_height, emit=_emit_log)
    out_mp4 = rep.get('output', f'{out_prefix}.mp4')
    if str(out_mp4) != str(out_path) and Path(out_mp4).is_file():
        try:
            Path(out_mp4).replace(out_path)
            out_mp4 = str(out_path)
        except OSError:
            pass
    emit.emit('vae_mux_completed', output=str(out_mp4))
    result = {'output': str(out_mp4), 'seconds': round(_time.time() - t0, 2), 'mode': request.mode, 'timing': rep}
    emit.emit('completed', **{k: result[k] for k in ('output', 'seconds', 'mode')})
    return result
