# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/hunyuan_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import gc
import os
import re
import time
import types
from pathlib import Path
_HY_LAYER_RE = re.compile('^(?:[^/]+/)?model\\.layers\\.(\\d+)\\.')
_FALSE_ENV = {'0', 'false', 'off', 'no', 'n'}

def _env_enabled(name: str, default: bool=True) -> bool:
    val = os.environ.get(name)
    if val is None or str(val).strip() == '':
        return default
    return str(val).strip().lower() not in _FALSE_ENV

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, '') or default)
    except Exception:
        return default

def _is_hy_vae_path(path: str) -> bool:
    return path == 'vae' or path.startswith('vae.')

def _patch_transformers_compat():
    try:
        import inspect
        import transformers.cache_utils as _tc
        if not hasattr(_tc, 'StaticLayer'):
            return
        li = _tc.StaticLayer.lazy_initialization
        if getattr(li, '_hy_patched', False):
            return
        params = inspect.signature(li).parameters
        needs = 'value_states' in params and params['value_states'].default is inspect.Parameter.empty
        if needs:

            def _li(self, key_states, value_states=None, *a, **k):
                return li(self, key_states, key_states if value_states is None else value_states, *a, **k)
            _li._hy_patched = True
            _tc.StaticLayer.lazy_initialization = _li
    except Exception as e:
        print('[HY-STREAM] (cache compat patch skipped:', type(e).__name__, e, ')', flush=True)

def _load_meta_hunyuan(model_dir: str):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    _patch_transformers_compat()
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = 'sdpa'
    if hasattr(cfg, 'moe_impl'):
        cfg.moe_impl = 'default'
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    import json, os
    gcf = os.path.join(model_dir, 'generation_config.json')
    if os.path.exists(gcf):
        for k, v in json.load(open(gcf, encoding='utf-8')).items():
            setattr(model.generation_config, k, v)
    if not hasattr(cfg, 'model_version'):
        cfg.model_version = None
    return (model, cfg)

def _force_stream_device_metadata(model, device: str):
    import torch
    dev = torch.device(device)
    for obj in (model, getattr(model, 'model', None)):
        if obj is None:
            continue
        try:
            setattr(type(obj), 'device', property(lambda _self, _dev=dev: _dev))
            setattr(obj, '_hy_stream_device', dev)
        except Exception as e:
            print(f'[HY-STREAM] device metadata patch skipped for {type(obj).__name__}: {type(e).__name__}: {str(e)[:120]}', flush=True)

def hy_group_by_layer(idx):
    layers, globs = ({}, [])
    for name, ref in idx.refs.items():
        m = _HY_LAYER_RE.search(name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (layers, globs)

def _patch_vision_conditioning_device(model):
    vision = getattr(model, 'vision_model', None)
    if vision is None or getattr(vision, '_hy_mask_device_patched', False):
        return
    original = vision.forward

    def forward(pixel_values, attention_mask=None, spatial_shapes=None, **kwargs):
        if attention_mask is not None and attention_mask.device != pixel_values.device:
            attention_mask = attention_mask.to(device=pixel_values.device)
        return original(pixel_values, attention_mask=attention_mask, spatial_shapes=spatial_shapes, **kwargs)
    vision.forward = forward
    vision._hy_mask_device_patched = True

def build_streamed_hunyuan(model_dir: str, device: str='cuda', n_gpu_buffers: int=3):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    from .build_streamed_transformer import StreamingWeightLoader
    model, cfg = _load_meta_hunyuan(model_dir)
    _patch_vision_conditioning_device(model)
    tf = model.model
    idx = build_index(model_dir)
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = hy_group_by_layer(idx)
    pb = PortableBackend(pin_memory=True)

    def _pp(name):
        return name.split('/', 1)[1] if '/' in name else name
    lazy_vae = _env_enabled('HY_LAZY_VAE', True)
    vae_tiling = _env_enabled('HY_VAE_TILING', True)
    unload_vae_after = _env_enabled('HY_UNLOAD_VAE_AFTER_DECODE', True)
    release_globals_before_vae = _env_enabled('HY_RELEASE_GLOBALS_BEFORE_VAE', True)
    pre_vae_free_gb = _env_float('HY_PRE_VAE_FREE_GB', 3.0)
    pre_vae_wait_s = _env_float('HY_PRE_VAE_WAIT_S', 45.0)
    vae_refs = [ref for ref in global_refs if _is_hy_vae_path(_pp(ref.name))]
    resident_global_refs = [ref for ref in global_refs if not _is_hy_vae_path(_pp(ref.name))] if lazy_vae else list(global_refs)

    def _enable_vae_tiling():
        if not vae_tiling:
            return
        vae = getattr(model, 'vae', None)
        if vae is None:
            return
        try:
            if hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
            if hasattr(vae, 'enable_temporal_tiling'):
                vae.enable_temporal_tiling()
            print('[HY-STREAM] VAE tiled decode enabled', flush=True)
        except Exception as e:
            print(f'[HY-STREAM] VAE tiling enable skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)
    for ref in resident_global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path in sd_keys:
            set_module_tensor_to_device(model, path, device, value=pb.read_tensor(ref, device, direct=True))

    def _install_lazy_vae():
        vae = getattr(model, 'vae', None)
        vae_paths = [StreamingWeightLoader._param_path(ref) for ref in vae_refs if StreamingWeightLoader._param_path(ref) in sd_keys]
        if vae is None:
            return
        if not lazy_vae or not vae_paths:
            _enable_vae_tiling()
            return
        if getattr(vae, '_hy_lazy_wrapped', False):
            return
        state = {'loaded': False}
        streamer_released = {'done': False}
        globals_released = {'done': False}

        def _free_host_gb():
            try:
                from cosmos_windows import mem_guard as _mg
                return _mg.free_gb()
            except Exception:
                return float('inf')

        def _cuda_gc():
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass

        def _release_non_vae_globals_for_decode():
            if globals_released['done'] or not release_globals_before_vae:
                return
            globals_released['done'] = True
            released = 0
            for ref in resident_global_refs:
                path = StreamingWeightLoader._param_path(ref)
                if path in sd_keys and (not _is_hy_vae_path(path)):
                    try:
                        set_module_tensor_to_device(model, path, 'meta')
                        released += 1
                    except Exception as e:
                        print(f'[HY-STREAM] non-VAE global release skipped for {path}: {type(e).__name__}: {str(e)[:100]}', flush=True)
            _cuda_gc()
            print(f'[HY-STREAM] released {released} non-VAE resident globals before VAE decode (free={_free_host_gb():.1f}GB)', flush=True)

        def _wait_for_vae_headroom():
            deadline = time.time() + max(0.0, pre_vae_wait_s)
            while True:
                free = _free_host_gb()
                if free >= pre_vae_free_gb:
                    print(f'[HY-STREAM] VAE decode RAM gate OK: {free:.1f}GB free (need >= {pre_vae_free_gb:.1f})', flush=True)
                    return
                if time.time() >= deadline:
                    raise RuntimeError(f'Not enough host RAM to lazy-load Hunyuan VAE safely: {free:.1f}GB free, need >= {pre_vae_free_gb:.1f}GB. The diffusion pass finished, but decode was stopped before the watchdog could kill the process. Close apps or lower resolution; override with HY_PRE_VAE_FREE_GB if you know this machine has headroom.')
                print(f'[HY-STREAM] waiting for VAE decode RAM headroom: {free:.1f}GB free (need >= {pre_vae_free_gb:.1f})', flush=True)
                _cuda_gc()
                time.sleep(2.0)

        def _release_streamer_for_decode():
            if streamer_released['done']:
                return
            streamer_released['done'] = True
            try:
                streamer.close()
                print('[HY-STREAM] DirectStorage buffers released before VAE decode', flush=True)
            except Exception as e:
                print(f'[HY-STREAM] streamer release before VAE decode skipped: {type(e).__name__}: {str(e)[:120]}', flush=True)
            _cuda_gc()

        def _load_vae(release_streamer: bool=False):
            if state['loaded']:
                return
            if release_streamer:
                _release_streamer_for_decode()
                _release_non_vae_globals_for_decode()
                _wait_for_vae_headroom()
            print(f'[HY-STREAM] lazy-loading VAE ({len(vae_paths)} tensors) for tiled decode...', flush=True)
            for ref in vae_refs:
                path = StreamingWeightLoader._param_path(ref)
                if path in sd_keys:
                    set_module_tensor_to_device(model, path, device, value=pb.read_tensor(ref, device, direct=True))
            state['loaded'] = True
            model._hy_vae_loaded = True
            _enable_vae_tiling()

        def _unload_vae():
            if not state['loaded']:
                return
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            for path in vae_paths:
                set_module_tensor_to_device(model, path, 'meta')
            state['loaded'] = False
            model._hy_vae_loaded = False
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print('[HY-STREAM] VAE evicted after decode', flush=True)

        def _wrap_method(method_name):
            raw = getattr(type(vae), method_name, None)
            if raw is None:
                return

            def _wrapped(self, *args, _raw=raw, _method=method_name, **kwargs):
                _load_vae(release_streamer=_method == 'decode')
                try:
                    return _raw(self, *args, **kwargs)
                finally:
                    if unload_vae_after:
                        _unload_vae()
            setattr(vae, method_name, types.MethodType(_wrapped, vae))
        _enable_vae_tiling()
        _wrap_method('decode')
        _wrap_method('encode')
        _wrap_method('forward')
        vae._hy_lazy_wrapped = True
        model._hy_vae_lazy = True
        model._hy_vae_loaded = False
        model._hy_load_vae = _load_vae
        model._hy_unload_vae = _unload_vae
        print(f'[HY-STREAM] lazy VAE enabled ({len(vae_paths)} tensors; unload_after_decode={unload_vae_after})', flush=True)
    streamer = DirectStorageStreamer(idx, component='', n_gpu_buffers=n_gpu_buffers, device=device, layer_groups=layer_refs)
    from .ds_backend import enforce_ds_mode
    enforce_ds_mode(streamer, 'Hunyuan', 'HUNYUAN')
    n_layers = len(tf.layers)
    lazy_note = f', {len(vae_refs)} lazy VAE tensors' if lazy_vae and vae_refs else ''
    print(f'[HY-STREAM] {n_layers} layers streamed, {len(resident_global_refs)} resident globals{lazy_note}, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    _install_lazy_vae()
    layer_paths = {il: [_pp(r.name) for r in refs if _pp(r.name) in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    set_module_tensor_to_device(model, p, device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for p in layer_paths[il]:
                set_module_tensor_to_device(model, p, 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(tf.layers):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *a, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kw):
            _b()
            try:
                out = _fwd(_layer, hidden_states, *a, **kw)
            finally:
                _e()
            return out
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.eval()
    _force_stream_device_metadata(model, device)
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[HY-STREAM] wrapped {n_layers} HunyuanImage3DecoderLayers (bind/evict per layer; native batch dim for N images). Streams ALL 64 experts/layer = optimal for image gen (8194 tok/forward hits 64/64 experts; routed-expert FETCH_ACTIVE saves ~0% here, only helps few-token AR/text).', flush=True)
    return (model, streamer)

def build_streamed_backward_hunyuan(model_dir: str, device: str='cuda', n_gpu_buffers: int=1):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .build_streamed_transformer import StreamingWeightLoader
    from .safetensors_index import build_index
    from ..streamed_backward import apply_streamed_multi
    model, cfg = _load_meta_hunyuan(model_dir)
    _patch_vision_conditioning_device(model)
    tf = model.model
    idx = build_index(model_dir)
    sd_keys = set(model.state_dict().keys())
    layer_refs, global_refs = hy_group_by_layer(idx)
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path in sd_keys:
            set_module_tensor_to_device(model, path, device, value=pb.read_tensor(ref, device, direct=True))
    streamer = DirectStorageStreamer(idx, component='', n_gpu_buffers=n_gpu_buffers, device=device, layer_groups=layer_refs)
    n_layers = len(tf.layers)
    print(f"[HY-BWD] mode={streamer.mode} {n_layers} layers, {len(streamer.tbufs)} GPU buf(s), {len(global_refs)} resident globals (reverse-capable; {('prefetch ON' if n_gpu_buffers > 1 else 'single-buffer=validated reverse')})", flush=True)

    def _pp(name):
        return name.split('/', 1)[1] if '/' in name else name

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = model.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    layer_paths = {il: [_pp(r.name) for r, _ in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in streamer.plan}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    set_module_tensor_to_device(model, _resolve(p), device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for p in layer_paths[il]:
                set_module_tensor_to_device(model, _resolve(p), 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(tf.layers):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *args, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kwargs):
            call = lambda lyr, h: _fwd(lyr, h, *args, **kwargs)[0]
            return (apply_streamed_multi(_layer, call, hidden_states, bind=_b, evict=_e),)
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    model.eval()
    model.config.use_cache = False
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    tf.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    print(f'[HY-BWD] wrapped {n_layers} HunyuanImage3DecoderLayers in StreamedLayerMulti — attach LoRA, forward (use_cache=False), then streamer.new_pass(reverse=True) before loss.backward()', flush=True)
    return (model, streamer)
if __name__ == '__main__':
    import sys
    md = sys.argv[1] if len(sys.argv) > 1 else ''
    from .safetensors_index import build_index
    model, cfg = _load_meta_hunyuan(md)
    print(f'meta model: {type(model).__name__}, {len(model.model.layers)} layers, hidden={cfg.hidden_size}, experts={cfg.num_experts}')
    idx = build_index(md)
    layers, globs = idx.group_by_layer('')
    print(f"index: {len(layers)} layer-groups (of {len(set(idx.refs)) and 'local shards'}), {len(globs)} globals")
    nexp = sum((1 for r in layers.get(0, []) if 'experts.' in r.name))
    print(f'layer 0: {len(layers.get(0, []))} tensors ({nexp} expert tensors) — the MoE bulk')
    print('RESULT', 'OK' if len(layers) == cfg.num_hidden_layers else 'LAYER_COUNT_MISMATCH')
