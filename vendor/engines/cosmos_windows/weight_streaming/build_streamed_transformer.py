# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/build_streamed_transformer.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from pathlib import Path
from .safetensors_index import build_index
from .loader import StreamingWeightLoader

def _load_meta_transformer(model_dir: str):
    from accelerate import init_empty_weights
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    tdir = str(Path(model_dir) / 'transformer')
    cfg = Cosmos3OmniTransformer.load_config(tdir)
    with init_empty_weights():
        tf = Cosmos3OmniTransformer.from_config(cfg)
    return tf

def keyset_report(model_dir: str):
    tf = _load_meta_transformer(model_dir)
    sd_keys = set(tf.state_dict().keys())
    idx = build_index(model_dir)
    layer_refs, global_refs = idx.group_by_layer('transformer')
    idx_paths = set()
    for refs in layer_refs.values():
        for r in refs:
            idx_paths.add(StreamingWeightLoader._param_path(r))
    for r in global_refs:
        idx_paths.add(StreamingWeightLoader._param_path(r))
    missing_in_tree = idx_paths - sd_keys
    extra_in_tree = sd_keys - idx_paths
    return {'n_layers': len(layer_refs), 'n_idx_paths': len(idx_paths), 'n_state_dict': len(sd_keys), 'missing_in_tree': sorted(missing_in_tree), 'extra_in_tree': sorted(extra_in_tree)}

def build_streamed_transformer(model_dir: str, window: int=2, device: str='cuda'):
    tf = _load_meta_transformer(model_dir)
    idx = build_index(model_dir)
    loader = StreamingWeightLoader(tf, idx, component='transformer', window=window, device=device).register()
    loader.materialize_globals()
    return (tf, loader)

def build_ra_streamed_transformer(model_dir: str, device: str='cuda', keep_resident=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    tf = _load_meta_transformer(model_dir)
    idx = build_index(model_dir)
    sd_keys = set(tf.state_dict().keys())
    layer_refs, global_refs = idx.group_by_layer('transformer')
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path in sd_keys:
            set_module_tensor_to_device(tf, path, device, value=pb.read_tensor(ref, device))

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = tf.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    keep = set(keep_resident or ())
    done = set()

    def make_pre(il, refs):

        def hook(mod, args):
            if il in keep and il in done:
                return
            for ref in refs:
                p = StreamingWeightLoader._param_path(ref)
                if p in sd_keys:
                    set_module_tensor_to_device(tf, _resolve(p), device, value=pb.read_tensor(ref, device))
            if il in keep:
                done.add(il)
        return hook

    def make_post(il, refs):

        def hook(mod, args, output):
            if il in keep:
                return output
            for ref in refs:
                p = StreamingWeightLoader._param_path(ref)
                if p in sd_keys:
                    set_module_tensor_to_device(tf, _resolve(p), 'meta')
            return output
        return hook
    for il, layer in enumerate(tf.layers):
        refs = layer_refs.get(il, [])
        layer.register_forward_pre_hook(make_pre(il, refs))
        layer.register_forward_hook(make_post(il, refs))
    print(f'[RA] random-access (mmap) streamed transformer: {len(tf.layers)} layers, {len(keep)} kept resident — backward-capable', flush=True)
    return (tf, None)

def build_ds_streamed_transformer(model_dir: str, device: str='cuda', n_gpu_buffers: int=3, keep_resident=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    tf = _load_meta_transformer(model_dir)
    idx = build_index(model_dir)
    sd_keys = set(tf.state_dict().keys())
    _layer_refs, global_refs = idx.group_by_layer('transformer')
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path in sd_keys:
            set_module_tensor_to_device(tf, path, device, value=pb.read_tensor(ref, device))
    streamer = DirectStorageStreamer(idx, component='transformer', n_gpu_buffers=n_gpu_buffers, device=device)
    print(f"[DS] mode={streamer.mode} (3=DSTORAGE), {streamer.n} layers, {len(streamer.tbufs)} GPU bufs (prefetch {('ON' if n_gpu_buffers > 1 else 'OFF/single-buffer')})")

    def _pp(name):
        return name.split('/', 1)[1] if '/' in name else name

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = tf.get_submodule(parent)
        except AttributeError:
            return path
        if hasattr(m, 'base_layer'):
            return f'{parent}.base_layer.{leaf}'
        return path
    layer_paths = {il: [_pp(r.name) for r, _ in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in range(streamer.n)}
    keep = set(keep_resident or ())
    resident_done = set()

    def make_pre(il):

        def hook(mod, args):
            if il in keep and il in resident_done:
                return
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    set_module_tensor_to_device(tf, _resolve(p), device, value=t.clone() if il in keep else t)
            if il in keep:
                resident_done.add(il)
        return hook

    def make_post(il):

        def hook(mod, args, output):
            if il in keep:
                streamer.release_layer(il)
                return output
            for p in layer_paths[il]:
                set_module_tensor_to_device(tf, _resolve(p), 'meta')
            streamer.release_layer(il)
            return output
        return hook
    tf.register_forward_pre_hook(lambda m, a: streamer.new_pass())
    for il, layer in enumerate(tf.layers):
        layer.register_forward_pre_hook(make_pre(il))
        layer.register_forward_hook(make_post(il))
    return (tf, streamer)

def materialize_meta_lora(tf, device: str='cuda', dtype=None, std: float=0.02):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    dt = dtype or torch.float32
    n = 0
    for name, p in list(tf.named_parameters()):
        if p.device.type == 'meta' and 'lora_' in name:
            if 'lora_B' in name:
                val = torch.zeros(tuple(p.shape), device=device, dtype=dt)
            else:
                val = torch.randn(tuple(p.shape), device=device, dtype=dt) * std
            set_module_tensor_to_device(tf, name, device, value=val)
            tf.get_parameter(name).requires_grad_(True)
            n += 1
    return n

def build_streamed_backward_transformer(model_dir: str, device: str='cuda', n_gpu_buffers: int=1):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from ..streamed_backward import apply_streamed_multi
    tf = _load_meta_transformer(model_dir)
    idx = build_index(model_dir)
    sd_keys = set(tf.state_dict().keys())
    _layer_refs, global_refs = idx.group_by_layer('transformer')
    pb = PortableBackend(pin_memory=True)
    for ref in global_refs:
        path = StreamingWeightLoader._param_path(ref)
        if path in sd_keys:
            set_module_tensor_to_device(tf, path, device, value=pb.read_tensor(ref, device))
    streamer = DirectStorageStreamer(idx, component='transformer', n_gpu_buffers=n_gpu_buffers, device=device)
    print(f"[DS-BWD] mode={streamer.mode} {streamer.n} layers, {len(streamer.tbufs)} GPU buf(s) (reverse-capable; prefetch {('ON' if n_gpu_buffers > 1 else 'OFF/single-buffer=validated reverse')})", flush=True)

    def _pp(name):
        return name.split('/', 1)[1] if '/' in name else name

    def _resolve(path):
        parent, _, leaf = path.rpartition('.')
        try:
            m = tf.get_submodule(parent)
        except AttributeError:
            return path
        return f'{parent}.base_layer.{leaf}' if hasattr(m, 'base_layer') else path
    planned = set(streamer.plan.keys())
    missing = [il for il in range(streamer.n) if il not in planned]
    if missing:
        print(f"[DS-BWD] WARNING: {len(missing)} layer(s) missing from the streamer plan {missing[:8]}{('...' if len(missing) > 8 else '')} — the model download is INCOMPLETE; those layers cannot be streamed and training will fail at their forward. Re-download the transformer shards.", flush=True)
    layer_paths = {il: [_pp(r.name) for r, _ in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in planned}

    def make_bind(il):

        def bind():
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    set_module_tensor_to_device(tf, _resolve(p), device, value=t)
        return bind

    def make_evict(il):

        def evict():
            for p in layer_paths[il]:
                set_module_tensor_to_device(tf, _resolve(p), 'meta')
            streamer.release_layer(il)
        return evict
    for il, layer in enumerate(tf.layers):
        if il not in planned:
            continue
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(und_seq, gen_seq, rotary_emb, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict):
            call = lambda lyr, u, g: _fwd(lyr, u, g, rotary_emb)
            return apply_streamed_multi(_layer, call, und_seq, gen_seq, bind=_b, evict=_e)
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    tf.register_forward_pre_hook(lambda m, a: streamer.new_pass())
    print(f'[DS-BWD] wrapped {len(tf.layers)} blocks in StreamedLayerMulti — call streamer.new_pass(reverse=True) after forward, before loss.backward()', flush=True)
    return (tf, streamer)
if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else ''
    rep = keyset_report(model)
    print(f'model: {model}')
    print(f"transformer layers: {rep['n_layers']}")
    print(f"index param paths: {rep['n_idx_paths']}  | meta state_dict keys: {rep['n_state_dict']}")
    print(f"MISSING in module tree (FATAL if >0): {len(rep['missing_in_tree'])}")
    for k in rep['missing_in_tree'][:20]:
        print(f'  MISSING: {k}')
    print(f"extra in module tree (buffers/tied, usually OK): {len(rep['extra_in_tree'])}")
    for k in rep['extra_in_tree'][:20]:
        print(f'  extra: {k}')
    ok = len(rep['missing_in_tree']) == 0
    print(f"KEYSET ASSERT: {('PASS (every checkpoint weight maps to a module param)' if ok else 'FAIL')}")
    sys.exit(0 if ok else 1)
