# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from pathlib import Path
import hashlib
import json
import os
import re
_BLOCK = re.compile('^transformer_blocks\\.(\\d+)\\.')
_SCALE = '.studio_int8_scale'

def block_groups(index, *, block_prefix='transformer_blocks'):
    pattern = re.compile('^' + re.escape(block_prefix) + '\\.(\\d+)\\.')
    groups = {}
    for name, ref in index.refs.items():
        match = pattern.match(name)
        if match:
            groups.setdefault(int(match[1]), []).append(ref)
    if not groups or sorted(groups) != list(range(len(groups))):
        raise RuntimeError('Qwen transformer blocks are missing or non-contiguous')
    return groups

def _validate_blocks(module, index, *, block_prefix='transformer_blocks'):
    groups = block_groups(index, block_prefix=block_prefix)
    if len(groups) != len(getattr(module, block_prefix)):
        raise RuntimeError('Qwen block count does not match the checkpoint')
    expected = {name for name in module.state_dict() if name.startswith(block_prefix + '.')}
    found = {name for name in index.refs if name.startswith(block_prefix + '.') and (not name.endswith(_SCALE))}
    if expected != found:
        raise RuntimeError(f'Qwen block coverage mismatch: missing={sorted(expected - found)}, extra={sorted(found - expected)}')
    return groups

def _streamer(index, groups, device):
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    stream = DirectStorageStreamer(index, layer_groups=groups, n_gpu_buffers=1, device=device, record_compute_done=True)
    if stream.mode != 3:
        stream.close()
        raise RuntimeError('Qwen requires the DirectStorage transport (mode 3)')
    return stream

def quantize_block(values):
    import torch
    packed = {}
    for name, tensor in values.items():
        if tensor.ndim == 2 and name.endswith('.weight'):
            scale = tensor.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-08) / 127
            packed[name] = (tensor.float() / scale).round().clamp(-127, 127).to(torch.int8)
            packed[name + _SCALE] = scale
        else:
            packed[name] = tensor.clone()
    return packed

def _int8_pack(source, index, device, *, block_prefix='transformer_blocks', pack_dir=None):
    from safetensors.torch import save_file
    import torch
    files = sorted({str(ref.file) for ref in index.refs.values()})
    identity = [(f, Path(f).stat().st_size, Path(f).stat().st_mtime_ns) for f in files]
    fingerprint = hashlib.sha256(json.dumps(['qwen-row-int8-v1', identity]).encode()).hexdigest()[:20]
    root = Path(pack_dir or os.environ.get('STUDIO_QWEN_IMAGE21_PACK_DIR', '')) / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    groups = block_groups(index, block_prefix=block_prefix)
    complete = root / 'complete.json'
    if complete.is_file() and all(((root / f'block-{i:03}.safetensors').is_file() for i in groups)):
        return root
    stream = _streamer(index, groups, device)
    try:
        stream.new_pass()
        for i in sorted(groups):
            values = stream.wait_layer(i)
            packed = quantize_block(values)
            cpu = {name: tensor.cpu().contiguous() for name, tensor in packed.items()}
            dest = root / f'block-{i:03}.safetensors'
            staging = dest.with_suffix('.partial')
            save_file(cpu, str(staging))
            staging.replace(dest)
            del cpu, packed, values
            torch.cuda.synchronize(device)
            stream.release_layer(i)
            print(f'[status] Preparing INT8 block pack: {i + 1}/{len(groups)}', flush=True)
    finally:
        stream.close()
        torch.cuda.empty_cache()
    complete.write_text(json.dumps({'fingerprint': fingerprint, 'blocks': len(groups), 'source': str(source)}))
    return root

class QwenBlockWeights:

    def __init__(self, module, source, *, streaming, precision, device='cuda:0', defer_resident=False, block_prefix='transformer_blocks', pack_dir=None):
        import torch
        from cosmos_windows.weight_streaming.safetensors_index import build_index
        index = build_index(source)
        self.block_prefix, self.pack_dir = (block_prefix, pack_dir)
        if precision == 'int8' and (not (defer_resident and (not streaming))):
            index = build_index(_int8_pack(source, index, device, block_prefix=block_prefix, pack_dir=pack_dir))
        groups = _validate_blocks(module, index, block_prefix=block_prefix)
        self.module, self.device = (module, device)
        self._source, self._index, self._groups = (source, index, groups)
        self._streaming, self._precision = (streaming, precision)
        self._deferred_pack = precision == 'int8' and defer_resident and (not streaming)
        self.stream = None if defer_resident and (not streaming) else _streamer(index, groups, device)
        self.resident = None
        self.binds = self.evictions = 0
        if not streaming and (not defer_resident):
            self.materialize_resident()
        self._next_layer = 0
        self.handles = [module.register_forward_pre_hook(lambda m, a: self.new_pass())]
        for i, block in enumerate(getattr(module, block_prefix)):
            names = [ref.name for ref in groups[i] if not ref.name.endswith(_SCALE)]
            self.handles.append(block.register_forward_pre_hook(self._pre(i)))
            self.handles.append(block.register_forward_hook(self._post(i, names), always_call=True))
        print(f"[status] Qwen {precision.upper()} blocks: {('DirectStorage streaming' if streaming else 'quantized resident')}; only the active block materialized", flush=True)

    def materialize_resident(self):
        if self._streaming or self.resident is not None:
            return
        import torch
        if self._deferred_pack:
            from cosmos_windows.weight_streaming.safetensors_index import build_index
            packed_index = build_index(_int8_pack(self._source, self._index, self.device, block_prefix=self.block_prefix, pack_dir=self.pack_dir))
            packed_groups = _validate_blocks(self.module, packed_index, block_prefix=self.block_prefix)
            self._index, self._groups = (packed_index, packed_groups)
            self._deferred_pack = False
        if self.stream is None:
            self.stream = _streamer(self._index, self._groups, self.device)
        resident = {}
        try:
            self.stream.new_pass()
            for i in sorted(self._groups):
                resident[i] = {name: value.clone() for name, value in self.stream.wait_layer(i).items()}
                torch.cuda.synchronize(self.device)
                self.stream.release_layer(i)
        finally:
            self.stream.close()
            self.stream = None
        self.resident = resident

    def evict_resident(self):
        if self._streaming:
            return
        import torch
        for refs in self._groups.values():
            for ref in refs:
                if ref.name.endswith(_SCALE):
                    continue
                parent_name, _, leaf = ref.name.rpartition('.')
                parent = self.module.get_submodule(parent_name)
                old = parent._parameters[leaf]
                parent._parameters[leaf] = torch.nn.Parameter(torch.empty_like(old, device='meta'), requires_grad=False)
        self.resident = None

    def new_pass(self):
        self._next_layer = 0
        if not self._streaming and self.resident is None:
            raise RuntimeError('Qwen resident blocks have not been materialized after prompt encoding')
        if self.stream is not None:
            self.stream.new_pass()

    def _pre(self, i):

        def bind(mod, args):
            import torch
            if self.stream is not None:
                for skipped in range(self._next_layer, i):
                    self.stream.wait_layer(skipped)
                    self.stream.release_layer(skipped)
            values = self.resident[i] if self.resident is not None else self.stream.wait_layer(i)
            self._next_layer = i + 1
            for name, value in values.items():
                if name.endswith(_SCALE):
                    continue
                parent_name, _, leaf = name.rpartition('.')
                parent = self.module.get_submodule(parent_name)
                if value.dtype == torch.int8:
                    value = (value.to(torch.float32) * values[name + _SCALE]).to(torch.bfloat16)
                parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            self.binds += 1
        return bind

    def _post(self, i, names):

        def evict(mod, args, output):
            import torch
            for name in names:
                parent_name, _, leaf = name.rpartition('.')
                parent = self.module.get_submodule(parent_name)
                old = parent._parameters[leaf]
                parent._parameters[leaf] = torch.nn.Parameter(torch.empty_like(old, device='meta'), requires_grad=False)
            if self.stream is not None:
                torch.cuda.synchronize(self.device)
                self.stream.release_layer(i)
            self.evictions += 1
            return output
        return evict

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.resident = None
