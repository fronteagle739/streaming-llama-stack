# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import hashlib
import json
import math
import re
import struct
from contextlib import ExitStack
from pathlib import Path
import torch

def _stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

def variant_config(arch, root):
    root = Path(root)

    def read(name):
        path = root / name
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError('Krea source is missing a bounded ' + name)
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('Krea source configuration must be an object')
        return value
    pipeline = read('model_index.json')
    config = read('transformer/config.json')
    if pipeline.get('_class_name') != 'Krea2Pipeline' or config.get('_class_name') != 'Krea2Transformer2DModel':
        raise ValueError('Krea native streaming requires the original Krea2 transformer/pipeline layout')
    if type(pipeline.get('is_distilled')) is not bool:
        raise ValueError('Krea source must state its actual distillation mode')
    if arch == 'krea2-turbo' and (not pipeline['is_distilled']):
        raise ValueError('Krea Turbo requires a distilled checkpoint')
    if arch == 'krea2-raw' and pipeline['is_distilled']:
        raise ValueError('Krea Raw requires an undistilled checkpoint')
    for key, maximum in [('num_layers', 128), ('num_attention_heads', 128), ('attention_head_dim', 256), ('in_channels', 256), ('intermediate_size', 65536), ('text_hidden_dim', 8192)]:
        if type(config.get(key)) is not int or not 1 <= config[key] <= maximum:
            raise ValueError('Krea transformer configuration exceeds the supported ' + key + ' bound')
    if config['num_attention_heads'] * config['attention_head_dim'] > 8192:
        raise ValueError('Krea transformer hidden width exceeds its native bound')
    if pipeline.get('transformer') != ['diffusers', 'Krea2Transformer2DModel'] or pipeline.get('text_encoder') != ['transformers', 'Qwen3VLModel']:
        raise ValueError('Krea source components differ from the supported original layout')
    return (pipeline, config)

def preflight_single_file(path):
    from .boogu_krea import lora_transformer_class
    from . import krea2_diffusion as native
    with torch.device('meta'):
        model = lora_transformer_class('krea2')()
    streamer = native._Krea2TransformerStreamer(model, str(path), dtype=torch.bfloat16, device='cpu')
    try:
        expected = model.state_dict()
        for target, (key, reshape) in streamer.entries.items():
            shape = streamer._sf.get_slice(key).get_shape()
            descriptor = torch.empty(tuple(shape), device='meta')
            descriptor = native._reshape_mapped_value(descriptor, reshape)
            if tuple(descriptor.shape) != tuple(expected[target].shape):
                raise ValueError('Krea converted checkpoint shape mismatch: ' + target)
        return {'format': 'converted-Krea', 'tensors_checked': len(streamer.entries), 'source': str(path)}
    finally:
        streamer.close()

class KreaHFStreamer:

    def __init__(self, model, transformer_dir, *, dtype, device):
        self.model = model
        self.root = Path(transformer_dir).resolve()
        self.dtype = dtype
        self.device = str(device)
        self.should_cancel = None
        self.closed = False
        self._wrappers = []
        self._active = set()
        self._active_bytes = 0
        self.blocks_bound = 0
        self.peak_active_blocks = 0
        self.peak_block_bytes = 0
        self.peak_rss_bytes = 0
        self.peak_cuda_bytes = 0
        self.events = []
        self.event_count = 0
        self.stamps = {}
        self.headers = {}
        self.globals = []
        self.groups = {}
        index = self.root / 'diffusion_pytorch_model.safetensors.index.json'
        if not index.is_file() or index.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Krea native streaming requires a bounded sharded safetensors index')
        document = json.loads(index.read_text(encoding='utf-8'))
        self.weight_map = document.get('weight_map')
        if not isinstance(self.weight_map, dict) or not 1 <= len(self.weight_map) <= 50000:
            raise ValueError('Krea native streaming index has no valid weight map')
        expected = model.state_dict()
        self._expected_dtypes = {key: value.dtype for key, value in expected.items()}
        self._floating = {key: value.is_floating_point() for key, value in expected.items()}
        if set(expected) != set(self.weight_map):
            raise ValueError('Krea native streaming checkpoint keys do not match the configured transformer')
        if any((value.device.type != 'meta' for value in expected.values())):
            raise ValueError('Krea native streaming requires an empty meta transformer')
        filenames = set(self.weight_map.values())
        if len(filenames) > 128:
            raise ValueError('Krea native streaming exceeds the shard limit')
        header_bytes = 0
        for filename in filenames:
            if not isinstance(filename, str) or Path(filename).name != filename or (not filename.endswith('.safetensors')):
                raise ValueError('Krea checkpoint shard must be a local safetensors filename')
            path = self.root / filename
            if path.is_symlink() or path.resolve().parent != self.root:
                raise ValueError('Krea checkpoint shard escapes its selected source')
            self.stamps[filename] = _stamp(path)
            with path.open('rb') as source:
                length_bytes = source.read(8)
                if len(length_bytes) != 8:
                    raise ValueError('Krea safetensors header is truncated')
                length = struct.unpack('<Q', length_bytes)[0]
                if not 2 <= length <= 8 * 1024 * 1024:
                    raise ValueError('Krea safetensors header exceeds its limit')
                header_bytes += length
                if header_bytes > 16 * 1024 * 1024:
                    raise ValueError('Krea checkpoint exceeds the aggregate header budget')
                raw = source.read(length)
                if len(raw) != length:
                    raise ValueError('Krea safetensors header is truncated')
            header = json.loads(raw)
            if not isinstance(header, dict):
                raise ValueError('Krea safetensors header is malformed')
            self.headers[filename] = {'sha256': hashlib.sha256(raw).hexdigest(), 'tensors': header}
        for key, filename in self.weight_map.items():
            tensor = self.headers[filename]['tensors'].get(key)
            if not isinstance(tensor, dict) or tensor.get('shape') != list(expected[key].shape):
                raise ValueError(f'Krea checkpoint tensor shape mismatch: {key}')
            if tensor.get('dtype') not in {'F16', 'BF16', 'F32', 'I64', 'I32', 'BOOL'}:
                raise ValueError(f'Krea checkpoint encoding is unsupported for {key}')
            if expected[key].is_floating_point() and tensor.get('dtype') not in {'F16', 'BF16', 'F32'}:
                raise ValueError(f'Krea checkpoint floating tensor encoding mismatch: {key}')
            if not expected[key].is_floating_point() and tensor.get('dtype') != {torch.int64: 'I64', torch.int32: 'I32', torch.bool: 'BOOL'}.get(expected[key].dtype):
                raise ValueError(f'Krea checkpoint integer tensor encoding mismatch: {key}')
            offsets = tensor.get('data_offsets')
            if not isinstance(offsets, list) or len(offsets) != 2 or any((type(value) is not int for value in offsets)):
                raise ValueError(f'Krea checkpoint offsets are malformed: {key}')
            if not 0 <= offsets[0] <= offsets[1] <= self.stamps[filename][2] - 8:
                raise ValueError(f'Krea checkpoint offsets exceed their shard: {key}')
            match = re.match('^transformer_blocks\\.(\\d+)\\.', key)
            if match:
                self.groups.setdefault(int(match.group(1)), []).append(key)
            else:
                self.globals.append(key)
        if set(self.groups) != set(range(len(model.transformer_blocks))):
            raise ValueError('Krea checkpoint does not contain every configured transformer block')
        from safetensors import safe_open
        for filename in sorted(filenames):
            with safe_open(str(self.root / filename), framework='pt', device='cpu') as handle:
                selected = {key for key, shard in self.weight_map.items() if shard == filename}
                if set(handle.keys()) != selected:
                    raise ValueError('Krea shard/index tensor inventory differs')

    def _check(self):
        if self.closed:
            raise ValueError('Krea native streamer is closed')
        if self.should_cancel and self.should_cancel():
            raise RuntimeError('generation cancelled')
        for filename, expected in self.stamps.items():
            if _stamp(self.root / filename) != expected:
                raise ValueError('Krea checkpoint source changed during generation')

    def _bind(self, keys):
        self._check()
        from accelerate.utils import set_module_tensor_to_device
        from .boogu_krea import _base_tensor_key
        from safetensors import safe_open
        with ExitStack() as contexts:
            handles = {filename: contexts.enter_context(safe_open(str(self.root / filename), framework='pt', device='cpu')) for filename in {self.weight_map[key] for key in keys}}
            for key in keys:
                if self.should_cancel and self.should_cancel():
                    raise RuntimeError('generation cancelled')
                value = handles[self.weight_map[key]].get_tensor(key)
                target_dtype = self._target_dtype(key)
                value = value.to(device=self.device, dtype=target_dtype)
                set_module_tensor_to_device(self.model, _base_tensor_key(self.model, key), self.device, value=value, dtype=target_dtype)
                self.measure()

    def _target_dtype(self, key):
        if not self._floating[key]:
            return self._expected_dtypes[key]
        keep = getattr(self.model, '_keep_in_fp32_modules', []) or []
        return torch.float32 if any((pattern in key for pattern in keep)) else self.dtype

    def measure(self):
        import psutil
        self.peak_rss_bytes = max(self.peak_rss_bytes, psutil.Process().memory_info().rss)
        if torch.device(self.device).type == 'cuda':
            self.peak_cuda_bytes = max(self.peak_cuda_bytes, torch.cuda.memory_allocated(self.device))

    def _evict(self, keys):
        from .boogu_krea import _base_tensor_key, _set_module_tensor_to_meta
        for key in keys:
            if not _set_module_tensor_to_meta(self.model, _base_tensor_key(self.model, key)):
                raise RuntimeError('Krea could not evict its owned tensor ' + key)

    def materialize_globals(self):
        self._bind(self.globals)

    def evict_globals(self):
        self._evict(self.globals)

    def _event(self, kind, block):
        self.event_count += 1
        if len(self.events) < 4096:
            self.events.append({'event': kind, 'block': block, 'active_blocks': len(self._active)})

    def register(self):
        if self._wrappers:
            raise ValueError('Krea block wrappers are already registered')
        for index, block in enumerate(self.model.transformer_blocks):
            original = block.forward
            keys = self.groups[index]

            def forward(*args, _original=original, _keys=keys, _index=index, **kwargs):
                self._check()
                if self._active:
                    raise RuntimeError('Krea native streamer cannot run overlapping transformer blocks')
                self._active.add(_index)
                try:
                    self._bind(_keys)
                    self.blocks_bound += 1
                    amount = sum((math.prod(self.headers[self.weight_map[key]]['tensors'][key]['shape']) * torch.empty((), dtype=self._target_dtype(key)).element_size() for key in _keys))
                    self.peak_block_bytes = max(self.peak_block_bytes, amount)
                    self.peak_active_blocks = max(self.peak_active_blocks, len(self._active))
                    self.measure()
                    self._event('bind', _index)
                    return _original(*args, **kwargs)
                finally:
                    self._evict(_keys)
                    self._active.discard(_index)
                    self._event('evict', _index)
            block.forward = forward
            self._wrappers.append((block, original, forward))
        self.model._krea_hf_streamer = self

    def receipt(self):
        return {'implementation': 'original-HF-Krea-block-bind-evict', 'source': str(self.root), 'blocks_bound': self.blocks_bound, 'peak_active_blocks': self.peak_active_blocks, 'peak_block_bytes': self.peak_block_bytes, 'peak_rss_bytes_process': self.peak_rss_bytes, 'peak_cuda_allocated_bytes_process': self.peak_cuda_bytes, 'closed': self.closed, 'events': list(self.events), 'event_count': self.event_count, 'events_truncated': self.event_count > len(self.events), 'header_sha256': {name: header['sha256'] for name, header in self.headers.items()}}

    def close(self):
        if self.closed:
            return
        errors = []
        for block, original, wrapper in self._wrappers:
            if block.forward is wrapper:
                block.forward = original
        self._wrappers.clear()
        for keys in [*self.groups.values(), self.globals]:
            try:
                self._evict(keys)
            except Exception as error:
                errors.append(str(error))
        self._active.clear()
        if errors:
            raise RuntimeError('Krea native streamer cleanup failed: ' + '; '.join(errors))
        self.closed = True
        if getattr(self.model, '_krea_hf_streamer', None) is self:
            delattr(self.model, '_krea_hf_streamer')
