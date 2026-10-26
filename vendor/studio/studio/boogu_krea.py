# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import re
import shutil
import sys
import importlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Optional
import torch
_VENDORED = Path(__file__).parent / 'vendored'
ARCH_DEFAULTS = {'krea2': {'steps': 28, 'guidance': 4.5, 'loader': 'krea2'}, 'krea2-turbo': {'steps': 8, 'guidance': 0.0, 'loader': 'krea2'}, 'krea2-raw': {'steps': 28, 'guidance': 4.5, 'loader': 'krea2'}, 'boogu': {'steps': 50, 'guidance': 4.0, 'loader': 'boogu'}, 'boogu-turbo': {'steps': 4, 'guidance': 1.0, 'loader': 'boogu'}, 'boogu-edit': {'steps': 50, 'guidance': 4.0, 'loader': 'boogu'}}

def _loader_for(arch: str) -> str:
    a = (arch or '').lower()
    return 'boogu' if a.startswith('boogu') else 'krea2'

def _ensure_boogu_package():
    if importlib.util.find_spec('boogu') is None:
        if not (_VENDORED / 'boogu/models/transformers/transformer_boogu.py').is_file():
            raise ValueError('Boogu transformer software is not installed')
        if str(_VENDORED) not in sys.path:
            sys.path.append(str(_VENDORED))

def lora_transformer_class(arch):
    if _loader_for(arch) == 'boogu':
        _ensure_boogu_package()
        from boogu.models.transformers import BooguImageTransformer2DModel
        return BooguImageTransformer2DModel
    try:
        module = importlib.import_module('diffusers.models.transformers.transformer_krea2')
    except ModuleNotFoundError as error:
        if error.name != 'diffusers.models.transformers.transformer_krea2':
            raise
        name = 'diffusers.models.transformers._studio_krea2_preflight'
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, _VENDORED / 'krea2/transformer_krea2.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return module.Krea2Transformer2DModel

def _prepare_kernel_runtime() -> None:
    try:
        from .config import configure_model_storage_env, settings
        configure_model_storage_env(settings.models_root)
    except Exception:
        pass
    if not sys.platform.startswith('win'):
        return
    try:
        import transformers.integrations.finegrained_fp8 as _fg
    except Exception:
        return
    if getattr(_fg, '_STUDIO_WINDOWS_DEEPGEMM_DISABLED', False):
        return

    def _no_windows_deepgemm():
        raise ImportError('DeepGEMM Hub kernels do not publish a Windows build; using finegrained-fp8 fallback.')
    try:
        _fg._load_deepgemm_kernel.cache_clear()
    except Exception:
        pass
    _fg._load_deepgemm_kernel = _no_windows_deepgemm
    _fg._STUDIO_WINDOWS_DEEPGEMM_DISABLED = True

def _ensure_krea2_installed() -> None:
    import diffusers
    diff = Path(diffusers.__file__).parent
    tdst = diff / 'models' / 'transformers' / 'transformer_krea2.py'
    if not tdst.exists():
        shutil.copy(_VENDORED / 'krea2' / 'transformer_krea2.py', tdst)
    pdst = diff / 'pipelines' / 'krea2'
    if not pdst.exists():
        pdst.mkdir(parents=True, exist_ok=True)
        for f in (_VENDORED / 'krea2' / 'pipeline').glob('*.py'):
            shutil.copy(f, pdst / f.name)
    tu = diff / 'utils' / 'torch_utils.py'
    src = tu.read_text(encoding='utf-8')
    if 'KREA2_DTYPE_SHIM' not in src:
        tu.write_text(src + '\n\n# KREA2_DTYPE_SHIM (override the broken/slow upstream maybe_adjust_dtype_for_device)\ndef maybe_adjust_dtype_for_device(dtype, device):\n    import torch as _t\n    return _t.float32 if dtype == _t.float64 else dtype\n', encoding='utf-8')
    import diffusers.utils.torch_utils as _tu_mod
    _tu_mod.maybe_adjust_dtype_for_device = lambda dtype, device: torch.float32 if dtype == torch.float64 else dtype
KREA2_COMPONENTS = ('text_encoder', 'transformer', 'vae')

def _krea2_build_empty(name: str, folder: Path):
    from .direct_vram import empty
    if name == 'text_encoder':
        from transformers import AutoConfig, Qwen3VLModel
        config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
        return empty(lambda: Qwen3VLModel._from_config(config))
    if name == 'transformer':
        import diffusers
        cls = diffusers.Krea2Transformer2DModel
    elif name == 'vae':
        from diffusers import AutoencoderKLQwenImage as cls
    else:
        raise ValueError(f'unknown Krea 2 component {name!r}')
    config = cls.load_config(str(folder))
    return empty(lambda: cls.from_config(config))

def _krea2_load_component(ckpt, name: str, dtype, device):
    from .direct_vram import load_into
    folder = Path(ckpt) / name
    module = _krea2_build_empty(name, folder)
    return load_into(module, folder, device, dtype, keep_fp32=getattr(module, '_keep_in_fp32_modules', None) or ())

def _direct_from_pretrained(cls, folder, dtype, device):
    from .direct_vram import empty, load_into
    folder = Path(folder)
    if hasattr(cls, 'load_config'):
        config = cls.load_config(str(folder))
        config = {k: v for k, v in config.items() if not k.startswith('_')} if cls.__module__.startswith('boogu') else config
        module = empty(lambda: cls(**config) if cls.__module__.startswith('boogu') else cls.from_config(config))
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
        module = empty(lambda: cls._from_config(config))
    return load_into(module, folder, device, dtype, keep_fp32=getattr(module, '_keep_in_fp32_modules', None) or ())

@contextmanager
def _qwen3vl_direct(device, dtype):
    from transformers import Qwen3VLForConditionalGeneration
    original = vars(Qwen3VLForConditionalGeneration).get('from_pretrained')
    hits = []

    def load(path, *args, subfolder=None, torch_dtype=None, **kwargs):
        folder = Path(path) / subfolder if subfolder else Path(path)
        hits.append(str(folder))
        return _direct_from_pretrained(Qwen3VLForConditionalGeneration, folder, torch_dtype or dtype, device)
    Qwen3VLForConditionalGeneration.from_pretrained = load
    try:
        yield hits
    finally:
        if original is None:
            del Qwen3VLForConditionalGeneration.from_pretrained
        else:
            Qwen3VLForConditionalGeneration.from_pretrained = original

def _krea2_ensure(pipe, name: str, device):
    if getattr(pipe, name, None) is None:
        setattr(pipe, name, _krea2_load_component(pipe._studio_krea_ckpt, name, pipe._studio_krea_dtype, device))
    return getattr(pipe, name)

def _krea2_release(pipe, name: str) -> None:
    resident = getattr(pipe, '_studio_krea_resident', None)
    if resident is None:
        resident = () if getattr(pipe, '_studio_staged_disposable', False) else KREA2_COMPONENTS
    if name not in resident:
        setattr(pipe, name, None)
        _cleanup_cuda()

def _load_krea2(ckpt: str, dtype: torch.dtype, *, arch='krea2', device='cpu', stream_transformer=False, resident=KREA2_COMPONENTS):
    from .krea_hf_stream import variant_config
    variant_config(arch, ckpt)
    if stream_transformer:
        from .krea_hf_stream import KreaHFStreamer
        _, config = variant_config(arch, ckpt)
        transformer_class = lora_transformer_class(arch)
        with torch.device('meta'):
            transformer = transformer_class.from_config(config).to(dtype=dtype)
        streamer = KreaHFStreamer(transformer, Path(ckpt) / 'transformer', dtype=dtype, device=device)
        try:
            import diffusers
            import types
            diffusers.models.Krea2Transformer2DModel = transformer_class
            diffusers.Krea2Transformer2DModel = transformer_class
            package = 'diffusers.pipelines.krea2'
            if package not in sys.modules:
                module = types.ModuleType(package)
                module.__path__ = [str(_VENDORED / 'krea2/pipeline')]
                sys.modules[package] = module
            from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
            streamer.register()
            pipe = Krea2Pipeline.from_pretrained(ckpt, torch_dtype=dtype, transformer=transformer, text_encoder=None, local_files_only=True)
            pipe._studio_krea_hf_streamed = True
            pipe._studio_staged_gpu = True
            pipe._studio_staged_device = str(device)
            pipe._studio_krea_text_dir = str(Path(ckpt) / 'text_encoder')
            pipe._studio_krea_text_dtype = dtype
            return pipe
        except BaseException:
            streamer.close()
            raise
    _ensure_krea2_installed()
    import diffusers
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
    diffusers.models.Krea2Transformer2DModel = Krea2Transformer2DModel
    diffusers.Krea2Transformer2DModel = Krea2Transformer2DModel
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
    diffusers.Krea2Pipeline = Krea2Pipeline
    components = {name: _krea2_load_component(ckpt, name, dtype, device) if name in resident else None for name in KREA2_COMPONENTS}
    pipe = Krea2Pipeline.from_pretrained(ckpt, torch_dtype=dtype, local_files_only=True, **components)
    pipe._studio_krea_ckpt = str(ckpt)
    pipe._studio_krea_dtype = dtype
    pipe._studio_krea_resident = tuple(resident)
    return pipe

def _child_module(module, name: str):
    if name.isdigit() and hasattr(module, '__getitem__'):
        try:
            return module[int(name)]
        except Exception:
            pass
    return getattr(module, name)

def _module_parent(root, key: str):
    parts = key.split('.')
    module = root
    for part in parts[:-1]:
        module = _child_module(module, part)
    return (module, parts[-1])

def _base_tensor_key(root, key: str) -> str:
    module, name = _module_parent(root, key)
    if hasattr(module, 'get_base_layer') and name in {'weight', 'bias'}:
        parent, _, _ = key.rpartition('.')
        return f'{parent}.base_layer.{name}'
    return key

def _set_module_tensor_to_meta(root, key: str) -> bool:
    try:
        module, name = _module_parent(root, key)
        param = getattr(module, '_parameters', {}).get(name)
        if param is not None:
            module._parameters[name] = torch.nn.Parameter(torch.empty(tuple(param.shape), device='meta', dtype=param.dtype), requires_grad=param.requires_grad)
            return True
        buf = getattr(module, '_buffers', {}).get(name)
        if buf is not None:
            module._buffers[name] = torch.empty(tuple(buf.shape), device='meta', dtype=buf.dtype)
            return True
    except Exception:
        return False
    return False

def _boogu_group(key: str) -> tuple[str, int] | None:
    m = re.match('^(context_refiner|noise_refiner|ref_image_refiner|double_stream_layers|single_stream_layers)\\.(\\d+)\\.', key)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))

class _BooguTransformerStreamer:

    def __init__(self, model, transformer_dir: Path, *, dtype: torch.dtype, device: str):
        self.model = model
        self.transformer_dir = Path(transformer_dir)
        self.dtype = dtype
        self.device = str(device)
        st_index = self.transformer_dir / 'diffusion_pytorch_model.safetensors.index.json'
        bin_index = self.transformer_dir / 'diffusion_pytorch_model.bin.index.json'
        if st_index.is_file():
            index_path = st_index
            self._format = 'safetensors'
        elif bin_index.is_file():
            index_path = bin_index
            self._format = 'bin'
        else:
            raise RuntimeError(f'Boogu native streaming requires a sharded safetensors or bin index under {self.transformer_dir}')
        index = json.loads(index_path.read_text(encoding='utf-8'))
        self.weight_map: dict[str, str] = dict(index.get('weight_map') or {})
        self._cms: dict[str, Any] = {}
        self._handles: dict[str, Any] = {}
        if self._format == 'safetensors':
            from safetensors import safe_open
            for filename in sorted(set(self.weight_map.values())):
                cm = safe_open(str(self.transformer_dir / filename), framework='pt', device='cpu')
                self._cms[filename] = cm
                self._handles[filename] = cm.__enter__()
        else:
            from torch.serialization import safe_globals
            from zipfile import is_zipfile
            try:
                from torchao.quantization import Float8Tensor
                safe = [Float8Tensor]
            except Exception:
                safe = []
            for filename in sorted(set(self.weight_map.values())):
                path = self.transformer_dir / filename
                kwargs = {'map_location': 'cpu', 'weights_only': True}
                if is_zipfile(str(path)):
                    kwargs['mmap'] = True
                with safe_globals(safe):
                    self._handles[filename] = torch.load(str(path), **kwargs)
        expected = set(model.state_dict().keys())
        indexed = set(self.weight_map.keys())
        missing = sorted(expected - indexed)
        unexpected = sorted(indexed - expected)
        if missing or unexpected:
            raise RuntimeError(f'Boogu streamed transformer key mismatch: missing={missing[:6]} unexpected={unexpected[:6]}')
        self.groups: dict[tuple[str, int], list[str]] = {}
        self.globals: list[str] = []
        for key in sorted(indexed):
            group = _boogu_group(key)
            if group is None:
                self.globals.append(key)
            else:
                self.groups.setdefault(group, []).append(key)

    def close(self) -> None:
        for cm in self._cms.values():
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        self._cms.clear()
        self._handles.clear()

    def _read_target(self, key: str):
        filename = self.weight_map[key]
        if self._format == 'safetensors':
            value = self._handles[filename].get_tensor(key)
        else:
            value = self._handles[filename][key]
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.dtype)
        return value.to(device=self.device)

    def _bind_targets(self, targets: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device
        for key in targets:
            value = self._read_target(key)
            target = _base_tensor_key(self.model, key)
            module, name = _module_parent(self.model, target)
            want = tuple(getattr(module, name).shape)
            if tuple(value.shape) != want:
                raise RuntimeError(f'Boogu tensor shape mismatch for {key}: checkpoint {tuple(value.shape)} != module {want}')
            assign_dtype = self.dtype if value.is_floating_point() else None
            set_module_tensor_to_device(self.model, target, self.device, value=value, dtype=assign_dtype)

    def _evict_targets(self, targets: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device
        for key in targets:
            target = _base_tensor_key(self.model, key)
            if not _set_module_tensor_to_meta(self.model, target):
                set_module_tensor_to_device(self.model, target, 'meta')

    def materialize_globals(self) -> int:
        self._bind_targets(self.globals)
        return len(self.globals)

    def materialize_non_main_blocks(self) -> int:
        n = self.materialize_globals()
        for (name, _i), keys in self.groups.items():
            if name in ('context_refiner', 'noise_refiner', 'ref_image_refiner'):
                self._bind_targets(keys)
                n += len(keys)
        return n

    def register(self) -> None:

        def _wrap(module, targets: list[str]):
            raw_forward = type(module).forward
            streamer = self

            def wrapped(*args, _module=module, _raw_forward=raw_forward, _targets=targets, **kwargs):
                streamer._bind_targets(_targets)
                try:
                    return _raw_forward(_module, *args, **kwargs)
                finally:
                    streamer._evict_targets(_targets)
            module.forward = wrapped
            module._boogu_raw_forward = raw_forward
        for name, modules in (('context_refiner', self.model.context_refiner), ('noise_refiner', self.model.noise_refiner), ('ref_image_refiner', self.model.ref_image_refiner), ('double_stream_layers', self.model.double_stream_layers), ('single_stream_layers', self.model.single_stream_layers)):
            for i, block in enumerate(modules):
                targets = sorted(self.groups.get((name, i), []))
                if targets:
                    _wrap(block, targets)
        self.model._boogu_streamer = self
_BOOGU_DOUBLE_RE = re.compile('(?:^|[/.])double_stream_layers\\.(\\d+)\\.')
_BOOGU_SINGLE_RE = re.compile('(?:^|[/.])single_stream_layers\\.(\\d+)\\.')

def _boogu_group_main_blocks(idx, num_double: int):
    blocks, globs = ({}, [])
    for name, ref in idx.refs.items():
        md = _BOOGU_DOUBLE_RE.search(name)
        ms = _BOOGU_SINGLE_RE.search(name)
        if md:
            blocks.setdefault(int(md.group(1)), []).append(ref)
        elif ms:
            blocks.setdefault(num_double + int(ms.group(1)), []).append(ref)
        else:
            globs.append(ref)
    ordered = {i: blocks[k] for i, k in enumerate(sorted(blocks))}
    return (ordered, globs)

class _BooguMainBlockIndex:

    def __init__(self, idx, num_double: int):
        self._idx = idx
        self._num_double = num_double

    def group_by_layer(self, component=''):
        return _boogu_group_main_blocks(self._idx, self._num_double)

    def __getattr__(self, k):
        return getattr(self._idx, k)

def _attach_ds_boogu_main_blocks(transformer, transformer_dir: Path, dtype, device: str, n_gpu_buffers: int=3):
    import sys as _sys
    import os as _os
    from accelerate.utils import set_module_tensor_to_device
    repo = _os.environ.get('STUDIO_COSMOS_WINDOWS_PATH', '')
    if repo and repo not in _sys.path:
        _sys.path.insert(0, repo)
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    num_double = len(transformer.double_stream_layers)
    n_main = num_double + len(transformer.single_stream_layers)
    idx = build_index(str(transformer_dir))
    sd_keys = set(transformer.state_dict().keys())
    streamer = DirectStorageStreamer(_BooguMainBlockIndex(idx, num_double), component='', n_gpu_buffers=n_gpu_buffers, device=device)
    if getattr(streamer, 'mode', None) != 3 and _os.environ.get('BOOGU_ALLOW_NON_DIRECTSTORAGE') != '1':
        try:
            streamer.close()
        finally:
            raise RuntimeError(f"Boogu DS streamer mode={getattr(streamer, 'mode', '?')} is not DirectStorage(3) â€” R1 requires the optimised NVMe->GPU path. Set BOOGU_ALLOW_NON_DIRECTSTORAGE=1 for diagnostics.")
    if streamer.n != n_main:
        streamer.close()
        raise RuntimeError(f'Boogu DS block count {streamer.n} != {n_main} main blocks (double_stream+single_stream grouping mismatch)')

    def _pp(name):
        return name.split('/', 1)[1] if '/' in name else name
    block_paths = {il: [_pp(r.name) for r, _ao in streamer.plan[il][0] if _pp(r.name) in sd_keys] for il in range(streamer.n)}

    def make_pre(il):

        def hook(_mod, _args):
            for name, t in streamer.wait_layer(il).items():
                p = _pp(name)
                if p in sd_keys:
                    if t.is_floating_point():
                        t = t.to(dtype)
                    set_module_tensor_to_device(transformer, _base_tensor_key(transformer, p), device, value=t, dtype=dtype if t.is_floating_point() else None)
        return hook

    def make_post(il):

        def hook(_mod, _args, output):
            for p in block_paths[il]:
                set_module_tensor_to_device(transformer, _base_tensor_key(transformer, p), 'meta')
            streamer.release_layer(il)
            return output
        return hook
    transformer.register_forward_pre_hook(lambda m, a: streamer.new_pass())
    for i, block in enumerate(transformer.double_stream_layers):
        block.register_forward_pre_hook(make_pre(i))
        block.register_forward_hook(make_post(i))
    for j, block in enumerate(transformer.single_stream_layers):
        il = num_double + j
        block.register_forward_pre_hook(make_pre(il))
        block.register_forward_hook(make_post(il))
    transformer._boogu_ds_streamer = streamer
    return streamer

def _load_boogu_streamed_transformer(ckpt: str, dtype: torch.dtype, device: str):
    from accelerate import init_empty_weights
    from boogu.models.transformers import BooguImageTransformer2DModel
    transformer_dir = Path(ckpt) / 'transformer'
    cfg = json.loads((transformer_dir / 'config.json').read_text(encoding='utf-8'))
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with init_empty_weights():
        transformer = BooguImageTransformer2DModel(**cfg)
    streamer = _BooguTransformerStreamer(transformer, transformer_dir, dtype=dtype, device=device)
    if getattr(streamer, '_format', None) == 'safetensors':
        n_resident = streamer.materialize_non_main_blocks()
        streamer.close()
        ds = _attach_ds_boogu_main_blocks(transformer, transformer_dir, dtype, device)
        transformer._boogu_streamer = ds
        print(f'[boogu] DS-streamed DiT: {n_resident} resident (globals+refiners), {ds.n} main blocks via DirectStorage (mode={ds.mode})', flush=True)
    else:
        n_globals = streamer.materialize_globals()
        streamer.register()
        n_blocks = sum((len(v) for v in streamer.groups.values()))
        print(f'[boogu] streamed transformer (bin legacy): {n_globals} globals resident, {n_blocks} block tensors bind/evict', flush=True)
    return transformer.eval()

def _load_boogu(ckpt: str, dtype: torch.dtype, *, device: str='cpu', stream_transformer: bool=False):
    _ensure_boogu_package()
    if str(_VENDORED) not in sys.path:
        sys.path.insert(0, str(_VENDORED))
    sys.modules.setdefault('transformer_boogu', importlib.import_module('boogu.models.transformers.transformer_boogu'))
    sys.modules.setdefault('scheduling_flow_match_euler_discrete_time_shifting', importlib.import_module('boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting'))
    from boogu.models.transformers import BooguImageTransformer2DModel
    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
    kwargs: dict[str, Any] = {'torch_dtype': dtype, 'trust_remote_code': True}
    transformer_dir = Path(ckpt) / 'transformer'
    if stream_transformer:
        kwargs['transformer'] = _load_boogu_streamed_transformer(ckpt, dtype, device)
        import psutil as _ps
        _p = _ps.Process()

        def _uss():
            try:
                return _p.memory_full_info().uss / 1000000000.0
            except Exception:
                return -1.0
        _u0 = _uss()
        with _qwen3vl_direct(device, dtype) as _loaded:
            pipe = BooguImagePipeline.from_pretrained(ckpt, **kwargs)
        _hits = {'n': len(_loaded)}
        try:
            pipe.vae.to(device)
            _mllm_dev = str(next(pipe.mllm.parameters()).device)
        except Exception:
            _mllm_dev = '?'
        print(f"[boogu] streaming mllm on {_mllm_dev}; qwen3vl from_pretrained patch hits={_hits['n']}; build USS {_u0:.1f}->{_uss():.1f}GB", flush=True)
        pipe._studio_boogu_mllm_on_gpu = True
        pipe._studio_mllm_dir = str(Path(ckpt) / 'mllm')
        pipe._studio_mllm_dtype = dtype
        return pipe
    if (transformer_dir / 'diffusion_pytorch_model.bin.index.json').is_file():
        raise RuntimeError('DIRECT_VRAM_UNAVAILABLE: this Boogu checkpoint is .bin; DirectStorage needs safetensors. Convert it (or use native streaming) instead of a host-RAM load.')
    kwargs['transformer'] = _direct_from_pretrained(BooguImageTransformer2DModel, transformer_dir, dtype, device)
    with _qwen3vl_direct(device, dtype):
        pipe = BooguImagePipeline.from_pretrained(ckpt, **kwargs)
    pipe.vae.to(device)
    return pipe

def _krea2_resident_components(mode: str):
    if mode in ('staged', 'staged_disposable'):
        return ()
    if mode in ('resident_lazy', 'lazy_resident'):
        return ('transformer', 'vae')
    return KREA2_COMPONENTS

def build_pipeline(arch: str, ckpt: str, device: str='cuda:0', mode: str='resident', dtype: Optional[torch.dtype]=None):
    loader = _loader_for(arch)
    if loader == 'krea2' and sys.platform == 'win32' and (mode in ('streaming', 'stream', 'streamed', 'native_streaming')):
        raise ValueError('KREA_STREAMING_NEEDS_NATIVE_FILE: on Windows Krea streaming runs on the native DirectStorage path (adapter routes it there); the memory-mapped diffusers streamer is Mac-only')
    if mode in ('sequential', 'model_offload'):
        raise ValueError(f'CPU_OFFLOAD_REFUSED: {arch} mode {mode!r} parks weights in host RAM; use resident, staged or streaming (all direct to VRAM)')
    stream_boogu = loader == 'boogu' and mode in ('streaming', 'stream', 'streamed', 'native_streaming')
    stream_krea = loader == 'krea2' and mode in ('streaming', 'stream', 'streamed', 'native_streaming')
    from . import ram_preflight
    if stream_boogu or stream_krea:
        ram_preflight.preflight_resident_ram(6, f'{arch} native streaming â€” DiT on DirectStorage + mllm now loads/runs DIRECT-to-VRAM and is FREED (not CPU-parked) after encode, so the host-RAM commit is only ~2.2GB (USS-confirmed 2026-07-05); the residual is reclaimable safetensors mmap that evicts under pressure', env_prefix='BOOGU_KREA_STREAM', advice='Free host RAM before a boogu streaming gen if this refuses (needs only ~6GB now that the mllm is VRAM-resident + phased-freed, down from the pre-fix ~16GB CPU commit).')
    _prepare_kernel_runtime()
    dtype = dtype or torch.bfloat16
    pipe = _load_boogu(ckpt, dtype, device=device, stream_transformer=stream_boogu) if loader == 'boogu' else _load_krea2(ckpt, dtype, arch=arch, device=device, stream_transformer=stream_krea, resident=_krea2_resident_components(mode))
    if mode in ('staged', 'staged_disposable') and loader == 'krea2':
        pipe._studio_staged_gpu = True
        pipe._studio_staged_disposable = mode == 'staged_disposable'
        pipe._studio_staged_device = str(device)
    elif stream_krea:
        pipe._studio_staged_device = str(device)
    elif stream_boogu:
        pipe._studio_boogu_streamed = True
        pipe._studio_stream_device = str(device)
        if not getattr(pipe, '_studio_boogu_mllm_on_gpu', False):
            pipe.mllm.to('cpu')
            pipe.vae.to('cpu')
    elif mode in ('resident_lazy', 'lazy_resident') and loader == 'krea2':
        pipe._studio_lazy_resident = True
        pipe._studio_lazy_device = str(device)
    return pipe

def _cleanup_cuda() -> None:
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _boogu_ensure_mllm(pipe, dev) -> None:
    m = getattr(pipe, 'mllm', None)
    if m is not None:
        try:
            pipe.mllm.to(dev)
            return
        except Exception:
            pipe.mllm = None
    mdir = getattr(pipe, '_studio_mllm_dir', None)
    if not mdir:
        raise RuntimeError('boogu phased mllm reload: _studio_mllm_dir not set on the pipe')
    from transformers import Qwen3VLForConditionalGeneration
    dt = getattr(pipe, '_studio_mllm_dtype', torch.bfloat16)
    pipe.mllm = _direct_from_pretrained(Qwen3VLForConditionalGeneration, mdir, dt, str(dev))

def _boogu_free_mllm(pipe) -> None:
    try:
        pipe.mllm = None
    except Exception:
        pass
    _cleanup_cuda()

def _generate_krea2_staged(pipe, *, prompt: str, negative: str, width: int, height: int, steps: int, guidance: float, generator, device: str, callback_on_step_end=None, loras=None, receipt=None, should_cancel=None, two_pass=None):
    import numpy as np
    from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
    dev = torch.device(str(getattr(pipe, '_studio_staged_device', device) or device))
    batch_size = 1
    num_images_per_prompt = 1
    max_sequence_length = 512
    with torch.no_grad():
        _check_cancel(should_cancel)
        streamed = bool(getattr(pipe, '_studio_krea_hf_streamed', False))
        streamer = getattr(pipe.transformer, '_krea_hf_streamer', None) if streamed else None
        if streamed:
            pipe.text_encoder = _krea2_load_component(Path(pipe._studio_krea_text_dir).parent, 'text_encoder', pipe._studio_krea_text_dtype, str(dev))
            if any((parameter.device.type != dev.type or (dev.index is not None and parameter.device.index != dev.index) for parameter in pipe.text_encoder.parameters())):
                raise RuntimeError('Krea streaming text encoder did not load directly onto its selected device')
            streamer.should_cancel = should_cancel
            streamer.measure()
        else:
            _krea2_ensure(pipe, 'text_encoder', dev)
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt=prompt, device=dev, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        negative_prompt_embeds = negative_prompt_embeds_mask = None
        if guidance > 0:
            negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(prompt=negative or '', device=dev, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        if streamed:
            pipe.text_encoder = None
        else:
            _krea2_release(pipe, 'text_encoder')
        _cleanup_cuda()
        _check_cancel(should_cancel)
        if streamed:
            streamer.materialize_globals()
        else:
            _krea2_ensure(pipe, 'transformer', dev)
        num_channels_latents = pipe.transformer.config.in_channels // pipe.patch_size ** 2
        latents = pipe.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, dev, generator, None)
        grid_height = height // (pipe.vae_scale_factor * pipe.patch_size)
        grid_width = width // (pipe.vae_scale_factor * pipe.patch_size)
        position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, dev)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        image_seq_len = latents.shape[1]
        if pipe.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(image_seq_len, pipe.scheduler.config.get('base_image_seq_len', 256), pipe.scheduler.config.get('max_image_seq_len', 6400), pipe.scheduler.config.get('base_shift', 0.5), pipe.scheduler.config.get('max_shift', 1.15))
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, steps, dev, sigmas=sigmas, mu=mu)
        pipe._num_timesteps = len(timesteps)
        pipe.scheduler.set_begin_index(0)
        from .generation_loras import phase_loras
        from .two_pass import adapter_gate
        with phase_loras(pipe.transformer, loras or [], two_pass, device=str(dev), receipt=receipt) as apply, adapter_gate(two_pass, len(timesteps), apply, first_pass=loras or []) as gate:
            for i, t in enumerate(timesteps):
                _check_cancel(should_cancel)
                timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)
                noise_pred = pipe.transformer(hidden_states=latents, encoder_hidden_states=prompt_embeds, timestep=timestep, position_ids=position_ids, encoder_attention_mask=prompt_embeds_mask, attention_kwargs=None, return_dict=False)[0]
                if guidance > 0:
                    neg_noise_pred = pipe.transformer(hidden_states=latents, encoder_hidden_states=negative_prompt_embeds, timestep=timestep, position_ids=position_ids, encoder_attention_mask=negative_prompt_embeds_mask, attention_kwargs=None, return_dict=False)[0]
                    noise_pred = noise_pred + guidance * (noise_pred - neg_noise_pred)
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if gate:
                    gate.complete(i, {'latents': latents})
                if callback_on_step_end is not None:
                    callback_on_step_end(pipe, i, t, {'latents': latents})
        if streamed:
            streamer.evict_globals()
        else:
            _krea2_release(pipe, 'transformer')
        _cleanup_cuda()
        _check_cancel(should_cancel)
        if streamed:
            pipe.vae.to(dev)
        else:
            _krea2_ensure(pipe, 'vae', dev)
        latents = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = pipe.image_processor.postprocess(image, output_type='pil')
        if not streamed:
            _krea2_release(pipe, 'vae')
        _cleanup_cuda()
        return image[0]

def _generate_boogu_streamed(pipe, *, prompt: str, negative: str, width: int, height: int, steps: int, guidance: float, generator, device: str, reference_images: Optional[list]=None, on_step: Optional[Callable[[int, int], None]]=None):
    from boogu.models.transformers.rope import BooguImageRotaryPosEmbed
    from boogu.pipelines.boogu.pipeline_boogu import MomentumRollingSum
    dev = torch.device(str(getattr(pipe, '_studio_stream_device', device) or device))
    batch_size = 1
    num_images_per_instruction = 1
    flat_refs = list(reference_images or [])
    input_images = [flat_refs] if flat_refs else []
    pipe.user_set_pipe_device = str(dev)
    pipe.execution_device = str(dev)
    pipe._text_guidance_scale = float(guidance)
    pipe._image_guidance_scale = 1.0
    pipe._empty_instruction_guidance_scale = 0.0
    pipe._cfg_range = (0.0, 1.0)

    def step_func(i: int, total: int) -> None:
        if on_step is not None:
            on_step(i + 1, total)
    with torch.no_grad():
        _boogu_ensure_mllm(pipe, dev)
        instruction_embeds, instruction_attention_mask, negative_instruction_embeds, negative_instruction_attention_mask, empty_instruction_embeds, empty_instruction_attention_mask = pipe.encode_instruction([prompt], guidance > 1.0, negative_instruction=[negative or ''], input_images=input_images, num_images_per_instruction=num_images_per_instruction, device=dev, max_sequence_length=1280, truncate_instruction_sequence=False, use_rewrite_text_instruction=False, task_type=pipe._get_task_type_by_input_images(input_images))
        _boogu_free_mllm(pipe)
        _cleanup_cuda()
        dtype = getattr(pipe.vae, 'dtype', torch.bfloat16)
        if input_images:
            pipe.vae.to(dev)
        ref_latents = pipe.prepare_image(images=input_images, batch_size=batch_size, num_images_per_instruction=num_images_per_instruction, max_input_image_pixels=height * width, max_side_length=2 * max(height, width), device=dev, dtype=dtype)
        if input_images:
            pipe.vae.to('cpu')
            _cleanup_cuda()
        input_images, width, height, ori_width, ori_height = pipe._resolve_output_and_original_size(input_images=input_images, ref_latents=ref_latents, align_res=True, width=width, height=height, max_input_image_pixels=height * width, max_images_per_sample=max(len(reference_images or []), 0), img_scale_num=pipe.vae_scale_factor * 2)
        if len(input_images) == 0:
            pipe._image_guidance_scale = 1.0
        latent_channels = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(batch_size * num_images_per_instruction, latent_channels, height, width, instruction_embeds.dtype, dev, generator, None)
        freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(pipe.transformer.config.axes_dim_rope, pipe.transformer.config.axes_lens, theta=10000)
        pipe._studio_stage_vae_decode = True
        try:
            image = pipe.processing(latents=latents, ref_latents=ref_latents, instruction_embeds=instruction_embeds, freqs_cis=freqs_cis, negative_instruction_embeds=negative_instruction_embeds, instruction_attention_mask=instruction_attention_mask, negative_instruction_attention_mask=negative_instruction_attention_mask, num_inference_steps=steps, timesteps=None, device=dev, dtype=dtype, verbose=False, step_func=step_func, empty_instruction_embeds=empty_instruction_embeds, empty_instruction_attention_mask=empty_instruction_attention_mask, tg_momentum_state=MomentumRollingSum(), ig_momentum_state=MomentumRollingSum(), eg_momentum_state=MomentumRollingSum())
        finally:
            pipe._studio_stage_vae_decode = False
        image = torch.nn.functional.interpolate(image, size=(ori_height, ori_width), mode='bilinear')
        out = pipe.image_processor.postprocess(image, output_type='pil')
        pipe.vae.to('cpu')
        _cleanup_cuda()
        return out[0]

def _mk_callback(on_step: Optional[Callable[[int, int], None]]):
    if on_step is None:
        return None

    def _cb(_pipe, step: int, _timestep, kw):
        total = getattr(_pipe, 'num_timesteps', None) or getattr(_pipe, '_num_timesteps', 0)
        on_step(step + 1, total)
        return kw
    return _cb

def _generate(pipe, arch: str, *, prompt: str, negative: str='', width: int=1024, height: int=1024, steps: Optional[int]=None, guidance: Optional[float]=None, seed: int=0, device: str='cuda:0', reference_images: Optional[list]=None, on_step: Optional[Callable[[int, int], None]]=None, loras=None, receipt=None, should_cancel=None, two_pass=None, apply_loras=None):
    _prepare_kernel_runtime()
    defaults = ARCH_DEFAULTS.get((arch or '').lower(), ARCH_DEFAULTS['krea2'])
    steps = int(steps if steps is not None else defaults['steps'])
    guidance = float(guidance if guidance is not None else defaults['guidance'])
    gen = torch.Generator(device).manual_seed(int(seed))
    width, height = (width // 16 * 16, height // 16 * 16)
    cb = _mk_callback(on_step)
    if _loader_for(arch) == 'boogu':
        if getattr(pipe, '_studio_boogu_streamed', False):
            return _generate_boogu_streamed(pipe, prompt=prompt, negative=negative or '', width=width, height=height, steps=steps, guidance=guidance, generator=gen, device=device, reference_images=reference_images, on_step=on_step)
        kw = dict(instruction=prompt, negative_instruction=negative or '', height=height, width=width, max_input_image_pixels=height * width, max_input_image_side_length=2 * max(height, width), num_inference_steps=steps, text_guidance_scale=guidance, generator=gen, step_func=lambda step, total: on_step(step + 1, total) if on_step else None)
        if reference_images:
            kw['input_images'] = reference_images
        out = pipe(**kw)
    else:
        if getattr(pipe, '_studio_staged_gpu', False):
            return _generate_krea2_staged(pipe, prompt=prompt, negative=negative or '', width=width, height=height, steps=steps, guidance=guidance, generator=gen, device=device, callback_on_step_end=cb, loras=loras, receipt=receipt, should_cancel=should_cancel, two_pass=two_pass)
        if getattr(pipe, '_studio_lazy_resident', False):
            dev = torch.device(str(getattr(pipe, '_studio_lazy_device', device) or device))
            with torch.no_grad():
                _krea2_ensure(pipe, 'text_encoder', dev)
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt=prompt, device=dev)
                negative_prompt_embeds = negative_prompt_embeds_mask = None
                if guidance > 0:
                    negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(prompt=negative or '', device=dev)
                _krea2_release(pipe, 'text_encoder')
                _cleanup_cuda()
                from .two_pass import run_pipeline
                out = run_pipeline(pipe, dict(prompt=None, negative_prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask, negative_prompt_embeds=negative_prompt_embeds, negative_prompt_embeds_mask=negative_prompt_embeds_mask, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, generator=gen, output_type='pil', callback_on_step_end=cb), two_pass, apply_loras, first_pass=loras or [])
            return out.images[0]
        from .two_pass import run_pipeline
        out = run_pipeline(pipe, dict(prompt=prompt, negative_prompt=negative or None, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, generator=gen, output_type='pil', callback_on_step_end=cb), two_pass, apply_loras, first_pass=loras or [])
    return out.images[0]

def _check_cancel(should_cancel):
    if should_cancel and should_cancel():
        raise RuntimeError('generation cancelled')

def generate(pipe, arch, *, loras=None, receipt=None, should_cancel=None, on_step=None, two_pass=None, **kwargs):
    _check_cancel(should_cancel)
    device = kwargs.get('device', 'cuda:0')
    staged = bool(getattr(pipe, '_studio_staged_gpu', False)) and _loader_for(arch) != 'boogu'
    hook = pipe.transformer.register_forward_pre_hook(lambda *_: _check_cancel(should_cancel)) if getattr(pipe, 'transformer', None) is not None else None

    def progress(step, total):
        _check_cancel(should_cancel)
        if on_step:
            on_step(step, total)
        _check_cancel(should_cancel)
    try:
        from .generation_loras import phase_loras
        from .krea2_diffusion import _krea2_sda_schedule
        if not staged:
            context = phase_loras(pipe.transformer, loras or [], two_pass, device=device, receipt=receipt)
        else:
            context = nullcontext(None)
        schedule = _krea2_sda_schedule(pipe, two_pass) if two_pass else nullcontext()
        with context as apply, schedule:
            result = _generate(pipe, arch, on_step=progress, loras=loras, receipt=receipt, should_cancel=should_cancel, two_pass=two_pass, apply_loras=apply, **kwargs)
            _check_cancel(should_cancel)
            return result
    finally:
        if hook is not None:
            hook.remove()
