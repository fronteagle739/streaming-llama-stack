# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
H3_LORA_FORMAT = 'minimax-h3-streamed-lora-v1'
H3_DATASET_FORMAT = 'minimax-h3-video-dataset-v1'
DEFAULT_TARGETS = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')

@dataclass(frozen=True)
class H3DatasetRecord:
    prompt: str
    target_video: Path
    target_audio: Path | None = None
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_images: tuple[Path, ...] = ()
    source_video: Path | None = None

@dataclass(frozen=True)
class H3LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_TARGETS

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError('LoRA rank must be at least 1')
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('LoRA alpha must be finite and positive')
        if not 0 <= self.dropout < 1:
            raise ValueError('LoRA dropout must be in [0, 1)')
        if not self.targets:
            raise ValueError('at least one LoRA target is required')

def _resolve_media(manifest: Path, raw: object, field: str, *, required: bool=False) -> Path | None:
    if raw is None:
        if required:
            raise ValueError(f'{field} is required')
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f'{field} must be a non-empty path string')
    path = Path(raw)
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f'{field} file does not exist: {path}')
    return path

def load_h3_dataset_manifest(path: str | Path) -> tuple[H3DatasetRecord, ...]:
    manifest = Path(path).resolve()
    records: list[H3DatasetRecord] = []
    for line_number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError('record must be an object')
            prompt = raw.get('prompt')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('prompt must be a non-empty string')
            references = raw.get('reference_images', [])
            if not isinstance(references, list):
                raise ValueError('reference_images must be an array')
            record = H3DatasetRecord(prompt=prompt.strip(), target_video=_resolve_media(manifest, raw.get('target_video'), 'target_video', required=True), target_audio=_resolve_media(manifest, raw.get('target_audio'), 'target_audio'), first_frame=_resolve_media(manifest, raw.get('first_frame'), 'first_frame'), last_frame=_resolve_media(manifest, raw.get('last_frame'), 'last_frame'), reference_images=tuple((_resolve_media(manifest, item, f'reference_images[{index}]', required=True) for index, item in enumerate(references))), source_video=_resolve_media(manifest, raw.get('source_video'), 'source_video'))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f'invalid H3 dataset record at line {line_number}: {exc}') from exc
        records.append(record)
    if not records:
        raise ValueError('H3 dataset manifest contains no records')
    return tuple(records)

def dataset_manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class H3LoRALinear(nn.Linear):

    def __init__(self, source: nn.Linear, config: H3LoRAConfig, *, device: str | torch.device):
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None, device='meta', dtype=source.weight.dtype)
        self.weight = source.weight
        if source.bias is not None:
            self.bias = source.bias
        self.lora_A = nn.Parameter(torch.empty(config.rank, source.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(source.out_features, config.rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scale = config.alpha / config.rank
        self.lora_strength = 1.0
        self.lora_dropout = float(config.dropout)
        self.inference_adapters: list[tuple[torch.Tensor, torch.Tensor, float, frozenset]] = []
        self.lora_passes: frozenset = frozenset()
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, value):
        from .minimax_h3_int8 import adapter_active
        base = F.linear(value, self.weight, self.bias)
        if adapter_active(getattr(self, 'lora_passes', ())):
            adapter_input = F.dropout(value.float(), self.lora_dropout, self.training)
            adapter = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
            result = base + adapter.to(base.dtype) * (self.lora_scale * self.lora_strength)
        else:
            result = base
        for down, up, scale, passes in self.inference_adapters:
            if not adapter_active(passes):
                continue
            extra = F.linear(F.linear(value.float(), down), up)
            result = result + extra.to(result.dtype) * scale
        return result

    def add_inference_adapter(self, down, up, scale: float, passes=()) -> None:
        self.inference_adapters.append((down.detach(), up.detach(), float(scale), frozenset(passes)))

def inject_h3_lora(model: nn.Module, config: H3LoRAConfig, *, device='cuda') -> tuple[str, ...]:
    config.validate()
    replaced: list[str] = []
    blocks = getattr(model, 'blocks', None)
    if blocks is None:
        raise ValueError('H3 model does not expose blocks')
    for block_index, block in enumerate(blocks):
        for target in config.targets:
            parent = block
            parts = target.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            leaf = parts[-1]
            source = getattr(parent, leaf)
            if not isinstance(source, nn.Linear):
                raise ValueError(f'LoRA target blocks.{block_index}.{target} is not linear')
            setattr(parent, leaf, H3LoRALinear(source, config, device=device))
            replaced.append(f'blocks.{block_index}.{target}')
    return tuple(replaced)

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name.endswith('.lora_A') or name.endswith('.lora_B')}

def save_h3_lora_checkpoint(path: str | Path, model: nn.Module, *, config: H3LoRAConfig, dataset_sha256: str, step: int, optimizer=None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    payload = {'format': H3_LORA_FORMAT, 'config': asdict(config), 'dataset_sha256': dataset_sha256, 'step': int(step), 'lora': lora_state_dict(model), 'optimizer': optimizer.state_dict() if optimizer is not None else None}
    torch.save(payload, temporary)
    temporary.replace(destination)

def load_h3_lora_checkpoint(path: str | Path, model: nn.Module, *, expected_dataset_sha256: str | None=None, optimizer=None) -> int:
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if payload.get('format') != H3_LORA_FORMAT:
        raise ValueError('unsupported H3 LoRA checkpoint format')
    if expected_dataset_sha256 and payload.get('dataset_sha256') != expected_dataset_sha256:
        raise ValueError('dataset manifest changed since the H3 LoRA checkpoint')
    missing, unexpected = model.load_state_dict(payload['lora'], strict=False)
    unexpected = [name for name in unexpected if name.endswith(('.lora_A', '.lora_B'))]
    expected = set(lora_state_dict(model))
    loaded = set(payload['lora'])
    if unexpected or loaded != expected:
        raise ValueError(f'LoRA checkpoint keys do not match model: unexpected={unexpected}')
    if optimizer is not None and payload.get('optimizer') is not None:
        optimizer.load_state_dict(payload['optimizer'])
    return int(payload['step'])

def _load_h3_lora_artifact(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        return torch.load(path, map_location='cpu', weights_only=False)
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
    try:
        config = json.loads(metadata['config'])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid H3 LoRA safetensors metadata: {path}') from exc
    return {'format': metadata.get('format'), 'config': config, 'dataset_sha256': metadata.get('dataset_sha256'), 'step': int(metadata.get('step', '0')), 'lora': load_file(path, device='cpu'), 'optimizer': None}
H3_EXTERNAL_APPLICATION = 'W_eff = W + lora_B @ lora_A'

def _load_external_h3_lora(path: Path, *, device='cpu') -> dict[str, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file
    if str(device).startswith('cuda'):
        from .safetensors_index import build_index
        from .ds_backend import DirectStorageStreamer, enforce_ds_mode
        index = build_index(path)
        groups = {i: [ref] for i, ref in enumerate(index.refs.values())}
        from .backends import _torch_dtype
        offsets, size = ({}, 0)
        for ref in index.refs.values():
            size = (size + 255) // 256 * 256
            offsets[ref.name] = size
            size += ref.nbytes
        arena = torch.empty(size, dtype=torch.uint8, device=device)
        streamer = DirectStorageStreamer(index, component='lora', device=device, layer_groups=groups, n_gpu_buffers=1, record_compute_done=True)
        enforce_ds_mode(streamer, 'H3 LoRA', 'MINIMAX_H3')
        flat = {}
        try:
            for i in groups:
                for name, value in streamer.wait_layer(i).items():
                    ref = index.refs[name]
                    offset = offsets[name]
                    owned = arena[offset:offset + ref.nbytes].view(_torch_dtype(ref.dtype)).view(ref.shape)
                    owned.copy_(value)
                    flat[name] = owned
                torch.cuda.synchronize(device)
                streamer.release_layer(i)
            torch.cuda.synchronize(device)
        finally:
            streamer.close()
    else:
        flat = load_file(str(path), device='cpu')
    out: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    for key, tensor in flat.items():
        if key.endswith('.alpha'):
            alphas[_kohya_stem(key[:-len('.alpha')])] = float(tensor.reshape(()).item())
            continue
        name = key[:-len('.weight')] if key.endswith('.weight') else key
        if name.endswith(('.lora_A.default', '.lora_B.default')):
            name = name[:-len('.default')]
        for suffix, slot in (('.lora_A', 'A'), ('.lora_B', 'B'), ('.lora_down', 'A'), ('.lora_up', 'B')):
            if name.endswith(suffix):
                out.setdefault(_kohya_stem(name[:-len(suffix)]), {})[slot] = tensor
                break
        else:
            raise ValueError(f'external H3 LoRA {path} has a key that is neither lora_A/lora_B nor lora_down/lora_up: {key!r}')
    incomplete = [m for m, pair in out.items() if not {'A', 'B'} <= set(pair)]
    if incomplete:
        raise ValueError(f'external H3 LoRA {path}: {len(incomplete)} module(s) missing an A/B half, e.g. {incomplete[:3]}')
    if not out:
        raise ValueError(f'external H3 LoRA {path} contains no lora_A/lora_B tensors')
    for module_name, alpha in alphas.items():
        pair = out.get(module_name)
        if pair is None:
            raise ValueError(f'external H3 LoRA {path}: alpha for {module_name!r} has no matching lora_down/lora_up pair')
        rank = int(pair['A'].shape[0])
        pair['scale'] = alpha / rank
    out = _strip_diffusion_model_prefix(out, path)
    return out
_KOHYA_PREFIXES = ('lora_unet_', 'lora_te_', 'lora_transformer_')

def _kohya_stem(name: str) -> str:
    for prefix in _KOHYA_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
_COMFY_ROOT_PREFIX = 'diffusion_model.'

def _strip_diffusion_model_prefix(pairs: dict[str, dict[str, torch.Tensor]], path: Path) -> dict[str, dict[str, torch.Tensor]]:
    prefixed = [m for m in pairs if m.startswith(_COMFY_ROOT_PREFIX)]
    if not prefixed:
        return pairs
    if len(prefixed) != len(pairs):
        plain = [m for m in pairs if not m.startswith(_COMFY_ROOT_PREFIX)]
        raise ValueError(f'external H3 LoRA {path}: {len(prefixed)} module(s) carry the {_COMFY_ROOT_PREFIX!r} prefix and {len(plain)} do not, e.g. {prefixed[:2]} vs {plain[:2]} — refusing to guess the layout of a mixed-convention adapter')
    renamed = {m[len(_COMFY_ROOT_PREFIX):]: ab for m, ab in pairs.items()}
    if len(renamed) != len(pairs):
        raise ValueError(f'external H3 LoRA {path}: stripping {_COMFY_ROOT_PREFIX!r} collapsed {len(pairs)} modules onto {len(renamed)} names')
    return renamed

def _h3_adapter_spec(entry):
    if len(entry) == 3:
        path, weight, passes = entry
    else:
        (path, weight), passes = (entry, ())
    return (Path(path), float(weight), frozenset((int(p) for p in passes or ())))

def attach_external_h3_lora_files(model: nn.Module, adapters: Iterable[tuple[str | Path, float]], *, device='cuda') -> tuple[str, ...]:
    specs = [_h3_adapter_spec(entry) for entry in adapters]
    if not specs:
        return ()
    for path, strength, _passes in specs:
        if not math.isfinite(strength):
            raise ValueError(f'non-finite H3 LoRA strength for {path}')
    underscore_index: dict[str, str] | None = None
    for path, strength, passes in specs:
        pairs = _load_external_h3_lora(path, device=device)
        skipped: list[str] = []
        applied_here = 0
        for module_name, ab in sorted(pairs.items()):
            down, up = (ab['A'], ab['B'])
            try:
                target = model.get_submodule(module_name)
            except AttributeError as exc:
                if underscore_index is None:
                    underscore_index = {}
                    for real, _mod in model.named_modules():
                        if real:
                            underscore_index.setdefault(real.replace('.', '_'), real)
                resolved = underscore_index.get(module_name)
                if resolved is None:
                    raise ValueError(f'external H3 LoRA {path} targets {module_name!r}, which does not exist in this model — refusing to apply a partial adapter') from exc
                module_name = resolved
                target = model.get_submodule(module_name)
            if not isinstance(target, nn.Linear):
                raise ValueError(f'external H3 LoRA {path}: {module_name!r} is a {type(target).__name__}, not nn.Linear')
            rank = int(down.shape[0])
            if down.shape[1] != target.in_features or up.shape[0] != target.out_features or up.shape[1] != rank:
                skipped.append(f'{module_name} (adapter A{tuple(down.shape)} B{tuple(up.shape)} vs linear in={target.in_features} out={target.out_features})')
                continue
            applied_here += 1
            adapter_scale = float(ab.get('scale', 1.0))
            from .minimax_h3_int8 import H3ConvRotLinear, H3Int8ConvRotLinear, H3PackedConvRotLinear
            if isinstance(target, (H3Int8ConvRotLinear, H3ConvRotLinear)):
                target.add_inference_adapter(down.to(device=device, dtype=down.dtype if isinstance(target, H3PackedConvRotLinear) else torch.float32), up.to(device=device, dtype=up.dtype if isinstance(target, H3PackedConvRotLinear) else torch.float32), strength * adapter_scale, passes)
            elif not isinstance(target, H3LoRALinear):
                cfg = H3LoRAConfig(rank=rank, alpha=float(rank) * adapter_scale)
                parent_name, _, leaf = module_name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapper = H3LoRALinear(target, cfg, device=device)
                setattr(parent, leaf, wrapper)
                wrapper.lora_A.data.copy_(down.to(device=device, dtype=torch.float32))
                wrapper.lora_B.data.copy_(up.to(device=device, dtype=torch.float32))
                wrapper.lora_scale = adapter_scale
                wrapper.lora_strength = strength
                wrapper.lora_A.requires_grad_(False)
                wrapper.lora_B.requires_grad_(False)
                wrapper.lora_passes = frozenset(passes)
            else:
                target.add_inference_adapter(down.to(device=device, dtype=torch.float32), up.to(device=device, dtype=torch.float32), strength * adapter_scale, passes)
        if skipped:
            print(f'WARNING: external H3 LoRA {path}: applied {applied_here} module(s), SKIPPED {len(skipped)} whose shapes do not match this checkpoint:', file=sys.stderr, flush=True)
            for entry in skipped:
                print(f'    - {entry}', file=sys.stderr, flush=True)
        if pairs and applied_here == 0:
            raise ValueError(f"external H3 LoRA {path}: none of its {len(pairs)} modules match this checkpoint (first: {(skipped[0] if skipped else 'n/a')}) -- refusing to pretend it was applied")
    model.eval()
    return tuple((str(p) for p, _, _ in specs))

def attach_h3_lora_files(model: nn.Module, adapters: Iterable[tuple[str | Path, float]], *, device='cuda') -> tuple[str, ...]:
    specs = [_h3_adapter_spec(entry) for entry in adapters]
    if not specs:
        return ()
    external = []
    for path, _, _ in specs:
        if path.suffix.lower() == '.safetensors':
            from safetensors import safe_open
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                md = handle.metadata() or {}
            external.append('config' not in md)
        else:
            external.append(False)
    if any(external):
        if not all(external):
            raise ValueError('cannot stack external and native H3 LoRA artifacts in one request; load them separately')
        return attach_external_h3_lora_files(model, specs, device=device)
    payloads = []
    for path, strength, _passes in specs:
        if not math.isfinite(strength):
            raise ValueError(f'non-finite H3 LoRA strength for {path}')
        payload = _load_h3_lora_artifact(path)
        if payload.get('format') != H3_LORA_FORMAT:
            raise ValueError(f'unsupported H3 LoRA artifact: {path}')
        config = H3LoRAConfig(**payload['config'])
        config.validate()
        payloads.append((path, strength, payload, config, _passes))
    first_config = payloads[0][3]
    for path, _, _, config, _ in payloads[1:]:
        if config.targets != first_config.targets:
            raise ValueError(f'H3 LoRA target mismatch in {path}')
    replaced = inject_h3_lora(model, first_config, device=device)
    modules = dict(model.named_modules())
    expected_keys = set()
    for module_name in replaced:
        expected_keys.update((f'{module_name}.lora_A', f'{module_name}.lora_B'))
    for adapter_index, (path, strength, payload, config, passes) in enumerate(payloads):
        state = payload.get('lora', {})
        if set(state) != expected_keys:
            raise ValueError(f'H3 LoRA keys do not match the model: {path}')
        for module_name in replaced:
            module = modules[module_name]
            down = state[f'{module_name}.lora_A'].to(device=device, dtype=torch.float32)
            up = state[f'{module_name}.lora_B'].to(device=device, dtype=torch.float32)
            if adapter_index == 0:
                module.lora_A.data.copy_(down)
                module.lora_B.data.copy_(up)
                module.lora_scale = config.alpha / config.rank
                module.lora_strength = strength
                module.lora_A.requires_grad_(False)
                module.lora_B.requires_grad_(False)
                module.lora_passes = frozenset(passes)
            else:
                module.add_inference_adapter(down, up, strength * config.alpha / config.rank, passes)
    model.eval()
    return tuple((str(path) for path, *_ in payloads))
