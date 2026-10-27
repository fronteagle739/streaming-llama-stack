from __future__ import annotations
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
DEFAULT_LAYER = 78
DEFAULT_SIDECAR_SUFFIX = '-mtp-layer78'

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

@dataclass(frozen=True)
class MtpSidecarInfo:
    sidecar_dir: Path
    model_path: Path
    config_path: Path
    manifest_path: Path | None
    source_layer: int
    tensor_prefix: str
    tensor_count: int
    metadata: dict[str, str]
    model_type: str
    source_repo: str
    weights_source: str
    tensor_name_format: str
    quantization: dict[str, Any]
    index_share_for_mtp_iteration: bool
    num_nextn_predict_layers: int
    n_routed_experts: int
    expert_tensor_count: int
    has_shared_head_head: bool
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        shared = 'yes' if self.index_share_for_mtp_iteration else 'no'
        head = 'yes' if self.has_shared_head_head else 'no'
        return f'{self.sidecar_dir} layer={self.source_layer} tensors={self.tensor_count} experts={self.n_routed_experts} expert_tensors={self.expert_tensor_count} index_share={shared} format={self.tensor_name_format} shared_head_head={head}'

def is_quantized_mtp_sidecar(info: MtpSidecarInfo) -> bool:
    return bool(getattr(info, 'quantization', {}) or {})

def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f'{path} is not a JSON object')
    return value

def read_safetensors_header(path: Path) -> tuple[dict[str, TensorInfo], dict[str, str]]:
    with path.open('rb') as f:
        hdr_len_b = f.read(8)
        if len(hdr_len_b) != 8:
            raise ValueError(f'{path} is too short for a safetensors header')
        hdr_len, = struct.unpack('<Q', hdr_len_b)
        header_b = f.read(hdr_len)
    if len(header_b) != hdr_len:
        raise ValueError(f'{path} safetensors header is truncated')
    header = json.loads(header_b.decode('utf-8'))
    if not isinstance(header, dict):
        raise ValueError(f'{path} safetensors header is not a JSON object')
    metadata = header.get('__metadata__', {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    tensors: dict[str, TensorInfo] = {}
    max_data_end = 0
    for name, value in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(value, dict):
            raise ValueError(f'{path} tensor entry {name!r} is not an object')
        shape = tuple((int(x) for x in value.get('shape', ())))
        offsets = tuple((int(x) for x in value.get('data_offsets', ())))
        if len(offsets) != 2:
            raise ValueError(f'{path} tensor entry {name!r} has bad data_offsets')
        if offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ValueError(f'{path} tensor entry {name!r} has invalid data_offsets')
        max_data_end = max(max_data_end, offsets[1])
        dtype = str(value.get('dtype', ''))
        tensors[name] = TensorInfo(name, dtype, shape, (offsets[0], offsets[1]))
    expected_size = 8 + hdr_len + max_data_end
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f'{path} safetensors data is truncated: file size {actual_size} < expected {expected_size}')
    return (tensors, {str(k): str(v) for k, v in metadata.items()})

def resolve_sidecar_dir(model_path: str | Path, sidecar: str | Path) -> Path | None:
    spec = str(sidecar).strip()
    if spec.lower() in ('', 'off', 'none', 'disabled', 'false', '0'):
        return None
    base = Path(model_path)
    if spec.lower() == 'auto':
        candidates = [Path(str(base) + DEFAULT_SIDECAR_SUFFIX), base.with_name(base.name + DEFAULT_SIDECAR_SUFFIX), Path(str(base) + DEFAULT_SIDECAR_SUFFIX + '-mxfp4'), base.with_name(base.name + DEFAULT_SIDECAR_SUFFIX + '-mxfp4'), base.with_name(base.name + '-mtp')]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]
    return Path(spec)

def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get('text_config')
    if isinstance(text, dict):
        return text
    return config

def _required_tensors(layer: int) -> set[str]:
    p = f'model.layers.{layer}'
    return {f'{p}.eh_proj.weight', f'{p}.enorm.weight', f'{p}.hnorm.weight', f'{p}.input_layernorm.weight', f'{p}.post_attention_layernorm.weight', f'{p}.self_attn.q_a_proj.weight', f'{p}.self_attn.q_a_layernorm.weight', f'{p}.self_attn.q_b_proj.weight', f'{p}.self_attn.kv_a_proj_with_mqa.weight', f'{p}.self_attn.kv_a_layernorm.weight', f'{p}.self_attn.kv_b_proj.weight', f'{p}.self_attn.o_proj.weight', f'{p}.self_attn.indexer.k_norm.bias', f'{p}.self_attn.indexer.k_norm.weight', f'{p}.self_attn.indexer.weights_proj.weight', f'{p}.self_attn.indexer.wk.weight', f'{p}.self_attn.indexer.wq_b.weight', f'{p}.mlp.gate.e_score_correction_bias', f'{p}.mlp.gate.weight', f'{p}.mlp.shared_experts.down_proj.weight', f'{p}.mlp.shared_experts.gate_proj.weight', f'{p}.mlp.shared_experts.up_proj.weight', f'{p}.shared_head.norm.weight'}

def _expect_shape(tensors: dict[str, TensorInfo], name: str, expected: Iterable[int], errors: list[str]) -> None:
    info = tensors.get(name)
    if info is None:
        errors.append(f'missing tensor: {name}')
        return
    exp = tuple((int(x) for x in expected))
    if info.shape != exp:
        errors.append(f'bad shape for {name}: expected {exp}, got {info.shape}')

def _ceil_div(value: int, denom: int) -> int:
    return (int(value) + int(denom) - 1) // int(denom)

def _mlx_quant_weight_shape(expected: Iterable[int], quantization: dict[str, Any]) -> tuple[int, ...]:
    exp = tuple((int(x) for x in expected))
    if not exp:
        return exp
    bits = int(quantization.get('bits', 4) or 4)
    pack = max(1, 32 // bits)
    return exp[:-1] + (_ceil_div(exp[-1], pack),)

def _mlx_quant_scale_shape(expected: Iterable[int], quantization: dict[str, Any]) -> tuple[int, ...]:
    exp = tuple((int(x) for x in expected))
    if not exp:
        return exp
    group_size = int(quantization.get('group_size', 32) or 32)
    return exp[:-1] + (_ceil_div(exp[-1], group_size),)

def _expect_mlx_runtime_weight(tensors: dict[str, TensorInfo], base: str, expected_dense: Iterable[int], quantized: bool, quantization: dict[str, Any], errors: list[str]) -> None:
    weight_name = f'{base}.weight'
    if not quantized:
        _expect_shape(tensors, weight_name, expected_dense, errors)
        return
    _expect_shape(tensors, weight_name, _mlx_quant_weight_shape(expected_dense, quantization), errors)
    _expect_shape(tensors, f'{base}.scales', _mlx_quant_scale_shape(expected_dense, quantization), errors)
    weight = tensors.get(weight_name)
    if weight is not None and weight.dtype != 'U32':
        errors.append(f'bad dtype for {weight_name}: expected U32, got {weight.dtype}')
    scales = tensors.get(f'{base}.scales')
    if scales is not None and scales.dtype != 'U8':
        errors.append(f'bad dtype for {base}.scales: expected U8, got {scales.dtype}')

def _expert_tensor_names(layer: int, n_experts: int) -> set[str]:
    p = f'model.layers.{layer}.mlp.experts'
    names: set[str] = set()
    for expert_id in range(n_experts):
        for proj in ('down_proj', 'gate_proj', 'up_proj'):
            names.add(f'{p}.{expert_id}.{proj}.weight')
    return names

def mlx_runtime_direct_tensor_map(layer: int) -> dict[str, str]:
    p = f'model.layers.{layer}'
    direct = {f'{p}.eh_proj.weight': 'eh_proj.weight', f'{p}.enorm.weight': 'enorm.weight', f'{p}.hnorm.weight': 'hnorm.weight', f'{p}.shared_head.norm.weight': 'shared_head_norm.weight', f'{p}.input_layernorm.weight': 'mtp_block.input_layernorm.weight', f'{p}.post_attention_layernorm.weight': 'mtp_block.post_attention_layernorm.weight', f'{p}.self_attn.q_a_proj.weight': 'mtp_block.self_attn.q_a_proj.weight', f'{p}.self_attn.q_a_layernorm.weight': 'mtp_block.self_attn.q_a_layernorm.weight', f'{p}.self_attn.q_b_proj.weight': 'mtp_block.self_attn.q_b_proj.weight', f'{p}.self_attn.kv_a_proj_with_mqa.weight': 'mtp_block.self_attn.kv_a_proj_with_mqa.weight', f'{p}.self_attn.kv_a_layernorm.weight': 'mtp_block.self_attn.kv_a_layernorm.weight', f'{p}.self_attn.o_proj.weight': 'mtp_block.self_attn.o_proj.weight', f'{p}.self_attn.indexer.k_norm.bias': 'mtp_block.self_attn.indexer.k_norm.bias', f'{p}.self_attn.indexer.k_norm.weight': 'mtp_block.self_attn.indexer.k_norm.weight', f'{p}.self_attn.indexer.weights_proj.weight': 'mtp_block.self_attn.indexer.weights_proj.weight', f'{p}.self_attn.indexer.wk.weight': 'mtp_block.self_attn.indexer.wk.weight', f'{p}.self_attn.indexer.wq_b.weight': 'mtp_block.self_attn.indexer.wq_b.weight', f'{p}.mlp.gate.e_score_correction_bias': 'mtp_block.mlp.gate.e_score_correction_bias', f'{p}.mlp.gate.weight': 'mtp_block.mlp.gate.weight', f'{p}.mlp.shared_experts.down_proj.weight': 'mtp_block.mlp.shared_experts.down_proj.weight', f'{p}.mlp.shared_experts.gate_proj.weight': 'mtp_block.mlp.shared_experts.gate_proj.weight', f'{p}.mlp.shared_experts.up_proj.weight': 'mtp_block.mlp.shared_experts.up_proj.weight'}
    return dict(sorted(direct.items()))

def mlx_runtime_quantizable_bases() -> set[str]:
    return {'eh_proj', 'mtp_block.self_attn.q_a_proj', 'mtp_block.self_attn.q_b_proj', 'mtp_block.self_attn.kv_a_proj_with_mqa', 'mtp_block.self_attn.embed_q', 'mtp_block.self_attn.unembed_out', 'mtp_block.self_attn.o_proj', 'mtp_block.self_attn.indexer.weights_proj', 'mtp_block.self_attn.indexer.wk', 'mtp_block.self_attn.indexer.wq_b', 'mtp_block.mlp.shared_experts.down_proj', 'mtp_block.mlp.shared_experts.gate_proj', 'mtp_block.mlp.shared_experts.up_proj', 'mtp_block.mlp.switch_mlp.down_proj', 'mtp_block.mlp.switch_mlp.gate_proj', 'mtp_block.mlp.switch_mlp.up_proj'}

def mlx_runtime_expected_tensors(layer: int, n_experts: int, quantized: bool) -> set[str]:
    names = set(mlx_runtime_direct_tensor_map(layer).values())
    names.add('mtp_block.self_attn.embed_q.weight')
    names.add('mtp_block.self_attn.unembed_out.weight')
    names.add('mtp_block.mlp.switch_mlp.down_proj.weight')
    names.add('mtp_block.mlp.switch_mlp.gate_proj.weight')
    names.add('mtp_block.mlp.switch_mlp.up_proj.weight')
    if quantized:
        for base in mlx_runtime_quantizable_bases():
            names.add(f'{base}.scales')
    return names

def validate_sidecar(sidecar_dir: str | Path) -> MtpSidecarInfo:
    sidecar = Path(sidecar_dir)
    config_path = sidecar / 'config.json'
    model_path = sidecar / 'model.safetensors'
    manifest_path = sidecar / 'glm52_mtp_manifest.json'
    if not sidecar.is_dir():
        raise FileNotFoundError(f'MTP sidecar directory not found: {sidecar}')
    if not config_path.is_file():
        raise FileNotFoundError(f'MTP sidecar config not found: {config_path}')
    if not model_path.is_file():
        raise FileNotFoundError(f'MTP sidecar model not found: {model_path}')
    config = _load_json(config_path)
    text_config = _text_config(config)
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    tensors, metadata = read_safetensors_header(model_path)
    layer = int(config.get('source_layer', manifest.get('source_layer', DEFAULT_LAYER)))
    prefix = str(config.get('tensor_prefix', f'model.layers.{layer}.'))
    model_type = str(config.get('model_type', ''))
    source_repo = str(config.get('source_repo', metadata.get('source_repo', '')))
    weights_source = str(config.get('weights_source', 'hf_original_layer78'))
    tensor_name_format = str(config.get('tensor_name_format', 'hf_original'))
    quantization = config.get('quantization') or config.get('quantization_config') or {}
    if not isinstance(quantization, dict):
        quantization = {}
    n_experts = int(text_config.get('n_routed_experts', 0) or 0)
    hidden = int(text_config.get('hidden_size', 0) or 0)
    q_lora = int(text_config.get('q_lora_rank', 0) or 0)
    kv_lora = int(text_config.get('kv_lora_rank', 0) or 0)
    qk_nope = int(text_config.get('qk_nope_head_dim', 0) or 0)
    qk_rope = int(text_config.get('qk_rope_head_dim', 0) or 0)
    v_head = int(text_config.get('v_head_dim', 0) or 0)
    n_head = int(text_config.get('num_attention_heads', 0) or 0)
    index_heads = int(text_config.get('index_n_heads', 0) or 0)
    index_head_dim = int(text_config.get('index_head_dim', 0) or 0)
    moe_hidden = int(text_config.get('moe_intermediate_size', 0) or 0)
    shared_experts = int(text_config.get('n_shared_experts', 1) or 1)
    errors: list[str] = []
    warnings: list[str] = []
    if model_type != 'glm_moe_dsa_mtp':
        errors.append(f"unexpected model_type {model_type!r}; expected 'glm_moe_dsa_mtp'")
    if not prefix.startswith(f'model.layers.{layer}.'):
        errors.append(f'tensor_prefix {prefix!r} does not match source_layer {layer}')
    if int(metadata.get('source_layer', layer)) != layer:
        errors.append(f"safetensors metadata source_layer={metadata.get('source_layer')} != {layer}")
    is_mlx_runtime = tensor_name_format == 'mlx_mtp_runtime' or weights_source == 'mlx_runtime_layer78'
    if is_mlx_runtime:
        quantized_runtime = bool(quantization)
        missing = sorted(mlx_runtime_expected_tensors(layer, n_experts, quantized_runtime).difference(tensors))
    else:
        missing = sorted(_required_tensors(layer).difference(tensors))
    errors.extend((f'missing tensor: {name}' for name in missing))
    if n_experts <= 0:
        errors.append('text_config.n_routed_experts is missing or zero')
    elif not is_mlx_runtime:
        expert_names = _expert_tensor_names(layer, n_experts)
        missing_experts = sorted(expert_names.difference(tensors))
        if missing_experts:
            errors.append(f'missing {len(missing_experts)} expert tensors; first={missing_experts[0]}')
    p = f'model.layers.{layer}'
    if is_mlx_runtime:
        quantized_runtime = bool(quantization)
        if hidden:
            _expect_mlx_runtime_weight(tensors, 'eh_proj', (hidden, hidden * 2), quantized_runtime, quantization, errors)
            _expect_shape(tensors, 'enorm.weight', (hidden,), errors)
            _expect_shape(tensors, 'hnorm.weight', (hidden,), errors)
            _expect_shape(tensors, 'shared_head_norm.weight', (hidden,), errors)
            _expect_shape(tensors, 'mtp_block.input_layernorm.weight', (hidden,), errors)
            _expect_shape(tensors, 'mtp_block.post_attention_layernorm.weight', (hidden,), errors)
            _expect_shape(tensors, 'mtp_block.mlp.gate.weight', (n_experts, hidden), errors)
        if q_lora:
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.q_a_proj', (q_lora, hidden), quantized_runtime, quantization, errors)
            _expect_shape(tensors, 'mtp_block.self_attn.q_a_layernorm.weight', (q_lora,), errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.q_b_proj', (n_head * (qk_nope + qk_rope), q_lora), quantized_runtime, quantization, errors)
        if kv_lora:
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.kv_a_proj_with_mqa', (kv_lora + qk_rope, hidden), quantized_runtime, quantization, errors)
            _expect_shape(tensors, 'mtp_block.self_attn.kv_a_layernorm.weight', (kv_lora,), errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.embed_q', (n_head, kv_lora, qk_nope), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.unembed_out', (n_head, v_head, kv_lora), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.o_proj', (hidden, n_head * v_head), quantized_runtime, quantization, errors)
        if index_head_dim:
            _expect_shape(tensors, 'mtp_block.self_attn.indexer.k_norm.bias', (index_head_dim,), errors)
            _expect_shape(tensors, 'mtp_block.self_attn.indexer.k_norm.weight', (index_head_dim,), errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.indexer.wk', (index_head_dim, hidden), quantized_runtime, quantization, errors)
        if index_heads and hidden:
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.indexer.weights_proj', (index_heads, hidden), quantized_runtime, quantization, errors)
        if index_heads and index_head_dim and q_lora:
            _expect_mlx_runtime_weight(tensors, 'mtp_block.self_attn.indexer.wq_b', (index_heads * index_head_dim, q_lora), quantized_runtime, quantization, errors)
        if moe_hidden and hidden:
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.switch_mlp.down_proj', (n_experts, hidden, moe_hidden), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.switch_mlp.gate_proj', (n_experts, moe_hidden, hidden), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.switch_mlp.up_proj', (n_experts, moe_hidden, hidden), quantized_runtime, quantization, errors)
            shexp = moe_hidden * shared_experts
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.shared_experts.down_proj', (hidden, shexp), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.shared_experts.gate_proj', (shexp, hidden), quantized_runtime, quantization, errors)
            _expect_mlx_runtime_weight(tensors, 'mtp_block.mlp.shared_experts.up_proj', (shexp, hidden), quantized_runtime, quantization, errors)
    else:
        if hidden:
            _expect_shape(tensors, f'{p}.eh_proj.weight', (hidden, hidden * 2), errors)
            _expect_shape(tensors, f'{p}.enorm.weight', (hidden,), errors)
            _expect_shape(tensors, f'{p}.hnorm.weight', (hidden,), errors)
            _expect_shape(tensors, f'{p}.input_layernorm.weight', (hidden,), errors)
            _expect_shape(tensors, f'{p}.post_attention_layernorm.weight', (hidden,), errors)
            _expect_shape(tensors, f'{p}.mlp.gate.weight', (n_experts, hidden), errors)
        if q_lora:
            _expect_shape(tensors, f'{p}.self_attn.q_a_proj.weight', (q_lora, hidden), errors)
            _expect_shape(tensors, f'{p}.self_attn.q_a_layernorm.weight', (q_lora,), errors)
            _expect_shape(tensors, f'{p}.self_attn.q_b_proj.weight', (n_head * (qk_nope + qk_rope), q_lora), errors)
        if kv_lora:
            _expect_shape(tensors, f'{p}.self_attn.kv_a_proj_with_mqa.weight', (kv_lora + qk_rope, hidden), errors)
            _expect_shape(tensors, f'{p}.self_attn.kv_a_layernorm.weight', (kv_lora,), errors)
            _expect_shape(tensors, f'{p}.self_attn.kv_b_proj.weight', (n_head * (qk_nope + v_head), kv_lora), errors)
        if index_heads and index_head_dim:
            _expect_shape(tensors, f'{p}.self_attn.indexer.k_norm.bias', (index_head_dim,), errors)
            _expect_shape(tensors, f'{p}.self_attn.indexer.k_norm.weight', (index_head_dim,), errors)
            _expect_shape(tensors, f'{p}.self_attn.indexer.weights_proj.weight', (index_heads, hidden), errors)
            _expect_shape(tensors, f'{p}.self_attn.indexer.wk.weight', (index_head_dim, hidden), errors)
            _expect_shape(tensors, f'{p}.self_attn.indexer.wq_b.weight', (index_heads * index_head_dim, q_lora), errors)
        if moe_hidden and hidden:
            for suffix, shape in (('down_proj.weight', (hidden, moe_hidden)), ('gate_proj.weight', (moe_hidden, hidden)), ('up_proj.weight', (moe_hidden, hidden))):
                _expect_shape(tensors, f'{p}.mlp.experts.0.{suffix}', shape, errors)
            shexp = moe_hidden * shared_experts
            _expect_shape(tensors, f'{p}.mlp.shared_experts.down_proj.weight', (hidden, shexp), errors)
            _expect_shape(tensors, f'{p}.mlp.shared_experts.gate_proj.weight', (shexp, hidden), errors)
            _expect_shape(tensors, f'{p}.mlp.shared_experts.up_proj.weight', (shexp, hidden), errors)
    if f'{p}.shared_head.head.weight' not in tensors and 'shared_head_head.weight' not in tensors:
        warnings.append('shared_head.head.weight is absent; runtime must use target/base lm_head for logits')
    expected_count = len(mlx_runtime_expected_tensors(layer, n_experts, bool(quantization))) if is_mlx_runtime else len(_required_tensors(layer)) + 3 * n_experts
    if n_experts and len(tensors) < expected_count:
        errors.append(f'tensor_count {len(tensors)} is below expected minimum {expected_count}')
    if errors:
        raise ValueError('Invalid GLM-5.2 MTP sidecar:\n  ' + '\n  '.join(errors))
    return MtpSidecarInfo(sidecar_dir=sidecar, model_path=model_path, config_path=config_path, manifest_path=manifest_path if manifest_path.is_file() else None, source_layer=layer, tensor_prefix=prefix, tensor_count=len(tensors), metadata=metadata, model_type=model_type, source_repo=source_repo, weights_source=weights_source, tensor_name_format=tensor_name_format, quantization={str(k): v for k, v in quantization.items()}, index_share_for_mtp_iteration=bool(config.get('index_share_for_mtp_iteration', text_config.get('index_share_for_mtp_iteration', False))), num_nextn_predict_layers=int(config.get('num_nextn_predict_layers', 0) or 0), n_routed_experts=n_experts, expert_tensor_count=3 * n_experts if is_mlx_runtime and all((f'mtp_block.mlp.switch_mlp.{proj}.weight' in tensors for proj in ('down_proj', 'gate_proj', 'up_proj'))) else sum((1 for name in tensors if f'{p}.mlp.experts.' in name)), has_shared_head_head=f'{p}.shared_head.head.weight' in tensors or 'shared_head_head.weight' in tensors, warnings=tuple(warnings))

def build_gguf_tensor_plan(sidecar_dir: str | Path) -> dict[str, Any]:
    info = validate_sidecar(sidecar_dir)
    if info.tensor_name_format != 'hf_original':
        raise ValueError(f'GGUF tensor planning requires an hf_original sidecar; got tensor_name_format={info.tensor_name_format!r}')
    cfg = _load_json(info.config_path)
    text = _text_config(cfg)
    layer = info.source_layer
    p = f'model.layers.{layer}'
    b = f'blk.{layer}'
    n_head = int(text['num_attention_heads'])
    kv_lora = int(text['kv_lora_rank'])
    qk_nope = int(text['qk_nope_head_dim'])
    v_head = int(text['v_head_dim'])
    n_experts = int(text['n_routed_experts'])
    direct = {f'{p}.eh_proj.weight': f'{b}.nextn.eh_proj.weight', f'{p}.enorm.weight': f'{b}.nextn.enorm.weight', f'{p}.hnorm.weight': f'{b}.nextn.hnorm.weight', f'{p}.shared_head.norm.weight': f'{b}.nextn.shared_head_norm.weight', f'{p}.input_layernorm.weight': f'{b}.attn_norm.weight', f'{p}.post_attention_layernorm.weight': f'{b}.ffn_norm.weight', f'{p}.self_attn.q_a_proj.weight': f'{b}.attn_q_a.weight', f'{p}.self_attn.q_a_layernorm.weight': f'{b}.attn_q_a_norm.weight', f'{p}.self_attn.q_b_proj.weight': f'{b}.attn_q_b.weight', f'{p}.self_attn.kv_a_proj_with_mqa.weight': f'{b}.attn_kv_a_mqa.weight', f'{p}.self_attn.kv_a_layernorm.weight': f'{b}.attn_kv_a_norm.weight', f'{p}.self_attn.o_proj.weight': f'{b}.attn_output.weight', f'{p}.self_attn.indexer.k_norm.bias': f'{b}.indexer.k_norm.bias', f'{p}.self_attn.indexer.k_norm.weight': f'{b}.indexer.k_norm.weight', f'{p}.self_attn.indexer.weights_proj.weight': f'{b}.indexer.proj.weight', f'{p}.self_attn.indexer.wk.weight': f'{b}.indexer.attn_k.weight', f'{p}.self_attn.indexer.wq_b.weight': f'{b}.indexer.attn_q_b.weight', f'{p}.mlp.gate.e_score_correction_bias': f'{b}.exp_probs_b.bias', f'{p}.mlp.gate.weight': f'{b}.ffn_gate_inp.weight', f'{p}.mlp.shared_experts.down_proj.weight': f'{b}.ffn_down_shexp.weight', f'{p}.mlp.shared_experts.gate_proj.weight': f'{b}.ffn_gate_shexp.weight', f'{p}.mlp.shared_experts.up_proj.weight': f'{b}.ffn_up_shexp.weight'}
    if info.has_shared_head_head:
        direct[f'{p}.shared_head.head.weight'] = f'{b}.nextn.shared_head_head.weight'
    return {'format': 'glm52_mtp_gguf_tensor_plan.v1', 'arch': 'glm-dsa', 'source_layer': layer, 'num_nextn_predict_layers': info.num_nextn_predict_layers, 'index_share_for_mtp_iteration': info.index_share_for_mtp_iteration, 'direct': dict(sorted(direct.items())), 'kv_b_split': {'source': f'{p}.self_attn.kv_b_proj.weight', 'source_shape': [n_head * (qk_nope + v_head), kv_lora], 'k_target': f'{b}.attn_k_b.weight', 'k_shape': [qk_nope, kv_lora, n_head], 'v_target': f'{b}.attn_v_b.weight', 'v_shape': [kv_lora, v_head, n_head], 'split_axis': 0, 'k_rows': n_head * qk_nope, 'v_rows': n_head * v_head}, 'expert_merge': {'count': n_experts, 'gate_target': f'{b}.ffn_gate_exps.weight', 'up_target': f'{b}.ffn_up_exps.weight', 'down_target': f'{b}.ffn_down_exps.weight', 'gate_sources': [f'{p}.mlp.experts.{idx}.gate_proj.weight' for idx in range(n_experts)], 'up_sources': [f'{p}.mlp.experts.{idx}.up_proj.weight' for idx in range(n_experts)], 'down_sources': [f'{p}.mlp.experts.{idx}.down_proj.weight' for idx in range(n_experts)]}}

def write_gguf_tensor_plan(sidecar_dir: str | Path, output_path: str | Path) -> Path:
    plan = build_gguf_tensor_plan(sidecar_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, sort_keys=True)
        f.write('\n')
    return out
