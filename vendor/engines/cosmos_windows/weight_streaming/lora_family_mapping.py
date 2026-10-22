# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/lora_family_mapping.py
# Machine fallback paths cleared; no model data included.
import re
_KREA_BLOCK = {'attn.to_q': 'attn.wq', 'attn.to_k': 'attn.wk', 'attn.to_v': 'attn.wv', 'attn.to_gate': 'attn.gate', 'attn.to_out.0': 'attn.wo', 'ff.gate': 'mlp.gate', 'ff.up': 'mlp.up', 'ff.down': 'mlp.down'}
_KREA_TOP = {'img_in': 'first', 'time_embed.linear_1': 'tmlp.0', 'time_embed.linear_2': 'tmlp.2', 'time_mod_proj': 'tproj.1', 'txt_in.linear_1': 'txtmlp.1', 'txt_in.linear_2': 'txtmlp.3', 'text_fusion.projector': 'txtfusion.projector', 'final_layer.linear': 'last.linear'}

def _krea_aliases(transformer):
    aliases = {}
    for name, _ in transformer.named_modules():
        native = _KREA_TOP.get(name)
        match = re.fullmatch('(transformer_blocks|text_fusion.layerwise_blocks|text_fusion.refiner_blocks)\\.(\\d+)\\.(.+)', name)
        if match and match[3] in _KREA_BLOCK:
            group = {'transformer_blocks': 'blocks', 'text_fusion.layerwise_blocks': 'txtfusion.layerwise_blocks', 'text_fusion.refiner_blocks': 'txtfusion.refiner_blocks'}[match[1]]
            native = f'{group}.{match[2]}.{_KREA_BLOCK[match[3]]}'
        if native:
            aliases[native] = name
            aliases['lora_unet_' + native.replace('.', '_')] = name
    return aliases

def adapter_key_plan(transformer, key, shape):
    from cosmos_windows.gen_with_lora import _normalize_adapter_key
    normalized = _normalize_adapter_key(key, 'default')
    normalized = normalized.replace('.lora_down.weight', '.lora_A.default.weight')
    normalized = normalized.replace('.lora_up.weight', '.lora_B.default.weight')
    match = re.fullmatch('(.+)(\\.lora_([AB])\\.default\\.weight|\\.alpha)', normalized)
    if not match:
        return [(normalized, shape, None)]
    module, suffix, side = match.groups()
    family = type(transformer).__name__
    if family == 'Krea2Transformer2DModel':
        module = _krea_aliases(transformer).get(module, module)
    elif family == 'Flux2Transformer2DModel':
        double = re.fullmatch('double_blocks\\.(\\d+)\\.(.+)', module)
        single = re.fullmatch('single_blocks\\.(\\d+)\\.(linear[12])', module)
        if double:
            index, tail = double.groups()
            prefix = f'transformer_blocks.{index}'
            if tail in {'img_attn.qkv', 'txt_attn.qkv'}:
                parts = ['attn.to_q', 'attn.to_k', 'attn.to_v'] if tail.startswith('img') else ['attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj']
                widths = [transformer.get_submodule(prefix + '.' + part).out_features for part in parts]
                if side == 'B' and (len(shape) != 2 or shape[0] != sum(widths)):
                    raise ValueError(f'Packed Flux2 QKV LoRA shape does not match {module}: {shape}')
                offset, result = (0, [])
                for part, width in zip(parts, widths):
                    rows = (offset, offset + width) if side == 'B' else None
                    converted = (width, shape[1]) if side == 'B' else shape
                    result.append((prefix + '.' + part + suffix, converted, rows))
                    offset += width
                return result
            tail = {'img_attn.proj': 'attn.to_out.0', 'txt_attn.proj': 'attn.to_add_out', 'img_mlp.0': 'ff.linear_in', 'img_mlp.2': 'ff.linear_out', 'txt_mlp.0': 'ff_context.linear_in', 'txt_mlp.2': 'ff_context.linear_out'}.get(tail, tail)
            module = prefix + '.' + tail
        elif single:
            index, tail = single.groups()
            module = f'single_transformer_blocks.{index}.attn.' + ('to_qkv_mlp_proj' if tail == 'linear1' else 'to_out')
    return [(module + suffix, shape, None)]
