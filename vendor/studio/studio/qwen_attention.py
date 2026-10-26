# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations

def omit_redundant_padding_mask(module, args, kwargs):
    mask = kwargs.get('encoder_hidden_states_mask')
    if mask is not None and bool(mask.bool().all().item()):
        kwargs = dict(kwargs, encoder_hidden_states_mask=None)
    return (args, kwargs)

def flash_attention_function():
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        return flash_attn_func
    except (ImportError, OSError):
        return None

def make_flash_processor(flash):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor, _qwenimage21_prepare_qkv

    class FlashProcessor(QwenImage21AttnProcessor):

        def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
            if key_valid is not None or (segments is None and attention_mask is not None):
                return super().__call__(attn, hidden_states, attention_mask, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid)
            query, key, value, length = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
            if segments is None:
                result = flash(query, key, value, dropout_p=0.0, causal=False)
            else:
                outputs = []
                for start, end, is_text in segments:
                    outputs.append(flash(query[:, start:end], key[:, :end], value[:, :end], dropout_p=0.0, causal=is_text))
                prefix = segments[-1][1] if segments else 0
                if prefix < query.shape[1]:
                    outputs.append(flash(query[:, prefix:], key, value, dropout_p=0.0, causal=False))
                result = torch.cat(outputs, dim=1)
            result = result[:, :length].flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](result))
    return FlashProcessor()

def configure_attention(transformer):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    flash = flash_attention_function()
    transformer.set_attn_processor(make_flash_processor(flash) if flash else QwenImage21AttnProcessor())
    transformer.register_forward_pre_hook(omit_redundant_padding_mask, with_kwargs=True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    transformer._studio_attention = 'flash_attn2' if flash else 'sdpa'
    print('[status] Qwen attention: ' + ('compiled FlashAttention 2; block-causal prefill and cached decode' if flash else 'fused SDPA (compiled FA2 unavailable in this interpreter)'), flush=True)

def configure_text_encoder(encoder):
    captured = {}
    text_model = getattr(encoder.model, 'language_model', encoder.model)

    def before(module, args, kwargs):
        captured.clear()
        return (args, dict(kwargs, output_hidden_states=False, logits_to_keep=1))

    def capture(module, args, output):
        captured['last'] = args[0]

    def after(module, args, output):
        if 'last' not in captured:
            raise RuntimeError('Qwen text encoder did not publish its final hidden state')
        output.hidden_states = (captured.pop('last'),)
        return output
    encoder.register_forward_pre_hook(before, with_kwargs=True)
    text_model.norm.register_forward_hook(capture)
    encoder.register_forward_hook(after)
