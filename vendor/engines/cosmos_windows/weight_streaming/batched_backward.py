# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/batched_backward.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import torch
from ..streamed_backward import apply_streamed_batch

def pre_blocks(tf, pack):
    cfg = tf.config
    input_ids = pack['input_ids']
    text_indexes = pack['text_indexes']
    position_ids = pack['position_ids']
    und_len = pack['und_len']
    sequence_length = pack['sequence_length']
    vision_tokens = pack['vision_tokens']
    vision_token_shapes = pack['vision_token_shapes']
    vision_sequence_indexes = pack['vision_sequence_indexes']
    vision_mse_loss_indexes = pack['vision_mse_loss_indexes']
    vision_timesteps = pack['vision_timesteps']
    vision_noisy_frame_indexes = pack['vision_noisy_frame_indexes']
    packed_text_embedding = tf.embed_tokens(input_ids)
    target_dtype = packed_text_embedding.dtype
    hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, cfg.hidden_size))
    hidden_states[text_indexes] = packed_text_embedding
    packed_tokens_vision, original_latent_shapes = tf._patchify_and_pack_latents(vision_tokens)
    packed_tokens_vision = tf.proj_in(packed_tokens_vision)
    timesteps_vision = vision_timesteps * cfg.timestep_scale
    packed_timestep_embeds_vision = tf.time_embedder(tf.time_proj(timesteps_vision)).to(target_dtype)
    packed_tokens_vision = tf._apply_timestep_embeds_to_noisy_tokens(packed_tokens=packed_tokens_vision, packed_timestep_embeds=packed_timestep_embeds_vision, noisy_frame_indexes=vision_noisy_frame_indexes, token_shapes=vision_token_shapes)
    hidden_states[vision_sequence_indexes] = packed_tokens_vision
    cos, sin = tf.rotary_emb(position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1), device=hidden_states.device, dtype=hidden_states.dtype)
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    und_seq = hidden_states[:und_len]
    gen_seq = hidden_states[und_len:]
    rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
    post_ctx = dict(vision_mse_loss_indexes=vision_mse_loss_indexes, vision_token_shapes=vision_token_shapes, vision_noisy_frame_indexes=vision_noisy_frame_indexes, original_latent_shapes=original_latent_shapes)
    return (und_seq, gen_seq, rotary_emb, post_ctx)

def post_blocks(tf, und_seq, gen_seq, post_ctx):
    und_out = tf.norm(und_seq)
    gen_out = tf.norm_moe_gen(gen_seq)
    last_hidden_state = torch.cat([und_out, gen_out], dim=0)
    preds_vision_packed = tf.proj_out(last_hidden_state[post_ctx['vision_mse_loss_indexes']])
    preds_vision = tf._unpatchify_and_unpack_latents(preds_vision_packed, token_shapes_vision=post_ctx['vision_token_shapes'], noisy_frame_indexes_vision=post_ctx['vision_noisy_frame_indexes'], original_latent_shapes=post_ctx['original_latent_shapes'])
    return preds_vision

def batched_block_loop(tf, streamer, pre_states, reverse_streamer=None, offload_saved=False):
    N = len(pre_states)
    us = [p[0] for p in pre_states]
    vs = [p[1] for p in pre_states]
    rots = [p[2] for p in pre_states]
    for il, layer in enumerate(tf.layers):
        if not hasattr(layer, '_sb_raw_forward'):
            continue
        raw = layer._sb_raw_forward
        call_fns = [lambda lyr, u, v, _r=rots[i]: raw(lyr, u, v, _r) for i in range(N)]
        outs = apply_streamed_batch(layer, call_fns, list(zip(us, vs)), bind=layer._sb_bind, evict=layer._sb_evict, offload=offload_saved)
        us = [o[0] for o in outs]
        vs = [o[1] for o in outs]
    return list(zip(us, vs))

def batched_train_forward(tf, streamer, packs, post=True):
    streamer.new_pass()
    pre = [pre_blocks(tf, pk) for pk in packs]
    post_states = batched_block_loop(tf, streamer, [(u, v, r) for u, v, r, _c in pre])
    if not post:
        return (post_states, [c for _u, _v, _r, c in pre])
    return [post_blocks(tf, us, vs, pre[i][3]) for i, (us, vs) in enumerate(post_states)]
