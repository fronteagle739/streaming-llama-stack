# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_vae_decode_config.py
# Machine fallback paths cleared; no model data included.
import math
import os

def decode_tile_size():
    value = int(os.environ.get('MINIMAX_H3_VAE_DECODE_TILE', '0'))
    if value != 0 and (value < 256 or value > 448 or value % 16):
        raise ValueError('MINIMAX_H3_VAE_DECODE_TILE must be 0 or a multiple of 16 in [256, 448]')
    return value

def decode_stack_tiles():
    value = os.environ.get('MINIMAX_H3_VAE_DECODE_STACK', '0')
    if value not in ('0', '1'):
        raise ValueError('MINIMAX_H3_VAE_DECODE_STACK must be 0 or 1')
    return value == '1'

def decode_reuse_workspace():
    value = os.environ.get('MINIMAX_H3_VAE_DECODE_REUSE', '0')
    if value not in ('0', '1'):
        raise ValueError('MINIMAX_H3_VAE_DECODE_REUSE must be 0 or 1')
    return value == '1'

def decode_offload_weights():
    value = os.environ.get('MINIMAX_H3_VAE_DECODE_OFFLOAD', '0')
    if value not in ('0', '1'):
        raise ValueError('MINIMAX_H3_VAE_DECODE_OFFLOAD must be 0 or 1')
    return value == '1'

def decode_cache_weights():
    value = os.environ.get('MINIMAX_H3_VAE_DECODE_CACHE', '0')
    if value not in ('0', '1'):
        raise ValueError('MINIMAX_H3_VAE_DECODE_CACHE must be 0 or 1')
    return value == '1'

def configure_video_decode(model, *, width, height):
    tile = decode_tile_size()
    if tile:
        if tile < model.decoder_tile_size or tile % model.vae_ratio:
            raise ValueError('MINIMAX_H3_VAE_DECODE_TILE must enlarge the released tile and align to vae_ratio')
        model.decoder_tile_size = tile
    if decode_stack_tiles():
        if max(width, height) > 448:
            raise ValueError('Stacked H3 decode is limited to canvases up to 448 pixels per side')
        model.stack_tiling = True

def decode_tile_extra_bytes(width=448, height=448):
    tile = decode_tile_size()
    tile = tile or 256
    stack = 4 if decode_stack_tiles() else 1
    area = min(width, tile) * min(height, tile)
    original = min(width, 256) * min(height, 256)
    if decode_stack_tiles() and max(width, height) > 448:
        raise ValueError('Stacked H3 decode is limited to canvases up to 448 pixels per side')
    return math.ceil(500000000 * max(0, area / original * stack - 1)) + (500000000 if decode_reuse_workspace() else 0) + (decode_cache_limit_bytes() if decode_cache_weights() and (not decode_offload_weights()) else 0)

def decode_cache_limit_bytes():
    from .minimax_h3_vae_fast import decoder_mode
    return 2750000000 if decoder_mode() == 'int8_convrot' else 1500000000
