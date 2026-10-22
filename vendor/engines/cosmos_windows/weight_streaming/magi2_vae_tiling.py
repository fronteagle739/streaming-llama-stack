# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_vae_tiling.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
Span = Tuple[int, int]

def plan_1d_tiles(size: int, tile: int, overlap: int) -> List[Span]:
    if size <= 0:
        raise ValueError(f'size must be > 0, got {size}')
    if tile <= 0:
        raise ValueError(f'tile must be > 0, got {tile}')
    if overlap < 0 or overlap >= tile:
        raise ValueError(f'overlap must be in [0, tile); got overlap={overlap}, tile={tile}')
    if tile >= size:
        return [(0, size)]
    step = tile - overlap
    tiles: List[Span] = []
    start = 0
    while True:
        end = min(start + tile, size)
        tiles.append((start, end))
        if end >= size:
            break
        start += step
    return tiles

@dataclass(frozen=True)
class SpatialTile:
    h: Span
    w: Span
    is_h_first: bool
    is_h_last: bool
    is_w_first: bool
    is_w_last: bool

def plan_spatial_tiles(h_lat: int, w_lat: int, tile_lat: int, overlap_lat: int) -> List[SpatialTile]:
    hs = plan_1d_tiles(h_lat, tile_lat, overlap_lat)
    ws = plan_1d_tiles(w_lat, tile_lat, overlap_lat)
    out: List[SpatialTile] = []
    for i, (h0, h1) in enumerate(hs):
        for j, (w0, w1) in enumerate(ws):
            out.append(SpatialTile(h=(h0, h1), w=(w0, w1), is_h_first=i == 0, is_h_last=i == len(hs) - 1, is_w_first=j == 0, is_w_last=j == len(ws) - 1))
    return out

def plan_temporal_chunks(t_lat: int, first_chunk: int, step: int) -> List[Span]:
    if t_lat <= 0:
        raise ValueError(f't_lat must be > 0, got {t_lat}')
    if first_chunk <= 0 or step <= 0:
        raise ValueError('first_chunk and step must be > 0')
    chunks: List[Span] = []
    start = 0
    size = first_chunk
    while start < t_lat:
        end = min(start + size, t_lat)
        chunks.append((start, end))
        start = end
        size = step
    return chunks

def blend_ramp_profile(length: int, ramp_left: int, ramp_right: int) -> List[float]:
    if length <= 0:
        raise ValueError('length must be > 0')
    w = [1.0] * length
    rl = min(ramp_left, length)
    rr = min(ramp_right, length)
    for i in range(rl):
        w[i] = min(w[i], (i + 1) / (rl + 1))
    for i in range(rr):
        w[length - 1 - i] = min(w[length - 1 - i], (i + 1) / (rr + 1))
    return w

def estimate_tile_peak_bytes(tile_h_lat: int, tile_w_lat: int, t_pixels: int, *, spatial_upscale: int, hidden_channels: int, dtype_bytes: int, activation_factor: float=6.0) -> int:
    ph = tile_h_lat * spatial_upscale
    pw = tile_w_lat * spatial_upscale
    return int(ph * pw * max(1, t_pixels) * hidden_channels * dtype_bytes * activation_factor)

def choose_tile_lat(h_lat: int, w_lat: int, *, spatial_upscale: int, hidden_channels: int, dtype_bytes: int, budget_bytes: int, t_pixels: int, overlap_lat: int, min_tile_lat: int=16, activation_factor: float=6.0) -> int:
    upper = max(h_lat, w_lat)
    tile = upper
    while tile > min_tile_lat:
        peak = estimate_tile_peak_bytes(min(tile, h_lat), min(tile, w_lat), t_pixels, spatial_upscale=spatial_upscale, hidden_channels=hidden_channels, dtype_bytes=dtype_bytes, activation_factor=activation_factor)
        if peak <= budget_bytes:
            return tile
        tile = max(min_tile_lat, min(tile - max(1, upper // 8), tile - 1))
    peak_min = estimate_tile_peak_bytes(min(min_tile_lat, h_lat), min(min_tile_lat, w_lat), t_pixels, spatial_upscale=spatial_upscale, hidden_channels=hidden_channels, dtype_bytes=dtype_bytes, activation_factor=activation_factor)
    if peak_min <= budget_bytes:
        if min_tile_lat <= overlap_lat:
            raise ValueError(f'min_tile_lat {min_tile_lat} must exceed overlap_lat {overlap_lat}')
        return min_tile_lat
    raise MemoryError(f'VAE decode cannot fit even a {min_tile_lat}x{min_tile_lat} latent tile (~{peak_min / 1000000000.0:.2f}GB modeled) in the {budget_bytes / 1000000000.0:.2f}GB VAE VRAM budget. Lower the resolution/clip, raise the VAE budget, or add temporal chunking — refusing rather than OOMing (no silent quality-affecting fallback).')

@dataclass(frozen=True)
class TileDecodePlan:
    tile_lat: int
    overlap_lat: int
    spatial_upscale: int
    spatial_tiles: Tuple[SpatialTile, ...]
    temporal_chunks: Tuple[Span, ...]

    def summary(self) -> dict:
        return {'tile_lat': self.tile_lat, 'overlap_lat': self.overlap_lat, 'spatial_upscale': self.spatial_upscale, 'num_spatial_tiles': len(self.spatial_tiles), 'num_temporal_chunks': len(self.temporal_chunks)}

def plan_tiled_decode(h_lat: int, w_lat: int, t_lat: int, *, spatial_upscale: int, hidden_channels: int, dtype_bytes: int, budget_bytes: int, t_pixels: int, overlap_lat: int=2, min_tile_lat: int=16, temporal: Optional[Tuple[int, int]]=None) -> TileDecodePlan:
    tile = choose_tile_lat(h_lat, w_lat, spatial_upscale=spatial_upscale, hidden_channels=hidden_channels, dtype_bytes=dtype_bytes, budget_bytes=budget_bytes, t_pixels=t_pixels, overlap_lat=overlap_lat, min_tile_lat=min_tile_lat)
    tiles = plan_spatial_tiles(h_lat, w_lat, tile, overlap_lat)
    chunks: Tuple[Span, ...] = ()
    if temporal is not None:
        chunks = tuple(plan_temporal_chunks(t_lat, temporal[0], temporal[1]))
    return TileDecodePlan(tile_lat=tile, overlap_lat=overlap_lat, spatial_upscale=spatial_upscale, spatial_tiles=tiles, temporal_chunks=chunks)

def _ramp_1d(torch, length: int, ramp_left: int, ramp_right: int, device, dtype):
    prof = blend_ramp_profile(length, ramp_left, ramp_right)
    return torch.tensor(prof, device=device, dtype=dtype)

def tiled_decode(latent, decode_fn: Callable, plan: TileDecodePlan, *, out_channels: int=3):
    import torch
    if latent.dim() != 5:
        raise ValueError(f'latent must be [B,C,T,H,W]; got shape {tuple(latent.shape)}')
    B, C, T, Hl, Wl = latent.shape
    su = plan.spatial_upscale
    ov_px = plan.overlap_lat * su

    def _decode_span(lat_span):
        Hp, Wp = (Hl * su, Wl * su)
        out = None
        wsum = None
        for st in plan.spatial_tiles:
            h0, h1 = st.h
            w0, w1 = st.w
            tile_lat = lat_span[:, :, :, h0:h1, w0:w1]
            px = decode_fn(tile_lat)
            if out is None:
                Tp = px.shape[2]
                out = torch.zeros(B, px.shape[1], Tp, Hp, Wp, device=px.device, dtype=torch.float32)
                wsum = torch.zeros(1, 1, 1, Hp, Wp, device=px.device, dtype=torch.float32)
            ph0, ph1 = (h0 * su, h1 * su)
            pw0, pw1 = (w0 * su, w1 * su)
            rh = _ramp_1d(torch, ph1 - ph0, 0 if st.is_h_first else ov_px, 0 if st.is_h_last else ov_px, px.device, torch.float32)
            rw = _ramp_1d(torch, pw1 - pw0, 0 if st.is_w_first else ov_px, 0 if st.is_w_last else ov_px, px.device, torch.float32)
            w2d = torch.outer(rh, rw).view(1, 1, 1, ph1 - ph0, pw1 - pw0)
            out[:, :, :, ph0:ph1, pw0:pw1] += px.float() * w2d
            wsum[:, :, :, ph0:ph1, pw0:pw1] += w2d
            del px
        out /= wsum.clamp_min(1e-08)
        return out
    if not plan.temporal_chunks:
        return _decode_span(latent)
    parts = [_decode_span(latent[:, :, c0:c1, :, :]) for c0, c1 in plan.temporal_chunks]
    return torch.cat(parts, dim=2)
