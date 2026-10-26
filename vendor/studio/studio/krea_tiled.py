# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
def starts(length, tile, overlap):
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError('Tile size must exceed its non-negative overlap')
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values

def predict(pipe, latents, prompt_embeds, timestep, position_ids, prompt_mask, *, grid_height, grid_width, tile, overlap, should_cancel=None):
    import torch
    if latents.shape[1] != grid_height * grid_width:
        raise ValueError('Packed latent does not match the complete image grid')
    pixels = latents.reshape(latents.shape[0], grid_height, grid_width, -1)
    total = torch.zeros_like(pixels, dtype=torch.float32)
    weights = torch.zeros((1, grid_height, grid_width, 1), device=latents.device, dtype=torch.float32)
    text_len = prompt_embeds.shape[1]
    image_ids = position_ids[text_len:].reshape(grid_height, grid_width, 3)
    for y in starts(grid_height, tile, overlap):
        for x in starts(grid_width, tile, overlap):
            if should_cancel and should_cancel():
                raise RuntimeError('generation cancelled')
            h, w = (min(tile, grid_height - y), min(tile, grid_width - x))
            chunk = pixels[:, y:y + h, x:x + w].reshape(latents.shape[0], h * w, -1)
            ids = torch.cat((position_ids[:text_len], image_ids[y:y + h, x:x + w].reshape(-1, 3)))
            result = pipe.transformer(hidden_states=chunk, encoder_hidden_states=prompt_embeds, timestep=timestep, position_ids=ids, encoder_attention_mask=prompt_mask, return_dict=False)[0]
            wy = torch.hann_window(h + 2, periodic=False, device=latents.device, dtype=torch.float32)[1:-1]
            wx = torch.hann_window(w + 2, periodic=False, device=latents.device, dtype=torch.float32)[1:-1]
            weight = (wy[:, None] * wx[None, :])[None, :, :, None]
            total[:, y:y + h, x:x + w] += result.reshape(latents.shape[0], h, w, -1).float() * weight
            weights[:, y:y + h, x:x + w] += weight
    return (total / weights).reshape_as(latents).to(latents.dtype)
