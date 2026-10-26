# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations

def full_lokr_bases(keys: set[str]) -> list[str]:
    bases = sorted({key.rsplit('.', 1)[0] for key in keys if '.lokr_' in key})
    for base in bases:
        required = {base + '.lokr_w1', base + '.lokr_w2'}
        present = {key for key in keys if key.startswith(base + '.lokr_')}
        if present != required or base + '.dora_scale' in keys:
            raise RuntimeError(f'streamed Krea2 LoKr requires full lokr_w1/lokr_w2 Linear factors; unsupported or incomplete variant at {base!r}')
    return bases

def load_full_lokr(handle, base: str, in_features: int, out_features: int, device, dtype):
    w1, w2 = (handle.get_tensor(base + '.lokr_w1'), handle.get_tensor(base + '.lokr_w2'))
    if w1.ndim != 2 or w2.ndim != 2 or w1.shape[0] * w2.shape[0] != out_features or (w1.shape[1] * w2.shape[1] != in_features):
        raise RuntimeError(f'LoKr tensor shapes {tuple(w1.shape)}/{tuple(w2.shape)} do not match Linear ({out_features}, {in_features}) at {base!r}')
    return (w1.to(device=device, dtype=dtype), w2.to(device=device, dtype=dtype))

def full_lokr_linear(input, w1, w2):
    import torch.nn.functional as F
    w1, w2 = (w1.to(input.device, input.dtype), w2.to(input.device, input.dtype))
    grouped = input.reshape(*input.shape[:-1], w1.shape[1], w2.shape[1])
    inner = F.linear(grouped, w2)
    outer = F.linear(inner.transpose(-1, -2), w1).transpose(-1, -2)
    return outer.reshape(*input.shape[:-1], w1.shape[0] * w2.shape[0])
