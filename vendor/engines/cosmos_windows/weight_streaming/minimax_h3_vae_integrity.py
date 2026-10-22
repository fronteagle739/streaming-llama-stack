# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_vae_integrity.py
# Machine fallback paths cleared; no model data included.
import contextlib
import math
import torch

def verify_decoder_upload(model, index, paths, *, chunk_bytes=4 * 1024 * 1024):
    from .minimax_h3_vae_mux import _meta_dtype, _model_name
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    paths = set(paths)
    state = model.state_dict()
    mismatches, checked_bytes, seen = ([], 0, set())
    with contextlib.ExitStack() as stack:
        sources = {}
        for ref in index.refs.values():
            name = _model_name(ref.name)
            if name not in paths:
                continue
            seen.add(name)
            actual = state[name].detach().reshape(-1)
            dtype = _meta_dtype(ref.dtype)
            width = torch.empty((), dtype=dtype).element_size()
            if actual.is_meta or actual.numel() != math.prod(ref.shape):
                mismatches.append(name)
                continue
            if ref.file not in sources:
                sources[ref.file] = stack.enter_context(open(ref.file, 'rb'))
            source = sources[ref.file]
            source.seek(ref.offset)
            elements = max(1, chunk_bytes // width)
            for start in range(0, actual.numel(), elements):
                count = min(elements, actual.numel() - start)
                raw = bytearray(source.read(count * width))
                if len(raw) != count * width:
                    raise IOError(f'Short VAE verification read: {name}')
                expected = torch.frombuffer(raw, dtype=dtype).to(actual.dtype)
                observed = actual[start:start + count].view(torch.uint8).cpu()
                checked_bytes += count * width
                if not torch.equal(observed, expected.view(torch.uint8)):
                    mismatches.append(name)
                    break
    unverified = sorted(paths - seen)
    return dict(mismatches=mismatches, unverified=unverified, byte_mismatch_count=len(mismatches), unverified_count=len(unverified), checked_tensors=len(seen), checked_bytes=checked_bytes)
