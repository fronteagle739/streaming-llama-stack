# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_activation_chunk.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
_CHUNK_STATE = {'chunk_tokens': 0, 'min_tokens': 0, 'patched': False, 'targets': []}
_ALLOW_NO_DISPATCHER: dict = {}

def _make_chunked(orig):

    def chunked(self, x, modality_dispatcher=None):
        import torch
        ct = int(_CHUNK_STATE['chunk_tokens'])
        mt = int(_CHUNK_STATE['min_tokens'])
        L = x.shape[0]
        if ct <= 0 or L <= ct or L <= mt:
            return orig(self, x, modality_dispatcher)
        if modality_dispatcher is None:
            if not _ALLOW_NO_DISPATCHER.get(id(orig)):
                return orig(self, x, modality_dispatcher)
            return torch.cat([orig(self, x[s:min(s + ct, L)], None) for s in range(0, L, ct)], dim=0)
        mm = getattr(modality_dispatcher, 'permuted_modality_mapping', None)
        num_mod = getattr(modality_dispatcher, 'num_modalities', None)
        if mm is None or num_mod is None:
            return orig(self, x, modality_dispatcher)
        from inference.model.magi2_preview import ModalityDispatcher
        outs = []
        for s in range(0, L, ct):
            e = min(s + ct, L)
            sub = ModalityDispatcher(mm[s:e].contiguous(), int(num_mod))
            outs.append(orig(self, x[s:e], sub))
        return torch.cat(outs, dim=0)
    chunked._magi2_orig = orig
    return chunked

def install_chunked_shared_mlp(chunk_tokens: int=8192, min_tokens: int=20000, emit=None) -> dict:
    report = {'patched': False, 'chunk_tokens': chunk_tokens, 'min_tokens': min_tokens, 'targets': [], 'note': None}
    if chunk_tokens is None or int(chunk_tokens) <= 0:
        _CHUNK_STATE['chunk_tokens'] = 0
        report['note'] = 'chunking disabled (chunk_tokens<=0); reference forwards'
        return report
    try:
        from inference.model.magi2_preview import MLP, MultiHeadMoELayer, MultiModalityRMSNorm
    except Exception as exc:
        report['note'] = f'cannot import reference model: {type(exc).__name__}: {exc}'
        return report
    _CHUNK_STATE['chunk_tokens'] = int(chunk_tokens)
    _CHUNK_STATE['min_tokens'] = int(min_tokens)
    if not _CHUNK_STATE['patched']:
        targets = ((MLP, 'forward', False), (MultiHeadMoELayer, '_shared_expert_forward', False), (MultiModalityRMSNorm, 'forward', True))
        for cls, meth, no_disp in targets:
            orig = getattr(cls, meth)
            if no_disp:
                _ALLOW_NO_DISPATCHER[id(orig)] = True
            setattr(cls, meth, _make_chunked(orig))
            _CHUNK_STATE['targets'].append(f'{cls.__name__}.{meth}')
        _CHUNK_STATE['patched'] = True
    report['patched'] = True
    report['targets'] = list(_CHUNK_STATE['targets'])
    report['note'] = f"shared-expert MLP token-chunked at {chunk_tokens} rows on {report['targets']} (sequences <= {min_tokens} tokens use the reference path)"
    if emit is not None:
        emit.emit('stage_log', stage='mlp_token_chunk', message=report['note'], chunk_tokens=chunk_tokens, min_tokens=min_tokens, targets=report['targets'])
    else:
        print(f"[MAGI2-MLP-CHUNK] {report['note']}", flush=True)
    return report

def uninstall_chunked_shared_mlp() -> None:
    if _CHUNK_STATE['patched']:
        try:
            from inference.model.magi2_preview import MLP, MultiHeadMoELayer, MultiModalityRMSNorm
            for cls, meth in ((MLP, 'forward'), (MultiHeadMoELayer, '_shared_expert_forward'), (MultiModalityRMSNorm, 'forward')):
                cur = getattr(cls, meth, None)
                orig = getattr(cur, '_magi2_orig', None)
                if orig is not None:
                    setattr(cls, meth, orig)
        except Exception:
            pass
    _CHUNK_STATE['patched'] = False
    _CHUNK_STATE['chunk_tokens'] = 0
    _CHUNK_STATE['targets'] = []
