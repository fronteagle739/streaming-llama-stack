# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_moe_binding.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
CORE_MOE_PARAMS: Tuple[str, ...] = ('W_gate', 'W_up', 'W_down', 'gate', 'router.expert_bias', 'router.expert_bias_ema')
PREVIEW_HIDDEN = 3072
PREVIEW_MOE_HEADS = 12
PREVIEW_NUM_EXPERTS = 256
PREVIEW_TOP_K = 6
PREVIEW_D_EXPERT = 1280
PREVIEW_ROUTE_SCALE = 4.9
PREVIEW_NUM_LAYERS = 40

def core_moe_ckpt_name(layer_idx: int, attr: str) -> str:
    return f'block.layers.{layer_idx}.mlp.moe_mlp.{attr}'

def install_moe_kernel_sm120_patch(emit=None) -> dict:
    report = {'patched': False, 'reason': None, 'config': None}
    try:
        import torch
        import inference.flash_mh_moe.triton.mh_moe_fwd as K
    except Exception as exc:
        report['reason'] = f'cannot import MoE kernel: {type(exc).__name__}: {exc}'
        return report
    if getattr(K, '_magi2_sm120_patched', False):
        report.update(patched=True, reason='already patched', config=K._magi2_sm120_config)
        return report
    if not torch.cuda.is_available():
        report['reason'] = 'no CUDA device'
        return report
    major, _ = torch.cuda.get_device_capability(0)
    optin = None
    try:
        props = torch.cuda.get_device_properties(0)
        optin = getattr(props, 'shared_memory_per_block_optin', None) or getattr(props, 'sharedMemPerBlockOptin', None)
    except Exception:
        optin = None
    needs_fix = optin is not None and optin < 120 * 1024 or (optin is None and major >= 12)
    orig = K._select_block_config
    if not needs_fix:
        report['reason'] = f'arch major={major} optin={optin} has enough shared memory; unpatched'
        return report
    bt, bdh, bde, _ns, nw = orig()
    safe = (bt, bdh, bde, 1, nw)

    def _sm120_select_block_config():
        return safe
    K._select_block_config = _sm120_select_block_config
    K._magi2_sm120_patched = True
    K._magi2_sm120_config = safe
    report.update(patched=True, reason=f'sm_{major}0 shared-mem cap (optin={optin}) -> num_stages=1', config=safe, original=(bt, bdh, bde, _ns, nw))
    if emit is not None:
        emit.emit('stage_log', stage='moe_kernel_patch', message=f'MoE Triton kernel patched for SM120 (num_stages 2->1): {safe}')
    return report

def build_preview_moe_config(*, num_layers: int=PREVIEW_NUM_LAYERS, params_dtype=None):
    import torch
    from inference.model.magi2_preview import CoreMultiHeadMoEConfig
    return CoreMultiHeadMoEConfig(hidden_size=PREVIEW_HIDDEN, num_heads=PREVIEW_MOE_HEADS, num_experts=PREVIEW_NUM_EXPERTS, top_k=PREVIEW_TOP_K, expert_intermediate_size=PREVIEW_D_EXPERT, num_layers=num_layers, params_dtype=params_dtype if params_dtype is not None else torch.bfloat16, score_func='sigmoid', route_norm=True, route_scale=PREVIEW_ROUTE_SCALE)

def make_core_moe(*, num_layers: int=PREVIEW_NUM_LAYERS, params_dtype=None):
    import torch
    from inference.model.magi2_preview import CoreMultiHeadMoE
    cfg = build_preview_moe_config(num_layers=num_layers, params_dtype=params_dtype)
    with torch.device('meta'):
        moe = CoreMultiHeadMoE(cfg)
    if moe.ep_size != 1:
        raise RuntimeError(f'CoreMultiHeadMoE.ep_size={moe.ep_size}; single-GPU streaming requires ep_size==1 (no expert-parallel process group). Do not initialise psm groups.')
    return moe

def bind_core_moe(moe, views: Dict[str, object], device: str) -> None:
    from accelerate.utils import set_module_tensor_to_device
    for attr, tensor in views.items():
        set_module_tensor_to_device(moe, attr, device, value=tensor)

def evict_core_moe(moe) -> None:
    from accelerate.utils import set_module_tensor_to_device
    for attr in CORE_MOE_PARAMS:
        set_module_tensor_to_device(moe, attr, 'meta')

def read_core_moe_views_direct(preview_dir, layer_idx: int, device: str) -> Dict[str, object]:
    from pathlib import Path
    from safetensors import safe_open
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    index = build_index(Path(preview_dir))
    out: Dict[str, object] = {}
    by_file: Dict[str, List[Tuple[str, str]]] = {}
    for attr in CORE_MOE_PARAMS:
        name = core_moe_ckpt_name(layer_idx, attr)
        ref = index[name]
        by_file.setdefault(ref.file, []).append((attr, name))
    for file, items in by_file.items():
        with safe_open(file, framework='pt', device=device) as f:
            for attr, name in items:
                out[attr] = f.get_tensor(name)
    return out

def core_moe_views_from_layer(layer_views: Dict[str, object], layer_idx: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for attr in CORE_MOE_PARAMS:
        name = core_moe_ckpt_name(layer_idx, attr)
        if name not in layer_views:
            raise KeyError(f'streamed layer {layer_idx} is missing {name}; got {sorted(layer_views)[:6]}...')
        out[attr] = layer_views[name]
    return out

def _module_layers(transformer):
    block = getattr(transformer, 'block', None)
    layers = getattr(block, 'layers', None) if block is not None else None
    if layers is None:
        raise ValueError('preview Transformer must expose .block.layers (a ModuleList)')
    return layers

class InLoopLayerVerifier:
    HEAD_BYTES = 128

    def __init__(self, index, tensors_per_layer: int=2, elems: int=8, emit=None):
        import os as _o
        self.index = index
        self.tensors_per_layer = int(_o.environ.get('MAGI2_VERIFY_TENSORS', '8'))
        self.offsets = _o.environ.get('MAGI2_VERIFY_OFFSETS', 'head,mid,tail').split(',')
        self.elems = elems
        self.emit = emit
        self._fh: Dict[str, object] = {}
        self._expected: Dict[str, bytes] = {}
        self._picks: Dict[int, List[str]] = {}
        self.checked = 0
        self.mismatches: List[str] = []

    def _probe_offsets(self, nbytes: int) -> List[tuple]:
        n = min(self.HEAD_BYTES, nbytes)
        out = []
        for label in self.offsets:
            label = label.strip()
            if label == 'head':
                off = 0
            elif label == 'mid':
                off = max(0, nbytes // 2 - n // 2)
            elif label == 'tail':
                off = max(0, nbytes - n)
            else:
                continue
            out.append((label, off))
        return out or [('head', 0)]

    def _expected_sample(self, name, byte_off: int=0) -> bytes:
        key = f'{name}@{byte_off}'
        if key not in self._expected:
            ref = self.index[name]
            fh = self._fh.get(ref.file)
            if fh is None:
                fh = open(ref.file, 'rb', buffering=0)
                self._fh[ref.file] = fh
            n = min(self.HEAD_BYTES, max(0, ref.nbytes - byte_off))
            fh.seek(ref.offset + byte_off)
            buf = bytearray(n)
            got = fh.readinto(memoryview(buf))
            if got != n:
                raise RuntimeError(f'verifier short read for {name}@{byte_off}: {got} != {n}')
            self._expected[key] = bytes(buf)
        return self._expected[key]

    def _identify_delivered(self, name: str, got: bytes) -> str:
        import re
        pat = re.sub('layers\\.\\d+\\.', 'layers.N.', name)
        hits = []
        for cand in self.index.refs:
            if re.sub('layers\\.\\d+\\.', 'layers.N.', cand) == pat and cand != name:
                try:
                    if self._expected_sample(cand) == got:
                        hits.append(cand)
                except Exception:
                    pass
        return f'delivered bytes belong to {hits}' if hits else "delivered bytes match NO layer's head bytes (torn transfer or garbage)"

    def _pick_names(self, layer_idx: int, views: Dict[str, object]) -> List[str]:
        if layer_idx in self._picks:
            return self._picks[layer_idx]
        allnames = sorted((n for n in views if n in self.index.refs))
        k = max(1, min(self.tensors_per_layer, len(allnames)))
        experts = [n for n in allnames if 'expert' in n.lower()]
        others = [n for n in allnames if 'expert' not in n.lower()]
        picks: List[str] = []
        if experts:
            n_exp = max(1, k // 2)
            stride = max(1, len(experts) // n_exp)
            picks.extend(experts[::stride][:n_exp])
        n_oth = k - len(picks)
        if n_oth > 0 and others:
            stride = max(1, len(others) // n_oth)
            picks.extend(others[::stride][:n_oth])
        self._picks[layer_idx] = picks or allnames[:k]
        return self._picks[layer_idx]

    def check_layer(self, layer_idx: int, views: Dict[str, object]) -> None:
        import torch
        for name in self._pick_names(layer_idx, views):
            v = views[name]
            nbytes = self.index[name].nbytes
            for label, off in self._probe_offsets(nbytes):
                exp = self._expected_sample(name, off)
                if not exp:
                    continue
                flat_u8 = v.reshape(-1).view(torch.uint8)[off:off + len(exp)]
                got = bytes(flat_u8.cpu().numpy())
                if exp == got:
                    self.checked += 1
                    continue
                whose = self._identify_delivered(name, got) if off == 0 else f'probe at {label} offset {off} of {nbytes} bytes differs while earlier probes matched -> TRUNCATED/PARTIAL transfer, not a wrong-layer serve'
                msg = f'LAYER-DELIVERY MISMATCH at layer {layer_idx}, tensor {name!r} probe {label}@{off}: the ring delivered bytes that do NOT match the byte-index ground truth (expected {list(exp[:8])}, got {list(got[:8])}). {whose}. This is finite-but-WRONG weight delivery (wrong layer / stale buffer / truncated transfer), the signature of a black render from a run that reports success.'
                self.mismatches.append(msg)
                if self.emit is not None:
                    self.emit(event='layer_verify_failed', layer=layer_idx, tensor=name, probe=label, byte_offset=off, delivered_from=whose, message=msg)
                raise RuntimeError(msg)

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass
        self._fh.clear()

def build_streamed_magi2_preview(transformer, index, streamer, *, device: str='cuda', emit=None):
    from accelerate.utils import set_module_tensor_to_device
    layers = _module_layers(transformer)
    layer_refs, global_refs = index.group_by_layer('')
    state_keys = set(transformer.state_dict().keys())
    missing_globals = []
    for ref in global_refs:
        if ref.name in state_keys:
            from safetensors import safe_open
            with safe_open(ref.file, framework='pt', device=device) as f:
                set_module_tensor_to_device(transformer, ref.name, device, value=f.get_tensor(ref.name))
        else:
            missing_globals.append(ref.name)
    if missing_globals and emit is not None:
        emit.emit('stage_log', stage='bind_globals', message=f'{len(missing_globals)} indexed global tensors not in model state (e.g. {missing_globals[:4]})')
    layer_paths: Dict[int, List[str]] = {lid: [r.name for r in refs if r.name in state_keys] for lid, refs in layer_refs.items()}
    import os as _os
    overlap = _os.environ.get('MAGI2_RING_OVERLAP', '0').strip().lower() not in ('0', 'false', 'no', 'off')
    ring_depth = int(getattr(getattr(streamer, 'plan', None), 'ring_depth', 2) or 2)
    n_layers = len(layers)
    _cursor = {'next': 0}
    _verifier = None
    if _os.environ.get('MAGI2_VERIFY_LAYERS', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
        _verifier = InLoopLayerVerifier(index, emit=emit.emit if hasattr(emit, 'emit') else emit)
        print('[MAGI2-VERIFY] in-loop layer delivery verification ENABLED', flush=True)

    def make_bind(layer_idx):

        def bind():
            if not overlap:
                streamer.prefetch(layer_idx)
            views = streamer.wait_layer(layer_idx)
            if _verifier is not None:
                _verifier.check_layer(layer_idx, views)
            for name, tensor in views.items():
                if name in state_keys:
                    set_module_tensor_to_device(transformer, name, device, value=tensor)
        return bind

    def make_evict(layer_idx):

        def evict():
            for name in layer_paths[layer_idx]:
                set_module_tensor_to_device(transformer, name, 'meta')
            streamer.evict(layer_idx)
            if overlap and _cursor['next'] < n_layers:
                streamer.prefetch(_cursor['next'])
                _cursor['next'] += 1
        return evict
    for layer_idx, layer in enumerate(layers):
        raw_forward = type(layer).forward
        bind, evict = (make_bind(layer_idx), make_evict(layer_idx))

        def wrapped(*args, _layer=layer, _forward=raw_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _forward(_layer, *args, **kwargs)
            finally:
                _evict()
        layer.forward = wrapped
        layer._magi2_stream_bind = bind
        layer._magi2_stream_evict = evict
        layer._magi2_raw_forward = raw_forward
    _boundary_empty = _os.environ.get('MAGI2_BOUNDARY_EMPTY_CACHE', '1').strip().lower() not in ('0', 'false', 'no', 'off')

    def _new_pass(_module, _args, _kwargs=None):
        if _boundary_empty:
            import torch as _t
            _t.cuda.synchronize()
            _t.cuda.empty_cache()
        streamer.new_pass()
        _cursor['next'] = 0
        if overlap:
            for _ in range(min(ring_depth, n_layers)):
                streamer.prefetch(_cursor['next'])
                _cursor['next'] += 1
    transformer.register_forward_pre_hook(_new_pass)
    transformer.to = lambda *a, **k: transformer
    transformer.eval()
    return (transformer, streamer)
