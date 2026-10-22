# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/streamed_text_encoder.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import re
import threading
import time
from typing import Callable

def host_rss_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

class RssProbe:

    def __init__(self, label: str='', interval_s: float=0.05, abort_mib: float | None=None):
        self.label = label
        self.interval_s = interval_s
        self.abort_mib = abort_mib
        self.baseline = 0.0
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._aborted = False

    def _run(self):
        while not self._stop.is_set():
            rss = host_rss_mib()
            if rss > self.peak:
                self.peak = rss
            if self.abort_mib is not None and rss - self.baseline > self.abort_mib:
                self._aborted = True
                print(f'[RSS-PROBE] ABORT-THRESHOLD HIT for {self.label!r}: RSS delta {rss - self.baseline:.0f} MiB > {self.abort_mib:.0f} MiB (peak {rss:.0f} MiB). This load is staging host RAM, not streaming.', flush=True)
                return
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.baseline = host_rss_mib()
        self.peak = self.baseline
        if self.baseline > 0.0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        rss = host_rss_mib()
        if rss > self.peak:
            self.peak = rss
        return False

    @property
    def delta_mib(self) -> float:
        return max(0.0, self.peak - self.baseline)

    @property
    def aborted(self) -> bool:
        return self._aborted

    def summary(self) -> str:
        if self.baseline <= 0.0:
            return f'[RSS-PROBE] {self.label}: psutil unavailable (probe disabled)'
        return f"[RSS-PROBE] {self.label}: baseline {self.baseline:.0f} MiB, peak {self.peak:.0f} MiB, delta {self.delta_mib:.0f} MiB{(' (ABORTED)' if self._aborted else '')}"

def build_streamed_text_encoder(encoder, model_dir: str, *, layer_regex: 're.Pattern | str', decoder_layers, ckpt_to_module: Callable[[str], 'str | None'], device: str='cuda', dtype=None, tie_lm_head_from: str | None=None, tie_map: 'dict[str, str] | None'=None, tensor_transform: 'Callable | None'=None, log_prefix: str='[TE-STREAM]'):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .safetensors_index import build_index
    dtype = dtype or torch.bfloat16
    layer_re = re.compile(layer_regex) if isinstance(layer_regex, str) else layer_regex
    sd_keys = set(encoder.state_dict().keys())
    idx = build_index(str(model_dir))
    layer_refs: dict[int, list] = {}
    global_refs: list = []
    for name, ref in idx.refs.items():
        m = layer_re.search(name)
        if m:
            layer_refs.setdefault(int(m.group(1)), []).append(ref)
        else:
            global_refs.append(ref)
    pb = PortableBackend(pin_memory=True)

    def _read(ref):
        t = pb.read_tensor(ref, device, direct=True)
        if tensor_transform is not None:
            t = tensor_transform(idx, ref, t)
        if t.is_floating_point():
            t = t.to(dtype)
        return t
    n_glob = 0
    loaded_modkeys: set[str] = set()
    for ref in global_refs:
        modk = ckpt_to_module(ref.name)
        if modk is None or modk not in sd_keys:
            continue
        t = _read(ref)
        set_module_tensor_to_device(encoder, modk, device, value=t, dtype=dtype if t.is_floating_point() else None)
        loaded_modkeys.add(modk)
        n_glob += 1
        del t
    effective_tie_map = dict(tie_map or {})
    if tie_lm_head_from:
        effective_tie_map.setdefault('lm_head.weight', tie_lm_head_from)
    state = encoder.state_dict()
    for dst, src in effective_tie_map.items():
        if dst in sd_keys and dst not in loaded_modkeys and (src in loaded_modkeys) and (state[src].device.type != 'meta'):
            set_module_tensor_to_device(encoder, dst, device, value=state[src], dtype=dtype)
            loaded_modkeys.add(dst)
    n_layers = len(layer_refs)
    layer_plan: dict[int, list[tuple[str, object]]] = {}
    for il, refs in layer_refs.items():
        plan = []
        for ref in refs:
            modk = ckpt_to_module(ref.name)
            if modk is not None and modk in sd_keys:
                plan.append((modk, ref))
        layer_plan[il] = plan

    def make_bind(il):

        def bind():
            for modk, ref in layer_plan[il]:
                t = _read(ref)
                set_module_tensor_to_device(encoder, modk, device, value=t, dtype=dtype if t.is_floating_point() else None)
        return bind

    def make_evict(il):

        def evict():
            for modk, _ref in layer_plan[il]:
                set_module_tensor_to_device(encoder, modk, 'meta')
        return evict
    for il, layer in enumerate(decoder_layers):
        cls_forward = type(layer).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(hidden_states, *a, _layer=layer, _fwd=cls_forward, _b=bind, _e=evict, **kw):
            _b()
            try:
                out = _fwd(_layer, hidden_states, *a, **kw)
            finally:
                _e()
            return out
        layer.forward = wrapped
        layer._sb_raw_forward = cls_forward
        layer._sb_bind = bind
        layer._sb_evict = evict
    streamed_modkeys = {modk for plan in layer_plan.values() for modk, _r in plan}
    meta_missing = [name for name, p in encoder.named_parameters() if p.device.type == 'meta' and name not in streamed_modkeys]
    if meta_missing:
        raise RuntimeError(f'{log_prefix} {len(meta_missing)} non-streamed tensors left on meta (would crash forward): {meta_missing[:8]}')
    encoder.eval()
    print(f'{log_prefix} streamed {n_layers} decoder layers off-NVMe, {n_glob} resident globals; peak text-encode VRAM ~= globals + a couple layers, not the full encoder.', flush=True)
    encoder._sb_streamed = True
    encoder._sb_reader = pb
    return encoder
