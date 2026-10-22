# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/loader.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import torch
from accelerate.utils import set_module_tensor_to_device
from .backends import get_backend

class StreamingWeightLoader:

    def __init__(self, model, index, component='transformer', backend='auto', window=2, device='cuda'):
        self.model = model
        self.device = device
        self.window = window
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.layer_refs, self.global_refs = index.group_by_layer(component)
        sd_keys = set(self.model.state_dict().keys())
        skipped = []
        for i in list(self.layer_refs.keys()):
            kept = [r for r in self.layer_refs[i] if self._param_path(r) in sd_keys]
            skipped += [self._param_path(r) for r in self.layer_refs[i] if self._param_path(r) not in sd_keys]
            self.layer_refs[i] = kept
        kept_globals = [r for r in self.global_refs if self._param_path(r) in sd_keys]
        skipped += [self._param_path(r) for r in self.global_refs if self._param_path(r) not in sd_keys]
        self.global_refs = kept_globals
        if skipped:
            print(f'[StreamingWeightLoader] skipping {len(skipped)} checkpoint tensors not in the module tree (e.g. {skipped[:3]}) — expected for action-gen heads under a T2I/T2V config.')
        self.n = len(self.layer_refs)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.resident: dict[int, dict[str, torch.Tensor]] = {}
        self.prefetched: dict[int, dict[str, torch.Tensor]] = {}
        self.copy_done: dict[int, torch.cuda.Event] = {}
        self.compute_done: dict[int, torch.cuda.Event] = {}
        self._handles = []

    @staticmethod
    def _param_path(ref) -> str:
        return ref.name.split('/', 1)[1] if '/' in ref.name else ref.name

    def materialize_globals(self):
        for ref in self.global_refs:
            t = self.backend.read_tensor(ref, self.device)
            set_module_tensor_to_device(self.model, self._param_path(ref), self.device, value=t)

    def _load_layer(self, i, into):
        out = {}
        with torch.cuda.stream(self.copy_stream):
            for ref in self.layer_refs[i]:
                out[self._param_path(ref)] = self.backend.read_tensor(ref, self.device, self.copy_stream)
        ev = torch.cuda.Event()
        ev.record(self.copy_stream)
        self.copy_done[i] = ev
        into[i] = out

    def _bind(self, i):
        for path, t in self.resident[i].items():
            set_module_tensor_to_device(self.model, path, self.device, value=t)

    def _evict(self, i):
        if i not in self.resident:
            return
        if i in self.compute_done:
            self.compute_done[i].synchronize()
        for path in self.resident[i]:
            set_module_tensor_to_device(self.model, path, 'meta')
        del self.resident[i]

    def _ensure(self, i):
        if i in self.resident:
            return
        if i in self.prefetched:
            self.resident[i] = self.prefetched.pop(i)
        else:
            self._load_layer(i, self.resident)
        self.copy_done[i].synchronize()
        self._bind(i)

    def _prefetch(self, i):
        for d in range(1, self.window + 1):
            j = (i + d) % self.n
            if j not in self.resident and j not in self.prefetched:
                self._load_layer(j, self.prefetched)

    def _pre(self, i):

        def hook(mod, args):
            self._ensure(i)
            torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
            self._prefetch(i)
        return hook

    def _post(self, i):

        def hook(mod, args, output):
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self.compute_done[i] = ev
            self._evict((i - self.window) % self.n)
            return output
        return hook

    def register(self):
        for i, layer in enumerate(self.model.layers):
            self._handles.append(layer.register_forward_pre_hook(self._pre(i)))
            self._handles.append(layer.register_forward_hook(self._post(i)))
        return self

    def unregister(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
