# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_shard_index.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from cosmos_windows.weight_streaming.safetensors_index import _DTYPE_SIZE, SafetensorsIndex, TensorRef, build_index
_LAYER_RE = re.compile('(?:^|[/.])layers\\.(\\d+)\\.')
_EXPERT_RE = re.compile('moe_mlp\\.W_(gate|up|down)$')
MAGI2_NUM_EXPERTS = 256
MAGI2_NUM_MOE_HEADS = 12
MAGI2_PROJS = ('gate', 'up', 'down')

@dataclass(frozen=True)
class ExpertRowRef:
    layer_idx: int
    flat_expert_id: int
    proj: str
    file: str
    offset: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]

@dataclass(frozen=True)
class MoELayerRef:
    layer_idx: int
    flatten: int
    num_experts: int
    num_heads: int
    tensors: Dict[str, TensorRef]

    def row_nbytes(self, proj: str) -> int:
        ref = self.tensors[proj]
        if ref.nbytes % self.flatten != 0:
            raise ValueError(f'layer {self.layer_idx} {proj}: nbytes {ref.nbytes} not divisible by flatten {self.flatten}')
        return ref.nbytes // self.flatten

    def expert_bytes(self) -> int:
        return sum((self.row_nbytes(p) for p in self.tensors))

    def layer_bytes(self) -> int:
        return sum((r.nbytes for r in self.tensors.values()))

class Magi2ShardIndex:

    def __init__(self, index: SafetensorsIndex, *, num_experts: int=MAGI2_NUM_EXPERTS, num_heads: int=MAGI2_NUM_MOE_HEADS) -> None:
        self.index = index
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.moe_layers: Dict[int, MoELayerRef] = {}
        self.non_expert: Dict[str, TensorRef] = {}
        self._build()

    def _build(self) -> None:
        staging: Dict[int, Dict[str, TensorRef]] = {}
        for name, ref in self.index.refs.items():
            em = _EXPERT_RE.search(name)
            lm = _LAYER_RE.search(name)
            if em and lm:
                layer = int(lm.group(1))
                staging.setdefault(layer, {})[em.group(1)] = ref
            else:
                self.non_expert[name] = ref
        for layer, tensors in staging.items():
            missing = [p for p in MAGI2_PROJS if p not in tensors]
            if missing:
                raise ValueError(f'MoE layer {layer} missing projections {missing}; found {sorted(tensors)}')
            flatten = tensors['gate'].shape[0]
            for p, r in tensors.items():
                if r.shape[0] != flatten:
                    raise ValueError(f'layer {layer} {p} dim0 {r.shape[0]} != gate dim0 {flatten}')
            if self.num_experts and flatten % self.num_experts != 0:
                raise ValueError(f'layer {layer} flatten {flatten} not divisible by num_experts {self.num_experts}')
            num_heads = flatten // self.num_experts if self.num_experts else self.num_heads
            self.moe_layers[layer] = MoELayerRef(layer_idx=layer, flatten=flatten, num_experts=self.num_experts, num_heads=num_heads, tensors=tensors)

    def moe_layer_ids(self) -> List[int]:
        return sorted(self.moe_layers)

    def flat_id(self, head: int, expert: int) -> int:
        if not 0 <= expert < self.num_experts:
            raise IndexError(f'expert {expert} out of range [0,{self.num_experts})')
        return head * self.num_experts + expert

    def decompose(self, flat_id: int) -> Tuple[int, int]:
        return divmod(flat_id, self.num_experts)

    def expert_row(self, layer_idx: int, flat_expert_id: int, proj: str) -> ExpertRowRef:
        layer = self.moe_layers[layer_idx]
        if not 0 <= flat_expert_id < layer.flatten:
            raise IndexError(f'flat_expert_id {flat_expert_id} out of range [0,{layer.flatten})')
        ref = layer.tensors[proj]
        row_nbytes = layer.row_nbytes(proj)
        return ExpertRowRef(layer_idx=layer_idx, flat_expert_id=flat_expert_id, proj=proj, file=ref.file, offset=ref.offset + flat_expert_id * row_nbytes, nbytes=row_nbytes, dtype=ref.dtype, shape=tuple(ref.shape[1:]))

    def expert_rows(self, layer_idx: int, flat_expert_id: int) -> List[ExpertRowRef]:
        return [self.expert_row(layer_idx, flat_expert_id, p) for p in MAGI2_PROJS]

    def expert_row_by_head(self, layer_idx: int, head: int, expert: int, proj: str) -> ExpertRowRef:
        return self.expert_row(layer_idx, self.flat_id(head, expert), proj)

    def expert_nbytes(self, layer_idx: Optional[int]=None) -> int:
        layer = self.moe_layers[layer_idx if layer_idx is not None else self.moe_layer_ids()[0]]
        return layer.expert_bytes()

    def total_expert_bytes(self) -> int:
        return sum((l.layer_bytes() for l in self.moe_layers.values()))

    def non_expert_bytes(self) -> int:
        return sum((r.nbytes for r in self.non_expert.values()))

    def summary(self) -> Dict[str, object]:
        ids = self.moe_layer_ids()
        first = self.moe_layers[ids[0]] if ids else None
        return {'moe_layers': len(ids), 'moe_layer_ids': ids, 'num_experts': self.num_experts, 'num_heads': first.num_heads if first else None, 'flatten_per_layer': first.flatten if first else None, 'expert_nbytes': self.expert_nbytes() if ids else 0, 'layer_expert_bytes': first.layer_bytes() if first else 0, 'total_expert_bytes': self.total_expert_bytes(), 'non_expert_bytes': self.non_expert_bytes(), 'non_expert_tensors': len(self.non_expert)}

def build_magi2_shard_index(preview_dir: str | Path, *, num_experts: int=MAGI2_NUM_EXPERTS, num_heads: int=MAGI2_NUM_MOE_HEADS) -> Magi2ShardIndex:
    index = build_index(preview_dir)
    return Magi2ShardIndex(index, num_experts=num_experts, num_heads=num_heads)

def dtype_size(dtype: str) -> int:
    return _DTYPE_SIZE[dtype]
if __name__ == '__main__':
    import json
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else ''
    si = build_magi2_shard_index(d)
    print(json.dumps(si.summary(), indent=2, default=str))
    ids = si.moe_layer_ids()
    if ids:
        r = si.expert_row(ids[0], 0, 'gate')
        print(f'layer {ids[0]} expert flat0 gate: file={Path(r.file).name} off={r.offset} nbytes={r.nbytes} shape={r.shape} dtype={r.dtype}')
