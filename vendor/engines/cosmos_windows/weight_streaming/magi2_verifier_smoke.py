# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_verifier_smoke.py
# Machine fallback paths cleared; no model data included.
import sys, tempfile, os, struct
sys.path.insert(0, '')
import torch
from cosmos_windows.weight_streaming.magi2_moe_binding import InLoopLayerVerifier

class Ref:

    def __init__(self, name, file, offset, nbytes):
        self.name, self.file, self.offset, self.nbytes = (name, file, offset, nbytes)

class Index:

    def __init__(self, refs):
        self.refs = refs

    def __getitem__(self, k):
        return self.refs[k]
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, 'shard.bin')
names = [f'block.layers.0.experts.{i}.w1.weight' for i in range(6)] + ['block.layers.0.attention.k_norm.weight', 'block.layers.0.mlp.w2.weight']
size = 4096
blob = bytearray()
refs = {}
for i, n in enumerate(names):
    refs[n] = Ref(n, path, i * size, size)
    blob += bytes([(i * 7 + j) % 251 for j in range(size)])
open(path, 'wb').write(bytes(blob))
index = Index(refs)
views = {n: torch.frombuffer(bytearray(blob[r.offset:r.offset + r.nbytes]), dtype=torch.uint8).clone() for n, r in refs.items()}
v = InLoopLayerVerifier(index, emit=None)
picks = v._pick_names(0, views)
n_exp = sum((1 for p in picks if 'expert' in p))
print(f'picks={len(picks)} experts_covered={n_exp} offsets={v.offsets}')
assert n_exp >= 1, 'verifier must probe expert tensors (they are the bulk of a MoE layer)'
assert len(v._probe_offsets(size)) == len(v.offsets)
v.check_layer(0, views)
print(f'clean pass OK, probes checked={v.checked}')
bad = views.copy()
name = next((p for p in picks if 'expert' in p))
t = views[name].clone()
t[-64:] = 0
bad[name] = t
try:
    v2 = InLoopLayerVerifier(index, emit=None)
    v2.check_layer(0, bad)
    print('FAIL: tail corruption NOT detected')
    sys.exit(1)
except RuntimeError as e:
    assert 'TRUNCATED' in str(e) or 'tail' in str(e), f'unexpected message: {e}'
    print('tail-corruption detected OK:', str(e)[:140])
print('VERIFIER SMOKE: PASS')
