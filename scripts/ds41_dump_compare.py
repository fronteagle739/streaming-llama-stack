"""Compare two DS4COMBO_V41_DUMP files (token-major reference vs layer-major).

usage: python -I ds41_dump_compare.py REF.bin TEST.bin

Per layer and point: max |a-b| and max |a-b| / max |a| over the common positions;
for the routed expert ids, how many positions select the same set of experts.
Points 7-10 are inside attention: q after rope, the cached kv row, the heads as the
output projection reads them (bf16, inverse rope), and the projection result.
"""
import struct
import sys

import numpy as np

POINTS = ['attn_in', 'after_attn', 'ffn_in', 'expert_ids', 'expert_w', 'routed', 'residual',
          'q_rope', 'kv_cached', 'heads_in', 'attn_proj']
MAGIC = 0x52313444


def load(path):
    out = {}
    data = open(path, 'rb').read()
    off = 0
    while off + 20 <= len(data):
        magic, layer, pos, point, words = struct.unpack_from('<5I', data, off)
        if magic != MAGIC:
            raise ValueError(f'{path}: bad record at byte {off}')
        off += 20
        raw = data[off:off + 4 * words]
        off += 4 * words
        dtype = '<i4' if point == 3 else '<f4'
        # first record wins: a position is dumped once per layer
        out.setdefault((layer, pos, point), np.frombuffer(raw, dtype=dtype).copy())
    return out


def main():
    ref, test = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(ref) & set(test))
    if not keys:
        sys.exit('no common (layer, position, point) records')
    layers = sorted({k[0] for k in keys})
    positions = sorted({k[1] for k in keys})
    print(f'positions {positions}; layers {layers[0]}..{layers[-1]}')
    print('layer ' + ' '.join(f'{p:>21}' for p in POINTS if p != 'expert_ids') + '  ids_same')
    print('      (each cell: max|a-b|  max|a-b|/max|a|)')
    first = None
    for layer in layers:
        cells, same, total = [], 0, 0
        for point, name in enumerate(POINTS):
            ks = [k for k in keys if k[0] == layer and k[2] == point]
            if point == 3:
                for k in ks:
                    total += 1
                    same += set(ref[k].tolist()) == set(test[k].tolist())
                continue
            if not ks:
                cells.append(f'{"-":>21}')
                continue
            abs_d = max(float(np.max(np.abs(ref[k] - test[k]))) for k in ks)
            scale = max(float(np.max(np.abs(ref[k]))) for k in ks) or 1.0
            rel = abs_d / scale
            cells.append(f'{abs_d:10.3g} {rel:10.3g}')
            if first is None and rel > 1e-3:
                first = (layer, name, rel)
        print(f'{layer:5d} ' + ' '.join(cells) + f'  {same}/{total}')
    worst = []
    for k in keys:
        if k[2] == 3 and set(ref[k].tolist()) != set(test[k].tolist()):
            worst.append(f'layer {k[0]} pos {k[1]}: ref {sorted(ref[k].tolist())} test {sorted(test[k].tolist())}')
    for line in worst[:20]:
        print('ids differ:', line)
    print('first point with relative diff > 1e-3:', first)


if __name__ == '__main__':
    main()
