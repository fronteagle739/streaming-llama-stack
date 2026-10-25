"""LLAMADECODE: draft-threshold hit rate for the indexer top-512, from recorded inputs.

Usage: python topk_draft_analysis.py <LLAMADECODE_TOPK_DUMP dir> <producer stderr.log>

The producer run must have LLAMADECODE_TOPK_DUMP=<dir> (and optionally
LLAMADECODE_TOPK_DUMP_STRIDE, default 64): every large multi-row top-k writes the
scores of every STRIDE-th query row plus the 512 ids it selected. stderr maps each
dump (seq) to the (layer, tile start) whose DS4PERF_TOPK_*_COMPLETE line follows it.

Per owner layer L (all but the first) and per sampled query row, with the previous
owner layer's ids for the SAME query (same tile start, same row; the compressed
index-key space is shared across owner layers), the owner's exact draft scheme is:
  tau = min(scores_L[draft ids]);  candidates = #{scores_L >= tau}
candidates >= 512 always; candidates/512 is what an exact draft select must sort.
Rows whose draft hits -inf (fewer than 512 visible keys) are counted separately.
It also re-derives the exact top-512 (value desc, index asc) from the recorded
scores and checks it equals the recorded ids.
"""
import re
import sys
from pathlib import Path

import numpy as np

K = 512


def load(path):
    raw = path.read_bytes()
    h = np.frombuffer(raw[:28], dtype='<u4')
    assert h[0] == 0x31444b54, path
    seq, nrows, ncols, k, stride, sampled = (int(v) for v in h[1:])
    rec = np.dtype([('row', '<u4'), ('s', '<f4', (ncols,)), ('ids', '<i4', (k,))])
    data = np.frombuffer(raw[28:], dtype=rec, count=sampled)
    return seq, k, data


def owner_map(stderr):
    """seq -> (layer, start), in log order."""
    pending, out = [], {}
    for line in Path(stderr).read_text(errors='replace').splitlines():
        m = re.search(r'LLAMADECODE_TOPK_DUMP seq=(\d+)', line)
        if m:
            pending.append(int(m.group(1)))
            continue
        m = re.search(r'DS4PERF_TOPK_(?:RADIX|FAST)_COMPLETE layer=(\d+) start=(\d+) count=\d+ actual=(\d+)', line)
        if m and pending and int(m.group(3)) > 0:
            for seq in pending:
                out[seq] = (int(m.group(1)), int(m.group(2)))
            pending = []
    return out


def exact_ids(s):
    flushed = np.where(np.abs(s) < np.float32(1.1754944e-38), np.float32(0), s)  # FTZ, -0 == +0
    order = np.lexsort((np.arange(s.size), -flushed.astype(np.float64)))
    return order[:K]


def main(dump_dir, stderr):
    where = owner_map(stderr)
    tiles = {}
    mismatches = checked = 0
    for p in sorted(Path(dump_dir).glob('topk-*.bin')):
        seq, k, data = load(p)
        if k != K or seq not in where:
            continue
        layer, start = where[seq]
        tiles[(layer, start)] = data
        for rec in data:
            if np.isnan(rec['s']).any():
                continue
            checked += 1
            mismatches += int(not np.array_equal(exact_ids(rec['s']), rec['ids']))
    print(f'recorded-input check: {checked} rows, exact top-512 != recorded ids in {mismatches}')
    layers = sorted({l for l, _ in tiles})
    print('owner layers seen:', layers)
    print('layer prev  rows  minus_inf  p10   p50   p90   p99   max   <=2x  <=4x  <=8x  overlap_p50')
    for prev, layer in zip(layers, layers[1:]):
        ratios, overlaps, inf_rows = [], [], 0
        for (l, start), cur in tiles.items():
            if l != layer or (prev, start) not in tiles:
                continue
            old = {int(r['row']): r['ids'] for r in tiles[(prev, start)]}
            for rec in cur:
                draft = old.get(int(rec['row']))
                if draft is None or np.isnan(rec['s']).any():
                    continue
                d = draft[draft >= 0]
                tau = rec['s'][d].min()
                if not np.isfinite(tau):
                    inf_rows += 1
                    continue
                ratios.append(int((rec['s'] >= tau).sum()) / K)
                overlaps.append(np.intersect1d(d, rec['ids']).size / K)
        if not ratios:
            print(f'{layer:5d} {prev:4d}  no matched rows')
            continue
        r = np.array(ratios)
        q = np.percentile(r, [10, 50, 90, 99])
        print(f'{layer:5d} {prev:4d} {r.size:5d} {inf_rows:9d} ' + ' '.join(f'{v:5.1f}' for v in q) +
              f' {r.max():6.1f} {np.mean(r <= 2):5.2f} {np.mean(r <= 4):5.2f} {np.mean(r <= 8):5.2f} {np.median(overlaps):8.2f}')
    return 0 if mismatches == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
