"""Decompose a DSV4.1 streamed-prefill trace into weight-stream and compute parts.

Input: the producer trace (stderr) of one run with LLAMADECODE_TILE_TIMING=1,
plus its prompt-only seconds. Every number comes from lines the producer
already prints: [dsv41-fetch] (per-stage DirectStorage bytes and fetch time),
LLAMADECODE_TILE_TIMING (per-tile host phases and GPU event time) and
DS4PERF_LAYER_WALL_LEDGER (per layer-pass wall, acquire wait, executor wall).

The layer passes are classified as sweep1 / sweep2 (layers 0-20, once per
row sweep) and tail (layers 21-39, cone rows only). For each pass the wall is
split into the part covered by that pass's own stage fetch and the part above
it. The rows add up to the layer wall; prompt minus layer wall is the work
outside layer passes (seed, head, admission).

Usage: python scripts/opus22_decompose.py TRACE PROMPT_SECONDS
"""
import re
import sys

SPECIAL = (0, 1, 2, 8, 14, 20)   # first, Engram and KV-source layers of the sweep


def parse(path):
    fetch, tiles, passes = [], [], []
    for line in open(path, errors='replace'):
        m = re.search(r'\[dsv41-fetch\] stage=(\d+) bytes=(\d+) fetch_ms=([\d.]+)', line)
        if m:
            fetch.append((int(m[1]), int(m[2]), float(m[3])))
        elif line.startswith('LLAMADECODE_TILE_TIMING'):
            tiles.append({k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)})
        elif line.startswith('DS4PERF_LAYER_WALL_LEDGER'):
            passes.append(({k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}, tiles))
            tiles = []
    return fetch, passes


def main(path, prompt):
    fetch, passes = parse(path)
    layer_stages = [f for f in fetch if f[0] not in (0, 41)]
    if len(layer_stages) != len(passes):
        raise ValueError('stage fetches %d do not match layer passes %d' % (len(layer_stages), len(passes)))
    seen, rows = set(), {}
    for (ledger, tiles), (stage, nbytes, fetch_ms) in zip(passes, layer_stages):
        layer = int(ledger['layer'])
        if stage != layer + 1:
            raise ValueError('stage %d is not layer %d' % (stage, layer))
        phase = 'tail' if layer >= 21 else ('sweep2' if layer in seen else 'sweep1')
        seen.add(layer)
        kind = phase + (' special' if phase != 'tail' and layer in SPECIAL else '')
        total = lambda key: sum(t[key] for t in tiles)
        host = total('host_prepare_ms') + total('host_upload_ms') + total('host_readback_ms') + total('host_cleanup_ms')
        r = rows.setdefault(kind, dict(passes=0, wall=0., covered=0., above=0., fetch=0., gb=0., wait=0., gpu=0., host=0.))
        wall = ledger['wall_ms']
        r['passes'] += 1
        r['wall'] += wall
        r['covered'] += min(wall, fetch_ms)
        r['above'] += max(0., wall - fetch_ms)
        r['fetch'] += fetch_ms
        r['gb'] += nbytes / 1e9
        r['wait'] += ledger['acquire_ms']
        r['gpu'] += total('gpu_ms')
        r['host'] += host + ledger['scaffold_ms']
    print('| pass kind | passes | wall s | covered by own stage fetch s | above fetch s | stage fetch s | GB | exposed weight wait s | GPU tile busy s | host-serial + scaffold s |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    layer_wall = 0.
    for kind, r in rows.items():
        layer_wall += r['wall'] / 1e3
        print('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.0f | %.2f | %.2f | %.2f |' % (
            kind, r['passes'], r['wall'] / 1e3, r['covered'] / 1e3, r['above'] / 1e3, r['fetch'] / 1e3,
            r['gb'], r['wait'] / 1e3, r['gpu'] / 1e3, r['host'] / 1e3))
    print('| outside layer passes | | %.2f | | | | | | | |' % (prompt - layer_wall))
    print('| **prompt** | | **%.2f** | | | | | | | |' % prompt)
    seconds = sum(f[2] for f in fetch) / 1e3
    gb = sum(f[1] for f in fetch) / 1e9
    print('\nAll stage fetches: %.1f GB in %.2f s of fetch time = %.2f GB/s' % (gb, seconds, gb / seconds))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
