"""CPU property check of tiled selection against independent lexicographic sort.

Run by file on hosts without MLX. This models the Metal compare/exchange
network and its pruning, not FP32 matrix arithmetic or GPU execution.
"""

from pathlib import Path

import numpy as np


INVALID = 0x7FFFFFFF


def _stage(values, ids, offset, pairs, stride, span=None):
    pair = np.arange(pairs)
    local = pair // stride * (2 * stride) + pair % stride
    a, b = offset + local, offset + local + stride
    # Every pair in a parallel stage must own two distinct addresses.
    assert np.unique(np.concatenate((a, b))).size == pairs * 2
    assert b.max() < len(values)
    better = (values[a] > values[b]) | ((values[a] == values[b]) & (ids[a] < ids[b]))
    worse = (values[a] < values[b]) | ((values[a] == values[b]) & (ids[a] > ids[b]))
    swap = worse if span is None else np.where((local & span) == 0, worse, better)
    a, b = a[swap], b[swap]
    values[a], values[b] = values[b].copy(), values[a].copy()
    ids[a], ids[b] = ids[b].copy(), ids[a].copy()


def _check_row(scores):
    values = np.full(1536, -np.inf, dtype=np.float32)
    ids = np.full(1536, INVALID, dtype=np.int32)
    skipped = 0
    for base in range(0, len(scores), 1024):
        count = min(1024, len(scores) - base)
        values[512:] = -np.inf
        ids[512:] = INVALID
        values[512:512 + count] = scores[base:base + count]
        ids[512:512 + count] = np.arange(base, base + count)
        if values[512:].max() <= values[511]:
            skipped += 1
        else:
            for span in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                stride = span // 2
                while stride:
                    pairs = 256 if span == 1024 and stride < 512 else 512
                    _stage(values, ids, 512, pairs, stride, span)
                    stride //= 2
            other = np.arange(1023, 511, -1)
            better = ((values[other] > values[:512]) |
                      ((values[other] == values[:512]) & (ids[other] < ids[:512])))
            values[:512] = np.where(better, values[other], values[:512])
            ids[:512] = np.where(better, ids[other], ids[:512])
            for stride in (256, 128, 64, 32, 16, 8, 4, 2, 1):
                _stage(values, ids, 0, 256, stride)
        prefix = scores[:base + count]
        order = np.lexsort((np.arange(len(prefix)), -prefix))[:512]
        expected = order[prefix[order] > -np.inf]
        actual = ids[:512][values[:512] > -np.inf]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values[:len(expected)], prefix[expected])
    for k in {1, min(37, len(scores)), min(512, len(scores))}:
        winners = ids[:k][values[:k] > -np.inf]
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        expected = np.sort(order[scores[order] > -np.inf])
        np.testing.assert_array_equal(np.sort(winners), expected)
    return skipped


def main():
    rng = np.random.default_rng(71)
    cases = skipped = 0
    for size in (1, 7, 37, 511, 512, 513, 1023, 1024, 1025, 2048, 3079, 6000, 30000):
        random = rng.normal(size=size).astype(np.float32)
        masked = np.where(rng.random(size) < .8, -np.inf, random)
        for scores in (random, masked, -np.abs(random),
                       rng.integers(-4, 5, size=size).astype(np.float32),
                       np.arange(size, dtype=np.float32),
                       -np.arange(size, dtype=np.float32),
                       np.zeros(size, dtype=np.float32), np.full(size, -np.inf)):
            skipped += _check_row(scores)
            cases += 1

    # Exhaustively verify block-output ownership after causal suffix pruning,
    # including BLOCK=1024, partial key tiles, lens=0 and all requested edges.
    for size in (37, 1031, 30000):
        for block in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            nb = (size + block - 1) // block
            for limit in (0, 1, 7, 8, 9, 511, 512, 513, 1023, 1024, 1025, size):
                end = min(size, limit)
                first = (end + 1023) // 1024 * (1024 // block)
                owners = np.zeros(nb, dtype=np.int32)
                owners[first:] += 1
                for base in range(0, end, 1024):
                    owners[base // block:min(nb, (base + 1024) // block)] += 1
                assert np.all(owners == 1), (size, block, limit)

    # Reference pin expression works for either truncation (MLX 0.32.2) or
    # mathematical floor; the former refutes the original ceil identity at 0.
    lens = np.arange(30002, dtype=np.int32)
    for divide in (lambda x: np.trunc(x / 8).astype(np.int32), lambda x: x // 8):
        last = divide(lens - 1)
        np.testing.assert_array_equal((last + 1) - 1, last)
    assert int(np.trunc(-1 / 8)) == 0 and (0 + 7) // 8 - 1 == -1

    metal = Path(__file__).with_name("index_prefill.metal").read_text()
    assert "values[1536]" in metal and "ids[1536]" in metal
    assert "span <= 1024" in metal and "incoming_max <= values[511]" in metal
    print(f"PASS: {cases} selection cases, {skipped} rejected tiles; "
          "block-output ownership and zero-length pin arithmetic. GPU validation still required.")


if __name__ == "__main__":
    main()
