#!/usr/bin/env python3
# Compare two logits dumps written by logits-dump: max abs diff and top-1 agreement per row.
# usage: compare_logits.py a.bin b.bin [--tol 1e-2]
import sys
import numpy as np


def load(path):
    with open(path, "rb") as f:
        n_rows, n_vocab = np.frombuffer(f.read(8), dtype=np.int32)
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(int(n_rows), int(n_vocab))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-2
    if a.shape != b.shape:
        print(f"FAIL shape {a.shape} vs {b.shape}")
        sys.exit(1)
    finite = np.isfinite(a).all() and np.isfinite(b).all()
    d = np.abs(a - b)
    top1 = (a.argmax(axis=1) == b.argmax(axis=1))
    scale = np.abs(a).max()
    ok = finite and top1.all() and d.max() < tol
    print(f"rows={a.shape[0]} vocab={a.shape[1]} max_abs_diff={d.max():.3e} mean_abs_diff={d.mean():.3e} "
          f"max_abs_logit={scale:.3e} top1_agree={int(top1.sum())}/{a.shape[0]} finite={finite} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
