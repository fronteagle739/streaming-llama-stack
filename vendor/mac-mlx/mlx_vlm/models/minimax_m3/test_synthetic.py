# Tiny-synthetic MLX coherence gate for the MiniMax-M3 block-sparse (MSA) attention — RUN ON A MAC
# (MLX is Metal-only). No weights: random-init a tiny minimax_m3 and check the MSA invariant that the
# RTX gguf engine validates: with a sequence shorter than topk*block_size (full block coverage) every
# block is selected, so the sparse output == the dense output; longer sequences must diverge (real
# pruning). Independent of the random weight values.
#
#   cd ./GitHub\mlx-vlm  (on the Mac: the synced clone)
#   python -m mlx_vlm.models.minimax_m3.test_synthetic
import os

import mlx.core as mx

from mlx_vlm.models.minimax_m3.minimax_m3 import Model, ModelArgs

ARGS = ModelArgs(
    hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
    num_attention_heads=8, num_key_value_heads=2, head_dim=16,
    num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=8,
    num_experts_per_tok=2, n_shared_experts=1, vocab_size=256,
    rotary_dim=8, indexer_n_head=2, indexer_head_size=16,
    indexer_top_k=2, sparse_block_size=4, sparse_init_blocks=0, sparse_local_blocks=1,
)
COVERAGE = ARGS.indexer_top_k * ARGS.sparse_block_size  # 8


def build():
    mx.random.seed(0)
    model = Model(ARGS)
    mx.eval(model.parameters())
    return model


def _toks(n_tok):
    return [(i * 7 + 1) % ARGS.vocab_size for i in range(n_tok)]


def fwd(model, n_tok, sparse):
    os.environ["M3_MLX_SPARSE"] = "1" if sparse else "0"
    logits = model(mx.array([_toks(n_tok)]))[0, -1]
    return int(mx.argmax(logits).item()), round(float(mx.max(logits).item()), 4)


def fwd_incremental(model, n_tok):
    """Last token's logits via prefill(0..n-2) + decode(1) using the M3Cache (decode-path MSA)."""
    os.environ["M3_MLX_SPARSE"] = "1"
    t = _toks(n_tok)
    cache = model.make_cache()
    if n_tok > 1:
        model(mx.array([t[:-1]]), cache=cache)
    logits = model(mx.array([[t[-1]]]), cache=cache)[0, -1]
    return int(mx.argmax(logits).item()), round(float(mx.max(logits).item()), 4)


def fwd_batch(model, n_tok):
    """B=2 batch of two distinct sequences; each row must equal that sequence's B=1 result (sequences
    are independent, so batching must not change per-row output)."""
    os.environ["M3_MLX_SPARSE"] = "1"
    t0 = [(i * 7 + 1) % ARGS.vocab_size for i in range(n_tok)]
    t1 = [(i * 5 + 3) % ARGS.vocab_size for i in range(n_tok)]
    batched = model(mx.array([t0, t1]))[:, -1]
    a0 = int(mx.argmax(model(mx.array([t0]))[0, -1]).item())
    a1 = int(mx.argmax(model(mx.array([t1]))[0, -1]).item())
    return int(mx.argmax(batched[0]).item()) == a0 and int(mx.argmax(batched[1]).item()) == a1


def main():
    model = build()
    print(f"MiniMax-M3 MLX coherence gate (coverage = topk*block_size = {COVERAGE}):")
    fail = 0
    for n in (4, 8, 9, 12, 16):
        s = fwd(model, n, True)
        d = fwd(model, n, False)
        if n <= COVERAGE:
            ok = s[0] == d[0]
            print(f"  n_tok={n:<3} {'PASS' if ok else 'FAIL'} full-coverage sparse{s} == dense{d}")
        else:
            ok = s[0] != d[0] or s[1] != d[1]
            print(f"  n_tok={n:<3} {'PASS' if ok else 'FAIL'} partial sparse{s} != dense{d}")
        fail += 0 if ok else 1
    print("MiniMax-M3 MLX decode-consistency (one-shot sparse == prefill+incremental-decode):")
    for n in (6, 12, 16):
        o = fwd(model, n, True)
        d = fwd_incremental(model, n)
        ok = o[0] == d[0]
        print(f"  n_tok={n:<3} {'PASS' if ok else 'FAIL'} oneshot{o} == incremental{d}")
        fail += 0 if ok else 1
    print("MiniMax-M3 MLX B>1 batching (each batch row == its B=1 result):")
    for n in (6, 12):
        ok = fwd_batch(model, n)
        print(f"  n_tok={n:<3} {'PASS' if ok else 'FAIL'} B=2 rows match per-sequence B=1")
        fail += 0 if ok else 1
    print("ALL PASS" if fail == 0 else f"FAILED ({fail})")
    raise SystemExit(fail)


if __name__ == "__main__":
    main()
