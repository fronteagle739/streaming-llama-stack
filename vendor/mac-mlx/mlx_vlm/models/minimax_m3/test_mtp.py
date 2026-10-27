# Tiny-synthetic MLX structural test for the MiniMax-M3 MTP (NextN draft) head — RUN ON A MAC (Metal-only).
# No weights: random-init a tiny M3MTPLayer and assert structural invariants independent of weight values:
#   * output shape;
#   * position-0 masking: the pos-0 token embedding is ignored (MTP has no previous token there);
#   * sanity: a NON-masked position DOES depend on its token embedding;
#   * B>1: each batch row matches its B=1 result (sequences are independent).
# The numerical RTX-vs-MLX MTP parity is a separate cross-check (scripts/m3_crosscheck_*); this guards the
# MLX MTP structure on its own.
#   cd <mlx-vlm clone>  (on the Mac: the synced clone)
#   python -m mlx_vlm.models.minimax_m3.test_mtp
import os

import mlx.core as mx

from mlx_vlm.models.minimax_m3.minimax_m3 import M3MTPLayer, ModelArgs

ARGS = ModelArgs(
    hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
    num_attention_heads=8, num_key_value_heads=2, head_dim=16,
    num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=8,
    num_experts_per_tok=2, n_shared_experts=1, vocab_size=256,
    rotary_dim=8, indexer_n_head=2, indexer_head_size=16,
    indexer_top_k=2, sparse_block_size=4, sparse_init_blocks=0, sparse_local_blocks=1,
    num_mtp_modules=1,
)


def main():
    os.environ["M3_MLX_SPARSE"] = "1"
    mx.random.seed(0)
    layer = M3MTPLayer(ARGS)
    mx.eval(layer.parameters())
    H, L = ARGS.hidden_size, 6
    pos = mx.arange(L)
    mx.random.seed(1)
    emb = mx.random.normal((1, L, H))
    prev = mx.random.normal((1, L, H))
    h = layer(emb, prev, pos)
    fail = 0

    ok = h.shape == (1, L, H)
    print(f"  {'PASS' if ok else 'FAIL'} output shape {h.shape} == (1,{L},{H})"); fail += 0 if ok else 1

    new0 = mx.random.normal((H,))
    h0 = layer(mx.concatenate([new0[None, None], emb[:, 1:]], axis=1), prev, pos)
    d0 = float(mx.max(mx.abs(h[0, 0] - h0[0, 0])).item())
    ok = d0 < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'} pos-0 output INVARIANT to pos-0 token embed (max|d|={d0:.2e})"); fail += 0 if ok else 1

    new3 = mx.random.normal((H,))
    h3 = layer(mx.concatenate([emb[:, :3], new3[None, None], emb[:, 4:]], axis=1), prev, pos)
    d3 = float(mx.max(mx.abs(h[0, 3] - h3[0, 3])).item())
    ok = d3 > 1e-4
    print(f"  {'PASS' if ok else 'FAIL'} pos-3 output DEPENDS on pos-3 token embed (max|d|={d3:.2e})"); fail += 0 if ok else 1

    emb_b = mx.concatenate([emb, mx.random.normal((1, L, H))], axis=0)
    prev_b = mx.concatenate([prev, mx.random.normal((1, L, H))], axis=0)
    hb = layer(emb_b, prev_b, pos)
    r0 = layer(emb_b[:1], prev_b[:1], pos)
    r1 = layer(emb_b[1:], prev_b[1:], pos)
    d = max(float(mx.max(mx.abs(hb[0] - r0[0])).item()), float(mx.max(mx.abs(hb[1] - r1[0])).item()))
    ok = d < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'} B=2 rows match per-sequence B=1 (max|d|={d:.2e})"); fail += 0 if ok else 1

    print("ALL PASS" if fail == 0 else f"FAILED ({fail})")
    raise SystemExit(fail)


if __name__ == "__main__":
    main()
