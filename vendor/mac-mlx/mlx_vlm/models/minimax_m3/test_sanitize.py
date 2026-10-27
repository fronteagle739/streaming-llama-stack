"""Structural validation of Model.sanitize() — HF M3 checkpoint layout -> the MLX param tree, NO real weights.

The RTX<->MLX cross-check loads PRE-remapped cc_mlx.npz, so it NEVER exercises sanitize(); this does.
Strategy: take the model's OWN param tree, REVERSE the sanitize (MLX names -> HF names, un-stack the SwitchGLU
experts into per-expert block_sparse_moe.experts.{x}.{w1,w2,w3}) to synthesize a checkpoint with the real HF
key names (confirmed vs the vLLM loader), then run sanitize() on it and assert it reproduces the EXACT param
tree (keys + shapes). A mismatch = a sanitize remap/stacking bug that would silently drop or mis-shape weights
at real-model load. (The +1 norm fold changes VALUES not keys/shapes, so we compare keys+shapes.)

  python -m mlx_vlm.models.minimax_m3.test_sanitize
"""
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.models.minimax_m3.minimax_m3 import Model, ModelArgs

ARGS = ModelArgs(
    hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
    num_attention_heads=8, num_key_value_heads=2, head_dim=16,
    num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=8,
    num_experts_per_tok=2, n_shared_experts=1, vocab_size=256,
    rotary_dim=8, indexer_n_head=2, indexer_head_size=16,
    indexer_top_k=2, sparse_block_size=4, sparse_init_blocks=0, sparse_local_blocks=1,
)

P2W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}  # inverse of sanitize's w2proj


def un_sanitize(params):
    """Inverse of sanitize: MLX param name -> HF checkpoint name(s). Un-stacks SwitchGLU experts back to
    per-expert block_sparse_moe.experts.{x}.{w1,w2,w3}, and renames the MoE gate / e_score / shared_experts
    back to block_sparse_moe.*. Dense mlp.{gate,up,down}_proj pass through unchanged (they are NOT MoE)."""
    hf = {}
    for k, v in params.items():
        if ".mlp.switch_mlp." in k:
            prefix, rest = k.split(".mlp.switch_mlp.")
            proj = rest.replace(".weight", "")
            for x in range(v.shape[0]):
                hf[f"{prefix}.block_sparse_moe.experts.{x}.{P2W[proj]}.weight"] = v[x]
        elif k.endswith(".mlp.e_score_correction_bias"):
            hf[k.replace(".mlp.e_score_correction_bias", ".block_sparse_moe.gate.e_score_correction_bias")] = v
        elif k.endswith(".mlp.gate.weight"):  # MoE router gate (NOT the dense mlp.gate_proj)
            hf[k.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")] = v
        elif ".mlp.shared_experts." in k:
            hf[k.replace(".mlp.", ".block_sparse_moe.")] = v
        else:
            hf[k] = v
    return hf


def main():
    mx.random.seed(0)
    model = Model(ARGS)
    mx.eval(model.parameters())
    params = dict(tree_flatten(model.parameters()))

    hf = un_sanitize(params)
    # the synthetic checkpoint must really use HF names (else the test is vacuous)
    assert any(".block_sparse_moe.experts." in k for k in hf), "test bug: no HF expert keys synthesized"
    assert not any(".switch_mlp." in k for k in hf), "test bug: MLX switch_mlp leaked into the HF checkpoint"

    san = model.sanitize(hf)
    pk, sk = set(params), set(san)
    missing = sorted(pk - sk)   # model params sanitize did NOT produce -> would be random/dropped at load
    extra = sorted(sk - pk)     # sanitize produced keys the model has no slot for -> silently ignored
    shape_mm = [(k, tuple(params[k].shape), tuple(san[k].shape))
                for k in sorted(pk & sk) if params[k].shape != san[k].shape]

    ok = not missing and not extra and not shape_mm
    if missing:
        print("MISSING (model params sanitize did NOT produce):", missing[:12])
    if extra:
        print("EXTRA (sanitize produced, no model slot):", extra[:12])
    for k, ps, ss in shape_mm[:12]:
        print(f"SHAPE {k}: model{ps} != sanitize{ss}")
    n_exp = sum(1 for k in hf if ".block_sparse_moe.experts." in k)
    print(f"params={len(params)} hf_keys={len(hf)} (experts un-stacked={n_exp}) sanitized={len(san)}")
    print("SANITIZE_TEST", "ALL PASS" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
