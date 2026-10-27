"""Weight-free coverage check for the DSpark drafter's ``sanitize`` KEY remap.

Runs the real DeepSeek-V4-Flash-DSpark checkpoint's ``mtp.*`` key set (from the
vendored 72,317-tensor manifest) through a pure-python MIRROR of
``DeepseekV4DSparkDraftModel.sanitize``'s key transforms, and asserts the result
is EXACTLY the DSpark drafter module's declared parameter surface — every
checkpoint tensor lands on a real module parameter, and every module parameter is
fed by the checkpoint (the two conditions ``load_weights(strict=True)`` enforces
on the Mac). No mlx / numpy / weights, so it runs on the Windows dev box:

  pytest mlx_vlm/models/deepseek_v4/test_dspark_sanitize_keys.py   (Mac)
  python mlx_vlm/models/deepseek_v4/test_dspark_sanitize_keys.py   (any box)

The ``_module_key_for`` mirror below MUST stay in sync with the ``sanitize``
implementation in
``mlx_vlm/speculative/drafters/deepseek_v4_dspark/deepseek_v4_dspark.py``; the
independently hand-derived ``_expected_module_surface`` (built from the module's
nn.Module structure, not from sanitize) is what makes a divergence in EITHER
observable.
"""

import os
import re

try:
    from mlx_vlm.models.deepseek_v4.checkpoint_layout import (
        expand_manifest,
        load_flash_config,
        load_manifest,
        num_draft_layers,
    )
except ImportError:  # no mlx on this box: import the (mlx-free) module by path
    import importlib.util

    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_layout.py")
    _spec = importlib.util.spec_from_file_location("_dsv4_checkpoint_layout", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    expand_manifest = _m.expand_manifest
    load_flash_config = _m.load_flash_config
    load_manifest = _m.load_manifest
    num_draft_layers = _m.num_draft_layers

_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _module_key_for(checkpoint_key: str) -> str:
    """Pure-python mirror of the DSpark sanitize key remap for ONE mtp.* key.

    Mirrors: strip mtp. prefix; ``.scale`` -> ``.scales`` (fp8/fp4 kept
    native-quantized); ``hc_{attn,ffn}_{fn,base,scale}`` -> ``{attn,ffn}_hc.*``;
    ``ffn.gate.bias`` -> ``ffn.gate.e_score_correction_bias``; shared-expert
    ``w1/w2/w3`` -> ``gate/down/up_proj``; routed ``ffn.experts.{e}.{w}`` collapsed
    into ``ffn.switch_mlp.{proj}``; ``hc_head_{fn,base,scale}`` -> ``hc_head.*``.
    """
    assert checkpoint_key.startswith("mtp."), checkpoint_key
    rest = checkpoint_key[len("mtp.") :]
    if rest.endswith(".scale"):  # quant scale (NOT the HC "_scale" params)
        rest = rest[: -len(".scale")] + ".scales"

    stage, field = rest.split(".", 1)

    field = field.replace("ffn.gate.bias", "ffn.gate.e_score_correction_bias")
    for sub in ("attn", "ffn"):
        for param in ("fn", "base", "scale"):
            field = field.replace(f"hc_{sub}_{param}", f"{sub}_hc.{param}")
    field = (
        field.replace("hc_head_fn", "hc_head.fn")
        .replace("hc_head_base", "hc_head.base")
        .replace("hc_head_scale", "hc_head.scale")
    )
    for old, new in _PROJ.items():
        field = field.replace(f"shared_experts.{old}.", f"shared_experts.{new}.")

    m = re.match(r"ffn\.experts\.\d+\.(w1|w2|w3)\.(weight|scales)$", field)
    if m:
        field = f"ffn.switch_mlp.{_PROJ[m.group(1)]}.{m.group(2)}"

    return f"stages.{stage}.{field}"


def _expected_module_surface(cfg) -> set:
    """The DSpark drafter module's declared parameter keys, derived
    independently from its nn.Module structure (deepseek_v4_dspark.py)."""
    n_stages = num_draft_layers(cfg)
    keys = set()

    def fp8(prefix):
        keys.update({f"{prefix}.weight", f"{prefix}.scales"})

    for s in range(n_stages):
        p = f"stages.{s}"
        # DSparkAttention (== LocalAttention surface): fp8 linears + norms + sink.
        for lin in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            fp8(f"{p}.attn.{lin}")
        keys.update(
            {
                f"{p}.attn.q_norm.weight",
                f"{p}.attn.kv_norm.weight",
                f"{p}.attn.attn_sink",
                f"{p}.attn_norm.weight",
                f"{p}.ffn_norm.weight",
            }
        )
        # Hyper-connections (attn + ffn), f32 params.
        for hc in ("attn_hc", "ffn_hc"):
            keys.update({f"{p}.{hc}.fn", f"{p}.{hc}.base", f"{p}.{hc}.scale"})
        # Bias-routed MoE: gate + routed experts (stacked) + shared experts.
        keys.update(
            {f"{p}.ffn.gate.weight", f"{p}.ffn.gate.e_score_correction_bias"}
        )
        for proj in _PROJ.values():
            fp8(f"{p}.ffn.switch_mlp.{proj}")  # fp4 experts also carry .scales
            fp8(f"{p}.ffn.shared_experts.{proj}")
        if s == 0:
            fp8(f"{p}.main_proj")
            keys.add(f"{p}.main_norm.weight")
        if s == n_stages - 1:
            keys.update(
                {
                    f"{p}.hc_head.fn",
                    f"{p}.hc_head.base",
                    f"{p}.hc_head.scale",
                    f"{p}.norm.weight",
                    f"{p}.markov_head.markov_w1.weight",
                    f"{p}.markov_head.markov_w2.weight",
                    f"{p}.confidence_head.proj.weight",
                }
            )
    return keys


def _real_mtp_keys() -> list:
    real = expand_manifest(load_manifest())
    return [k for k in real if k.startswith("mtp.")]


def test_every_checkpoint_key_maps_to_a_module_param():
    cfg = load_flash_config()
    expected = _expected_module_surface(cfg)
    mapped = {_module_key_for(k) for k in _real_mtp_keys()}
    unexpected = sorted(mapped - expected)
    assert not unexpected, f"sanitize would emit params the module lacks: {unexpected[:10]}"


def test_every_module_param_is_fed_by_the_checkpoint():
    cfg = load_flash_config()
    expected = _expected_module_surface(cfg)
    mapped = {_module_key_for(k) for k in _real_mtp_keys()}
    missing = sorted(expected - mapped)
    assert not missing, f"module params with no checkpoint source: {missing[:10]}"


def test_expert_stacking_collapses_all_routed_experts():
    cfg = load_flash_config()
    n_experts = cfg["n_routed_experts"]
    real = _real_mtp_keys()
    # Every routed-expert weight collapses onto exactly one switch_mlp param.
    per_stage_expert_weight_keys = [
        k for k in real if re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", k)
    ]
    n_stages = num_draft_layers(cfg)
    assert len(per_stage_expert_weight_keys) == n_stages * n_experts * 3
    collapsed = {_module_key_for(k) for k in per_stage_expert_weight_keys}
    assert len(collapsed) == n_stages * 3  # 3 projections per stage


def test_no_hash_routing_in_draft_stages():
    # Draft stages are bias-routed: gate.bias present, tid2eid absent.
    real = _real_mtp_keys()
    assert any(k.endswith("ffn.gate.bias") for k in real)
    assert not any("tid2eid" in k for k in real)
    for k in real:
        assert _module_key_for(k) != k or not k.startswith("mtp.")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"DSPARK_SANITIZE_KEYS_TEST ALL PASS ({len(fns)} tests)")


if __name__ == "__main__":
    main()
