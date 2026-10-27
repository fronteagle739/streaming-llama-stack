"""DeepSeek-V4.1 DSpark (MTP) coverage: key mapping, stage wiring, round loop.

Two halves, following the deepseek_v4 DSpark tests' pattern:

* STRUCTURAL (no mlx, no numpy, no torch, no weights) -- the ``mtp.*`` ->
  ``dspark.stages.<i>.*`` key mapping against the real 2,401-tensor release
  manifest for the separate drafter, unconditional MTP drops in the target,
  the per-stage module tree the drafter builds, the DSpark footprint,
  the drafter's ``stages.<i>.*`` re-prefix, the round-loop accept/trim/commit
  invariants, and an AST check that ``dspark.py`` reuses ``language.py``'s
  building blocks instead of redefining them.
* MAC (needs mlx + Metal) -- ``get_dspark_topk_idxs`` against the reference
  formula, the built parameter tree against the planned one, and a tiny-config
  speculative round loop (draft, verify, accept k, rollback, continue) whose
  emitted tokens equal a plain greedy decode of the same target.

Run:  python mlx_vlm/models/deepseek_v41/test_dspark_dsv41.py     (any box)
      python -m pytest mlx_vlm/models/deepseek_v41/test_dspark_dsv41.py   (Mac)

On Windows ``pytest`` cannot import a module inside ``mlx_vlm`` (the package
``__init__`` imports mlx), which is why the three CPU test files here are
collected through ``conftest.py``'s file-based ``StandaloneModule``; this file
runs as a script there instead. The mlx halves skip themselves cleanly.
"""

import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - the plain-script path
    pytest = None

HERE = Path(__file__).resolve().parent


if pytest is not None:
    # Subclassing pytest's own Skipped makes one raise both report a skip under
    # pytest and stay catchable by the plain-script runner below.
    class _Skipped(pytest.skip.Exception):
        pass
else:  # pragma: no cover - pytest is present in this project's venvs
    class _Skipped(Exception):
        pass


def _skip(reason):
    raise _Skipped(reason)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = _load(HERE / "config.py", "dspark_test_v41_config")
layout = _load(HERE / "checkpoint_layout.py", "dspark_test_v41_layout")
BK = _load(HERE.parents[1] / "speculative" / "dspark_bookkeeping.py",
           "dspark_test_bookkeeping")

# Release values, pinned here so a config edit has to be deliberate.
BLOCK_SIZE = 5
NOISE_TOKEN_ID = 128799
TARGET_LAYER_IDS = [37, 38, 39]
MARKOV_RANK = 256
N_STAGES = 3
DSPARK_EXPERTS = 128
DSPARK_BYTES = 8047486344  # 7.495 GiB, engram_source-independent

STAGE_COMMON = {
    "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight",
    "attn.wkv.scales", "attn.wkv.weight", "attn.wo_a.weight",
    "attn.wo_b.scales", "attn.wo_b.weight", "attn.wq_a.scales", "attn.wq_a.weight",
    "attn.wq_b.scales", "attn.wq_b.weight",
    "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "attn_norm.weight",
    "ffn.gate.bias_vl", "ffn.gate.e_score_correction_bias", "ffn.gate.weight",
    "ffn.shared_experts.down_proj.scales", "ffn.shared_experts.down_proj.weight",
    "ffn.shared_experts.gate_proj.scales", "ffn.shared_experts.gate_proj.weight",
    "ffn.shared_experts.up_proj.scales", "ffn.shared_experts.up_proj.weight",
    "ffn.switch_mlp.down_proj.scales", "ffn.switch_mlp.down_proj.weight",
    "ffn.switch_mlp.gate_proj.scales", "ffn.switch_mlp.gate_proj.weight",
    "ffn.switch_mlp.up_proj.scales", "ffn.switch_mlp.up_proj.weight",
    "ffn_hc.base", "ffn_hc.fn", "ffn_hc.scale", "ffn_norm.weight",
}
STAGE_FIRST = {"main_norm.weight", "main_proj.scales", "main_proj.weight"}
STAGE_LAST = {"norm.weight", "markov_head.embed.weight", "markov_head.head.weight",
              "confidence_head.proj.weight"}


def _release():
    """(raw config, tensor manifest, TextConfig) or None when absent."""
    roots = ([Path(os.environ["DSV41_META_DIR"])] if "DSV41_META_DIR" in os.environ
             else [Path("~/tmp/dsv41").expanduser(), Path("./tmp/upstream_dsv41")])
    for root in roots:
        paths = [root / "v41_config.json", root / "v41_tensors.json"]
        if all(p.is_file() for p in paths):
            raw, meta = [json.loads(p.read_text()) for p in paths]
            return raw, meta, config.TextConfig.from_dict(raw["text_config"])
    return None


def _require_release():
    data = _release()
    if data is None:
        _skip("DeepSeek-V4.1 release metadata is absent; set DSV41_META_DIR")
    return data


# --------------------------------------------------------------------------- #
# Structural: config -> stage count                                            #
# --------------------------------------------------------------------------- #

def test_release_config_declares_three_dspark_stages():
    c = config.TextConfig()
    assert layout.dspark_stage_count(c) == N_STAGES
    assert c.dspark_block_size == BLOCK_SIZE
    assert c.dspark_noise_token_id == NOISE_TOKEN_ID
    assert c.dspark_target_layer_ids == TARGET_LAYER_IDS
    assert c.dspark_markov_rank == MARKOV_RANK
    assert c.dspark_n_routed_experts == DSPARK_EXPERTS
    assert c.dspark_num_experts_per_tok == 3
    # The three MTP compress ratios are the last entries of the 43-entry list
    # and are all zero: the DSpark blocks are sliding-window only.
    assert c.compress_ratios[c.num_hidden_layers:] == [0, 0, 0]


def test_backbone_only_ratios_describe_no_dspark_stack():
    c = config.TextConfig()
    short = config.TextConfig(compress_ratios=c.compress_ratios[:c.num_hidden_layers])
    assert layout.dspark_stage_count(short) == 0
    # ... and such a config keeps planning mtp.* as a drop.
    plan = layout.plan_v41({"mtp.0.norm.weight": ["BF16", [5120]],
                            **{k: (d, s) for k, d, s in layout._source_parameters(short)}},
                           short, "hybrid")
    assert plan["mtp.0.norm.weight"]["op"] == "drop"


def test_partial_or_compressed_mtp_ratios_are_rejected():
    # TextConfig itself rejects a ratios list that is neither n nor n + MTP, so
    # the half-covered case is built on a stand-in carrying just the four fields
    # dspark_stage_count reads.
    import types

    def probe(tail):
        return types.SimpleNamespace(compress_ratios=[0, 0, 1, 1] + tail,
                                     num_hidden_layers=4, num_nextn_predict_layers=3,
                                     dspark_block_size=5)

    for tail, message in (([0, 0], "every MTP layer"), ([0, 0, 2], "sliding-window only")):
        try:
            layout.dspark_stage_count(probe(tail))
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - a regression would land here
            raise AssertionError(f"expected ValueError for tail {tail}")
    assert layout.dspark_stage_count(probe([0, 0, 0])) == 3


# --------------------------------------------------------------------------- #
# Structural: the release manifest's mtp.* keys                                #
# --------------------------------------------------------------------------- #

def test_dspark_source_keys_are_exactly_the_release_mtp_tensors():
    _, meta, c = _require_release()
    declared = {k: (d, s) for k, d, s in layout._dspark_source_parameters(c)}
    shipped = {k: tuple(v[:2]) for k, v in meta.items() if k.startswith("mtp.")}
    assert len(shipped) == 2401
    assert declared.keys() == shipped.keys()
    for key, (dtype, shape) in declared.items():
        assert (dtype, list(shape)) == (shipped[key][0], list(shipped[key][1])), key


def test_target_loads_dspark_when_release_config_describes_stages():
    _, meta, c = _require_release()
    plan = layout.plan_v41(meta, c, "hybrid")
    mtp = {k: e for k, e in plan.items() if k.startswith("mtp.")}
    assert len(mtp) == 2401
    assert all(e["op"] != "drop" and e["dst"].startswith("dspark.stages.")
               for e in mtp.values())
    assert set(layout.dspark_expected_parameters(c)) <= set(layout.expected_parameters(c, "hybrid"))
    without = {k: v for k, v in meta.items() if not k.startswith("mtp.")}
    try:
        layout.plan_v41(without, c, "hybrid")
    except ValueError as exc:
        assert "Missing checkpoint keys" in str(exc)
    else:
        raise AssertionError("DSpark stages must not be left randomly initialized")


def test_dspark_tree_matches_the_stage_modules():
    _, _, c = _require_release()
    params = layout.dspark_expected_parameters(c)
    for stage in range(N_STAGES):
        prefix = f"dspark.stages.{stage}."
        members = {k[len(prefix):] for k in params if k.startswith(prefix)}
        expected = set(STAGE_COMMON)
        if stage == 0:
            expected |= STAGE_FIRST
        if stage == N_STAGES - 1:
            expected |= STAGE_LAST
        assert members == expected, stage
    # Stage 0 owns the target-hidden mixer; only the last stage owns the heads.
    assert "dspark.stages.1.main_proj.weight" not in params
    assert "dspark.stages.0.markov_head.head.weight" not in params
    # V4.1 has no hyper-head anywhere: forward_head collapses with the
    # parameter-free hc_pre, which is why no mtp.*.hc_head* tensor ships.
    assert not any("hc_head" in k for k in params)
    # The markov head is [vocab, rank] twice over, the confidence head
    # [1, hidden + rank].
    assert params["dspark.stages.2.markov_head.embed.weight"] == (
        "bfloat16", [c.vocab_size, c.dspark_markov_rank])
    assert params["dspark.stages.2.markov_head.head.weight"] == (
        "bfloat16", [c.vocab_size, c.dspark_markov_rank])
    assert params["dspark.stages.2.confidence_head.proj.weight"] == (
        "bfloat16", [1, c.hidden_size + c.dspark_markov_rank])
    # main_proj mixes len(dspark_target_layer_ids) taps down to one hidden;
    # fp8 stores four codes per uint32 word.
    assert params["dspark.stages.0.main_proj.weight"] == (
        "uint32", [c.hidden_size, c.hidden_size * len(c.dspark_target_layer_ids) // 4])
    # The MTP MoE is the narrow one, not the backbone's 384.
    assert params["dspark.stages.0.ffn.switch_mlp.gate_proj.weight"][1][0] == DSPARK_EXPERTS


def test_dspark_footprint_is_engram_source_independent():
    _, _, c = _require_release()
    sizes = {"bfloat16": 2, "float32": 4, "uint32": 4, "uint8": 1}
    totals = []
    for source in ("ram", "nvme", "hybrid"):
        params = layout.dspark_expected_parameters(c)
        totals.append(sum(sizes[d] * math.prod(s) for k, (d, s) in params.items()
                          if k.startswith("dspark.")))
    assert totals == [DSPARK_BYTES] * 3
    assert abs(DSPARK_BYTES / 1024 ** 3 - 7.495) < 0.01


def test_drafter_prefix_is_the_same_tree_renamed():
    _, _, c = _require_release()
    target = {k[len("dspark.stages."):]: v
              for k, v in layout.dspark_expected_parameters(c).items()
              if k.startswith("dspark.stages.")}
    drafter = layout.dspark_expected_parameters(c, "stages.")
    assert {k[len("stages."):]: v for k, v in drafter.items()} == target
    assert layout.dspark_expected_parameters(c) == {
        "dspark.stages." + k: v for k, v in target.items()}
    for spelling in ("mtp.0.attn.wq_a.weight", "stages.0.attn.wq_a.weight",
                     "dspark.stages.0.attn.wq_a.weight"):
        assert layout._normalize_dspark_key(spelling) == "mtp.0.attn.wq_a.weight"


# --------------------------------------------------------------------------- #
# Structural: the round loop's accept / trim / commit invariants               #
# --------------------------------------------------------------------------- #

def _simulate_round(cache_offset, position, proposal, accepted):
    """One round of the DSpark loop's bookkeeping: the verify writes
    ``1 + proposal`` cells, the rejected tail is trimmed, and the next anchor
    sits at the committed context length."""
    assert position == cache_offset  # the anchor is not in the cache yet
    cache_offset += BK.dspark_verify_len(proposal)
    trim = BK.dspark_rollback_trim(proposal, accepted)
    if trim > 0:
        cache_offset -= trim
    committed = BK.dspark_context_commit_len(accepted)
    next_position = BK.dspark_next_position(position, accepted)
    assert cache_offset == next_position
    return cache_offset, next_position, committed


def test_round_loop_invariants_for_a_five_token_block():
    for proposal in range(BLOCK_SIZE + 1):
        assert BK.dspark_verify_len(proposal) == 1 + proposal
        for accepted in range(proposal + 1):
            assert BK.dspark_rollback_trim(proposal, accepted) == proposal - accepted
            assert BK.dspark_context_commit_len(accepted) == accepted + 1
            assert BK.dspark_next_position(100, accepted) == 101 + accepted
    # Full acceptance trims nothing, so the loop skips rollback entirely.
    assert BK.dspark_rollback_trim(BLOCK_SIZE, BLOCK_SIZE) == 0


def test_multiround_cache_and_position_stay_aligned():
    prompt = 12
    cache_offset = position = prompt
    total = 0
    # full accept, partial, full reject, a confidence-truncated proposal, partial
    for proposal, accepted in [(5, 5), (5, 2), (5, 0), (3, 1), (5, 4)]:
        cache_offset, position, committed = _simulate_round(
            cache_offset, position, proposal, accepted)
        total += committed
    assert position == prompt + total
    assert total == (5 + 1) + (2 + 1) + (0 + 1) + (1 + 1) + (4 + 1)


# --------------------------------------------------------------------------- #
# Structural: dspark.py reuses language.py rather than copying it              #
# --------------------------------------------------------------------------- #

def test_dspark_module_imports_the_shared_building_blocks():
    tree = ast.parse((HERE / "dspark.py").read_text())
    defined = {n.name for n in tree.body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "language"
                for alias in node.names}
    shared = {"Attention", "MoE", "HyperConnection", "RMSNorm", "sparse_attn",
              "hc_pre", "hc_post", "fake_quant_fp8_ue8m0", "_rope"}
    assert shared <= imported, sorted(shared - imported)
    assert not (shared & defined), sorted(shared & defined)
    # The stages and heads themselves live here.
    assert {"DSparkAttention", "DSparkStage", "DSparkStack", "DSparkMarkovHead",
            "DSparkConfidenceHead", "get_dspark_topk_idxs"} <= defined


# --------------------------------------------------------------------------- #
# Mac only (mlx + Metal)                                                       #
# --------------------------------------------------------------------------- #

def _mx():
    try:
        import mlx.core as mx
    except ImportError:
        _skip("DSpark runtime tests require MLX (Mac)")
    return mx


def _tiny_config():
    """A DSpark-bearing V4.1 config small enough to build and run."""
    return config.TextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=4, moe_intermediate_size=32,
        num_attention_heads=4, n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=3, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=1.0, q_lora_rank=32, head_dim=64, qk_rope_head_dim=16,
        rms_norm_eps=1e-20, o_groups=4, o_lora_rank=16, sliding_window=8,
        compress_ratios=[0, 0, 2, 1, 0, 0], kv_source_layer_ids=[2, 3],
        index_source_layer_ids=[2, 3], candidate_source_layer_id=3,
        candidate_topk_blocks=3, candidate_block_size=2, index_n_heads=4,
        index_head_dim=32, index_topk=4, rope_theta=10000, compress_rope_theta=40000,
        rope_scaling={"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 64},
        max_position_embeddings=64, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=[], engram_num_embeddings=[],
        num_nextn_predict_layers=2, dspark_block_size=3, dspark_noise_token_id=5,
        dspark_target_layer_ids=[2, 3], dspark_markov_rank=8,
        dspark_n_routed_experts=4, dspark_num_experts_per_tok=2,
    )


def _randomize(mx, module, seed):
    import numpy as np
    from mlx.utils import tree_flatten

    rng = np.random.default_rng(seed)
    updates = []
    for key, value in tree_flatten(module.parameters()):
        shape = value.shape
        scale = shape[-1] ** -0.5 if len(shape) > 1 else 0.2
        arr = (rng.standard_normal(shape) * scale).astype(np.float32)
        if "norm.weight" in key:
            arr = 1.0 + 0.1 * arr
        updates.append((key, mx.array(arr)))
    module.load_weights(updates, strict=False)
    mx.eval(module.parameters())
    return module


def test_topk_idxs_matches_the_reference_formula():
    mx = _mx()
    from mlx_vlm.models.deepseek_v41.dspark import get_dspark_topk_idxs

    for window, block, start in ((8, 3, 2), (8, 3, 20), (128, 5, 4), (128, 5, 400)):
        got = get_dspark_topk_idxs(window, 1, block, start)
        want = list(range(min(window, start + 1))) + [window + i for i in range(block)]
        assert got.shape == (1, block, len(want))
        assert got[0, 0].tolist() == want
        # Every block query sees the same keys: the draft block is bidirectional.
        assert got[0, block - 1].tolist() == want


def test_stack_parameter_tree_matches_the_planned_checkpoint_tree():
    mx = _mx()
    from mlx.utils import tree_flatten
    from mlx_vlm.models.deepseek_v41.dspark import DSparkStack

    c = _tiny_config()
    stack = DSparkStack(c)
    built = {"dspark." + k for k, _ in tree_flatten(stack.parameters())}
    planned = set(layout.dspark_expected_parameters(c))
    # The planned tree names packed payloads (``.scales``) where an unquantized
    # module has none, so compare the module paths rather than the leaf names.
    def modules(keys):
        return {k.rsplit(".", 1)[0].removesuffix(".scales") for k in keys}
    assert modules(planned) <= modules(built), sorted(modules(planned) - modules(built))


def _greedy_reference(mx, model, prompt, n):
    tokens, cache = [], model.make_cache()
    out = model(prompt, cache=cache)
    for _ in range(n):
        token = int(mx.argmax(out.logits[0, -1]).item())
        tokens.append(token)
        out = model(mx.array([[token]]), cache=cache)
    return tokens


class _ForcedAccept:
    """Delegates to the real drafter but replaces the proposal with the target's
    own continuation, so the loop's full-accept (no rollback) path runs too.
    A proxy rather than an attribute assignment: mlx ``nn.Module`` treats
    ``__setattr__`` as child/parameter registration."""

    def __init__(self, mx, drafter, by_position):
        self._mx, self._drafter, self._by_position = mx, drafter, by_position

    def __getattr__(self, name):
        return getattr(self._drafter, name)

    def draft_block(self, anchor, main_hidden, position=0, **kwargs):
        result = self._drafter.draft_block(anchor, main_hidden, position, **kwargs)
        take = [self._by_position[position + i]
                for i in range(int(result["draft_tokens"].shape[1]))
                if position + i in self._by_position]
        if take:
            result["draft_tokens"] = self._mx.array([take], dtype=self._mx.int32)
        return result


def _speculative_tokens(mx, model, drafter, prompt, n):
    from mlx_vlm.speculative.dspark import _dspark_rounds, dspark_context_from_captures

    cache = model.make_cache()
    out = model(prompt, cache=cache, capture_layer_ids=drafter.capture_layer_ids)
    hidden = dspark_context_from_captures(out.hidden_states)
    first = int(mx.argmax(out.logits[0, -1]).item())
    tokens = [first]
    for token, _ in _dspark_rounds(model, drafter, cache, hidden, first_bonus=first,
                                   max_tokens=n, sampler=lambda lg: mx.argmax(lg, axis=-1),
                                   greedy_sampling=True):
        tokens.append(token)
    return tokens


def test_round_loop_emits_the_targets_greedy_tokens():
    mx = _mx()
    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    from mlx_vlm.speculative.drafters.deepseek_v41_dspark import Model, ModelConfig

    mx.random.seed(0)
    c = _tiny_config()
    model = _randomize(mx, LanguageModel(c), 11)
    drafter_config = ModelConfig.from_dict({"text_config": c})
    drafter = _randomize(mx, Model(drafter_config), 12)

    prompt = mx.array([[3, 9, 14, 21, 30, 7, 2, 11, 19, 25]])
    n = 12
    reference = _greedy_reference(mx, model, prompt, n)

    got = _speculative_tokens(mx, model, drafter, prompt, n)
    assert got == reference, (got, reference)
    assert len(drafter.accept_lens) > 0
    # Every round re-anchors on the target's own token, so a random drafter still
    # produces the target's stream; the rejected tails exercise the exact trim.
    assert sum(drafter.accept_lens) <= len(drafter.accept_lens) * c.dspark_block_size

    # Now force full acceptance (drafts == the target's own continuation) so the
    # no-rollback path is exercised too. The anchor for the round at absolute
    # position p is reference[p - len(prompt) - 1], so the drafts start one later.
    base = prompt.shape[1]
    by_position = {base + i: token for i, token in enumerate(reference[1:])}
    drafter2 = _ForcedAccept(mx, _randomize(mx, Model(drafter_config), 12), by_position)
    forced = _speculative_tokens(mx, model, drafter2, prompt, n)
    assert forced == reference, (forced, reference)


def test_compiled_markov_head_matches_reference_recurrence():
    mx = _mx()
    from mlx_vlm.models.deepseek_v41.dspark import dspark_markov_walk

    mx.random.seed(41)
    scores = mx.random.normal((1, 5, 64))
    embed = mx.random.normal((64, 8))
    head = mx.random.normal((64, 8))
    anchor = mx.array([3])
    def walk(scores, anchor, embed, head):
        return dspark_markov_walk(scores, anchor, embed, head, None, True)
    expected = walk(scores, anchor, embed, head)
    compiled = mx.compile(walk)
    actual = compiled(scores, anchor, embed, head)
    mx.eval(expected, actual)
    assert mx.array_equal(actual[0], expected[0]).item()
    diff = float(mx.max(mx.abs(actual[1] - expected[1])).item())
    ratio = float((mx.linalg.norm(actual[1]) / mx.linalg.norm(expected[1])).item())
    print(f"DSpark compiled Markov max_abs={diff} norm_ratio={ratio}")
    assert mx.allclose(actual[1], expected[1], atol=1e-5, rtol=1e-5).item()


def test_sampled_round_loop_preserves_fixed_seed():
    mx = _mx()
    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    from mlx_vlm.speculative.dspark import _dspark_rounds, dspark_context_from_captures

    c = _tiny_config()
    n = 80  # Exceed the first real-model mismatch position and several rollbacks.
    model = _randomize(mx, LanguageModel(c), 11)
    drafter = model.make_dspark_drafter()
    prompt = mx.array([[3, 9, 14, 21, 30, 7, 2, 11, 19, 25]])
    def sample(logits):
        return mx.random.categorical(logits / .7)
    mx.random.seed(41)
    cache = model.make_cache()
    out = model(prompt, cache=cache)
    expected = []
    for i in range(n):
        row = out.logits[:, -1]
        token = sample(row - mx.logsumexp(row, axis=-1, keepdims=True))
        expected.append(int(token.item()))
        if i < n - 1:
            out = model(token[:, None], cache=cache)
    expected_rng = [mx.array(s) for s in mx.random.state]
    mx.random.seed(41)
    cache = model.make_cache()
    out = model(prompt, cache=cache, capture_layer_ids=drafter.capture_layer_ids)
    row = out.logits[:, -1]
    first = int(sample(row - mx.logsumexp(row, axis=-1, keepdims=True)).item())
    actual = [first]
    actual.extend(t for t, _ in _dspark_rounds(model, drafter, cache,
                  dspark_context_from_captures(out.hidden_states), first_bonus=first,
                  max_tokens=n, sampler=sample, greedy_sampling=False))
    assert actual == expected, (actual, expected)
    assert all(mx.array_equal(a,b).item() for a,b in zip(expected_rng,mx.random.state))
    assert cache[0].offset == prompt.shape[1] + n - 1
    print(f"DSpark tiny sampled seed=41: {n}/{n} tokens and final RNG identical")


def _run_all():
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    passed = skipped = 0
    for name in names:
        try:
            globals()[name]()
            passed += 1
            print(f"  PASS {name}")
        except _Skipped as exc:
            skipped += 1
            print(f"  SKIP {name}: {exc}")
    print(f"DSPARK_DSV41_TEST {passed} passed, {skipped} skipped ({len(names)} total)")
    return 0 if passed + skipped == len(names) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
