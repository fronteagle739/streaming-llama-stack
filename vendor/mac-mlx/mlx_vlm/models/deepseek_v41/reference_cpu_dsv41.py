"""CPU oracle for the unedited DeepSeek reference, with quantization bypassed.

Import this file by path (no MLX/package import required). Override DSV41_REF_DIR
and DSV41_TOKENIZER_DIR to use another local copy. This is a bf16 oracle, not a
bitwise CUDA emulator; see REFERENCE_CPU_NOTES.md for the numerical contract.
"""
from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
import sys
import types

import torch


SCRATCH = Path(
    "C:./AppData/Local/Temp/claude/"
    "d--GitHub-RTX-ACCELERATED-MAC-PREFILL-LLAMA/"
    "5008b92d-5137-4669-8b83-1f5d882708cd/scratchpad"
)
def resolve_dir(env_var, *candidates):
    """Honor explicit overrides, otherwise use the first existing local directory."""
    if env_var in os.environ:
        return Path(os.environ[env_var]).expanduser()
    paths = [Path(path).expanduser() for path in candidates]
    return next((path for path in paths if path.is_dir()), paths[-1])


REF_DIR = resolve_dir(
    "DSV41_REF_DIR", "./tmp/upstream_dsv41/pipenetwork/docs/reference",
    "~/tmp/dsv41/pipenetwork/docs/reference", SCRATCH / "dsv41_ref",
)
TOKENIZER_DIR = resolve_dir(
    "DSV41_TOKENIZER_DIR", SCRATCH / "dsv41_tok",
    "./Models/deepseek-ai/DeepSeek-V4.1-Flash",
)
SEED = 41
FINGERPRINT_IDS = [[1, 73, 204, 17, 981, 32, 67, 441, 9, 352, 1024, 22, 513, 777, 28, 91]]


def _identity_quant(x, block_size, scale_dtype, inplace):
    if x.size(-1) % block_size:
        raise ValueError("quantization block must divide the last dimension")
    if inplace:
        return x  # Same object and storage, including noncontiguous views.
    scales = torch.ones(*x.shape[:-1], x.size(-1) // block_size,
                        dtype=torch.float32, device=x.device).to(scale_dtype)
    return x, scales


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Intentional bf16-mode identity, including the reference's in-place ABI."""
    return _identity_quant(x, block_size, scale_dtype, inplace)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Identity mode returns unpacked x and unit scales, never packed FP4."""
    return _identity_quant(x, block_size, scale_dtype, inplace)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """FP32 block products, then activation and 2-D weight scales, then sum."""
    k, n = a.size(-1), b.size(0)
    out = torch.zeros(*a.shape[:-1], n, dtype=torch.float32, device=a.device)
    for j, lo in enumerate(range(0, k, block_size)):
        product = a[..., lo:lo + block_size].float() @ b[:, lo:lo + block_size].float().T
        bs = b_s.float()[torch.arange(n, device=b.device) // block_size, j]
        out += product * a_s[..., j].float().unsqueeze(-1) * bs
    return out.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """Per-32-K products; unpack low then high E2M1 nibbles if necessary."""
    if b.dtype == torch.float4_e2m1fn_x2:
        raw = b.view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                            -0., -.5, -1, -1.5, -2, -3, -4, -6], device=b.device)
        b = lut[codes]
    out = torch.zeros(*a.shape[:-1], b.size(0), dtype=torch.float32, device=a.device)
    for j, lo in enumerate(range(0, a.size(-1), 32)):
        product = a[..., lo:lo + 32].float() @ b[:, lo:lo + 32].float().T
        out += product * a_s[..., lo // act_block_size].float().unsqueeze(-1) * b_s[:, j].float()
    return out.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Reproduce the 64-slot online softmax, including BF16 exp rounding."""
    b, s, h, d = q.shape
    maximum = torch.full((b, s, h), -1e30, dtype=torch.float32, device=q.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q, dtype=torch.float32)
    batch = torch.arange(b, device=q.device)[:, None, None]
    for lo in range(0, topk_idxs.size(-1), 64):
        idx = topk_idxs[..., lo:lo + 64].long()
        valid = idx != -1
        rows = kv[batch, idx.clamp_min(0)].masked_fill(~valid.unsqueeze(-1), 0).float()
        scores = torch.einsum("bshd,bskd->bshk", q.float(), rows) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(-2), -torch.inf)
        new_max = torch.maximum(maximum, scores.amax(-1))
        correction = (maximum - new_max).exp()
        exp = (scores - new_max.unsqueeze(-1)).exp()
        denominator = denominator * correction + exp.sum(-1)
        numerator = numerator * correction.unsqueeze(-1) + torch.einsum(
            "bshk,bskd->bshd", exp.to(torch.bfloat16).float(), rows)
        maximum = new_max
    denominator += (attn_sink.float() - maximum).exp()
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype).contiguous()


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    h = hc_mult
    mixes, hc_scale, hc_base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = (mixes[..., :h] * hc_scale[0] + hc_base[:h]).sigmoid() + eps
    post = 2 * (mixes[..., h:2*h] * hc_scale[1] + hc_base[h:2*h]).sigmoid()
    comb = (mixes[..., 2*h:] * hc_scale[2] + hc_base[2*h:]).unflatten(-1, (h, h))
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _load(path, name):
    # Same path-loader pattern as test_engram_hash_parity.py; no MLX imports.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need their module during exec.
    spec.loader.exec_module(module)
    return module


def _load_reference():
    names = ("kernel", "engram", "vision", "image_processor")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        kernel = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                     "sparse_attn", "hc_split_sinkhorn"):
            setattr(kernel, name, globals()[name])
        sys.modules["kernel"] = kernel
        for name in names[1:]:
            _load(REF_DIR / f"{name}.py", name)
        return _load(REF_DIR / "model.py", "_dsv41_original_cpu_model")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# Missing local inputs must not abort pytest collection. Tests check these paths
# before constructing a model; the numerical shims remain usable without them.
REFERENCE_FILES = ("model.py", "kernel.py", "engram.py", "vision.py", "image_processor.py")
reference = _load_reference() if all((REF_DIR / name).is_file() for name in REFERENCE_FILES) else None
ModelArgs = reference.ModelArgs if reference is not None else None
Transformer = reference.Transformer if reference is not None else None


def tiny_args(tokenizer):
    return ModelArgs(
        dtype="bf16", expert_dtype=None, temperature=0, max_batch_size=2,
        max_seq_len=64, vocab_size=len(tokenizer), dim=64, n_layers=6,
        n_mtp_layers=0, moe_inter_dim=64, n_heads=4, head_dim=32,
        rope_head_dim=8, q_lora_rank=32, o_groups=2, o_lora_rank=16,
        n_routed_experts=4, n_activated_experts=2, window_size=8,
        compress_ratios=(0, 2, 2, 1, 1, 0), kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4), index_n_heads=2, index_head_dim=32,
        index_topk=4, candidate_source_layer=3, candidate_topk_blocks=2,
        candidate_block_size=2, engram_layer_ids=(1, 4),
        engram_num_embeddings=(4200, 4200), engram_vocab_size=1000,
        engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32,
        engram_compressed_vocab_size=99092, vision_n_layers=0, dspark_block_size=0,
    )


def build_reference(args: ModelArgs, tokenizer, *, seed: int = SEED) -> Transformer:
    """Build original classes on CPU; seed all parameters, preserving mixed dtypes.

    Random draws are FP32 normal(0, .02), then cast; norm weights stay one,
    Engram scales are random E8M0 bytes 126..128, HC scales are .1.
    RNG/default dtype/device are restored. Each model owns a shared runtime.
    """
    if args.dtype != "bf16" or args.expert_dtype is not None:
        raise ValueError("This oracle requires dtype='bf16', expert_dtype=None")
    if args.vision_enabled or args.dspark_block_size:
        raise ValueError("This harness supports the text backbone only")
    layout = reference.EngramLayout.from_args(args)
    if layout:
        for rows, primes in zip(layout.num_embeddings, layout.primes):
            if rows < sum(sum(group) for group in primes):
                raise ValueError("Engram table must contain all prime buckets")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"), reference.set_dtype(torch.bfloat16):
        torch.manual_seed(seed)
        model = Transformer(args, tokenizer)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype == torch.float8_e8m0fnu:
                    param.view(torch.uint8).copy_(torch.randint(126, 129, param.shape, dtype=torch.uint8))
                elif name.endswith(("q_weight", "k_weight")) or (name.endswith("weight") and param.ndim == 1):
                    param.fill_(1)
                elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
                    param.fill_(.1)
                else:
                    param.copy_((torch.randn(param.shape, dtype=torch.float32) * .02).to(param.dtype))
    model.requires_grad_(False).eval()
    model._cpu_runtime = reference.SharedAttentionRuntime()
    model._cpu_args = args
    model._cpu_seed = seed
    return model


@torch.inference_mode()
def forward_logits(model, input_ids, start_pos=0):
    """Return (FP32 logits [B,S,V], reference output token IDs [B,S]).

    A temporary head pre-hook requests full_logits=True without copying or
    rewriting Transformer.forward. Temperature zero gives deterministic argmax.
    Calls are serial: the original model uses process-global attention state.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device="cpu")
    if ids.ndim != 2 or ids.size(1) == 0 or (start_pos and ids.size(1) != 1):
        raise ValueError("Expected nonempty [B,S] prefill or [B,1] decode")
    if start_pos < 0 or start_pos + ids.size(1) > model.max_seq_len:
        raise ValueError("Positions exceed the model cache")
    def all_positions(module, inputs, kwargs):
        return inputs, dict(kwargs, full_logits=True)
    previous = reference.shared_attn
    reference.shared_attn = model._cpu_runtime
    hook = model.head.register_forward_pre_hook(all_positions, with_kwargs=True)
    try:
        with torch.device("cpu"), reference.set_dtype(torch.bfloat16):
            output_ids, logits, _ = model(ids, start_pos)
        return logits.float(), output_ids
    finally:
        hook.remove()
        reference.shared_attn = previous


def run_split(model, ids, cut):
    """Prefill once, then one-token decode; return concatenated logits and IDs."""
    ids = torch.as_tensor(ids, dtype=torch.long, device="cpu")
    if not 1 <= cut <= ids.size(1):
        raise ValueError("cut must be inside the sequence")
    parts = [forward_logits(model, ids[:, :cut], 0)]
    parts.extend(forward_logits(model, ids[:, p:p+1], p) for p in range(cut, ids.size(1)))
    return tuple(torch.cat([part[i] for part in parts], dim=1) for i in (0, 1))


def make_fingerprint(model, ids=FINGERPRINT_IDS):
    """JSON-ready whole-sequence evidence; never writes/updates a golden file."""
    from dataclasses import asdict
    import hashlib
    import json
    import numpy as np

    logits, tokens = forward_logits(model, ids, 0)
    probes = [[0, pos, col] for pos in range(logits.size(1))
              for col in (0, 1, 41, 1024, logits.size(-1) - 1)]
    hashes = {name: hashlib.sha256((REF_DIR / name).read_bytes().replace(b"\r", b"")).hexdigest()
              for name in ("model.py", "kernel.py", "engram.py", "vision.py", "image_processor.py")}
    tok_hashes = {name: hashlib.sha256((TOKENIZER_DIR / name).read_bytes()).hexdigest()
                  for name in ("tokenizer.json", "tokenizer_config.json")}
    return dict(
        schema_version=1, seed=model._cpu_seed, torch_version=torch.__version__,
        platform=platform.system(),
        numpy_version=np.__version__, num_threads=torch.get_num_threads(),
        contract="Original text backbone; bf16 quantization bypass; 64-slot attention with bf16 exp rounding",
        initialization="FP32 normal(0,.02) then cast; norms=1; HC scales=.1; Engram scale bytes uniform 126..128",
        args=json.loads(json.dumps(asdict(model._cpu_args))), source_sha256=hashes,
        tokenizer_sha256=tok_hashes, input_ids=torch.as_tensor(ids).tolist(),
        shape=list(logits.shape), sum_float64=logits.double().sum().item(),
        sum_squares_float64=logits.double().square().sum().item(),
        per_position_sum_float64=logits.double().sum(-1).tolist(),
        probe_indices=probes, probe_values=[logits[tuple(p)].item() for p in probes],
        output_token_ids=tokens.tolist(),
    )
