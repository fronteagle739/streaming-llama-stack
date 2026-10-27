import argparse
import asyncio
import inspect
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .models.base import create_attention_mask
from .utils import (
    _pipeline_local_files,
    _pipeline_target,
    get_model_path,
    load_model,
    load_processor,
)


@dataclass(frozen=True)
class _PipelineGroup:
    rank_value: int
    size_value: int

    def rank(self) -> int:
        return self.rank_value

    def size(self) -> int:
        return self.size_value


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_json(sock: socket.socket, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _tune_rpc_socket(sock: socket.socket) -> None:
    # Apply on accepted sockets too: otherwise the rank1 response can wait on
    # Nagle/delayed ACK even though the initiating rank0 socket is tuned.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, option, 1 << 20)
        except OSError:
            pass  # OS buffer ceilings differ; framing and NODELAY still apply.


def _send_buffers(sock: socket.socket, buffers) -> None:
    """One scatter/gather write for metadata + tensor, handling short writes."""
    pending = [memoryview(value).cast("B") for value in buffers if len(value)]
    if not hasattr(sock, "sendmsg"):
        for value in pending:
            sock.sendall(value)
        return
    while pending:
        sent = sock.sendmsg(pending)
        if sent <= 0:
            raise ConnectionError("socket closed while sending")
        while pending and sent >= len(pending[0]):
            sent -= len(pending.pop(0))
        if pending and sent:
            pending[0] = pending[0][sent:]


def _recv_json(sock: socket.socket) -> dict:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


def _dtype_name(dtype) -> str:
    for name in (
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool_",
    ):
        if dtype == getattr(mx, name, None):
            return name
    return str(dtype).split(".")[-1]


def _coerce_boundary_dtype(array: mx.array, boundary_dtype: str) -> mx.array:
    if boundary_dtype == "preserve":
        return array
    return array.astype(getattr(mx, boundary_dtype))


def _array_to_wire(array: mx.array) -> tuple[np.ndarray, str]:
    dtype = _dtype_name(array.dtype)
    if array.dtype == mx.bfloat16:
        # NumPy has no portable bfloat16 dtype. Bitcast through uint16 so the
        # RPC boundary preserves the activation dtype instead of promoting to fp32.
        return np.ascontiguousarray(np.array(array.view(mx.uint16))), dtype
    return np.ascontiguousarray(np.array(array)), dtype


def _wire_to_array(data: bytes, dtype_name: str, shape: tuple[int, ...]) -> mx.array:
    if dtype_name == "bfloat16":
        np_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        return mx.array(np_array).view(mx.bfloat16)
    np_dtype = np.dtype("bool" if dtype_name == "bool_" else dtype_name)
    np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    return mx.array(np_array)


def _send_array(
    sock: socket.socket, array: mx.array, *, boundary_dtype: str = "preserve"
) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    start = time.perf_counter()
    array = _coerce_boundary_dtype(array, boundary_dtype)
    timing["coerce_s"] = time.perf_counter() - start
    start = time.perf_counter()
    np_array, dtype = _array_to_wire(array)
    timing["host_copy_s"] = time.perf_counter() - start
    meta = {
        "shape": list(np_array.shape),
        "dtype": dtype,
        "nbytes": int(np_array.nbytes),
    }
    start = time.perf_counter()
    header = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    timing["header_send_s"] = time.perf_counter() - start
    start = time.perf_counter()
    _send_buffers(sock, (struct.pack("!I", len(header)) + header,
                         memoryview(np_array).cast("B")))
    timing["payload_send_s"] = time.perf_counter() - start
    timing.update(meta)
    return timing


def _recv_array_timed(sock: socket.socket) -> tuple[mx.array, dict[str, Any]]:
    timing: dict[str, Any] = {}
    start = time.perf_counter()
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    meta = json.loads(_recv_exact(sock, size).decode("utf-8"))
    timing["header_recv_s"] = time.perf_counter() - start
    start = time.perf_counter()
    data = _recv_exact(sock, int(meta["nbytes"]))
    timing["payload_recv_s"] = time.perf_counter() - start
    start = time.perf_counter()
    array = _wire_to_array(data, meta["dtype"], tuple(meta["shape"]))
    timing["device_copy_s"] = time.perf_counter() - start
    timing.update(
        {
            "hidden_nbytes": int(meta["nbytes"]),
            "hidden_wire_dtype": meta["dtype"],
            "hidden_wire_shape": list(meta["shape"]),
        }
    )
    return array, timing


def _recv_array(sock: socket.socket) -> mx.array:
    array, _timing = _recv_array_timed(sock)
    return array


def _assert_gpu_runtime(module, *, rank: int, require_gpu: bool) -> None:
    info = _device_summary(module)
    print(
        "RPC_GPU_ASSERT "
        + json.dumps({"rank": rank, "require_gpu": require_gpu, **info}, separators=(",", ":")),
        flush=True,
    )
    if not require_gpu:
        return
    default_device = str(info.get("default_device") or "").lower()
    param_devices = " ".join(str(k).lower() for k in info.get("param_devices", {}))
    if "gpu" not in default_device and "gpu" not in param_devices:
        raise RuntimeError(
            "MLX RPC rank is not GPU-backed "
            f"(rank={rank}, default_device={info.get('default_device')}, "
            f"param_devices={info.get('param_devices')})"
        )


def _cache_values(cache) -> Iterable[mx.array]:
    if cache is None:
        return
    for attr in ("keys", "values", "buf_kv", "buf_gate", "pooled"):
        value = getattr(cache, attr, None)
        if isinstance(value, mx.array):
            yield value
    state = getattr(cache, "state", None)
    if isinstance(state, (list, tuple)):
        for item in state:
            yield from _cache_values(item)
    if hasattr(cache, "caches"):
        for item in cache.caches:
            yield from _cache_values(item)
        return
    if isinstance(cache, (list, tuple)):
        for item in cache:
            yield from _cache_values(item)


def _profile_arrays(value) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _profile_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _profile_arrays(item)


def _eval_with_cache(hidden: mx.array, cache) -> None:
    mx.eval(hidden, *list(_cache_values(cache)))


def _quant_module_summary(module) -> dict[str, Any]:
    try:
        flat_modules = tree_flatten(
            module.leaf_modules(), is_leaf=nn.Module.is_module
        )
    except Exception:
        return {}

    combos: dict[tuple[str, int, int, str], int] = {}
    for _path, leaf in flat_modules:
        bits = getattr(leaf, "bits", None)
        group_size = getattr(leaf, "group_size", None)
        if bits is None or group_size is None:
            continue
        mode = str(getattr(leaf, "mode", "affine"))
        key = (type(leaf).__name__, int(bits), int(group_size), mode)
        combos[key] = combos.get(key, 0) + 1

    return {
        "total": sum(combos.values()),
        "combos": [
            {
                "class": cls,
                "bits": bits,
                "group_size": group_size,
                "mode": mode,
                "count": count,
            }
            for (cls, bits, group_size, mode), count in sorted(combos.items())
        ],
    }


def _device_summary(module) -> dict[str, Any]:
    arrays = [
        (name, value)
        for name, value in tree_flatten(module.parameters())
        if isinstance(value, mx.array)
    ]
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for name, value in arrays:
        dtype = _dtype_name(value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        device = getattr(value, "device", None)
        if callable(device):
            try:
                device = device()
            except TypeError:
                pass
        device_name = str(device)
        device_counts[device_name] = device_counts.get(device_name, 0) + 1
        if len(samples) < 8:
            samples.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": dtype,
                    "device": device_name,
                }
            )
    info: dict[str, Any] = {
        "default_device": str(mx.default_device()),
        "param_count": len(arrays),
        "param_dtypes": dtype_counts,
        "param_devices": device_counts,
        "samples": samples,
        "has_metal": hasattr(mx, "metal"),
    }
    if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
        info["metal_available"] = bool(mx.metal.is_available())
    return info


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _configure_memory_limits(*, wired_limit_mb: int | None, memory_limit_mb: int | None):
    def max_recommended_working_set_bytes() -> int | None:
        for owner in (mx, getattr(mx, "metal", None)):
            if owner is None:
                continue
            info_fn = getattr(owner, "device_info", None)
            if info_fn is None:
                continue
            try:
                info = info_fn()
            except Exception:
                continue
            if isinstance(info, dict):
                value = info.get("max_recommended_working_set_size")
                if value:
                    return int(value)
        return None

    def apply_limit(label: str, setter, mb: int | None):
        if not mb or setter is None:
            return
        limit = int(mb) * 1024 * 1024
        try:
            setter(limit)
        except ValueError as exc:
            max_ws = max_recommended_working_set_bytes()
            over_working_set = (
                "wired" in label
                and "working set" in str(exc).lower()
                and max_ws is not None
                and limit > max_ws
            )
            if over_working_set:
                max_mb = int(max_ws / 1024 / 1024)
                try:
                    setter(max_ws)
                except ValueError as clamp_exc:
                    print(
                        f"RPC_MEMORY_LIMIT_SKIPPED {label}_mb={int(mb)} "
                        f"max_recommended_mb={max_mb} error={exc} "
                        f"clamp_error={clamp_exc}",
                        flush=True,
                    )
                    return
                print(
                    f"RPC_MEMORY_LIMIT_CLAMPED {label}_mb={int(mb)} "
                    f"applied_mb={max_mb} error={exc}",
                    flush=True,
                )
                return
            raise
        print(f"RPC_MEMORY_LIMIT {label}_mb={int(mb)}", flush=True)

    apply_limit("mx_wired_limit", getattr(mx, "set_wired_limit", None), wired_limit_mb)
    if hasattr(mx, "metal"):
        apply_limit(
            "metal_wired_limit",
            getattr(mx.metal, "set_wired_limit", None),
            wired_limit_mb,
        )
        apply_limit(
            "metal_memory_limit",
            getattr(mx.metal, "set_memory_limit", None),
            memory_limit_mb,
        )
    apply_limit("mx_memory_limit", getattr(mx, "set_memory_limit", None), memory_limit_mb)


def _force_residency(module, *, label: str) -> None:
    flat_params = tree_flatten(module.parameters())
    total = len(flat_params)
    start = time.perf_counter()
    for idx, (name, value) in enumerate(flat_params, 1):
        if not isinstance(value, mx.array):
            continue
        mx.eval(mx.sum(value))
        if idx == 1 or idx == total or idx % 100 == 0:
            print(
                "RPC_FORCE_RESIDENCY "
                f"{label} {idx}/{total} name={name} "
                f"elapsed_s={time.perf_counter() - start:.3f}",
                flush=True,
            )


def _load_pipeline_rank(
    model: str,
    *,
    rank: int,
    size: int,
    pipeline_layers: str,
    force_residency: bool,
    require_gpu: bool,
):
    os.environ["MLX_VLM_PIPELINE_LAYERS"] = pipeline_layers
    model_path = get_model_path(model)
    group = _PipelineGroup(rank, size)

    raw_config = json.loads((Path(model_path) / "config.json").read_text())
    if raw_config.get("model_type") == "mimo_v2":
        from .models.mimo_v2.raw_loader import is_native, load_native

        if is_native(raw_config):
            remote = None
            descriptor = Path(model_path) / "studio-tensor-source.json"
            if rank == 1 and descriptor.exists():
                remote = json.loads(descriptor.read_text())
                remote["token"] = os.environ.get("STUDIO_PAIR_TOKEN", "")
            loaded = load_native(model_path, pipeline_group=group, remote=remote)
            # The first-stage worker receives token IDs and encoded features.
            # It needs neither the media processor nor PyTorch dependencies.
            processor = (load_processor(model_path, True, eos_token_ids=raw_config.get("eos_token_id"))
                         if rank == 0 else None)
            if force_residency:
                _force_residency(loaded.language_model, label=f"rank={rank}")
            _assert_gpu_runtime(loaded.language_model, rank=rank, require_gpu=require_gpu)
            inner = loaded.language_model
            capacity_tokens = int(os.environ.get("MLX_MIMO_CAPACITY_TOKENS", "0"))
            if capacity_tokens:
                from .models.mimo_v2.capacity import probe_capacity
                def capacity_forward(cache):
                    if rank == 1:
                        return _rank1_forward(inner, [1], cache)
                    hidden = mx.zeros((1, 1, inner.args.hidden_size), dtype=mx.bfloat16)
                    return _rank0_logits(inner, hidden, cache, [1])
                probe_capacity(inner, capacity_tokens, capacity_forward,
                    limit_bytes=int(os.environ.get("MLX_MIMO_CAPACITY_LIMIT", "0")))
            print("RPC_QUANT_MODULES " + json.dumps({"rank": rank, **_quant_module_summary(inner)}), flush=True)
            print(f"RPC rank {rank}: materialized native MiMo", flush=True)
            return loaded, processor, inner.start_idx, inner.end_idx

    skeleton = load_model(model_path, lazy=True, strict=False, load_weights=False)
    _check_v41_rpc_support(skeleton.language_model)
    config = skeleton.config.to_dict()
    inner = _pipeline_target(skeleton)
    inner.pipeline(group)
    local_files = _pipeline_local_files(
        skeleton, inner, model_path, config, group
    )
    start_idx = getattr(inner, "start_idx", None)
    end_idx = getattr(inner, "end_idx", None)
    del skeleton, inner

    loaded = load_model(
        model_path,
        lazy=True,
        strict=False,
        weight_files=local_files,
    )
    inner = _pipeline_target(loaded)
    inner.pipeline(group)
    processor = load_processor(
        model_path, True, eos_token_ids=config.get("eos_token_id", None)
    )
    print(
        f"RPC rank {rank}: loading layers {start_idx}:{end_idx} "
        f"from {Path(model_path)}",
        flush=True,
    )
    mx.eval(loaded.language_model.parameters())
    if force_residency:
        _force_residency(loaded.language_model, label=f"rank={rank}")
    loaded.eval()
    print(
        "RPC_DEVICE_INFO "
        + json.dumps(
            {"rank": rank, **_device_summary(loaded.language_model)},
            separators=(",", ":"),
        ),
        flush=True,
    )
    quant_summary = _quant_module_summary(loaded.language_model)
    if quant_summary.get("total"):
        print(
            "RPC_QUANT_MODULES "
            + json.dumps({"rank": rank, **quant_summary}, separators=(",", ":")),
            flush=True,
        )
    _assert_gpu_runtime(loaded.language_model, rank=rank, require_gpu=require_gpu)
    print(f"RPC rank {rank}: materialized", flush=True)
    return loaded, processor, int(start_idx), int(end_idx)


@lru_cache(maxsize=None)
def _accepted_params_for_type(layer_type: type) -> set[str]:
    try:
        return set(inspect.signature(layer_type.__call__).parameters)
    except (TypeError, ValueError):
        return set()


def _core_model(language_model):
    return getattr(language_model, "model", language_model)


def _local_layers(language_model) -> list[Any]:
    for obj in (language_model, _core_model(language_model)):
        if hasattr(obj, "pipeline_layers"):
            return list(obj.pipeline_layers)
    if hasattr(language_model, "layers"):
        return list(language_model.layers)
    raise ValueError("model does not expose pipeline layers")


def _token_embedding(language_model):
    candidates = (
        (language_model, "input_embeds"),
        (_core_model(language_model), "embed_tokens"),
        (_core_model(language_model), "wte"),
        (_core_model(language_model), "embeddings"),
        (language_model, "embed_tokens"),
    )
    for module, attr in candidates:
        if hasattr(module, attr):
            return getattr(module, attr)
    raise ValueError("model does not expose token embeddings")


def _input_embeds(language_model, tokens: mx.array) -> mx.array:
    embedding = _token_embedding(language_model)
    hidden = embedding(tokens)
    return _prepare_pipeline_hidden(language_model, hidden)


def _prepare_pipeline_hidden(language_model, hidden: mx.array) -> mx.array:
    core = _core_model(language_model)
    args = getattr(core, "args", getattr(language_model, "args", None))
    hc_mult = getattr(args, "hc_mult", None)
    if hc_mult and hidden.ndim == 3 and hasattr(core, "hc_head"):
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (hidden.shape[0], hidden.shape[1], int(hc_mult), hidden.shape[2]),
        )
        hidden = mx.contiguous(hidden)
    return hidden


def _first_cache_leaf(cache):
    if cache is None:
        return None
    if hasattr(cache, "caches"):
        return _first_cache_leaf(cache.caches[0] if cache.caches else None)
    if isinstance(cache, (list, tuple)):
        return _first_cache_leaf(cache[0] if cache else None)
    return cache


def _cache_offset(cache) -> int:
    leaf = _first_cache_leaf(cache)
    offset = getattr(leaf, "offset", 0)
    try:
        return int(offset)
    except (TypeError, ValueError):
        return 0


def _make_attention_mask(language_model, hidden: mx.array, cache, layer=None):
    core = _core_model(language_model)
    args = getattr(core, "args", getattr(language_model, "args", None))
    mask_hidden = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
    window_size = None
    if layer is not None and getattr(layer, "is_sliding_window", False):
        window_size = getattr(args, "sliding_window_size", None)
    elif hidden.ndim == 4:
        window_size = getattr(args, "sliding_window", None)
    if window_size is None and getattr(args, "use_sliding_window", False):
        window_size = getattr(args, "max_window_layers", None)

    kwargs = {}
    if window_size is not None:
        kwargs["window_size"] = window_size
    if hidden.ndim == 4:
        kwargs["return_array"] = True
    try:
        return create_attention_mask(mask_hidden, _first_cache_leaf(cache), **kwargs)
    except TypeError:
        kwargs.pop("return_array", None)
        return create_attention_mask(mask_hidden, _first_cache_leaf(cache), **kwargs)


def _precompute_layer_masks(language_model, hidden: mx.array, cache, layers):
    token_length = int(hidden.shape[1]) if hidden.ndim >= 2 else 0
    if token_length == 1 and _cache_offset(cache[0] if cache else None) > 0:
        return [None] * len(layers)

    if hidden.ndim == 4:
        mask = _make_attention_mask(
            language_model, hidden, cache[0] if cache else None, None
        )
        return [mask] * len(layers)

    if not any(getattr(layer, "is_sliding_window", False) for layer in layers):
        mask = _make_attention_mask(
            language_model, hidden, cache[0] if cache else None, None
        )
        return [mask] * len(layers)

    full_item = next(
        (
            (layer, layer_cache)
            for layer, layer_cache in zip(layers, cache)
            if not getattr(layer, "is_sliding_window", False)
        ),
        (None, cache[0] if cache else None),
    )
    swa_item = next(
        (
            (layer, layer_cache)
            for layer, layer_cache in zip(layers, cache)
            if getattr(layer, "is_sliding_window", False)
        ),
        (None, cache[0] if cache else None),
    )
    full_mask = _make_attention_mask(language_model, hidden, full_item[1], full_item[0])
    swa_mask = _make_attention_mask(language_model, hidden, swa_item[1], swa_item[0])
    return [
        swa_mask if getattr(layer, "is_sliding_window", False) else full_mask
        for layer in layers
    ]


def _call_pipeline_layer(layer, hidden: mx.array, mask, cache, tokens: mx.array):
    params = _accepted_params_for_type(type(layer))
    kwargs = {}
    if "mask" in params:
        kwargs["mask"] = mask
    if "cache" in params:
        kwargs["cache"] = cache
    if "input_ids" in params:
        kwargs["input_ids"] = tokens
    if kwargs:
        return layer(hidden, **kwargs)
    try:
        return layer(hidden, mask, cache, tokens)
    except TypeError:
        return layer(hidden, mask, cache)


def _profile_add(
    record: dict[str, Any], profile: dict[str, Any], phase: str, elapsed: float
) -> None:
    phases = record.setdefault("phases", {})
    phases[phase] = phases.get(phase, 0.0) + elapsed
    totals = profile.setdefault("phase_totals_s", {})
    totals[phase] = totals.get(phase, 0.0) + elapsed


def _profile_eval(
    record: dict[str, Any],
    profile: dict[str, Any],
    phase: str,
    *values,
) -> None:
    arrays: list[mx.array] = []
    for value in values:
        arrays.extend(_profile_arrays(value))
    start = time.perf_counter()
    if arrays:
        mx.eval(*arrays)
    _profile_add(record, profile, phase, time.perf_counter() - start)


def _profile_moe(
    module,
    hidden: mx.array,
    tokens: mx.array,
    record: dict[str, Any],
    profile: dict[str, Any],
):
    if _is_v41_module(module):
        # V4.1 returns (weights, indices) and combines in fp32. Its own body
        # is the profiling reference; the generic decomposition is different.
        out = module(hidden)
        _profile_eval(record, profile, "moe_total_v41", out)
        return out
    if not (hasattr(module, "gate") and hasattr(module, "switch_mlp")):
        out = module(hidden)
        _profile_eval(record, profile, "mlp_total", out)
        return out

    if getattr(module, "sharding_group", None) is not None:
        out = module(hidden, tokens)
        _profile_eval(record, profile, "moe_total_sharded", out)
        return out

    start = time.perf_counter()
    try:
        inds, scores = module.gate(hidden, tokens)
    except TypeError:
        inds, scores = module.gate(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "moe_gate_build", build_s)
    _profile_eval(record, profile, "moe_gate_eval", inds, scores)

    start = time.perf_counter()
    routed = module.switch_mlp(hidden, inds)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "moe_routed_build", build_s)
    _profile_eval(record, profile, "moe_routed_eval", routed)

    start = time.perf_counter()
    if module.__class__.__module__.endswith("deepseek_v4.language"):
        weighted = (routed * scores[..., None].astype(routed.dtype)).sum(-2)
    else:
        weighted = (routed * scores[..., None]).sum(axis=-2).astype(routed.dtype)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "moe_weighted_build", build_s)
    _profile_eval(record, profile, "moe_weighted_eval", weighted)

    if not hasattr(module, "shared_experts"):
        return weighted

    start = time.perf_counter()
    shared = module.shared_experts(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "moe_shared_build", build_s)
    _profile_eval(record, profile, "moe_shared_eval", shared)

    start = time.perf_counter()
    out = weighted + shared
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "moe_sum_build", build_s)
    _profile_eval(record, profile, "moe_sum_eval", out)
    return out


def _profile_deepseek_layer(
    layer,
    hidden: mx.array,
    mask,
    cache,
    tokens: mx.array,
    record: dict[str, Any],
    profile: dict[str, Any],
):
    _check_v41_rpc_support(layer)
    from .models.deepseek_v4.hyper_connection import hc_expand

    residual = hidden
    start = time.perf_counter()
    x, post, comb = layer.attn_hc(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_hc_build", build_s)
    _profile_eval(record, profile, "attn_hc_eval", x, post, comb)

    start = time.perf_counter()
    x = layer.attn_norm(x)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_norm_build", build_s)
    _profile_eval(record, profile, "attn_norm_eval", x)

    start = time.perf_counter()
    x = layer.attn(x, mask=mask, cache=cache)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_build", build_s)
    _profile_eval(record, profile, "attn_eval", x, cache)

    start = time.perf_counter()
    hidden = hc_expand(x, residual, post, comb)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_residual_build", build_s)
    _profile_eval(record, profile, "attn_residual_eval", hidden)

    residual = hidden
    start = time.perf_counter()
    x, post, comb = layer.ffn_hc(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "ffn_hc_build", build_s)
    _profile_eval(record, profile, "ffn_hc_eval", x, post, comb)

    start = time.perf_counter()
    x = layer.ffn_norm(x)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "ffn_norm_build", build_s)
    _profile_eval(record, profile, "ffn_norm_eval", x)

    x = _profile_moe(layer.ffn, x, tokens, record, profile)

    start = time.perf_counter()
    hidden = hc_expand(x, residual, post, comb)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "ffn_residual_build", build_s)
    _profile_eval(record, profile, "ffn_residual_eval", hidden)
    return hidden


def _profile_decoder_layer(
    layer,
    hidden: mx.array,
    mask,
    cache,
    tokens: mx.array,
    record: dict[str, Any],
    profile: dict[str, Any],
):
    start = time.perf_counter()
    x = layer.input_layernorm(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_norm_build", build_s)
    _profile_eval(record, profile, "attn_norm_eval", x)

    start = time.perf_counter()
    attn = layer.self_attn(x, mask, cache)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_build", build_s)
    _profile_eval(record, profile, "attn_eval", attn, cache)

    start = time.perf_counter()
    hidden = hidden + attn
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "attn_residual_build", build_s)
    _profile_eval(record, profile, "attn_residual_eval", hidden)

    start = time.perf_counter()
    x = layer.post_attention_layernorm(hidden)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "ffn_norm_build", build_s)
    _profile_eval(record, profile, "ffn_norm_eval", x)

    mlp = _profile_moe(layer.mlp, x, tokens, record, profile)

    start = time.perf_counter()
    hidden = hidden + mlp
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "ffn_residual_build", build_s)
    _profile_eval(record, profile, "ffn_residual_eval", hidden)
    return hidden


def _profile_pipeline_layer(
    layer,
    hidden: mx.array,
    mask,
    cache,
    tokens: mx.array,
    record: dict[str, Any],
    profile: dict[str, Any],
):
    if all(hasattr(layer, attr) for attr in ("attn_hc", "ffn_hc", "attn", "ffn")):
        return _profile_deepseek_layer(layer, hidden, mask, cache, tokens, record, profile)
    if all(
        hasattr(layer, attr)
        for attr in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    ):
        return _profile_decoder_layer(layer, hidden, mask, cache, tokens, record, profile)

    start = time.perf_counter()
    hidden = _call_pipeline_layer(layer, hidden, mask, cache, tokens)
    build_s = time.perf_counter() - start
    _profile_add(record, profile, "layer_fallback_build", build_s)
    _profile_eval(record, profile, "layer_fallback_eval", hidden, cache)
    return hidden


def _run_rank_layers(
    language_model,
    hidden: mx.array,
    cache,
    tokens: mx.array,
    profile: dict[str, Any] | None = None,
):
    _check_v41_rpc_support(language_model)
    layers = _local_layers(language_model)
    if cache is None:
        cache = [None] * len(layers)
    masks = _precompute_layer_masks(language_model, hidden, cache, layers)
    for local_idx, (layer, layer_cache, mask) in enumerate(zip(layers, cache, masks)):
        if profile is None:
            hidden = _call_pipeline_layer(layer, hidden, mask, layer_cache, tokens)
            continue
        record: dict[str, Any] = {
            "local_idx": local_idx,
            "class": type(layer).__name__,
            "input_dtype": _dtype_name(hidden.dtype),
            "input_shape": list(hidden.shape),
        }
        start = time.perf_counter()
        hidden = _profile_pipeline_layer(
            layer, hidden, mask, layer_cache, tokens, record, profile
        )
        record["total_s"] = time.perf_counter() - start
        record["output_dtype"] = _dtype_name(hidden.dtype)
        record["output_shape"] = list(hidden.shape)
        profile.setdefault("layers", []).append(record)
    mx.eval(hidden)
    return hidden


def _is_v41_module(module):
    # Inspect bases as well: the component profiler wraps module classes.
    return any(cls.__module__.endswith("deepseek_v41.language")
               for cls in type(module).__mro__)


def _check_v41_rpc_support(module):
    if _is_v41_module(module):
        from .models.deepseek_v41.distribution import reject_rpc_pipeline

        reject_rpc_pipeline()


def _rank1_forward(language_model, input_ids: list[int], cache, modal=None):
    tokens = mx.array([input_ids], dtype=mx.int32)
    hidden = _input_embeds(language_model, tokens)
    if modal is not None:
        positions, values = modal
        if values.ndim != 2 or values.shape != (len(positions), hidden.shape[-1]):
            raise ValueError("invalid multimodal embedding shape")
        if any(i < 0 or i >= len(input_ids) for i in positions):
            raise ValueError("multimodal position outside input chunk")
        hidden[:, mx.array(positions), :] = values.astype(hidden.dtype)
    return _run_rank_layers(language_model, hidden, cache, tokens)


def _final_norm(language_model, hidden: mx.array):
    if hasattr(language_model, "_final_norm"):
        return language_model._final_norm(hidden)
    core = _core_model(language_model)
    if hasattr(core, "hc_head"):
        hidden = core.hc_head(hidden)
    for module, attr in (
        (core, "norm"),
        (core, "ln_f"),
        (core, "final_layernorm"),
        (language_model, "norm"),
    ):
        if hasattr(module, attr):
            return getattr(module, attr)(hidden)
    return hidden


def _project_hidden_states(language_model, hidden: mx.array):
    if hasattr(language_model, "_project_hidden_states"):
        return language_model._project_hidden_states(hidden)
    if hasattr(language_model, "lm_head"):
        return language_model.lm_head(hidden)
    embedding = _token_embedding(language_model)
    if hasattr(embedding, "as_linear"):
        return embedding.as_linear(hidden)
    return None


def _rank0_logits(
    language_model,
    hidden: mx.array,
    cache,
    input_ids: list[int],
    timings: dict[str, float] | None = None,
    profile: dict[str, Any] | None = None,
):
    tokens = mx.array([input_ids], dtype=mx.int32)
    start = time.perf_counter()
    hidden = _run_rank_layers(language_model, hidden, cache, tokens, profile=profile)
    if timings is not None:
        timings["rank0_layers_s"] = time.perf_counter() - start
    start = time.perf_counter()
    logits = _project_hidden_states(language_model, _final_norm(language_model, hidden))
    if logits is None:
        raise ValueError("model does not expose an output projection")
    mx.eval(logits)
    if timings is not None:
        timings["rank0_project_s"] = time.perf_counter() - start
    return logits


def _listen_socket(host: str, port: int, backlog: int = 1) -> socket.socket:
    """Create the rank1 listener.

    `0.0.0.0` is the Studio launcher default. Bind that as an IPv6 dual-stack wildcard when the platform
    permits it so the rank0 app relay can use the MacBook's Thunderbolt IPv6 ULA while existing IPv4
    clients continue to work through IPv4-mapped sockets.
    """
    if host in ("", "0.0.0.0", "::"):
        try:
            server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            if hasattr(socket, "IPV6_V6ONLY"):
                server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("::", port))
            server.listen(backlog)
            return server
        except OSError:
            try:
                server.close()
            except Exception:
                pass
            if host == "::":
                raise

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    server = socket.socket(family, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(backlog)
        return server
    except Exception:
        server.close()
        raise


def rank1_server(args) -> None:
    _configure_memory_limits(
        wired_limit_mb=args.wired_limit_mb,
        memory_limit_mb=args.memory_limit_mb,
    )
    with _listen_socket(args.host, args.port, backlog=1) as server:
        print(f"RPC_LISTENING host={args.host} port={args.port}", flush=True)

        model, _processor, start_idx, end_idx = _load_pipeline_rank(
            args.model,
            rank=1,
            size=2,
            pipeline_layers=args.pipeline_layers,
            force_residency=args.force_residency,
            require_gpu=args.require_gpu,
        )
        language_model = model.language_model
        print(
            f"RPC_READY rank=1 layers={start_idx}:{end_idx}",
            flush=True,
        )

        client_index = 0
        while True:
            conn, addr = server.accept()
            _tune_rpc_socket(conn)
            client_index += 1
            print(f"RPC_SERVER_CLIENT_CONNECTED index={client_index} addr={addr}", flush=True)
            with conn:
                _send_json(
                    conn,
                    {
                        "ok": True,
                        "rank": 1,
                        "layers": [start_idx, end_idx],
                        "pipeline_layers": args.pipeline_layers,
                        "boundary_dtype": args.boundary_dtype,
                    },
                )
                cache = None
                step = 0
                try:
                    while True:
                        request_wait_start = time.perf_counter()
                        request = _recv_json(conn)
                        request_recv_s = time.perf_counter() - request_wait_start
                        op = request.get("op")
                        if op == "stop":
                            _send_json(conn, {"ok": True})
                            print(
                                f"RPC_SERVER_CLIENT_STOP index={client_index} step={step}",
                                flush=True,
                            )
                            break
                        if op != "forward":
                            raise ValueError(f"unknown RPC op {op!r} from {addr}")
                        if request.get("reset") or cache is None:
                            cache = language_model.make_cache()
                        input_ids = request["input_ids"]
                        modal = None
                        if request.get("modal_positions"):
                            values, _ = _recv_array_timed(conn)
                            modal = (request["modal_positions"], values)
                        cache_before = _cache_offset(cache)
                        start = time.perf_counter()
                        hidden = _rank1_forward(language_model, input_ids, cache, modal=modal)
                        rank1_s = time.perf_counter() - start
                        cache_after = _cache_offset(cache)
                        send_timing = _send_array(
                            conn, hidden, boundary_dtype=args.boundary_dtype
                        )
                        record = {
                            "client": client_index,
                            "step": step,
                            "input_tokens": len(input_ids),
                            "reset": bool(request.get("reset")),
                            "cache_offset_before": cache_before,
                            "cache_offset_after": cache_after,
                            "hidden_dtype": _dtype_name(hidden.dtype),
                            "hidden_shape": list(hidden.shape),
                            "boundary_dtype": args.boundary_dtype,
                            "request_recv_s": request_recv_s,
                            "rank1_compute_s": rank1_s,
                            "rank1_s": rank1_s,
                            **send_timing,
                        }
                        print(
                            "RPC_SERVER_FORWARD "
                            + json.dumps(record, separators=(",", ":")),
                            flush=True,
                        )
                        step += 1
                except (ConnectionError, OSError) as exc:
                    print(
                        "RPC_SERVER_CLIENT_DISCONNECTED "
                        f"index={client_index} step={step} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                except BaseException as exc:  # noqa: BLE001
                    print(
                        "RPC_SERVER_CLIENT_ERROR "
                        f"index={client_index} step={step} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                finally:
                    print(
                        f"RPC_SERVER_CLIENT_CLOSED index={client_index} next=accept",
                        flush=True,
                    )


def _connect(
    host: str, port: int, timeout: float, source_host: str | None = None,
    source_iface: str | None = None,
) -> socket.socket:
    deadline = time.time() + timeout
    last_error = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            # Scope the socket to the egress interface (macOS IP_BOUND_IF). The TB-bridge route to the peer
            # is INTERFACE-SCOPED (RTF_IFSCOPE on bridge0); a long-lived mlx_vlm serve process's connect can
            # fail the kernel route lookup with EHOSTUNREACH (errno 65) — emitting NO packet — even with a
            # valid ARP entry + a source-IP bind, while fresh processes resolve it. Binding the socket to the
            # interface forces the scoped route. (tcpdump 2026-06-16: 0 packets on the failing connect.)
            if source_iface and sys.platform == "darwin":
                try:
                    _IP_BOUND_IF = 25  # <netinet/in.h> on macOS
                    sock.setsockopt(socket.IPPROTO_IP, _IP_BOUND_IF,
                                    socket.if_nametoindex(source_iface))
                except OSError:
                    pass
            if source_host:
                sock.bind((source_host, 0))
            sock.connect((host, port))
            sock.settimeout(None)
            _tune_rpc_socket(sock)
            # TCP keepalive: this socket carries the rank0<->rank1 pipeline. During a long PROMPT-PROCESSING
            # (prefill) of a large prompt it can be quiet for minutes; without keepalive an idle/NAT drop or a
            # silently-dead peer leaves recv() blocked and/or closes the connection, which the OpenAI server
            # then surfaces as "[rpc stream error: ConnectionError: socket closed while receiving]". Keepalive
            # keeps the link warm + detects a genuinely dead peer. Options are best-effort (names differ by OS).
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except OSError:
                pass
            for _opt_name, _val in (("TCP_KEEPALIVE", 15), ("TCP_KEEPIDLE", 15),
                                    ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 4)):
                _opt = getattr(socket, _opt_name, None)
                if _opt is None:
                    continue
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, _opt, _val)
                except OSError:
                    pass
            return sock
        except OSError as exc:
            if sock is not None:
                sock.close()
            last_error = exc
            if attempts == 1 or attempts % 20 == 0:
                print(
                    f"RPC_CONNECT_RETRY host={host} port={port} "
                    f"attempt={attempts} error={exc!r}",
                    flush=True,
                )
            time.sleep(0.5)
    raise TimeoutError(f"could not connect to {host}:{port}: {last_error}")


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    encoded = tokenizer.encode(prompt)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return list(encoded)


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def client(args) -> None:
    _configure_memory_limits(
        wired_limit_mb=args.wired_limit_mb,
        memory_limit_mb=args.memory_limit_mb,
    )
    with _connect(
        args.server_host,
        args.server_port,
        args.connect_timeout,
        args.source_host,
        args.source_iface,
    ) as sock:
        hello = _recv_json(sock)
        if not hello.get("ok"):
            raise RuntimeError(f"rank1 rejected connection: {hello}")
        print(f"RPC_CONNECTED {hello}", flush=True)

        model, processor, start_idx, end_idx = _load_pipeline_rank(
            args.model,
            rank=0,
            size=2,
            pipeline_layers=args.pipeline_layers,
            force_residency=args.force_residency,
            require_gpu=args.require_gpu,
        )
        language_model = model.language_model
        tokenizer = _tokenizer(processor)
        rank0_cache = language_model.make_cache()
        input_ids = _encode_prompt(tokenizer, args.prompt)
        generated: list[int] = []
        token_records: list[dict[str, Any]] = []

        def forward(
            ids: list[int],
            *,
            reset: bool,
            decode_i: int | None = None,
        ) -> tuple[mx.array, dict[str, Any]]:
            detail: dict[str, Any] = {}
            start = time.perf_counter()
            _send_json(
                sock,
                {"op": "forward", "input_ids": ids, "reset": bool(reset)},
            )
            detail["send_s"] = time.perf_counter() - start
            start = time.perf_counter()
            hidden, recv_timing = _recv_array_timed(sock)
            detail["recv_s"] = time.perf_counter() - start
            detail.update(recv_timing)
            detail["hidden_dtype"] = _dtype_name(hidden.dtype)
            detail["hidden_shape"] = list(hidden.shape)
            profile = None
            if decode_i is not None and decode_i == args.profile_rank0_step:
                profile = {
                    "decode_i": decode_i,
                    "input_tokens": len(ids),
                    "hidden_dtype": _dtype_name(hidden.dtype),
                    "hidden_shape": list(hidden.shape),
                    "rank0_layers": [start_idx, end_idx],
                    "layers": [],
                    "phase_totals_s": {},
                }
            start = time.perf_counter()
            logits = _rank0_logits(
                language_model,
                hidden,
                rank0_cache,
                ids,
                timings=detail,
                profile=profile,
            )
            detail["rank0_s"] = time.perf_counter() - start
            if profile is not None:
                profile["rank0_layers_s"] = detail.get("rank0_layers_s")
                print(
                    "RPC_RANK0_PROFILE "
                    + json.dumps(profile, separators=(",", ":")),
                    flush=True,
                )
            return logits, detail

        prefill_start = time.perf_counter()
        logits, prefill_detail = forward(input_ids, reset=True)
        prefill_s = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()
        for token_index in range(args.max_tokens):
            token_start = time.perf_counter()
            sample_start = time.perf_counter()
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            sample_s = time.perf_counter() - sample_start
            generated.append(next_token)
            eos = getattr(tokenizer, "eos_token_id", None)
            forward_s = 0.0
            if eos is not None and next_token == eos:
                record = {
                    "i": token_index,
                    "id": next_token,
                    "sample_s": sample_s,
                    "forward_s": forward_s,
                    "wall_s": time.perf_counter() - token_start,
                }
                token_records.append(record)
                break
            forward_start = time.perf_counter()
            logits, forward_detail = forward(
                [next_token], reset=False, decode_i=token_index
            )
            forward_s = time.perf_counter() - forward_start
            record = {
                "i": token_index,
                "id": next_token,
                "sample_s": sample_s,
                "forward_s": forward_s,
                "wall_s": time.perf_counter() - token_start,
            }
            record.update(forward_detail)
            token_records.append(record)
        decode_s = time.perf_counter() - decode_start

        _send_json(sock, {"op": "stop"})
        _recv_json(sock)

    text = _decode_tokens(tokenizer, generated)
    total = args.prompt + text
    tok_s = len(generated) / decode_s if decode_s > 0 else 0.0
    warm_records = token_records[int(args.warmup_tokens) :]
    warm_decode_s = sum(float(record["wall_s"]) for record in warm_records)
    warm_tok_s = len(warm_records) / warm_decode_s if warm_decode_s > 0 else 0.0
    print("RPC_RESULT_BEGIN", flush=True)
    print(total, flush=True)
    print("RPC_RESULT_END", flush=True)
    print(
        "RPC_TOKEN_TIMES "
        + json.dumps(token_records, separators=(",", ":")),
        flush=True,
    )
    print(
        "RPC_PREFILL_TIMES "
        + json.dumps(prefill_detail, separators=(",", ":")),
        flush=True,
    )
    print(
        "RPC_STATS "
        f"rank0_layers={start_idx}:{end_idx} "
        f"rank1_layers={hello.get('layers')} "
        f"prompt_tokens={len(input_ids)} "
        f"generated_tokens={len(generated)} "
        f"prefill_s={prefill_s:.3f} "
        f"decode_s={decode_s:.3f} "
        f"tok_s={tok_s:.3f} "
        f"warmup_tokens={args.warmup_tokens} "
        f"warm_decode_tokens={len(warm_records)} "
        f"warm_decode_s={warm_decode_s:.3f} "
        f"warm_tok_s={warm_tok_s:.3f}",
        flush=True,
    )


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _messages_to_prompt(tokenizer, messages: list[dict[str, Any]], session=None, template_kwargs=None) -> str:
    template_kwargs = template_kwargs or {}
    has_media = any(isinstance(m.get("content"), list) and any(
        isinstance(p, dict) and p.get("type") in ("image_url", "input_audio", "audio_url", "video_url")
        for p in m["content"]) for m in messages if isinstance(m, dict))
    if has_media:
        if session is None:
            raise ValueError("multimodal messages require a configured media encoder")
        return session.prepare_media(messages, template_kwargs=template_kwargs)
    clean: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        clean.append({"role": role, "content": _message_content_text(message.get("content"))})
    if not clean:
        clean = [{"role": "user", "content": ""}]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(clean, tokenize=False, add_generation_prompt=True, **template_kwargs)
        except TypeError:
            try:
                return apply_template(clean, tokenize=False)
            except Exception:
                pass
        except Exception:
            pass
    lines = []
    for message in clean:
        lines.append(f"{message['role']}: {message['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


class _Rank0PipelineSession:
    def __init__(self, args) -> None:
        self.args = args
        self.sock = _connect(
            args.server_host,
            args.server_port,
            args.connect_timeout,
            args.source_host,
            args.source_iface,
        )
        # TCP can connect to a local relay before its remote peer arrives.
        # Bound the handshake as well as connect(), so a dead relay cannot hold
        # model initialization and daemon shutdown indefinitely.
        self.sock.settimeout(args.connect_timeout)
        try:
            self.hello = _recv_json(self.sock)
        except Exception:
            self.sock.close()
            raise
        finally:
            if self.sock.fileno() >= 0:
                self.sock.settimeout(None)
        if not self.hello.get("ok"):
            raise RuntimeError(f"rank1 rejected connection: {self.hello}")
        print(f"RPC_CONNECTED {self.hello}", flush=True)

        self.model, self.processor, self.start_idx, self.end_idx = _load_pipeline_rank(
            args.model,
            rank=0,
            size=2,
            pipeline_layers=args.pipeline_layers,
            force_residency=args.force_residency,
            require_gpu=args.require_gpu,
        )
        self.language_model = self.model.language_model
        self.tokenizer = _tokenizer(self.processor)
        self.lock = threading.Lock()
        self.closed = False
        self.media_encoder = None

    def prepare_media(self, messages, template_kwargs=None):
        if getattr(self.language_model, "model_type", None) != "mimo_v2":
            raise ValueError("this RPC model has no multimodal encoder")
        from .models.mimo_v2.multimodal import MiMoMediaEncoder
        with self.lock:
            if self.media_encoder is None:
                self.media_encoder = MiMoMediaEncoder(self.args.model, self.processor)
            return self.media_encoder.prepare(messages, template_kwargs=template_kwargs)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            _send_json(self.sock, {"op": "stop"})
            _recv_json(self.sock)
        except Exception as exc:
            print(f"RPC_STOP_WARNING {type(exc).__name__}: {exc}", flush=True)
        try:
            self.sock.close()
        except Exception:
            pass

    def _mark_broken(self, exc: BaseException) -> None:
        self.closed = True
        print(f"RPC_SESSION_BROKEN {type(exc).__name__}: {exc}", flush=True)
        try:
            self.sock.close()
        except Exception:
            pass

    def forward(
        self,
        rank0_cache,
        ids: list[int],
        *,
        reset: bool,
        decode_i: int | None = None,
        modal=None,
    ) -> tuple[mx.array, dict[str, Any]]:
        detail: dict[str, Any] = {}
        try:
            start = time.perf_counter()
            request = {"op": "forward", "input_ids": ids, "reset": bool(reset)}
            if modal is not None:
                request["modal_positions"] = modal[0]
            _send_json(self.sock, request)
            if modal is not None:
                _send_array(self.sock, modal[1], boundary_dtype="preserve")
            detail["send_s"] = time.perf_counter() - start
            start = time.perf_counter()
            hidden, recv_timing = _recv_array_timed(self.sock)
            detail["recv_s"] = time.perf_counter() - start
            detail.update(recv_timing)
        except (ConnectionError, OSError) as exc:
            self._mark_broken(exc)
            raise
        detail["hidden_dtype"] = _dtype_name(hidden.dtype)
        detail["hidden_shape"] = list(hidden.shape)
        profile = None
        if decode_i is not None and decode_i == self.args.profile_rank0_step:
            profile = {
                "decode_i": decode_i,
                "input_tokens": len(ids),
                "hidden_dtype": _dtype_name(hidden.dtype),
                "hidden_shape": list(hidden.shape),
                "rank0_layers": [self.start_idx, self.end_idx],
                "layers": [],
                "phase_totals_s": {},
            }
        start = time.perf_counter()
        logits = _rank0_logits(
            self.language_model,
            hidden,
            rank0_cache,
            ids,
            timings=detail,
            profile=profile,
        )
        detail["rank0_s"] = time.perf_counter() - start
        if profile is not None:
            profile["rank0_layers_s"] = detail.get("rank0_layers_s")
            print(
                "RPC_RANK0_PROFILE " + json.dumps(profile, separators=(",", ":")),
                flush=True,
            )
        return logits, detail

    def _prefill(self, cache, input_ids, max_tokens, prompt=None):
        limit = getattr(self.args, "max_context", 0)
        if limit and len(input_ids) + max_tokens > limit:
            raise ValueError(f"request needs {len(input_ids) + max_tokens} tokens; preset context is {limit}")
        chunk_size = getattr(self.args, "prefill_chunk_size", 0) or len(input_ids)
        details = {"chunks": 0, "prompt_tokens": len(input_ids)}
        for start in range(0, len(input_ids), chunk_size):
            modal = None
            if hasattr(prompt, "modal_positions"):
                selected = (prompt.modal_positions >= start) & (prompt.modal_positions < start + chunk_size)
                if selected.any():
                    modal = ((prompt.modal_positions[selected] - start).tolist(), mx.array(prompt.modal_embeddings[selected]))
            logits, _ = self.forward(cache, input_ids[start:start + chunk_size], reset=(start == 0), modal=modal)
            mx.eval(logits)
            details["chunks"] += 1
            print(f"RPC_PREFILL_PROGRESS tokens={min(start+chunk_size,len(input_ids))}/{len(input_ids)}", flush=True)
        return logits, details

    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        with self.lock:
            rank0_cache = self.language_model.make_cache()
            input_ids = getattr(prompt, "token_ids", None) or _encode_prompt(self.tokenizer, prompt)
            generated: list[int] = []
            token_records: list[dict[str, Any]] = []

            prefill_start = time.perf_counter()
            logits, prefill_detail = self._prefill(rank0_cache, input_ids, max_tokens, prompt)
            prefill_s = time.perf_counter() - prefill_start

            decode_start = time.perf_counter()
            for token_index in range(max(0, int(max_tokens))):
                token_start = time.perf_counter()
                sample_start = time.perf_counter()
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                sample_s = time.perf_counter() - sample_start
                generated.append(next_token)
                eos = getattr(self.tokenizer, "eos_token_id", None)
                forward_s = 0.0
                if eos is not None and next_token == eos:
                    token_records.append(
                        {
                            "i": token_index,
                            "id": next_token,
                            "sample_s": sample_s,
                            "forward_s": forward_s,
                            "wall_s": time.perf_counter() - token_start,
                        }
                    )
                    break
                forward_start = time.perf_counter()
                logits, forward_detail = self.forward(
                    rank0_cache, [next_token], reset=False, decode_i=token_index
                )
                forward_s = time.perf_counter() - forward_start
                record = {
                    "i": token_index,
                    "id": next_token,
                    "sample_s": sample_s,
                    "forward_s": forward_s,
                    "wall_s": time.perf_counter() - token_start,
                }
                record.update(forward_detail)
                token_records.append(record)
            decode_s = time.perf_counter() - decode_start

            text = _decode_tokens(self.tokenizer, generated)
            tok_s = len(generated) / decode_s if decode_s > 0 else 0.0
            warm_records = token_records[int(self.args.warmup_tokens) :]
            warm_decode_s = sum(float(record["wall_s"]) for record in warm_records)
            warm_tok_s = len(warm_records) / warm_decode_s if warm_decode_s > 0 else 0.0
            stats = {
                "rank0_layers": [self.start_idx, self.end_idx],
                "rank1_layers": self.hello.get("layers"),
                "prompt_tokens": len(input_ids),
                "generated_tokens": len(generated),
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "tok_s": tok_s,
                "warmup_tokens": int(self.args.warmup_tokens),
                "warm_decode_tokens": len(warm_records),
                "warm_decode_s": warm_decode_s,
                "warm_tok_s": warm_tok_s,
            }
            print(
                "RPC_REQUEST_STATS " + json.dumps(stats, separators=(",", ":")),
                flush=True,
            )
            return {
                "text": text,
                "generated": generated,
                "prompt_tokens": len(input_ids),
                "token_records": token_records,
                "prefill_detail": prefill_detail,
                "stats": stats,
            }

    def generate_deltas(self, prompt: str, *, max_tokens: int):
        with self.lock:
            rank0_cache = self.language_model.make_cache()
            input_ids = getattr(prompt, "token_ids", None) or _encode_prompt(self.tokenizer, prompt)
            generated: list[int] = []
            token_records: list[dict[str, Any]] = []
            prev_text = ""

            prefill_start = time.perf_counter()
            logits, prefill_detail = self._prefill(rank0_cache, input_ids, max_tokens, prompt)
            prefill_s = time.perf_counter() - prefill_start

            decode_start = time.perf_counter()
            for token_index in range(max(0, int(max_tokens))):
                token_start = time.perf_counter()
                sample_start = time.perf_counter()
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                sample_s = time.perf_counter() - sample_start
                generated.append(next_token)
                current_text = _decode_tokens(self.tokenizer, generated)
                if current_text.startswith(prev_text):
                    delta = current_text[len(prev_text):]
                else:
                    delta = _decode_tokens(self.tokenizer, [next_token])
                prev_text = current_text
                if delta:
                    yield delta
                eos = getattr(self.tokenizer, "eos_token_id", None)
                forward_s = 0.0
                if eos is not None and next_token == eos:
                    token_records.append(
                        {
                            "i": token_index,
                            "id": next_token,
                            "sample_s": sample_s,
                            "forward_s": forward_s,
                            "wall_s": time.perf_counter() - token_start,
                        }
                    )
                    break
                forward_start = time.perf_counter()
                logits, forward_detail = self.forward(
                    rank0_cache, [next_token], reset=False, decode_i=token_index
                )
                forward_s = time.perf_counter() - forward_start
                record = {
                    "i": token_index,
                    "id": next_token,
                    "sample_s": sample_s,
                    "forward_s": forward_s,
                    "wall_s": time.perf_counter() - token_start,
                }
                record.update(forward_detail)
                token_records.append(record)

            decode_s = time.perf_counter() - decode_start
            warm_records = token_records[int(self.args.warmup_tokens) :]
            warm_decode_s = sum(float(record["wall_s"]) for record in warm_records)
            stats = {
                "rank0_layers": [self.start_idx, self.end_idx],
                "rank1_layers": self.hello.get("layers"),
                "prompt_tokens": len(input_ids),
                "generated_tokens": len(generated),
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "tok_s": len(generated) / decode_s if decode_s > 0 else 0.0,
                "warmup_tokens": int(self.args.warmup_tokens),
                "warm_decode_tokens": len(warm_records),
                "warm_decode_s": warm_decode_s,
                "warm_tok_s": len(warm_records) / warm_decode_s if warm_decode_s > 0 else 0.0,
            }
            print(
                "RPC_REQUEST_STATS " + json.dumps(stats, separators=(",", ":")),
                flush=True,
            )
            print(
                "RPC_TOKEN_TIMES " + json.dumps(token_records, separators=(",", ":")),
                flush=True,
            )
            print(
                "RPC_PREFILL_TIMES " + json.dumps(prefill_detail, separators=(",", ":")),
                flush=True,
            )


class _LocalMiMoSession(_Rank0PipelineSession):
    """Single-machine native MiMo using the same media and generation path."""
    def __init__(self, args):
        self.args = args
        config = json.loads((Path(args.model) / "config.json").read_text())
        from .models.mimo_v2.raw_loader import is_native
        if not is_native(config):
            raise ValueError("--local-model requires a native MiMo checkpoint")
        self.model, self.processor, self.start_idx, self.end_idx = _load_pipeline_rank(
            args.model, rank=0, size=1, pipeline_layers=str(config['num_hidden_layers']),
            force_residency=args.force_residency, require_gpu=args.require_gpu)
        self.language_model = self.model.language_model
        self.tokenizer = _tokenizer(self.processor)
        self.lock = threading.Lock()
        self.closed = False
        self.media_encoder = None
        self.hello = {'layers': [], 'local': True}

    def close(self):
        self.closed = True

    def forward(self, rank0_cache, ids, *, reset, decode_i=None, modal=None):
        inputs = mx.array([ids])
        embeddings = _input_embeds(self.language_model, inputs)
        if modal is not None:
            embeddings[:, mx.array(modal[0]), :] = modal[1].astype(embeddings.dtype)
        return _rank0_logits(self.language_model, embeddings, rank0_cache, ids), {}


class _LazyRank0PipelineSession:
    """Bind the OpenAI HTTP server before doing heavyweight rank0 work.

    The live Studio launcher uses /health as the readiness gate for the HTTP
    process. Constructing _Rank0PipelineSession before uvicorn.run meant model
    load or a missing rank1 could prevent :8970 from binding at all.
    """

    def __init__(self, args) -> None:
        self.args = args
        self._session: _Rank0PipelineSession | None = None
        self._lock = threading.Lock()
        self.initializing = False
        self.last_error: str | None = None

    def get(self) -> _Rank0PipelineSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        with self._lock:
            session = self._session
            if session is not None and not session.closed:
                return session
            if self.last_error:
                raise RuntimeError(f"model initialization failed; restart after correcting it: {self.last_error}")
            self.initializing = True
            self.last_error = None
            print(
                "RPC_OPENAI_CONNECTING "
                f"server={self.args.server_host}:{self.args.server_port} "
                f"connect_timeout={self.args.connect_timeout}",
                flush=True,
            )
            try:
                session_class = _LocalMiMoSession if getattr(self.args, 'local_model', False) else _Rank0PipelineSession
                session = session_class(self.args)
                self._session = session
                print(
                    "RPC_OPENAI_READY "
                    f"rank0_layers={session.start_idx}:{session.end_idx} "
                    f"rank1_layers={session.hello.get('layers')}",
                    flush=True,
                )
                return session
            except BaseException as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"RPC_OPENAI_INIT_ERROR {self.last_error}", flush=True)
                raise
            finally:
                self.initializing = False

    def close(self) -> None:
        session = self._session
        if session is not None:
            session.close()

    def health(self, model_id: str) -> dict[str, Any]:
        session = self._session
        ready = bool(session is not None and not session.closed)
        payload: dict[str, Any] = {
            "ok": True,
            "engine": "mlx-vlm-rpc-pipeline",
            "model": model_id,
            "status": "ready" if ready else ("initializing" if self.initializing else "listening"),
            "connected": ready,
            "server": f"{self.args.server_host}:{self.args.server_port}",
        }
        if self.last_error:
            payload["last_error"] = self.last_error
        if ready and session is not None:
            payload["rank0_layers"] = [session.start_idx, session.end_idx]
            payload["rank1_layers"] = session.hello.get("layers")
        return payload


def _chat_completion_chunk(
    *,
    request_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def _chat_progress_chunk(
    *,
    request_id: str,
    created: int,
    model: str,
    text: str,
    step: int | None = None,
    total: int | None = None,
) -> str:
    progress: dict[str, Any] = {"text": text}
    if step is not None:
        progress["step"] = step
    if total is not None:
        progress["total"] = total
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "prefill_progress": progress,
    }
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def _stream_error_text(exc: BaseException) -> str:
    return f"[rpc stream error: {type(exc).__name__}: {str(exc)[:180]}]"


async def _stream_chat_completion_events(
    session,
    prompt: str,
    *,
    max_tokens: int,
    request_id: str,
    created: int,
    model: str,
    progress_interval_s: float,
    request_timeout_s: float,
    request=None,
    threaded: bool = True,
):
    if not threaded:
        yield _chat_completion_chunk(
            request_id=request_id,
            created=created,
            model=model,
            delta={"role": "assistant"},
        )
        yield _chat_progress_chunk(
            request_id=request_id,
            created=created,
            model=model,
            text="rpc prefill/page-in active",
            step=1,
        )

        request_timeout_s = max(0.0, float(request_timeout_s or 0.0))
        deadline = time.monotonic() + request_timeout_s if request_timeout_s > 0 else None
        try:
            for text in session.generate_deltas(prompt, max_tokens=max_tokens):
                if request is not None:
                    try:
                        if await request.is_disconnected():
                            print("RPC_STREAM_CLIENT_DISCONNECTED", flush=True)
                            return
                    except Exception:
                        pass
                if deadline is not None and time.monotonic() > deadline:
                    yield _chat_completion_chunk(
                        request_id=request_id,
                        created=created,
                        model=model,
                        delta={
                            "content": f"[rpc stream timeout after {request_timeout_s:.0f}s]"
                        },
                    )
                    break
                if text:
                    yield _chat_completion_chunk(
                        request_id=request_id,
                        created=created,
                        model=model,
                        delta={"content": str(text)},
                    )
        except BaseException as exc:  # noqa: BLE001
            print(f"RPC_STREAM_ERROR {type(exc).__name__}: {exc}", flush=True)
            yield _chat_completion_chunk(
                request_id=request_id,
                created=created,
                model=model,
                delta={"content": _stream_error_text(exc)},
            )

        yield _chat_completion_chunk(
            request_id=request_id,
            created=created,
            model=model,
            delta={},
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    items: queue.Queue[tuple[str, Any]] = queue.Queue()

    def worker() -> None:
        try:
            for text in session.generate_deltas(prompt, max_tokens=max_tokens):
                items.put(("delta", text))
        except BaseException as exc:  # noqa: BLE001
            items.put(("error", exc))
        finally:
            items.put(("done", None))

    thread = threading.Thread(target=worker, name="rpc-sse-decode", daemon=True)
    thread.start()

    yield _chat_completion_chunk(
        request_id=request_id,
        created=created,
        model=model,
        delta={"role": "assistant"},
    )

    progress_interval_s = max(0.5, float(progress_interval_s or 10.0))
    request_timeout_s = max(0.0, float(request_timeout_s or 0.0))
    started = time.monotonic()
    deadline = started + request_timeout_s if request_timeout_s > 0 else None
    progress_step = 0
    timed_out = False

    while True:
        if request is not None:
            try:
                if await request.is_disconnected():
                    print("RPC_STREAM_CLIENT_DISCONNECTED", flush=True)
                    return
            except Exception:
                pass

        timeout = progress_interval_s
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            timeout = min(timeout, max(0.05, remaining))

        try:
            kind, value = await asyncio.to_thread(items.get, True, timeout)
        except queue.Empty:
            progress_step += 1
            yield _chat_progress_chunk(
                request_id=request_id,
                created=created,
                model=model,
                text="rpc prefill/page-in active",
                step=progress_step,
            )
            continue

        if kind == "delta":
            if value:
                yield _chat_completion_chunk(
                    request_id=request_id,
                    created=created,
                    model=model,
                    delta={"content": str(value)},
                )
            continue
        if kind == "error":
            print(
                f"RPC_STREAM_ERROR {type(value).__name__}: {value}",
                flush=True,
            )
            yield _chat_completion_chunk(
                request_id=request_id,
                created=created,
                model=model,
                delta={"content": _stream_error_text(value)},
            )
            break
        if kind == "done":
            break

    if timed_out:
        try:
            close = getattr(session, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            print(f"RPC_STREAM_TIMEOUT_CLOSE_WARNING {type(exc).__name__}: {exc}", flush=True)
        yield _chat_completion_chunk(
            request_id=request_id,
            created=created,
            model=model,
            delta={"content": f"[rpc stream timeout after {request_timeout_s:.0f}s]"},
        )

    yield _chat_completion_chunk(
        request_id=request_id,
        created=created,
        model=model,
        delta={},
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


class _LazyMessageSession:
    def __init__(self, lazy: _LazyRank0PipelineSession, messages: list[dict[str, Any]], template_kwargs=None) -> None:
        self.lazy = lazy
        self.messages = messages
        self.template_kwargs = template_kwargs

    def close(self) -> None:
        self.lazy.close()

    def generate_deltas(self, _prompt: str, *, max_tokens: int):
        session = self.lazy.get()
        prompt = _messages_to_prompt(session.tokenizer, self.messages, session=session, template_kwargs=self.template_kwargs)
        yield from session.generate_deltas(prompt, max_tokens=max_tokens)


async def _stream_chat_completion_events_from_messages(
    lazy_session: _LazyRank0PipelineSession,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    request_id: str,
    created: int,
    model: str,
    progress_interval_s: float,
    request_timeout_s: float,
    request=None,
    threaded: bool = True,
    template_kwargs=None,
):
    async for event in _stream_chat_completion_events(
        _LazyMessageSession(lazy_session, messages, template_kwargs),
        "",
        max_tokens=max_tokens,
        request_id=request_id,
        created=created,
        model=model,
        progress_interval_s=progress_interval_s,
        request_timeout_s=request_timeout_s,
        request=request,
        threaded=threaded,
    ):
        yield event


def openai_serve(args) -> None:
    # Native MiMo prefill can occupy the GPU for seconds. Keep the HTTP loop
    # free to deliver progress, health probes and cancellation while it runs.
    try:
        config = json.loads((Path(args.model) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        config = {}
    if (config.get("model_type") == "mimo_v2"
            and config.get("quantization_config", {}).get("store_dtype") == "mxfp4"):
        args.threaded_stream = True
    _configure_memory_limits(
        wired_limit_mb=args.wired_limit_mb,
        memory_limit_mb=args.memory_limit_mb,
    )
    session = _LazyRank0PipelineSession(args)
    model_id = str(args.model)

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn

    app = FastAPI()
    # Streaming generators retain the model's synchronous lock across yields.
    # Queue a second HTTP request asynchronously before it can block the event
    # loop acquiring that lock and prevent the first stream from resuming.
    request_gate = asyncio.Lock()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        session.close()

    @app.get("/health")
    def health():
        payload = session.health(model_id)
        return JSONResponse(payload, status_code=200 if payload['status'] == 'ready' else 503)

    @app.get("/v1/models")
    def models():
        payload = session.health(model_id)
        if payload['status'] != 'ready':
            return JSONResponse(payload, status_code=503)
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "mlx-vlm-rpc",
            }],
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except BaseException as exc:  # noqa: BLE001
            print(f"RPC_REQUEST_JSON_ERROR {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            return JSONResponse({"error": "messages must be a list"}, status_code=400)
        raw_template = body.get("chat_template_kwargs") or {}
        if not isinstance(raw_template, dict):
            return JSONResponse({"error": "chat_template_kwargs must be an object"}, status_code=400)
        template_kwargs = {k: v for k, v in raw_template.items() if k == "enable_thinking" and isinstance(v, bool)}
        try:
            max_tokens = int(
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or args.max_tokens
            )
            request_model = str(body.get("model") or model_id)
        except BaseException as exc:  # noqa: BLE001
            print(f"RPC_REQUEST_ARG_ERROR {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse({"error": _stream_error_text(exc)}, status_code=400)
        request_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        if body.get("stream"):
            async def serialized_events():
                async with request_gate:
                    async for event in _stream_chat_completion_events_from_messages(
                        session, messages, template_kwargs=template_kwargs,
                        max_tokens=max_tokens, request_id=request_id, created=created,
                        model=request_model, progress_interval_s=args.stream_progress_interval,
                        request_timeout_s=args.request_timeout, request=request,
                        threaded=args.threaded_stream,
                    ):
                        yield event
            return StreamingResponse(
                serialized_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            async with request_gate:
                real_session = session.get()
                prompt = _messages_to_prompt(real_session.tokenizer, messages, session=real_session, template_kwargs=template_kwargs)
                result = real_session.generate(prompt, max_tokens=max_tokens)
        except BaseException as exc:  # noqa: BLE001
            print(f"RPC_NONSTREAM_ERROR {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse({"error": _stream_error_text(exc)}, status_code=503)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": len(result["generated"]),
                "total_tokens": result["prompt_tokens"] + len(result["generated"]),
            },
        }

    # PRE-WARM (regression fix for 660acc0 "Bind RPC OpenAI serve before rank0 load"): the lazy session keeps
    # /health responsive for the ring handshake, but deferring rank0's heavy 400GB+ load + force-residency to
    # the FIRST chat request meant that prompt ate the entire load off the model store — minutes at ~0.1 tok/s.
    # Warm up PROACTIVELY by self-issuing a tiny chat request through the REAL FastAPI -> decode path, so the
    # session is created + first-evaluated in the SAME request/worker context a live decode uses. (Calling
    # session.get() directly from a background thread instead bound MLX's GPU stream to that thread ->
    # "RuntimeError: There is no Stream(gpu, 0) in current thread" on the real decode.) Opt out: --no-prewarm.
    if getattr(args, "prewarm", True):
        def _prewarm() -> None:
            import urllib.request as _u
            warm_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
            payload = json.dumps({
                "model": model_id,
                "messages": [{"role": "user", "content": "warmup"}],
                "max_tokens": 1, "stream": False,
            }).encode()
            deadline = time.time() + 1800.0
            while time.time() < deadline:
                try:
                    req = _u.Request(warm_url, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
                    with _u.urlopen(req, timeout=1500) as resp:  # blocks through the full load + 1 token
                        resp.read()
                    print("RPC_PREWARM_DONE", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001 — retry until uvicorn binds / rank1 joins
                    if session.last_error:
                        print(f"RPC_PREWARM_FAILED {session.last_error}", flush=True)
                        return
                    time.sleep(1.0)
                    _last = exc
            print(f"RPC_PREWARM_GAVEUP {type(_last).__name__}: {_last}", flush=True)
        threading.Thread(target=_prewarm, name="rpc-rank0-prewarm", daemon=True).start()

    try:
        print(
            f"RPC_OPENAI_LISTENING host={args.host} port={args.port} model={model_id}",
            flush=True,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        print("RPC_UVICORN_RETURNED", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"RPC_UVICORN_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("RPC_OPENAI_SHUTDOWN closing_session", flush=True)
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MLX-VLM RPC pipeline proof")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--pipeline-layers", default="18,18")
    common.add_argument("--max-context", type=int, default=0)
    common.add_argument("--prefill-chunk-size", type=int, default=0)
    common.add_argument("--kv-bits", type=int, choices=(0, 8), default=0)
    common.add_argument("--wired-limit-mb", type=int, default=None)
    common.add_argument("--memory-limit-mb", type=int, default=None)
    common.add_argument("--force-residency", action="store_true")
    common.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail startup if MLX does not report a GPU-backed default device/parameter residency.",
    )
    common.add_argument(
        "--boundary-dtype",
        choices=("preserve", "float32", "bfloat16", "float16"),
        default="preserve",
        help="Activation dtype to use on the RPC boundary.",
    )

    server = sub.add_parser("server", parents=[common])
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8095)
    server.set_defaults(func=rank1_server)

    serve_parser = sub.add_parser("serve", parents=[common])
    serve_parser.add_argument("--server-host", default="127.0.0.1")
    serve_parser.add_argument("--local-model", action="store_true")
    serve_parser.add_argument("--server-port", type=int, default=8095)
    serve_parser.add_argument("--source-host", default=None)
    serve_parser.add_argument("--source-iface", default=None)
    serve_parser.add_argument("--connect-timeout", type=float, default=300.0)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8970)
    serve_parser.add_argument("--max-tokens", type=int, default=512)
    serve_parser.add_argument("--warmup-tokens", type=int, default=4)
    # Eagerly load + force-reside rank0 in the background right after binding (default ON) instead of on the
    # first chat request, so a large prompt never eats the whole load. --no-prewarm restores the lazy load.
    serve_parser.add_argument("--prewarm", dest="prewarm", action="store_true", default=True)
    serve_parser.add_argument("--no-prewarm", dest="prewarm", action="store_false")
    serve_parser.add_argument(
        "--stream-progress-interval",
        type=float,
        default=float(os.environ.get("MLX_RPC_STREAM_PROGRESS_INTERVAL", "10")),
        help="Seconds between in-band SSE progress packets while waiting for the first token.",
    )
    serve_parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("MLX_RPC_REQUEST_TIMEOUT", "1800")),
        help="Per-request stream timeout in seconds; set 0 to disable the server-side timeout.",
    )
    serve_parser.add_argument(
        "--threaded-stream",
        action="store_true",
        help=(
            "Run decode in a background thread to emit periodic prefill heartbeats. "
            "Default is inline decode because MLX/Metal is more stable there on "
            "the live 2-Mac serve path."
        ),
    )
    serve_parser.add_argument(
        "--profile-rank0-step",
        type=int,
        default=-1,
        help="Decode token index for one synchronized rank0 per-layer phase profile.",
    )
    serve_parser.set_defaults(func=openai_serve)

    client_parser = sub.add_parser("client", parents=[common])
    client_parser.add_argument("--server-host", required=True)
    client_parser.add_argument("--server-port", type=int, default=8095)
    client_parser.add_argument("--source-host", default=None)
    client_parser.add_argument("--source-iface", default=None)
    client_parser.add_argument("--connect-timeout", type=float, default=300.0)
    client_parser.add_argument("--prompt", default="The capital of France is")
    client_parser.add_argument("--max-tokens", type=int, default=8)
    client_parser.add_argument("--warmup-tokens", type=int, default=4)
    client_parser.add_argument(
        "--profile-rank0-step",
        type=int,
        default=-1,
        help="Decode token index for one synchronized rank0 per-layer phase profile.",
    )
    client_parser.set_defaults(func=client)

    args = parser.parse_args()
    if args.kv_bits:
        os.environ["MLX_MIMO_KV_BITS"] = str(args.kv_bits)
    if args.max_context and args.kv_bits == 8:
        os.environ["MLX_MIMO_CAPACITY_TOKENS"] = str(args.max_context)
        os.environ["MLX_MIMO_CAPACITY_LIMIT"] = str(int(getattr(args, "wired_limit_mb", 0) or 0) * 2**20)
    args.func(args)


if __name__ == "__main__":
    try:
        print(
            "RPC_PROCESS_START "
            + json.dumps({"argv": os.sys.argv, "pid": os.getpid()}, separators=(",", ":")),
            flush=True,
        )
        main()
        print("RPC_PROCESS_EXIT_NORMAL", flush=True)
    except SystemExit as exc:
        print(f"RPC_PROCESS_SYSTEM_EXIT code={exc.code!r}", flush=True)
        if getattr(exc, "__traceback__", None) is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        raise
    except BaseException as exc:  # noqa: BLE001
        print(f"RPC_PROCESS_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("RPC_PROCESS_FINAL", flush=True)
