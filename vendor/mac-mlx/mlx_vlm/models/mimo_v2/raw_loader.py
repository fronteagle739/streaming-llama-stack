"""Load native MiMo MXFP4 tensors a layer at a time, locally or over Studio HTTP.

The byte-range reader follows utils._InfShardReader: header index, seek/read one
tensor, preserve packed values and BF16 bit patterns. No full-shard fallback.
The HTTP path uses Studio's existing authenticated /dsv4/file Range transport.
"""

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .mimo_v2 import Model


def is_native(config):
    q = config.get("quantization_config") or {}
    return (config.get("model_type") == "mimo_v2"
            and q.get("store_dtype") == "mxfp4"
            and not config.get("quantization"))


class IndexedShardReader:
    """Plain-key equivalent of _InfShardReader, also accepting HTTP byte ranges."""

    def __init__(self, model_path, *, base_url="", source_model="", token=""):
        self.path = Path(model_path)
        self.base_url = base_url.rstrip("/")
        self.source_model = source_model
        self.token = token
        self.etags = {}
        self.index = {}
        self.bytes_read = 0
        self._system_http = False
        weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        for name in sorted(set(weight_map.values())):
            if Path(name).name != name or "\\" in name:
                raise ValueError("weight index must reference local shard basenames")
            length = struct.unpack("<Q", self.read_range(name, 0, 8))[0]
            if not 2 <= length <= 100_000_000:
                raise ValueError(f"invalid safetensors header: {name}")
            header = json.loads(self.read_range(name, 8, length))
            for key, info in header.items():
                if key == "__metadata__" or weight_map.get(key) != name:
                    continue
                start, end = info["data_offsets"]
                if not 0 <= start <= end:
                    raise ValueError(f"invalid tensor offsets: {key}")
                self.index[key] = (name, 8 + length + start, end - start,
                                   info["dtype"], info["shape"])
        missing = weight_map.keys() - self.index.keys()
        if missing:
            raise ValueError(f"weight index missing tensors in shard headers: {next(iter(missing))}")

    def read_range(self, name, start, size):
        if size == 0:
            return b""
        if self.base_url:
            query = urllib.parse.urlencode({"model": self.source_model, "name": name})
            headers = {"Range": f"bytes={start}-{start + size - 1}",
                       "X-Studio-Token": self.token}
            if name in self.etags:
                headers["If-Range"] = self.etags[name]
            req = urllib.request.Request(self.base_url + "/dsv4/file?" + query, headers=headers)
            if not getattr(self, '_system_http', False):
                try:
                    opened = urllib.request.urlopen(req, timeout=180)
                except urllib.error.URLError as exc:
                    if sys.platform != 'darwin' or getattr(exc.reason, 'errno', None) not in (51, 65):
                        raise
                    self._system_http = True
                    print('MIMO_TENSOR_TRANSPORT system-http (Python LAN route unavailable)', flush=True)
            if getattr(self, '_system_http', False):
                from .http_range import curl_range_response
                opened = curl_range_response(req, size)
            with opened as response:
                expected = f"bytes {start}-{start + size - 1}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Studio did not honor tensor byte range; refusing whole-shard download")
                etag = response.headers.get("ETag")
                if not etag or (name in self.etags and self.etags[name] != etag):
                    raise ValueError("checkpoint changed during tensor transfer")
                self.etags[name] = etag
                data = response.read(size + 1)
        else:
            with (self.path / name).open("rb") as source:
                source.seek(start)
                data = source.read(size)
        if len(data) != size:
            raise ValueError(f"short tensor read: {name} at {start}")
        self.bytes_read += size
        return data

    def read(self, key):
        name, start, size, dtype, shape = self.index[key]
        return self._array(self.read_range(name, start, size), dtype, shape, key)

    @staticmethod
    def _array(data, dtype, shape, key):
        # Same BF16 bitcast as _InfShardReader; FP8 remains raw bytes until
        # the upstream sanitizer applies block scales via mx.from_fp8.
        dtypes = {"BF16": "<u2", "F8_E4M3": "u1", "F8_E4M3FN": "u1",
                  "U8": "u1", "U32": "<u4", "F32": "<f4", "F16": "<f2"}
        if dtype not in dtypes:
            raise ValueError(f"unsupported native MiMo dtype {dtype}: {key}")
        array = mx.array(np.frombuffer(data, dtype=dtypes[dtype]))
        if dtype == "BF16":
            array = array.view(mx.bfloat16)
        return array.reshape(shape)

    def read_many(self, keys):
        # Coalesce adjacent required tensors (including tiny scale arrays) and
        # cap in-flight source buffers at four 64-MiB ranges. Never read an
        # off-rank tensor to make a larger transfer.
        groups = []
        for key in sorted(keys, key=lambda k: self.index[k][:2]):
            name, start, size, *_ = self.index[key]
            if (groups and groups[-1][0] == name and groups[-1][2] == start
                    and start + size - groups[-1][1] <= 64 * 2**20):
                groups[-1][2] = start + size
                groups[-1][3].append(key)
            else:
                groups.append([name, start, start + size, [key]])
        result = {}
        with ThreadPoolExecutor(max_workers=4 if self.base_url else 1) as pool:
            for offset in range(0, len(groups), 4):
                batch = groups[offset:offset + 4]
                pending = [pool.submit(self.read_range, n, s, e-s) for n, s, e, _ in batch]
                for (_, start, _, names), future in zip(batch, pending):
                    data = future.result()
                    for key in names:
                        _, pos, size, dtype, shape = self.index[key]
                        result[key] = self._array(memoryview(data)[pos-start:pos-start+size], dtype, shape, key)
        return result

    def fused_tp(self, args):
        for i, swa in enumerate(args.hybrid_layer_pattern):
            if swa:
                continue
            key = f"model.layers.{i}.self_attn.qkv_proj.weight"
            if key not in self.index:
                continue
            actual = self.index[key][4][0]
            padded = self.index[key + "_scale_inv"][4][0] * 128
            for tp in (1, 2, 4, 8, 16, 32):
                if args.num_attention_heads % tp or args.num_key_value_heads % tp:
                    continue
                rows = args.num_attention_heads // tp * args.head_dim
                rows += args.num_key_value_heads // tp * (args.head_dim + args.v_head_dim)
                if rows * tp == actual and -(-rows // 128) * 128 * tp == padded:
                    return tp
            raise ValueError(f"cannot identify native fused QKV layout at layer {i}")
        raise ValueError("native checkpoint has no full-attention QKV layout evidence")


def _install(module, weights):
    """Keep native experts packed; quantize other matrices once to affine Q8."""
    quants = {}
    for key, value in list(weights.items()):
        if not key.endswith(".weight") or value.ndim < 2:
            continue
        path = key[:-7]
        if path + ".scales" in weights:
            quants[path] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        elif ".gate." not in "." + key:
            packed, scales, biases = mx.quantize(value, group_size=64, bits=8)
            mx.eval(packed, scales, biases)
            weights[key] = packed
            weights[path + ".scales"] = scales
            weights[path + ".biases"] = biases
            quants[path] = {"group_size": 64, "bits": 8, "mode": "affine"}
    nn.quantize(module, class_predicate=lambda p, m: quants.get(p, False))
    module.load_weights(list(weights.items()), strict=True)
    mx.eval(module.parameters())


def load_native(model_path, *, pipeline_group=None, remote=None):
    """Materialize only the selected rank; remote rank requires metadata only."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    if not is_native(config):
        raise ValueError("native MiMo loader requires original MXFP4/FP8 weights")
    model = Model(ModelConfig.from_dict(config))
    lm = model.language_model
    if pipeline_group is not None:
        lm.pipeline(pipeline_group)
    reader = IndexedShardReader(model_path, **(remote or {}))
    lm.checkpoint_tp = reader.fused_tp(lm.args)
    selected = []
    for key in reader.index:
        if key.startswith("model.layers."):
            i = int(key.split(".")[2])
            if lm.start_idx <= i < lm.end_idx:
                selected.append(key)
        elif key.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
            selected.append(key)
    packed_bytes = sum(reader.index[k][2] for k in selected)
    device = mx.metal.device_info()
    ceiling = device.get("max_recommended_working_set_size", 0)
    # Additional layer assembly, dense conversion and cache/OS headroom.
    if ceiling and packed_bytes + 16 * 2**30 > ceiling:
        raise MemoryError(f"MiMo rank weights {packed_bytes / 2**30:.1f} GiB plus 16 GiB reserve exceed device ceiling")
    print(f"MIMO_LOAD rank={lm.pipeline_rank} layers={lm.start_idx}:{lm.end_idx} "
          f"packed_bytes={packed_bytes} qkv_tp={lm.checkpoint_tp} remote={bool(remote)}", flush=True)
    started = time.monotonic()
    old_cache = mx.set_cache_limit(256 * 2**20)
    try:
        for i in range(lm.start_idx, lm.end_idx):
            prefix = f"model.layers.{i}."
            raw = reader.read_many(k for k in selected if k.startswith(prefix))
            clean = lm.sanitize(raw)
            del raw
            weights = {k[len(prefix):]: v for k, v in clean.items()}
            _install(lm.model.layers[i], weights)
            del clean, weights
            mx.clear_cache()
            print(f"MIMO_LOAD layer={i} active_gib={mx.get_active_memory()/2**30:.2f} "
                  f"read_bytes={reader.bytes_read} elapsed_s={time.monotonic()-started:.2f}", flush=True)
        # Load globals separately using tiny wrappers so strict checking remains meaningful.
        for prefix, target in [("model.embed_tokens", lm.model.embed_tokens),
                               ("model.norm", lm.model.norm), ("lm_head", lm.lm_head)]:
            wrapper = nn.Module()
            wrapper.target = target
            raw = {k: reader.read(k) for k in selected if k.startswith(prefix + ".")}
            clean = lm.sanitize(raw)
            _install(wrapper, {"target." + k[len(prefix)+1:]: v for k, v in clean.items()})
            if prefix == "lm_head":
                lm.lm_head = wrapper.target
            elif prefix == "model.norm":
                lm.model.norm = wrapper.target
            else:
                lm.model.embed_tokens = wrapper.target
        model.eval()
        return model
    finally:
        mx.set_cache_limit(old_cache)
