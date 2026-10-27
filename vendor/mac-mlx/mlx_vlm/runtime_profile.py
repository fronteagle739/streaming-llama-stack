"""Explicit, local-file-triggered profiling of the already loaded receiver model.

No extra model is loaded. Timed component boundaries are a separate experiment:
their percentages must never be presented as shares of the asynchronous baseline.
"""

import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import time


def run_requested(model, prompt_ids):
    request = Path.home() / ".streaming-studio" / "runtime-profile.json"
    if not request.is_file():
        return
    config = json.loads(request.read_text())
    request.rename(request.with_name(f"runtime-profile.{time.time_ns()}.consumed.json"))
    if config.get("f7_census"):
        script = Path(__file__).resolve().parents[1] / "scripts" / "dsv41_f7_census.py"
        spec = importlib.util.spec_from_file_location("dsv41_f7_census", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(model, config)
    output = Path(config["output"])
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    lang = getattr(model, "language_model", model)
    is_v41 = "deepseek_v41" in type(lang).__module__
    # Local diagnostic controls only; never accepted from a network request.
    for name, value in config.get("environment", {}).items():
        if name not in {"DSV41_BOUNDED_CACHE", "DSV41_PREFILL_CHUNK", "DSV41_FAST_RMS_NORM"}:
            raise ValueError(f"Unsupported diagnostic environment control: {name}")
        os.environ[name] = str(value)
    if "break_compressed_retention" in config:
        if not is_v41:
            raise ValueError("Retention fault requires V4.1")
        language = importlib.import_module(type(lang).__module__)
        cls = language.DeepseekV41Cache
        if not hasattr(cls, "_diagnostic_original_finish"):
            cls._diagnostic_original_finish = cls.finish
        original = cls._diagnostic_original_finish
        if config["break_compressed_retention"]:
            def broken_finish(self, end_pos):
                original(self, end_pos)
                for layer in self.layers:
                    if layer.buffers["comp_kv"] is not None:
                        layer.buffers["comp_kv"] = mx.zeros_like(layer.buffers["comp_kv"])
                cls._diagnostic_fault_calls += 1
            cls._diagnostic_fault_calls = 0
            cls.finish = broken_finish
        else:
            cls.finish = original
        output.write_text(json.dumps({"fault_enabled": bool(config["break_compressed_retention"]),
                                     "fault_calls": getattr(cls, "_diagnostic_fault_calls", 0)}))
    if "break_sinkhorn" in config:
        if not is_v41:
            raise ValueError("Sinkhorn fault requires V4.1")
        language = importlib.import_module(type(lang).__module__)
        hc = importlib.import_module(type(lang).__module__.rsplit(".", 1)[0] + ".hyper_connection")
        if not hasattr(language, "_diagnostic_original_sinkhorn"):
            language._diagnostic_original_sinkhorn = language._hc_split_sinkhorn_ops
            language._diagnostic_original_metal_sinkhorn = hc.metal_sinkhorn
        original = language._diagnostic_original_sinkhorn
        if config["break_sinkhorn"]:
            def broken_split(*args):
                language._diagnostic_sinkhorn_fault_calls += 1
                return [mx.zeros_like(x) for x in original(*args)]
            language._diagnostic_sinkhorn_fault_calls = 0
            language._hc_split_sinkhorn_ops = broken_split
            hc.metal_sinkhorn = broken_split
        else:
            language._hc_split_sinkhorn_ops = original
            hc.metal_sinkhorn = language._diagnostic_original_metal_sinkhorn
        output.write_text(json.dumps({"fault_enabled": bool(config["break_sinkhorn"]),
                    "fault_calls": getattr(language, "_diagnostic_sinkhorn_fault_calls", 0)}))
    if config.get("control_only"):
        return
    ids = list(prompt_ids[:config.get("prompt_tokens", 256)])
    chunk = config.get("chunk", 256)
    steps = config.get("decode_steps", 16)
    result = {"model_class": str(type(lang)), "environment": {
        k: v for k, v in os.environ.items() if k.startswith("DSV41_") or k.startswith("LLAMA_MLX_ENGRAM")
        or k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")},
        "device": mx.device_info(), "runs": {}}

    def save():
        output.write_text(json.dumps(result, indent=2))

    def run(label, profiler=None):
        if is_v41:
            from mlx_vlm.custom_kernels.deepseek_v41.sparse_attention import status
            from mlx_vlm.models.deepseek_v41 import hyper_connection as hc_status
            kernel_before = status()
            hc_before = hc_status.SINKHORN_CALLS
        cache = lang.make_cache() if hasattr(lang, "make_cache") else make_prompt_cache(lang)
        records = []
        generated = []
        token = None
        for phase, batches in (("prefill", [ids[i:i + chunk] for i in range(0, len(ids), chunk)]),
                               ("decode", [None] * steps)):
            for batch in batches:
                started = time.perf_counter()
                x = mx.array([batch], mx.int32) if batch is not None else token[:, None]
                out = lang(x, cache=cache)
                built = time.perf_counter()
                logits = out.logits if hasattr(out, "logits") else out
                token = mx.argmax(logits[:, -1], axis=-1)
                mx.eval(token, [c.state for c in cache])
                ended = time.perf_counter()
                generated.append(token.item())
                records.append({"phase": phase, "tokens": len(batch) if batch is not None else 1,
                                "wall_s": ended - started, "build_s": built - started,
                                "eval_s": ended - built})
        summary = {}
        for phase in ("prefill", "decode"):
            rs = [r for r in records if r["phase"] == phase]
            wall = sum(r["wall_s"] for r in rs)
            summary[phase + "_tok_s"] = sum(r["tokens"] for r in rs) / wall
        result["runs"][label] = dict(summary, steps=records, generated=generated)
        if is_v41:
            result["runs"][label]["sinkhorn_invocations"] = hc_status.SINKHORN_CALLS - hc_before
            result["runs"][label]["sparse_invocations"] = {
                k: v - kernel_before[k] for k,v in status().items() if k != "metal_available"}
            result["runs"][label]["cache_bytes"] = cache[0].nbytes
        save()

    run("warmup")
    run("baseline")
    if config.get("experiments"):
        language = importlib.import_module(type(lang).__module__)
        hc = importlib.import_module(type(lang).__module__.rsplit(".", 1)[0] + ".hyper_connection")
        for experiment in config["experiments"]:
            restore = []
            if experiment.startswith("buffer_pool_"):
                old = mx.set_cache_limit(int(experiment.rsplit("_", 1)[1]) * 1024**3)
                restore.append(lambda limit=old: mx.set_cache_limit(limit))
            elif experiment == "unbounded_cache":
                old = os.environ.get("DSV41_BOUNDED_CACHE")
                os.environ["DSV41_BOUNDED_CACHE"] = "0"
                restore.append(lambda: os.environ.pop("DSV41_BOUNDED_CACHE", None) if old is None else os.environ.__setitem__("DSV41_BOUNDED_CACHE", old))
            elif experiment == "fast_rms_norm":
                old = os.environ.get("DSV41_FAST_RMS_NORM")
                os.environ["DSV41_FAST_RMS_NORM"] = "1"
                restore.append(lambda: os.environ.pop("DSV41_FAST_RMS_NORM", None) if old is None else os.environ.__setitem__("DSV41_FAST_RMS_NORM", old))
            elif experiment == "metal_sinkhorn":
                importlib.reload(hc)
                original = language._hc_split_sinkhorn_ops
                language._hc_split_sinkhorn_ops = hc.metal_sinkhorn
                restore.append(lambda fn=original: setattr(language, "_hc_split_sinkhorn_ops", fn))
            elif experiment == "expert_bf16":
                original = language.ClampedSwiGLU.__call__
                def activation(self, x, gate, _original=original):
                    return _original(self, x, gate).astype(x.dtype)
                language.ClampedSwiGLU.__call__ = activation
                restore.append(lambda: setattr(language.ClampedSwiGLU, "__call__", original))
            elif experiment == "compile_primitives":
                # The existing shapeless gate raises on argpartition's Slice.
                # Test identical source with ordinary shape-specialized compilation.
                import ast
                node = next(n for n in ast.parse(Path(language.__file__).read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == "_v41_expert_select")
                node.decorator_list = []
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), language.__file__, "exec"),
                     language.__dict__, namespace)
                original_select = language._v41_expert_select
                language._v41_expert_select = mx.compile(namespace[node.name])
                restore.append(lambda f=original_select: setattr(language, "_v41_expert_select", f))
                old = os.environ.get("DSV41_COMPILE_PRIMITIVES")
                os.environ["DSV41_COMPILE_PRIMITIVES"] = "1"
                restore.append(lambda: os.environ.pop("DSV41_COMPILE_PRIMITIVES", None) if old is None else os.environ.__setitem__("DSV41_COMPILE_PRIMITIVES", old))
            elif experiment == "compile_hc":
                old = hc.COMPILE_POST
                hc.COMPILE_POST = True
                original = language.hc_pre
                language.hc_pre = mx.compile(original)
                restore.extend([lambda: setattr(hc, "COMPILE_POST", old), lambda: setattr(language, "hc_pre", original)])
            elif experiment == "compile_fakequant":
                for name in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_ue8m0", "fake_quant_fp4_e4m3"):
                    fn = getattr(language, name)
                    setattr(language, name, mx.compile(fn))
                    restore.append(lambda n=name, f=fn: setattr(language, n, f))
            elif experiment in ("promote_wo_a", "promote_head", "demote_decode_weights"):
                modules = ([layer.attn.wo_a for layer in lang.layers]
                           if experiment == "promote_wo_a" else [lang.lm_head])
                if experiment == "demote_decode_weights":
                    modules = [layer.attn.wo_a for layer in lang.layers] + [lang.lm_head]
                for module in modules:
                    weight = module.weight
                    module.weight = weight.astype(mx.bfloat16 if experiment == "demote_decode_weights" else mx.float32)
                    mx.eval(module.weight)
                    restore.append(lambda m=module, w=weight: setattr(m, "weight", w))
            elif experiment == "reference_wo_a":
                import inspect
                import textwrap
                original = language.Attention.__call__
                source = textwrap.dedent(inspect.getsource(original))
                old = 'out.astype(mx.float32),\n                        self.wo_a.weight.astype(mx.float32)'
                # inspect dedents the class's four-space indentation.
                old = old.replace('\n                        ', '\n                    ')
                if source.count(old) != 1:
                    raise ValueError("Attention projection source did not match the audited contraction")
                namespace = {}
                exec(source.replace(old, 'out, self.wo_a.weight'), language.__dict__, namespace)
                language.Attention.__call__ = namespace['__call__']
                restore.append(lambda f=original: setattr(language.Attention, '__call__', f))
                for layer in lang.layers:
                    module = layer.attn.wo_a
                    weight = module.weight
                    module.weight = weight.astype(mx.bfloat16)
                    mx.eval(module.weight)
                    restore.append(lambda m=module, w=weight: setattr(m, "weight", w))
            elif experiment == "compile_moe":
                for layer in lang.layers:
                    module = layer.ffn
                    cls = type(module)
                    base = cls.__call__
                    def fn(x, image_mask, _module=module, _base=base):
                        return _base(_module, x, image_mask)
                    compiled = mx.compile(fn)
                    def call(self, x, image_mask=None, _compiled=compiled):
                        return _compiled(x, image_mask)
                    module.__class__ = type("Compiled" + cls.__name__, (cls,), {"__call__": call})
                    restore.append(lambda m=module, c=cls: setattr(m, "__class__", c))
            else:
                raise ValueError(f"Unknown experiment: {experiment}")
            try:
                run(experiment + "_warmup")
                run(experiment)
                result["runs"][experiment]["tokens_equal"] = result["runs"][experiment]["generated"] == result["runs"]["baseline"]["generated"]
            except Exception as exc:
                result["runs"][experiment] = {"error": repr(exc)}
                if config.get("activate", False):
                    for undo_experiment in reversed(restore):
                        undo_experiment()
                    raise
            finally:
                if not config.get("activate", False):
                    for undo_experiment in reversed(restore):
                        undo_experiment()
            save()
        return
    calls = {}
    undo = []
    operations = [(mx, n, n) for n in ("gather_qmm", "quantized_matmul", "dequantize", "from_fp8", "to_fp8")]
    operations.append((mx.fast, "rms_norm", "fast.rms_norm"))
    for scope, attr, name in operations:
        original = getattr(scope, attr)

        def wrapper(*args, _fn=original, _name=name, **kwargs):
            signature = _name + ":" + str([(tuple(a.shape), str(a.dtype)) for a in args if isinstance(a, mx.array)])
            if _name == "gather_qmm":
                signature += ":indices=" + str(kwargs["rhs_indices"].shape)
            item = calls.setdefault(signature, {"calls": 0, "logical_weight_bytes": 0})
            item["calls"] += 1
            if _name in ("gather_qmm", "quantized_matmul"):
                weights = [a for a in args[1:4] if isinstance(a, mx.array)]
                for name in ("scales", "biases"):
                    value = kwargs.get(name)
                    if isinstance(value, mx.array):
                        weights.append(value)
                if _name == "gather_qmm":
                    # Decode indices select distinct experts. For prefill this is an upper bound
                    # with reuse, not hardware DRAM traffic.
                    item["logical_weight_bytes"] += sum(a.nbytes // a.shape[0] for a in weights) * kwargs["rhs_indices"].size
                else:
                    item["logical_weight_bytes"] += sum(a.nbytes for a in weights)
            return _fn(*args, **kwargs)

        setattr(scope, attr, wrapper)
        undo.append((scope, attr, original))
    try:
        run("invocations")
        result["operator_invocations"] = calls
    finally:
        for scope, name, original in undo:
            setattr(scope, name, original)
    cp = cProfile.Profile()
    cp.enable()
    run("python_profile")
    cp.disable()
    cp.dump_stats(str(output) + ".pstats")
    stats = pstats.Stats(cp)
    result["python_self"] = [dict(file=k[0], line=k[1], function=k[2], calls=v[1], self_s=v[2], total_s=v[3])
                             for k, v in sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:80]]
    if is_v41:
        script = Path(__file__).resolve().parents[1] / "scripts" / "dsv41_profile.py"
        spec = importlib.util.spec_from_file_location("dsv41_cost_profile", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for eval_boundaries in (False, True):
            profiler = module.Profiler(eval_boundaries=eval_boundaries).install(model).enable()
            label = "boundary_costs" if eval_boundaries else "host_costs"
            try:
                run(label, profiler)
                result[label] = profiler.aggregate()
                result[label + "_summary"] = profiler.phase_summary()
            finally:
                profiler.uninstall()
            save()
        result["engram"] = lang.engram_stats()
        result["routing"] = {str(i): {"topk": l.ffn.gate.topk,
            "experts": l.ffn.gate.weight.shape[0], "switch": str(type(l.ffn.switch_mlp)),
            "projections": {n: str(type(getattr(l.ffn.switch_mlp, n, None)))
                            for n in ("gate_proj", "up_proj", "down_proj")}} for i,l in enumerate(lang.layers)}
    result["active_bytes"] = mx.get_active_memory()
    result["peak_bytes"] = mx.get_peak_memory()
    from mlx.utils import tree_flatten
    result["parameters"] = [{"name": n, "shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes}
                            for n, a in tree_flatten(lang.parameters())]
    save()
