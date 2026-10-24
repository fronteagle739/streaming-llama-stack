# Derived from Studio backend/studio/generation_loras.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; machine-specific fallback paths cleared, otherwise unchanged.
"""Configuration-only validation for the denoiser adapter paths implemented here."""
from pathlib import Path
from contextlib import contextmanager, ExitStack


def requested_loras(params):
    specs = list(params.loras or [])
    option = getattr(params, "two_pass", None)
    if option and option.enabled:
        specs.extend(option.loras)
    return specs


@contextmanager
def phase_loras(transformer, specs, option, *, device, receipt=None):
    """Own one stack at a time so a continuous phase handoff restores adapters correctly."""
    with ExitStack() as active:
        def apply(rows):
            active.close()
            records=[]
            if rows:
                from cosmos_windows.weight_streaming.lora_stack import stacked_adapters
                records=active.enter_context(stacked_adapters(transformer, rows, device=device))
            if receipt is not None:
                receipt.setdefault("adapter_phases", []).append(records)
                if "loras" not in receipt:
                    receipt["loras"] = records
        if not option or not option.enabled:
            apply(specs)
        yield apply


def available(runtime_root, *, custom_transformer=False):
    import importlib.util
    if not runtime_root or importlib.util.find_spec("peft") is None:
        return False
    path = Path(runtime_root) / "cosmos_windows/weight_streaming/lora_stack.py"
    if not path.is_file():
        return False
    spec = importlib.util.spec_from_file_location("_studio_generation_lora_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (getattr(module, "COSMOS_LORA_PROTOCOL", 0) == 1
            and (not custom_transformer or getattr(module, "DIFFUSERS_LORA_PROTOCOL", 0) == 1))


def preflight(params):
    specs = requested_loras(params)
    if not specs:
        return []
    from .generation_contract import family
    kind = family(params.model_arch or params.model or "cosmos3")
    if kind not in {"cosmos", "flux2", "flux-2", "klein", "glm", "boogu", "krea"}:
        return []
    from cosmos_windows.weight_streaming.lora_stack import preflight_model_loras
    root = Path(str(params.model or ""))
    if kind in {"boogu", "krea"}:
        import sys
        if kind == "krea" and sys.platform == "darwin":
            return []  # Mac Krea already uses the separate native LoKR backend.
        from .boogu_krea import lora_transformer_class
        return preflight_model_loras(root, specs, transformer_cls=lora_transformer_class(params.model_arch))
    if kind == "glm":
        import os
        import sys
        if not root.is_dir():
            root = Path(os.environ.get("STUDIO_GLM_MODEL") or
                        "")
        return preflight_model_loras(root, specs, transformer_class="GlmImageTransformer2DModel")
    if kind in {"flux2", "flux-2", "klein"}:
        import os
        if not root.is_dir():
            default = Path("")
            if not default.is_dir():
                default = Path("")
            root = Path((params.model_support_files or {}).get("components_repo")
                        or os.environ.get("STUDIO_FLUX2_KLEIN_DIR") or default)
        return preflight_model_loras(root, specs, transformer_class="Flux2Transformer2DModel")
    from cosmos_windows import cosmos_cli
    if str(params.model) in cosmos_cli.MODELS:
        root = Path(cosmos_cli.MODELS_ROOT) / cosmos_cli.MODELS[str(params.model)]["dir"]
    return preflight_model_loras(root, specs)
