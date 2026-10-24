# Derived from Studio backend/studio/two_pass.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; machine-specific fallback paths cleared, otherwise unchanged.
"""Gate model-only adapters inside one continuous denoising trajectory."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import time

SDA_FILE = "krea2_turbo_sda_v1.0_comfy.safetensors"
KROMA_MODEL = "Kroma-v0.3.1-Turbo-OPD-W8A8"
DEFAULT_STEPS = {"krea2": 8, "flux2": 4, "qwen": 40, "sdxl": 28}


@dataclass
class RuntimeSettings:
    enabled: bool
    split_step: int
    loras: list
    krea_sda_schedule: bool = False


def runtime_settings(value):
    """Strict worker parsing without importing the daemon's Pydantic dependency."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Two-pass settings must be an object")
    enabled, split = value.get("enabled", False), value.get("split_step", 2)
    schedule = value.get("krea_sda_schedule", False)
    if type(enabled) is not bool or type(schedule) is not bool:
        raise ValueError("Two-pass enable and schedule controls must be booleans")
    if type(split) is not int or split < 1:
        raise ValueError("First-pass split step must be a positive integer")
    rows = value.get("loras", [])
    if not isinstance(rows, list):
        raise ValueError("First-pass LoRAs must be a list")
    loras = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"].strip():
            raise ValueError("First-pass LoRA needs a file path")
        strength = row.get("weight", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ValueError("First-pass LoRA strength must be a finite number")
        loras.append({"path": row["path"], "weight": float(strength)})
    return RuntimeSettings(enabled, split, loras, schedule)


def model_family(arch):
    value = str(arch or "").lower()
    for prefix, result in (("krea", "krea2"), ("flux", "flux2"), ("klein", "flux2"),
                           ("qwen", "qwen"), ("sdxl", "sdxl"), ("z-image", "z-image")):
        if value.startswith(prefix):
            return result
    return ""


def _header(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Invalid LoRA safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError("Invalid LoRA safetensors header length")
        return json.loads(stream.read(length))


def validate_adapters(specs, family):
    """Read only headers before GPU admission; loaders also validate every target/shape."""
    for spec in specs:
        spec = spec.model_dump() if hasattr(spec, "model_dump") else spec
        path = str(spec.get("path") or "").strip()
        if not path or not Path(path).is_file():
            raise ValueError(f"First-pass LoRA file not found: {path}")
        if not math.isfinite(float(spec.get("weight", 1))):
            raise ValueError("First-pass LoRA strength must be finite")
        header = _header(path)
        keys = [key.lower() for key in header if key != "__metadata__"]
        if any(key.startswith(("lora_te", "text_encoder", "te.")) for key in keys):
            raise ValueError("Two-pass sampling requires model-only LoRAs; text encoder adapters would retain their conditioning in pass 2")
        markers = {
            "krea2": ("blocks.", "blocks_", "txtfusion", "txt_fusion", "text_fusion"),
            "flux2": ("double_blocks", "single_blocks", "single_transformer_blocks", "transformer_blocks"),
            "qwen": ("transformer_blocks",),
            "sdxl": ("lora_unet", "unet.", "down_blocks", "up_blocks"),
            "z-image": ("layers.", "layers_", "noise_refiner", "context_refiner"),
        }
        declared = " ".join(str(v).lower() for v in header.get("__metadata__", {}).values())
        declared_family = next((f for f, names in {
            "krea2": ("krea",), "flux2": ("flux", "klein"), "qwen": ("qwen-image", "qwen_image"),
            "sdxl": ("sdxl", "stable-diffusion-xl"), "z-image": ("z-image", "z_image"),
        }.items() if any(name in declared for name in names)), None)
        if declared_family and declared_family != family:
            raise ValueError(f"First-pass LoRA targets {declared_family}, but the selected model is {family}")
        # Qwen's img/txt attention paths distinguish it from Flux's joint attention.
        qwen_keys = any("img_attn" in key or "txt_attn" in key or "img_mlp" in key for key in keys)
        native_krea = declared_family == "krea2" or any(any(marker in key for marker in
            ("text_fusion", "txtfusion", "txt_fusion", ".ff.gate.", ".ff.up.", ".ff.down.")) for key in keys)
        flux_keys = any(any(marker in key for marker in
            ("double_blocks", "single_blocks", "single_transformer_blocks")) for key in keys)
        ambiguous_blocks = any("transformer_blocks" in key for key in keys) and not native_krea
        if family == "krea2" and (qwen_keys or flux_keys or ambiguous_blocks):
            raise ValueError("First-pass LoRA does not match the selected krea2 model family")
        if (family == "flux2" and qwen_keys) or (family == "qwen" and not qwen_keys and declared_family != "qwen"):
            raise ValueError(f"First-pass LoRA does not match the selected {family} model family")
        if not any(any(marker in key for marker in markers.get(family, ())) for key in keys):
            raise ValueError(f"First-pass LoRA does not match the selected {family} model family")


def validate_request(params):
    option = getattr(params, "two_pass", None)
    if option and option.krea_sda_schedule and model_family(params.model_arch) != "krea2":
        raise ValueError("The SDA noise schedule requires a Krea 2 model")
    if option is None or not option.enabled:
        return
    family = model_family(params.model_arch)
    if family not in DEFAULT_STEPS:
        raise ValueError(f"Two-pass model-only LoRA sampling has no continuous sampler route for {params.model_arch}")
    if params.modality != "t2i" or params.image_path:
        raise ValueError("Two-pass sampling currently requires text-to-image with a full noise schedule")
    steps = int(params.steps or DEFAULT_STEPS[family])
    if not 0 < option.split_step < steps:
        raise ValueError(f"First-pass split step must be between 1 and {steps - 1} for {steps} total steps")
    specs = list(params.loras or []) + list(option.loras)
    if not specs:
        raise ValueError("Select at least one first-pass LoRA for two-pass sampling")
    validate_adapters(specs, family)


def catalog():
    from .krea2_diffusion import _comfy_models_base
    import sys
    root = _comfy_models_base() / "loras" / "krea2"
    path = root / SDA_FILE
    result = {
        "defaults": {family: {"split_step": min(2, steps - 1), "strength": 1.0, "steps": steps}
                     for family, steps in DEFAULT_STEPS.items()},
        "presets": [{"id": "krea2-variation-sda-two-pass", "name": "Krea 2 — variation (SDA, two-pass)",
                     "family": "krea2", "installed": path.is_file(),
                     "params": {"steps": 8, "guidance": 1.0, "sampler": "euler", "sigmas": "simple",
                                "width": 1024, "height": 1024,
                                "two_pass": {"enabled": True, "split_step": 2,
                                             "krea_sda_schedule": True,
                                             "loras": [{"path": str(path), "weight": 1.0}]}}}],
    }
    # Kroma already includes Turbo OPD: do not merge another Turbo adapter.
    # Standard CFG 1 maps to native Krea guidance 0. The existing shifted
    # Euler schedule provides the author's mu=1.15, independently of gating.
    from .custom_models import get
    model = get(KROMA_MODEL)
    checkpoint = (model or {}).get("support_files", {}).get("dit") or (model or {}).get("path")
    result["presets"].append({
        "id": "kroma-int8", "name": "Kroma (int8)", "family": "krea2",
        "models": [KROMA_MODEL], "installed": bool(checkpoint and Path(checkpoint).is_file()),
        "missing_label": "model missing",
        "params": {"steps": 8, "guidance": 0.0, "sampler": "euler", "sigmas": "simple",
                   "width": 1024, "height": 1024,
                   "two_pass": {"enabled": False, "split_step": 2,
                                "krea_sda_schedule": True, "loras": []}},
    })
    return result


class AdapterGate:
    """Switch at callback k-1, without touching latents, RNG, or scheduler state."""
    def __init__(self, split_step, total, switch_off):
        if not 0 < int(split_step) < int(total):
            raise ValueError("Two-pass split must leave at least one step in each pass")
        self.split_step, self.total = int(split_step), int(total)
        self.switch_off = switch_off
        self.completed = 0
        self.switched = False

    def complete(self, step, values):
        if int(step) != self.completed:
            raise RuntimeError("Two-pass sampler callback skipped or repeated a denoise step")
        self.completed += 1
        if self.completed == self.split_step:
            started = time.perf_counter()
            self.switch_off()
            self.switch_seconds = time.perf_counter() - started
            self.switched = True
        return values

    def verify(self):
        if not self.switched or self.completed != self.total:
            raise RuntimeError(f"Two-pass sampler handoff was not executed: {self.completed}/{self.total} steps")


@contextmanager
def adapter_gate(option, total, apply, baseline=(), first_pass=()):
    if option is None or not option.enabled:
        yield None
        return
    extra = [spec.model_dump() if hasattr(spec, "model_dump") else spec for spec in option.loras]
    baseline = list(baseline)
    try:
        apply(baseline + list(first_pass) + extra)
        gate = AdapterGate(option.split_step, total, lambda: apply(baseline))
        yield gate
        gate.verify()
    finally:
        # Also restore after cancellation, loading errors, or decode failures.
        apply(baseline)


def run_pipeline(pipe, kwargs, option, apply, baseline=(), first_pass=()):
    pipe._studio_two_pass_receipt = None
    with adapter_gate(option, kwargs["num_inference_steps"], apply, baseline, first_pass) as gate:
        if gate:
            callback = kwargs.get("callback_on_step_end")
            def complete(p, step, timestep, values):
                gate.complete(step, values)
                return callback(p, step, timestep, values) if callback else values
            kwargs = {**kwargs, "callback_on_step_end": complete}
        result = pipe(**kwargs)
        if gate:
            gate.verify()
            scheduler = getattr(pipe, "scheduler", None)
            sigmas = getattr(scheduler, "sigmas", None)
            pipe._studio_two_pass_receipt = {
                "steps": gate.total, "split_step": gate.split_step,
                "first_pass_active_steps": list(range(gate.split_step)),
                "second_pass_active_steps": [], "completed_steps": gate.completed,
                "handoff": "same pipeline, latent, RNG and scheduler; no restart or new noise",
                "adapter_switch_seconds": gate.switch_seconds,
                "scheduler": type(scheduler).__name__,
                "sigmas": sigmas.detach().cpu().tolist() if hasattr(sigmas, "detach") else None,
            }
        return result


def apply_diffusers_loras(pipe, specs):
    """Unfused per-request model adapters; no weight or scheduler replacement."""
    if getattr(pipe, "_studio_two_pass_adapters", False):
        pipe.unload_lora_weights()
        pipe._studio_two_pass_adapters = False
    if not specs:
        return
    pipe._studio_two_pass_adapters = True
    names, strengths = [], []
    try:
        for index, spec in enumerate(specs):
            path = Path(spec["path"])
            name = f"studio_pass_{index}"
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name, local_files_only=True)
            names.append(name)
            strengths.append(float(spec.get("weight", 1.0)))
        pipe.set_adapters(names, adapter_weights=strengths)
    except Exception:
        pipe.unload_lora_weights()
        pipe._studio_two_pass_adapters = False
        raise
