# Derived from Studio backend/studio/generation_contract.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""Host-specific generation contract shared by admission and model controls.

Architecture entries describe implemented inputs. Model entries additionally check installed
files; they never load a checkpoint, download a component or claim trained output validation.
"""
from __future__ import annotations

import json
from pathlib import Path


ALIASES = {"sd-xl": "sdxl", "sd_xl": "sdxl", "illustrious": "sdxl", "noobai": "sdxl",
           "pony": "sdxl", "animagine": "sdxl", "h3": "minimax-h3", "zimage": "z-image",
           "domain-shuttle": "domainshuttle", "nano": "cosmos3", "super": "cosmos3",
           "super-t2i": "cosmos3", "super-i2v": "cosmos3", "3dgs": "tripo", "splat": "tripo",
           "image-to-3d": "tripo", "3d": "tripo"}
FAMILIES = ("cosmos", "flux2", "flux-2", "klein", "hunyuan", "glm", "krea2-native",
            "krea", "boogu", "sdxl", "anima", "z-image", "wan", "ltx", "bernini",
            "scail", "domainshuttle", "minimax-h3", "magi", "joyai", "ideogram",
            "ultratex", "tripo", "qwen-image-2.1", "qwenimage2.1", "moss-image")
ARCHITECTURES = (*FAMILIES, *ALIASES, "flux2-klein", "glm-image", "hunyuan-image-3",
                 "magi-2", "cosmos3", "cosmos3-t2i", "cosmos3-t2v", "cosmos3-i2v", "cosmos3-av")
MAC_FAMILIES = {"cosmos", "flux2", "flux-2", "klein", "hunyuan", "glm", "krea", "krea2-native", "sdxl", "tripo", "wan", "ltx"}
STACKED_CUDA = {"sdxl", "anima", "z-image", "wan", "bernini", "ltx", "krea2-native", "qwen-image-2.1", "qwenimage2.1", "minimax-h3"}
MODEL_DEFAULTS = {'moss-image': {'steps':50, 'guidance':3, 'width':1024, 'height':1024,
                                  'sampler':'euler', 'sigmas':'normal', 'weight_stream':False}}


class GenerationCapabilityError(ValueError, RuntimeError):
    """Invalid request for a backend; preserve the engine's existing RuntimeError interface."""


def family(arch):
    arch = str(arch or "").strip().lower()
    arch = ALIASES.get(arch, arch)
    return next((key for key in FAMILIES if arch.startswith(key)), "")


def architecture_contract(arch, platform, runtime_root, *, base=None):
    kind = family(arch)
    mac = platform == "darwin"
    available = bool(kind) and (not mac or kind in MAC_FAMILIES)
    result = {**(base or {}), "available": available, "platform": platform,
              "controlnet": False, "multi_lora": available and (kind in STACKED_CUDA if not mac else kind in {"krea", "krea2-native"}),
              "img2img": False, "inpaint_mask": False, "first_last_frame": False,
              "keyframes": False, "step_previews": False}
    from .two_pass import model_family, DEFAULT_STEPS
    result["two_pass"] = available and model_family(arch) in DEFAULT_STEPS and (not mac or kind != "sdxl")
    if kind in {"flux2", "flux-2", "klein"}:
        result["multi_lora"] = available
    if kind == "moss-image":
        from .moss_image import SAMPLERS, SCHEDULES
        result.update(modalities=["t2i","i2i"], img2img=available, multi_lora=available, samplers=list(SAMPLERS), schedules=list(SCHEDULES),
                      load_modes=['resident_cached','resident_lazy','streaming_lazy','streaming_nonlazy'],
                      load_mode_descriptions={
                          'resident_cached':'Keep the model and helper weights ready for subsequent images.',
                          'resident_lazy':'Load the encoder, denoiser and decoder in stages, releasing each after use. Denoiser weights stay on GPU during sampling.',
                          'streaming_lazy':'Stream denoiser layers from disk. Release the text encoder before sampling and load the decoder when needed.',
                          'streaming_nonlazy':'Stream denoiser layers from disk while keeping the text encoder and decoder on GPU through sampling.',
                      },
                      defaults=MODEL_DEFAULTS[kind])
    if not available:
        result.update(modalities=[], detail=f"{arch or 'Unknown architecture'} has no executable generation route on {platform}")
        return result
    if kind == "sdxl":
        result.update(modalities=["t2i"] if mac else ["t2i", "i2i"], img2img=not mac,
                      controlnet=True, multi_controlnet=True, control_kinds=["canny", "depth", "pose", "scribble"])
    elif kind in {"flux2", "flux-2", "klein", "glm", "krea", "krea2-native", "boogu", "hunyuan", "anima", "z-image", "ideogram", "ultratex"}:
        edit = kind in {"flux2", "flux-2", "klein", "glm", "krea2-native"} or (mac and kind == "krea") or (not mac and kind in {"boogu", "hunyuan", "anima", "z-image", "ideogram", "ultratex"})
        result.update(modalities=["t2i", "i2i"] if edit else ["t2i"], img2img=edit,
                      inpaint_mask=kind in {"krea2-native", "ideogram"} or (not mac and kind == "hunyuan") or (mac and kind == "krea"))
        if kind in {"flux2", "flux-2", "klein", "glm", "boogu"} or (kind == "krea" and not mac):
            from .generation_loras import available as loras_available
            result["multi_lora"] = (loras_available(runtime_root, custom_transformer=True)
                                    if kind in {"boogu", "krea"} else loras_available(runtime_root))
            if kind == "glm":
                result["lora_target_scope"] = "diffusion transformer linear projections"
    elif kind == "cosmos":
        from .cosmos_conditioning import features
        result.update(features(runtime_root))
        result.update(modalities=["t2i", "i2i", "t2v", "i2v", "av", "fflf", "multikeyframe"],
                      step_previews=not mac)
        from .generation_loras import available as loras_available
        result["multi_lora"] = loras_available(runtime_root)
    elif kind == "minimax-h3":
        result.update(modalities=["t2v", "i2v", "l2v", "fflf", "r2v", "ref2v", "v2v", "edit", "t2i", "i2i", "t2a", "ref2a"],
                      first_last_frame=True, img2img=True, official_modes=["t2va", "fl2va", "ref2va"])
    elif kind == "magi":
        # The configured runtime now defaults to the single-device stream-sdpa runner.
        # Its own dependency/asset/VRAM preflight still decides whether a selected model can run.
        result.update(modalities=["t2v", "i2v"], requires_multi_gpu=False, min_gpus=1,
                      requires_modules=["triton"], runner="stream-sdpa")
    elif kind in {"ltx", "wan", "joyai"}:
        if mac:
            from .generation_loras import available as loras_available
            result.update(modalities=["t2v"], multi_lora=loras_available(runtime_root),
                          audio_modes=["attach"], samplers=["flowmatch"], schedules=["normal"])
        else:
            result.update(modalities=["t2i", "i2i", "t2v", "i2v", "fflf", "multikeyframe", "av"],
                          first_last_frame=True, keyframes=True, img2img=True)
    return result


def _pipeline_missing(root, kind):
    missing = []
    index = root / "model_index.json"
    try:
        name = json.loads(index.read_text(encoding="utf-8")).get("_class_name")
    except (OSError, ValueError):
        return ["complete local pipeline model_index.json"]
    expected = {"cosmos": {"Cosmos3OmniPipeline"}, "wan": {"WanPipeline"}, "ltx": {"LTX2Pipeline"}}.get(
        kind, {"Flux2Pipeline", "Flux2KleinPipeline", "Flux2KleinKVPipeline"})
    if name not in expected:
        missing.append("compatible local pipeline class")
    components = ["transformer", "text_encoder", "tokenizer", "vae"] + ([] if kind == "cosmos" else ["scheduler"])
    if kind == "ltx":
        components += ["connectors", "audio_vae", "vocoder"]
    for name in components:
        directory = root / name
        if not directory.is_dir() or not any(directory.glob("*.json")):
            missing.append(f"{name} configuration")
        if name in {"transformer", "text_encoder", "vae", "connectors", "audio_vae", "vocoder"} and not any(directory.glob("*.safetensors")):
            missing.append(f"{name} weights")
    return missing


def model_contract(row, platform, runtime_root, *, modality=None, base=None):
    arch = row.arch or ("cosmos3" if row.name.lower() in ALIASES and family(row.name) == "cosmos" else "")
    features = architecture_contract(arch, platform, runtime_root, base=base)
    implemented = features.get("modalities")
    modalities = [m for m in row.modalities if implemented is None or m in implemented]
    missing = []
    if family(arch) == "sdxl":
        from .sdxl_control import installed_kinds
        kinds = installed_kinds(row.support_files)
        features.update(controlnet=bool(kinds), control_kinds=kinds)
    if row.status != "present":
        missing.append("installed checkpoint")
    selected = Path(row.path).expanduser() if row.path else None
    if selected is None and family(arch) == "tripo" and row.status == "present":
        try:
            from . import triposplat_backend
            selected = Path(triposplat_backend._ckpt_dir(triposplat_backend._repo_dir()))
        except (OSError, RuntimeError, ValueError):
            pass
    if selected is None or not selected.exists():
        missing.append("local checkpoint path")
    kind = family(arch)
    if kind == "moss-image" and selected is not None:
        from .moss_image import missing_components
        missing.extend(missing_components(selected))
    temporal_pipeline = kind in {"wan", "ltx"} and selected is not None and selected.is_dir()
    if temporal_pipeline:
        missing.extend(_pipeline_missing(selected, kind))
        features.update(modalities=["t2v"], first_last_frame=False, keyframes=False, img2img=False,
                        audio_modes=["attach"], samplers=["flowmatch"], schedules=["normal"])
        modalities = [m for m in row.modalities if m == "t2v"]
    from .custom_models import SUPPORT_SLOTS, comfy_support_defaults
    slots = SUPPORT_SLOTS.get(arch, SUPPORT_SLOTS.get(kind, []))
    support = {**comfy_support_defaults().get(arch, {}), **(row.support_files or {})}
    if kind == "magi" and support.get("engine") == "torchrun":
        features.update(runner="torchrun", requires_multi_gpu=True, min_gpus=8,
                        requires_modules=["magi_attention", "magi_compiler", "triton"])
    path_keys = {slot["key"] for slot in slots if slot.get("browse") in {"file", "folder"}}
    for slot in slots:
        if not temporal_pipeline and slot.get("required") and slot["key"] in path_keys:
            raw = support.get(slot["key"])
            if not raw or not Path(raw).expanduser().exists():
                missing.append(f"{slot['label']} local component")
    if platform == "darwin" and kind in {"cosmos", "flux2", "flux-2", "klein"}:
        root = Path(support.get("components_repo") or row.path or "")
        missing.extend(_pipeline_missing(root, kind))
    for key, raw in (row.support_files or {}).items():
        if key in path_keys and not key.startswith(("controlnet", "control_")) and key not in {"audio_encoder_codec", "audio_encoder", "audio_decoder"} and raw and not Path(raw).expanduser().exists():
            missing.append(f"{key} local component")
    if kind == "cosmos" and selected and selected.is_dir():
        try:
            config = json.loads((selected / "transformer/config.json").read_text())
            sound = bool(config.get("sound_gen") and any((selected / "sound_tokenizer").glob("*.safetensors")))
        except (OSError, ValueError):
            sound = False
        modes = ["attach", "mux"] + (["generate"] if sound else [])
        condition = False
        if sound and features.get("audio_condition"):
            try:
                from cosmos_windows.audio_condition import require_codec
                require_codec(support)
                condition = True
                modes.append("condition")
            except (ImportError, OSError, ValueError):
                pass
        features.update(audio_condition=condition, audio_modes=modes)
        if not sound:
            features.setdefault("prerequisites", {})["generate"] = ["joint sound checkpoint and complete sound tokenizer"]
    executable = bool(features["available"] and modalities and not missing and (modality is None or modality in modalities))
    return {"arch": arch, "features": features, "modalities": modalities,
            "executable": executable, "missing_prerequisites": missing}


def validate_request(params, platform, runtime_root, *, features=None):
    from .two_pass import validate_request as validate_two_pass
    validate_two_pass(params)
    arch = params.model_arch or ("cosmos3" if str(params.model or "").lower() in {"", "nano", "super", "super-t2i", "super-i2v"} else "")
    if not arch and family(params.model):
        arch = params.model
    # Legacy raw Mac single-file SDXL is inferred only when no explicit family was supplied.
    if not arch and platform == "darwin" and str(params.model).lower().endswith(".safetensors"):
        if not any(key in str(params.model).lower() for key in ("flux", "hunyuan", "klein")):
            arch = "sdxl"
    contract = features if features is not None else architecture_contract(arch, platform, runtime_root)
    if not contract["available"]:
        raise ValueError(contract["detail"])
    if contract.get('load_modes') and params.load_mode and params.load_mode not in contract['load_modes']:
        raise ValueError(f"{arch} does not implement load mode {params.load_mode}; use {', '.join(contract['load_modes'])}")
    if contract.get("modalities") is not None and params.modality not in contract["modalities"]:
        raise ValueError(f"{arch} does not execute {params.modality} on {platform}")
    if contract.get("audio_modes") is not None and params.audio_mode not in contract["audio_modes"]:
        raise ValueError(f"{arch} cannot execute audio {params.audio_mode} with its installed components on {platform}")
    timed = params.keyframes
    if timed and not contract["keyframes"]:
        # Normalization adds ordinary first/last endpoints to keyframes too. Preserve those
        # executable inputs without claiming that the backend consumes arbitrary middle frames.
        end = max(0, int(params.num_frames or 1) - 1)
        timed = [row for row in timed if not (
            ((row.frame == 0 or row.at_fraction == 0 or row.role == "first") and params.modality in {"i2v", "l2v", "fflf", "t2v", "r2v", "ref2v"})
            or ((row.frame == end or row.at_fraction == 1 or row.role == "last") and contract["first_last_frame"]))]
    from .generation_loras import requested_loras
    loras = requested_loras(params)
    for requested, supported, label in (
        (loras, contract["multi_lora"], "LoRA adapters"),
        (params.control_image or params.control_kind or params.control_guides, contract["controlnet"], "structural guides"),
        (params.mask_path, contract["inpaint_mask"], "masked inpainting"),
        (timed, contract["keyframes"], "timed keyframes"),
        (params.image2_path, contract["first_last_frame"], "final reference frame"),
    ):
        if requested and not supported:
            if label == "LoRA adapters":
                target = "the Cosmos pipeline" if family(arch) == "cosmos" else arch
                raise GenerationCapabilityError(f"LoRA adapters are not supported by {target} on {platform}")
            if label == "structural guides":
                raise GenerationCapabilityError(f"ControlNet/structural conditioning is not supported for {arch} on {platform}")
            raise GenerationCapabilityError(f"{arch} does not execute {label} on {platform}")
    if loras and features is None and family(arch) in {"cosmos", "flux2", "flux-2", "klein", "glm", "boogu", "krea"}:
        from .generation_loras import preflight
        preflight(params)
    if family(arch) == "sdxl" and (params.control_image or params.control_kind or params.control_guides):
        from .sdxl_control import preflight
        preflight(params)
    if family(arch) in {"wan", "ltx"} and (platform == "darwin" or Path(str(params.model)).is_dir()):
        from .temporal_video import preflight
        preflight(params)
        if platform != "darwin" and params.weight_stream:
            raise ValueError("Complete temporal pipelines use resident loading; turn weight streaming off")
    return contract
