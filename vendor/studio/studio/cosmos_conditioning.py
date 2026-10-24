# Derived from Studio backend/studio/cosmos_conditioning.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""Cosmos conditioning preflight shared by job admission and the engine seam."""
from __future__ import annotations

import math
from pathlib import Path


def is_cosmos(params):
    arch = str(params.model_arch or "").lower()
    model = str(params.model or "").lower()
    return (arch.startswith("cosmos") or arch in ("nano", "super", "super-t2i", "super-i2v")
            or (not arch and (not model or model in ("nano", "super", "super-t2i", "super-i2v"))))


def video_frames(params):
    return 17 if int(params.num_frames or 1) <= 1 else int(params.num_frames)


def validate(params):
    if not is_cosmos(params):
        return
    video = params.modality in ("t2v", "i2v", "v2v", "av", "fflf", "multikeyframe")
    if params.audio_mode in {"generate", "condition"} and not video:
        raise ValueError("Cosmos generated or conditioned audio requires a video operation")
    if params.denoise is not None and (not math.isfinite(params.denoise) or not 0 <= params.denoise <= 1):
        raise ValueError("Cosmos image edit strength must be between zero and one")
    if params.mask_path and params.modality != "i2i":
        raise ValueError("Cosmos masks require image editing")
    if params.modality == "i2i":
        if not params.image_path or not Path(params.image_path).is_file():
            raise ValueError("Cosmos image editing requires an existing source image")
        from PIL import Image
        for path in (params.image_path, params.mask_path):
            if path:
                with Image.open(path) as image:
                    image.verify()
    rows = [row.model_dump() for row in (params.keyframes or [])]
    if params.image2_path:
        rows.append(dict(image_path=params.image2_path, frame=video_frames(params) - 1))
    if rows:
        if not video or params.modality == "v2v":
            raise ValueError("Cosmos timed references require text/image-to-video or joint audio-video")
        from cosmos_windows.timed_frames import timeline
        timeline(rows, frames=video_frames(params), temporal_scale=1)
    if params.audio_mode == "condition":
        if not video:
            raise ValueError("Cosmos audio conditioning requires a generated video")
        if not params.audio_path or not Path(params.audio_path).is_file():
            raise ValueError("Cosmos audio conditioning requires an existing reference audio file")
        from cosmos_windows.audio_condition import require_codec
        require_codec(params.model_support_files)


def features(runtime_root):
    """Report executable integration; each request still checks its local assets."""
    runtime = Path(runtime_root or "") / "cosmos_windows"
    edit = (runtime / "image_edit.py").is_file()
    timed = (runtime / "timed_frames.py").is_file()
    audio = (runtime / "audio_condition.py").is_file()
    return dict(img2img=edit, inpaint_mask=edit, first_last_frame=timed,
                keyframes=timed, audio_condition=audio,
                audio_modes=["attach", "mux", "generate"] + (["condition"] if audio else []),
                prerequisites={"condition": ["joint audio-video Cosmos checkpoint", "paired complete Cosmos AVAE encoder codec"]},
                conditioning_provenance=timed or audio)
