# Derived from Studio backend/studio/models.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, StrictBool, model_validator
from .h3_upscale import H3UpscaleSettings
Modality = Literal['t2i', 't2v', 'i2v', 'l2v', 'v2v', 'i2i', 'av', 'fflf', 'multikeyframe', 't2t', 't+i2t', 'image-to-3d', 'r2v', 'ref2v', 'rv2v', 'mv2v', 'edit', 'animate', 'replace']
LOAD_MODES = ('resident_cached', 'resident_lazy', 'streaming_lazy', 'streaming_nonlazy')

def resolve_load_mode(params) -> tuple[str, bool, bool]:
    lm = getattr(params, 'load_mode', None)
    if lm not in LOAD_MODES:
        lm = 'streaming_lazy' if getattr(params, 'weight_stream', True) else 'resident_cached'
    return (lm, lm.startswith('resident'), lm in ('resident_lazy', 'streaming_lazy'))
AudioMode = Literal['attach', 'condition', 'mux', 'generate']

class CompositionBox(BaseModel):
    id: str
    type: Literal['obj', 'text'] = 'obj'
    label: str = ''
    desc: str = ''
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=1000)
    width: float = Field(gt=0, le=1000)
    height: float = Field(gt=0, le=1000)
    text: Optional[str] = None
    color: Optional[str] = None
    colors: list[str] = Field(default_factory=list)

class IdeogramPromptFields(BaseModel):
    high_level_description: Optional[str] = None
    aesthetics: Optional[str] = None
    lighting: Optional[str] = None
    medium: Optional[str] = None
    art_style: Optional[str] = None
    color_palette: list[str] = Field(default_factory=list)

class CompositionGuide(BaseModel):
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    background: str = ''
    boxes: list[CompositionBox] = Field(default_factory=list)
    ideogram: Optional[IdeogramPromptFields] = None

class ControlGuide(BaseModel):
    image_path: str = ''
    kind: str = 'canny'
    weight: float = 1.0
    source: str = 'control'
    role: Optional[str] = None

class VideoKeyframe(BaseModel):
    image_path: str
    frame: Optional[int] = Field(default=None, ge=0)
    at_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    role: Optional[Literal['first', 'middle', 'last']] = None
    prompt: Optional[str] = None
    strength: float = Field(default=1.0, ge=0, le=2)

class LoraSpec(BaseModel):
    path: str
    weight: float = 1.0
    name: Optional[str] = None
    expert: Optional[Literal['high', 'low', 'both']] = 'both'

class LoopStep(BaseModel):
    action: str = ''
    num_frames: Optional[int] = Field(default=None, ge=1)
    last_frame_path: Optional[str] = None
    tail_trim: Optional[int] = Field(default=None, ge=0)

class QwenImage21Point(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)

class QwenImage21RegionShape(BaseModel):
    type: Literal['circle', 'stroke', 'mask']
    cx: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    cy: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    radius: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    points: list[QwenImage21Point] = Field(default_factory=list)
    width: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    mask_path: Optional[str] = None

    @model_validator(mode='after')
    def validate_shape(self):
        if self.type == 'circle' and any((v is None for v in (self.cx, self.cy, self.radius))):
            raise ValueError('circle shape requires cx, cy, and radius')
        if self.type == 'stroke' and (len(self.points) < 2 or self.width is None):
            raise ValueError('stroke shape requires at least two points and width')
        if self.type == 'mask' and (not str(self.mask_path or '').strip()):
            raise ValueError('mask shape requires mask_path')
        return self

class QwenImage21Region(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    color: str = Field(min_length=1, max_length=40)
    instruction: str = Field(min_length=1, max_length=2000)
    shape: QwenImage21RegionShape

class TwoPassSettings(BaseModel):
    enabled: bool = False
    split_step: int = Field(default=2, ge=1)
    loras: list[LoraSpec] = Field(default_factory=list)
    krea_sda_schedule: bool = False

class ImagePaintCrop(BaseModel):
    x: int = Field(ge=0, le=65535)
    y: int = Field(ge=0, le=65535)
    width: int = Field(ge=1, le=65536)
    height: int = Field(ge=1, le=65536)

    @model_validator(mode='after')
    def bounded_frame(self) -> 'ImagePaintCrop':
        if self.x + self.width > 65536 or self.y + self.height > 65536:
            raise ValueError('crop falls outside the supported image coordinate range')
        return self

class ImagePaintEdit(BaseModel):
    overlay_path: str = Field(min_length=1, max_length=8192)
    combined_path: str = Field(min_length=1, max_length=8192)
    source_mask_path: Optional[str] = Field(default=None, min_length=1, max_length=8192)
    crop: Optional[ImagePaintCrop] = None

class GenerationParams(BaseModel):
    image_paint_edits: dict[str, ImagePaintEdit] = Field(default_factory=dict, max_length=64)
    prompt: str
    worker_id: Optional[str] = None
    negative: str = ''
    modality: Modality = 't2i'
    conversation_id: Optional[str] = None
    output_dir: Optional[str] = None
    width: int = 256
    height: int = 256
    steps: Optional[int] = None
    two_pass: Optional[TwoPassSettings] = None
    guidance: Optional[float] = None
    sdxl_prediction_type: Optional[Literal['epsilon', 'v_prediction']] = None
    sdxl_clip_skip: Optional[int] = Field(default=None, ge=0, le=10)
    wan_crossover_step: Optional[int] = Field(default=None, ge=0)
    wan_boundary_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    wan_shift: Optional[float] = Field(default=None, gt=0)
    sampler: str = 'unipc'
    sigmas: str = 'karras'
    seed: int = 42
    compute_dtype: Optional[str] = None
    model: str = 'nano'
    num_frames: int = 1
    fps: int = 24
    cfg_batch: bool = False
    weight_stream: bool = True
    h3_attention_backend: Literal['sdpa', 'sageattention', 'flash_attn2'] = 'flash_attn2'
    h3_sparse_attention: Literal['auto', 'on', 'off'] = 'auto'
    h3_sparse_method: Literal['sol-attn', 'sla'] = 'sol-attn'
    h3_reference_resolution: Optional[int] = None
    h3_max_short_edge: Optional[int] = Field(default=None, ge=32, le=8192)
    h3_prompt_enhance: bool = True
    h3_enhancer_backend: Optional[Literal['auto', 'endpoint', 'live-preset', 'local-reasoner', 'gguf']] = None
    h3_upscale: Optional[H3UpscaleSettings] = None
    prompt_already_enhanced: bool = False
    magi2_prompt_enhance: bool = True
    load_mode: Optional[Literal['resident_cached', 'resident_lazy', 'streaming_lazy', 'streaming_nonlazy']] = None
    image_path: Optional[str] = None
    image2_path: Optional[str] = None
    video_path: Optional[str] = None
    num_conditional_frames: Optional[int] = None
    reference_images: list[str] = Field(default_factory=list)
    qwen_weight_precision: Literal['bf16', 'int8'] = 'bf16'
    qwen_output_rgba: bool = False
    qwen_use_kv_cache: bool = True
    qwen_request_encoder_staging: bool = False
    qwen_source_width: Optional[int] = Field(default=None, gt=0, le=32768)
    qwen_source_height: Optional[int] = Field(default=None, gt=0, le=32768)
    qwen_regions: list[QwenImage21Region] = Field(default_factory=list, max_length=9)
    reference_video: Optional[str] = None
    source_video: Optional[str] = None
    reference_videos: list[str] = Field(default_factory=list)
    h3_seamless: bool = False
    h3_extend_from: Optional[str] = None
    reference_audios: list[str] = Field(default_factory=list)
    bernini_task: Optional[str] = None
    reference_domains: list = Field(default_factory=list)
    keyframes: list[VideoKeyframe] = Field(default_factory=list)
    audio_path: Optional[str] = None
    audio_mode: AudioMode = 'attach'
    denoise: Optional[float] = None
    krea2_i2i_tile_size: int = Field(default=0, ge=0, le=4096)
    krea2_i2i_tile_overlap: int = Field(default=256, ge=0, le=2048)
    hunyuan_native_edit: bool = False
    mask_path: Optional[str] = None
    krea2_edit: bool = False
    krea_request_process: bool = False
    krea_request_timeout_s: Optional[float] = Field(default=None, ge=1, le=10800)
    krea2_edit_reference_path: Optional[str] = None
    krea2_edit_lora_path: Optional[str] = None
    krea2_edit_grounding_px: int = Field(default=768, ge=0, le=4096)
    krea2_edit_recipe: Literal['identity', 'ostris'] = 'identity'
    loras: list[LoraSpec] = Field(default_factory=list)
    control_image: Optional[str] = None
    control_kind: Optional[str] = None
    control_weight: float = 1.0
    control_guides: list[ControlGuide] = Field(default_factory=list)
    composition_guide: Optional[CompositionGuide] = None
    composition_guide_path: Optional[str] = None
    model_arch: Optional[str] = None
    model_support_files: dict[str, str] = Field(default_factory=dict)
    reason: bool = False
    reasoner_model: Optional[str] = None
    reason_max_tokens: int = 200
    reason_temperature: Optional[float] = None
    reason_top_p: Optional[float] = None
    reason_system: Optional[str] = None
    ideo_preset: str = 'V4_QUALITY_48'
    ideo_quant: str = 'nf4'
    glm_rep_penalty: Optional[float] = None
    glm_rep_window: Optional[int] = None
    ltx_mode: Optional[str] = None
    ltx_360_strength: Optional[float] = None
    loop_steps: list[LoopStep] = Field(default_factory=list)
    loop_overlap: int = Field(default=5, ge=0)
    loop_tail_trim: int = Field(default=0, ge=0)
    project_ids: dict[str, str] = Field(default_factory=dict)
    history_thread_id: str = ''
    history_parent_entry_id: str = ''

    @model_validator(mode='after')
    def _capture_project_ids(self):
        if not self.project_ids:
            from .project_context import current
            ids = current()
            if ids:
                self.project_ids = ids
        return self

class GenResult(BaseModel):
    worker_receipt: Optional[dict[str, Any]] = None
    krea_worker_receipt: Optional[dict[str, Any]] = None
    path: str
    seconds: float
    modality: str
    width: int
    height: int
    kind: Literal['image', 'video', 'splat'] = 'image'
    seed: int
    engine: str
    model: Optional[str] = None
    model_arch: Optional[str] = None
    timings: Optional[dict[str, Any]] = None
    projection: Optional[str] = None
    splat: Optional[str] = None
    mesh: Optional[str] = None
    preview: Optional[str] = None
    views: Optional[list[str]] = None

class ModelInfo(BaseModel):
    name: str
    label: str
    status: str
    modalities: list[str]
    current: bool = False
    path: Optional[str] = None
    note: Optional[str] = None
    arch: Optional[str] = None
    support_files: Optional[dict[str, str]] = None
    checkpoint_format: Optional[str] = None
    checkpoint_components: Optional[list[str]] = None
    arch_hint: Optional[str] = None
    location_overridden: bool = False
