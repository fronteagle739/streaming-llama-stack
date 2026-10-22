# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,garbage_collection_threshold:0.7')
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from cosmos_windows.weight_streaming.magi2_prompt import DEFAULT_NEGATIVE_PROMPT, MAGI2_CLIP_SECONDS, MAGI2_MODES, load_prompt_text, magi2_enhance_presets, mode_config_defaults
MAGI2_PREVIEW_SHARDS = 56
MAGI2_REFINER_SHARDS = 3
MAGI2_OUTPUT_FPS = 12.5
MAGI2_OFFLOAD_MODES = ('cpu', 'gpu', 'roundtrip')
MAGI2_COMPONENTS = ('preview', 'refiner', 'text_encoder', 'vae', 'stable-audio-open-1.0', 'turbo_vae')
MAGI2_ENGINES = ('stream-sdpa', 'torchrun')
MAGI2_EXPERT_DTYPES = ('bf16', 'fp8')
RESOLUTION_PRESETS: dict[str, dict] = {'272p': {'config': 'configs/magi2_preview.json', 'pw': 448, 'ph': 256}, '640sq': {'config': 'configs/magi2_preview.json', 'pw': 640, 'ph': 640}, '800x448': {'config': 'configs/magi2_preview.json', 'pw': 800, 'ph': 448}, '540p': {'config': 'configs/magi2_preview.json', 'pw': 896, 'ph': 512}, '1080p': {'config': 'configs/magi2_refiner.json', 'pw': 896, 'ph': 512, 'rw': 1920, 'rh': 1088}}
MAGI2_REFERENCE_MODULES = ('torch', 'magi_attention', 'magi_compiler', 'triton')
MAGI2_STREAM_SDPA_MODULES = ('torch', 'triton')
MAGI2_MIN_VRAM_GB = float(os.environ.get('MAGI2_MIN_VRAM_GB', '28'))
MAGI2_DEFAULT_WORLD_SIZE = 8

@dataclass(frozen=True)
class Magi2GenerationRequest:
    model_dir: Path
    prompt: str
    mode: str
    out: Path
    image: Optional[Path] = None
    resolution: str = '1080p'
    seconds: float = MAGI2_CLIP_SECONDS
    steps: Optional[int] = None
    refiner_steps: Optional[int] = None
    enable_refiner: Optional[bool] = None
    seed: int = 42
    world_size: int = MAGI2_DEFAULT_WORLD_SIZE
    deterministic: bool = False
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    preview_width: Optional[int] = None
    preview_height: Optional[int] = None
    refiner_width: Optional[int] = None
    refiner_height: Optional[int] = None
    repo_root: Optional[Path] = None
    text_enc_offload: Optional[str] = None
    preview_offload: Optional[str] = None
    refiner_offload: Optional[str] = None
    vae_offload: Optional[str] = None
    master_port: int = 29500
    engine: str = 'stream-sdpa'
    prompt_file: Optional[Path] = None
    negative_prompt: Optional[str] = None
    batch_size: int = 1
    expert_dtype: str = 'bf16'
    compute_dtype: str = 'bf16'
    ring_budget_gb: float = 14.0
    min_vram_gb: float = MAGI2_MIN_VRAM_GB
    prompts: Tuple[str, ...] = ()
    images: Tuple[Optional[str], ...] = ()
    debug_events: bool = False

    def all_prompts(self) -> List[str]:
        return [self.prompt, *self.prompts] if self.prompts else [self.prompt]

    def all_images(self) -> List[Optional[str]]:
        if self.images:
            return [str(self.image) if self.image else None, *[str(i) if i else None for i in self.images]]
        return [str(self.image) if self.image else None]

def validate_magi2_generation_request(request: Magi2GenerationRequest, *, check_files: bool=True) -> list[str]:
    errors: list[str] = []
    if request.mode not in MAGI2_MODES:
        errors.append(f'unsupported MAGI-2 mode: {request.mode!r} (expected one of {MAGI2_MODES})')
    elif request.mode == 't2v':
        if request.image is not None:
            errors.append('t2v does not accept an --image; use i2v for a first-frame conditioned clip')
    elif request.mode == 'i2v':
        if request.image is None:
            errors.append('i2v requires --image (a first-frame still)')
    if request.resolution not in RESOLUTION_PRESETS:
        errors.append(f'resolution must be one of {tuple(RESOLUTION_PRESETS)}; got {request.resolution!r}')
    if request.engine not in MAGI2_ENGINES:
        errors.append(f'engine must be one of {MAGI2_ENGINES}; got {request.engine!r}')
    if request.expert_dtype not in MAGI2_EXPERT_DTYPES:
        errors.append(f'expert_dtype must be one of {MAGI2_EXPERT_DTYPES}; got {request.expert_dtype!r}')
    if request.batch_size < 1:
        errors.append('batch_size must be at least 1')
    if request.ring_budget_gb <= 0:
        errors.append('ring_budget_gb must be positive')
    if abs(request.seconds - MAGI2_CLIP_SECONDS) > 1e-06:
        errors.append(f'MAGI-2-preview only supports {MAGI2_CLIP_SECONDS:g}s clips; got {request.seconds:g}s')
    if request.steps is not None and request.steps < 1:
        errors.append('steps must be at least 1')
    if request.refiner_steps is not None and request.refiner_steps < 1:
        errors.append('refiner steps must be at least 1')
    if request.seed < 0:
        errors.append('seed must be non-negative')
    if request.world_size < 1:
        errors.append('world size (GPU count) must be at least 1')
    for name, value in (('text_enc_offload', request.text_enc_offload), ('preview_offload', request.preview_offload), ('refiner_offload', request.refiner_offload), ('vae_offload', request.vae_offload)):
        if value is not None and value not in MAGI2_OFFLOAD_MODES:
            errors.append(f'{name} must be one of {MAGI2_OFFLOAD_MODES}; got {value!r}')
    for dim_name, dim in (('output_width', request.output_width), ('output_height', request.output_height), ('preview_width', request.preview_width), ('preview_height', request.preview_height), ('refiner_width', request.refiner_width), ('refiner_height', request.refiner_height)):
        if dim is not None and dim <= 0:
            errors.append(f'{dim_name} must be positive')
    if not request.prompt.strip() and request.prompt_file is None:
        errors.append('prompt must not be empty (pass --prompt or --prompt-file)')
    if check_files:
        if not request.model_dir.exists():
            errors.append(f'checkpoint root does not exist: {request.model_dir}')
        if request.image is not None and (not request.image.exists()):
            errors.append(f'image does not exist: {request.image}')
        if request.prompt_file is not None and (not request.prompt_file.exists()):
            errors.append(f'prompt_file does not exist: {request.prompt_file}')
        if request.repo_root is not None and (not request.repo_root.exists()):
            errors.append(f'MAGI-2 repo root does not exist: {request.repo_root}')
    return errors

@dataclass(frozen=True)
class Magi2CheckpointLayout:
    root: str
    present_components: tuple[str, ...] = ()
    missing_components: tuple[str, ...] = ()
    preview_shards_found: int = 0
    refiner_shards_found: int = 0
    preview_index_present: bool = False
    refiner_index_present: bool = False
    complete: bool = False

def _count_shards(component_dir: Path, total: int) -> int:
    if not component_dir.is_dir():
        return 0
    found = 0
    for i in range(1, total + 1):
        shard = component_dir / f'model-{i:05d}-of-{total:05d}.safetensors'
        try:
            if shard.is_file() and shard.stat().st_size > 0:
                found += 1
        except OSError:
            pass
    return found

def _dir_has_payload(component_dir: Path) -> bool:
    if not component_dir.is_dir():
        return False
    for pat in ('*.safetensors', '*.pth', '*.ckpt', '*.bin', '*.pt', '*.json', '*.model'):
        for hit in component_dir.rglob(pat):
            try:
                if hit.is_file() and hit.stat().st_size > 0:
                    return True
            except OSError:
                pass
    return False

def inspect_magi2_checkpoint(model_dir: Path | str) -> Magi2CheckpointLayout:
    root = Path(model_dir)
    present: list[str] = []
    missing: list[str] = []
    for component in MAGI2_COMPONENTS:
        if _dir_has_payload(root / component):
            present.append(component)
        else:
            missing.append(component)
    preview_shards = _count_shards(root / 'preview', MAGI2_PREVIEW_SHARDS)
    refiner_shards = _count_shards(root / 'refiner', MAGI2_REFINER_SHARDS)
    preview_index = (root / 'preview' / f'model.safetensors.index.json').is_file()
    refiner_index = (root / 'refiner' / f'model.safetensors.index.json').is_file()
    complete = not missing and preview_shards == MAGI2_PREVIEW_SHARDS and (refiner_shards == MAGI2_REFINER_SHARDS)
    return Magi2CheckpointLayout(root=str(root), present_components=tuple(present), missing_components=tuple(missing), preview_shards_found=preview_shards, refiner_shards_found=refiner_shards, preview_index_present=preview_index, refiner_index_present=refiner_index, complete=complete)

def resolve_magi2_repo_root(explicit: Optional[Path]=None) -> Optional[Path]:
    candidates: list[Optional[Path]] = [explicit, Path(os.environ['MAGI2_REPO_ROOT']) if os.environ.get('MAGI2_REPO_ROOT') else None, Path(''), Path('')]
    for cand in candidates:
        if cand is None:
            continue
        try:
            if (cand / 'inference' / 'pipeline' / 'entry.py').is_file():
                return cand
        except OSError:
            pass
    return None

def _missing_modules(modules) -> list[str]:
    out: list[str] = []
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            out.append(module)
    return out

def magi2_runtime_blockers(world_size: int=MAGI2_DEFAULT_WORLD_SIZE) -> list[str]:
    blockers: list[str] = []
    for module in _missing_modules(MAGI2_REFERENCE_MODULES):
        blockers.append(f'required runtime module {module!r} is not importable (the reference torchrun path needs SandAI magi_attention + magi_compiler + a Triton MoE kernel on CUDA)')
    if 'torch' not in [m for m in MAGI2_REFERENCE_MODULES if m in _missing_modules(MAGI2_REFERENCE_MODULES)]:
        try:
            import torch
            available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception as exc:
            available = 0
            blockers.append(f'could not query CUDA device count: {exc}')
        else:
            if available < world_size:
                blockers.append(f'the torchrun engine uses context/expert parallelism and expects {world_size} GPUs; this host exposes {available}. Use --engine stream-sdpa for single-GPU streaming, or run on an 8x Hopper node.')
    return blockers

def magi2_stream_sdpa_blockers(model_dir: Path | str, *, min_vram_gb: float=MAGI2_MIN_VRAM_GB, repo_root: Optional[Path]=None) -> list[str]:
    blockers: list[str] = []
    layout = inspect_magi2_checkpoint(model_dir)
    if not layout.complete:
        blockers.append(f'checkpoint incomplete at {model_dir}: missing={list(layout.missing_components)}, preview {layout.preview_shards_found}/{MAGI2_PREVIEW_SHARDS}, refiner {layout.refiner_shards_found}/{MAGI2_REFINER_SHARDS} (download the full ~307GB sand-ai/MAGI-2-preview snapshot to D:)')
    for module in _missing_modules(MAGI2_STREAM_SDPA_MODULES):
        blockers.append(f'required module {module!r} not importable (stream-sdpa needs torch + the reference Triton MoE kernel; magi_attention/magi_compiler are shimmed away)')
    if resolve_magi2_repo_root(repo_root) is None:
        blockers.append('')
    if 'torch' not in _missing_modules(('torch',)):
        try:
            import torch
            if not torch.cuda.is_available():
                blockers.append('CUDA is not available (stream-sdpa needs one CUDA GPU)')
            else:
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / 1000000000.0
                if vram_gb + 0.5 < min_vram_gb:
                    blockers.append(f'GPU has {vram_gb:.1f}GB VRAM but stream-sdpa needs >= {min_vram_gb:.0f}GB (device: {props.name}); lower --ring-budget-gb or free VRAM')
        except Exception as exc:
            blockers.append(f'could not query CUDA device: {exc}')
    return blockers
MAGI2_SPILL_CEILING_GB = float(os.environ.get('MAGI2_SPILL_CEILING_GB', '30.5'))
MAGI2_ACT_GB_PER_TOKEN = float(os.environ.get('MAGI2_ACT_GB_PER_TOKEN', '3.32e-4'))
MAGI2_FIXED_GB = float(os.environ.get('MAGI2_FIXED_GB', '8.35'))
MAGI2_CHUNK_BASELINE = 4096.0

def packed_tokens_for(resolution: str, seconds: float=MAGI2_CLIP_SECONDS, txt_tokens: int=226) -> int:
    preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['540p'])
    pw, ph = (preset['pw'], preset['ph'])
    vae_stride, patch = ((4, 16, 16), (1, 2, 2))
    latent_h = ph // vae_stride[1] // patch[1] * patch[1]
    latent_w = pw // vae_stride[2] // patch[2] * patch[2]
    vll = (int(round(seconds * MAGI2_OUTPUT_FPS)) - 1) // vae_stride[0] + 1
    audio = int(round(seconds * 25.0))
    return vll * latent_h * latent_w + audio + txt_tokens

def project_peak_vram_gb(resolution: str, *, ring_depth: int=1, ring_layer_gb: float=5.7, mlp_chunk_tokens: int=0, seconds: float=MAGI2_CLIP_SECONDS, cfg_mode: str='batched', act_gb_per_token: Optional[float]=None) -> dict:
    tokens = packed_tokens_for(resolution, seconds)
    ring_gb = ring_depth * ring_layer_gb
    act_gb = (MAGI2_ACT_GB_PER_TOKEN if act_gb_per_token is None else act_gb_per_token) * tokens
    if str(cfg_mode).lower() == 'sequential':
        act_gb *= 0.5
    if mlp_chunk_tokens and mlp_chunk_tokens > 0:
        scale = min(1.0, float(mlp_chunk_tokens) / MAGI2_CHUNK_BASELINE)
        act_gb = act_gb * (0.85 + 0.15 * scale)
    projected = ring_gb + MAGI2_FIXED_GB + act_gb
    return {'projected_gb': round(projected, 2), 'tokens': tokens, 'ring_gb': round(ring_gb, 2), 'fixed_gb': MAGI2_FIXED_GB, 'activations_gb': round(act_gb, 2), 'ceiling_gb': MAGI2_SPILL_CEILING_GB, 'would_spill': projected > MAGI2_SPILL_CEILING_GB}

def _ring_depth2_allowed() -> bool:
    return os.environ.get('MAGI2_RING_ALLOW_DEPTH2', '0').strip().lower() in ('1', 'true', 'yes', 'on')

def choose_ring_config(resolution: str, *, mlp_chunk_tokens: int=0, cfg_mode: str='batched', seconds: float=MAGI2_CLIP_SECONDS, act_gb_per_token: Optional[float]=None) -> dict:
    projections = {}
    for depth in (1, 2):
        projections[depth] = project_peak_vram_gb(resolution, ring_depth=depth, mlp_chunk_tokens=mlp_chunk_tokens, seconds=seconds, cfg_mode=cfg_mode, act_gb_per_token=act_gb_per_token)
    if not projections[2]['would_spill'] and _ring_depth2_allowed():
        return {'ring_depth': 2, 'overlap': True, 'projections': projections, 'reason': f"depth-2 projects {projections[2]['projected_gb']:.1f}GB <= {MAGI2_SPILL_CEILING_GB:.1f}GB ceiling -> prefetch overlap ON (reclaims the streaming-overlap speedup)"}
    if not projections[2]['would_spill'] and (not _ring_depth2_allowed()) and (not projections[1]['would_spill']):
        return {'ring_depth': 1, 'overlap': False, 'projections': projections, 'reason': f"depth-2 fits ({projections[2]['projected_gb']:.1f}GB) but is DISABLED pending the stream-keyed protect gate (delivery race convicted 2026-08-06; MAGI2_RING_ALLOW_DEPTH2=1 to test) -> depth-1 at {projections[1]['projected_gb']:.1f}GB, overlap OFF"}
    if not projections[1]['would_spill']:
        return {'ring_depth': 1, 'overlap': False, 'projections': projections, 'reason': f"depth-2 would spill ({projections[2]['projected_gb']:.1f}GB) -> depth-1 at {projections[1]['projected_gb']:.1f}GB, overlap OFF"}
    return {'ring_depth': 1, 'overlap': False, 'projections': projections, 'reason': f"BOTH depths projected to spill (d1 {projections[1]['projected_gb']:.1f}GB, d2 {projections[2]['projected_gb']:.1f}GB) -> the spill gate will refuse this run; cut activations (cfg-mode sequential / smaller chunk / fused attention)"}
MAGI2_CONFIG_MATRIX = (('batched', 2, 1.0), ('batched', 1, 1.15), ('sequential', 2, 2.0), ('sequential', 1, 2.3))

def choose_run_config(resolution: str, *, mlp_chunk_tokens: int=0, seconds: float=MAGI2_CLIP_SECONDS, act_gb_per_token_by_cfg: Optional[dict]=None, act_gb_per_token: Optional[float]=None) -> dict:
    allow_d2 = _ring_depth2_allowed()
    matrix = []
    for cfg_mode, depth, cost in MAGI2_CONFIG_MATRIX:
        coeff = None
        if act_gb_per_token_by_cfg:
            coeff = act_gb_per_token_by_cfg.get(cfg_mode)
        if coeff is None:
            coeff = act_gb_per_token
        proj = project_peak_vram_gb(resolution, ring_depth=depth, mlp_chunk_tokens=mlp_chunk_tokens, seconds=seconds, cfg_mode=cfg_mode, act_gb_per_token=coeff)
        eligible = allow_d2 or depth == 1
        matrix.append({'cfg_mode': cfg_mode, 'ring_depth': depth, 'rel_cost': cost, 'projected_gb': proj['projected_gb'], 'fits': not proj['would_spill'], 'eligible': eligible, **({} if eligible else {'why_ineligible': 'depth-2 DISABLED pending stream-keyed protect gate (delivery race convicted 2026-08-06); MAGI2_RING_ALLOW_DEPTH2=1 to test'})})
    for cand in matrix:
        if cand['fits'] and cand['eligible']:
            return {'cfg_mode': cand['cfg_mode'], 'ring_depth': cand['ring_depth'], 'overlap': cand['ring_depth'] > 1, 'matrix': matrix, 'reason': f"fastest config that clears {MAGI2_SPILL_CEILING_GB:.1f}GB: {cand['cfg_mode']} CFG + depth-{cand['ring_depth']} at {cand['projected_gb']:.1f}GB (rel cost {cand['rel_cost']:.2f}x)"}
    return {'cfg_mode': 'sequential', 'ring_depth': 1, 'overlap': False, 'matrix': matrix, 'reason': 'NO config clears the ceiling — the spill gate will refuse this run. Cut activations further (smaller chunk / fused attention / lower resolution).'}

def spill_gate_blockers(request: Magi2GenerationRequest) -> list[str]:
    chunk = int(os.environ.get('MAGI2_MLP_CHUNK_TOKENS', '1024' if request.resolution in ('540p', '1080p') else '0'))
    depth = int(os.environ.get('MAGI2_RING_DEPTH', '1'))
    proj = project_peak_vram_gb(request.resolution, ring_depth=depth, mlp_chunk_tokens=chunk, seconds=request.seconds)
    if not proj['would_spill']:
        return []
    return [f"PROJECTED PEAK {proj['projected_gb']:.1f}GB EXCEEDS the {proj['ceiling_gb']:.1f}GB dedicated-VRAM ceiling -> this run WOULD SPILL into WDDM shared memory (crawls at ~0% util and starves the machine). Breakdown: ring {proj['ring_gb']:.1f}GB (depth {depth}) + fixed {proj['fixed_gb']:.1f}GB + activations {proj['activations_gb']:.1f}GB over {proj['tokens']} packed tokens. TURN A KNOB: lower MAGI2_MLP_CHUNK_TOKENS (currently {chunk}; try {max(512, chunk // 2)}), keep MAGI2_RING_DEPTH=1, drop --resolution, or raise MAGI2_SPILL_CEILING_GB only if you accept spilling."]

def build_magi2_torchrun_argv(request: Magi2GenerationRequest, *, repo_root: Path, python: str=sys.executable) -> list[str]:
    preset = RESOLUTION_PRESETS.get(request.resolution, RESOLUTION_PRESETS['1080p'])
    entry = str(repo_root / 'inference' / 'pipeline' / 'entry.py')
    argv: list[str] = [python, '-m', 'torch.distributed.run', '--nnodes=1', f'--nproc_per_node={request.world_size}', '--rdzv-backend=c10d', f'--rdzv-endpoint=localhost:{request.master_port}', entry, '--resolution', request.resolution, '--seconds', f'{request.seconds:g}', '--seed', str(request.seed), '--prompt', request.prompt, '--output', str(request.out.parent)]
    if request.mode == 'i2v' and request.image is not None:
        argv += ['--image', str(request.image)]
    if request.steps is not None:
        argv += ['--num-inference-steps', str(request.steps)]
    if request.refiner_steps is not None:
        argv += ['--refiner-num-inference-steps', str(request.refiner_steps)]
    if request.preview_width is not None:
        argv += ['--preview-width', str(request.preview_width)]
    if request.preview_height is not None:
        argv += ['--preview-height', str(request.preview_height)]
    if request.refiner_width is not None:
        argv += ['--refiner-width', str(request.refiner_width)]
    if request.refiner_height is not None:
        argv += ['--refiner-height', str(request.refiner_height)]
    if request.output_width is not None:
        argv += ['--output-width', str(request.output_width)]
    if request.output_height is not None:
        argv += ['--output-height', str(request.output_height)]
    if request.deterministic:
        argv += ['--deterministic']
    _ = preset
    return argv

def build_magi2_env(request: Magi2GenerationRequest, *, repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env['MAGI2_CKPT_ROOT'] = str(request.model_dir)
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = os.pathsep.join([str(repo_root), existing_pp]) if existing_pp else str(repo_root)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if request.deterministic:
        env['MAGI2_DETERMINISTIC'] = '1'
    if request.negative_prompt is not None:
        env['NEGATIVE_PROMPT'] = request.negative_prompt
    for var, value in (('MAGI2_TEXT_ENC_OFFLOAD_MODE', request.text_enc_offload), ('MAGI2_PREVIEW_OFFLOAD_MODE', request.preview_offload), ('MAGI2_REFINER_OFFLOAD_MODE', request.refiner_offload), ('MAGI2_VAE_OFFLOAD_MODE', request.vae_offload)):
        if value is not None:
            env[var] = value
    return env

def resolve_generation_params(request: Magi2GenerationRequest) -> dict:
    defaults = mode_config_defaults(request.mode, request.resolution)
    params = dict(defaults)
    if request.steps is not None:
        params['steps'] = request.steps
    if request.refiner_steps is not None and params.get('uses_refiner'):
        params['refiner_steps'] = request.refiner_steps
    if request.preview_width is not None:
        params['preview_width'] = request.preview_width
    if request.preview_height is not None:
        params['preview_height'] = request.preview_height
    if request.refiner_width is not None:
        params['refiner_width'] = request.refiner_width
    if request.refiner_height is not None:
        params['refiner_height'] = request.refiner_height
    params['output_width'] = request.output_width
    params['output_height'] = request.output_height
    params['negative_prompt'] = request.negative_prompt if request.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
    params['batch_size'] = request.batch_size
    params['expert_dtype'] = request.expert_dtype
    params['compute_dtype'] = request.compute_dtype
    params['enable_refiner'] = request.enable_refiner
    params['ring_budget_gb'] = request.ring_budget_gb
    params['seed'] = request.seed
    params['engine'] = request.engine
    return params

def _emit_json_event(enabled: bool, event: str, **payload) -> None:
    if enabled:
        print(json.dumps({'event': event, 'time': time.time(), **payload}), flush=True)
_TQDM_STEP = re.compile('(?P<step>\\d+)\\s*/\\s*(?P<total>\\d+)\\s*\\[.*?(?:(?P<sit>[\\d.]+)it/s|(?P<spi>[\\d.]+)s/it)')
_MAGI2_SAMPLE = re.compile('\\[magi2\\]\\s+sample\\s+(?P<idx>\\d+)/(?P<total>\\d+)')
_MAGI2_STAGE = re.compile('\\[magi2\\]\\s+(?P<text>.+)')

def _translate_child_line(line: str) -> Optional[dict]:
    s = line.strip()
    if not s:
        return None
    m = _TQDM_STEP.search(s)
    if m:
        payload: dict = {'event': 'denoise_step_complete', 'step': int(m['step']), 'steps': int(m['total'])}
        if m.group('sit'):
            rate = float(m['sit'])
            if rate > 0:
                payload['seconds'] = round(1.0 / rate, 3)
        elif m.group('spi'):
            payload['seconds'] = round(float(m['spi']), 3)
        return payload
    m = _MAGI2_SAMPLE.search(s)
    if m:
        return {'event': 'text_encoding_started', 'message': f"sample {m['idx']}/{m['total']}"}
    m = _MAGI2_STAGE.search(s)
    if m:
        return {'event': 'stage_log', 'message': m['text'].strip()}
    return None

def generate_magi2(request: Magi2GenerationRequest, *, json_events: bool=False, python: Optional[str]=None) -> dict:
    errors = validate_magi2_generation_request(request)
    if errors:
        raise ValueError('; '.join(errors))
    if request.engine == 'torchrun':
        return _generate_via_torchrun(request, json_events=json_events, python=python)
    return _generate_via_stream_sdpa(request, json_events=json_events)

def _generate_via_stream_sdpa(request: Magi2GenerationRequest, *, json_events: bool) -> dict:
    from cosmos_windows.weight_streaming.magi2_runtime import Magi2EventEmitter
    emit = Magi2EventEmitter(enabled=json_events, debug=request.debug_events)
    params = resolve_generation_params(request)
    emit.emit('request_validated', mode=request.mode, engine='stream-sdpa', output=str(request.out), resolution=request.resolution, seconds=request.seconds, steps=params['steps'], refiner_steps=params.get('refiner_steps'), batch_size=request.batch_size, expert_dtype=request.expert_dtype)
    layout = inspect_magi2_checkpoint(request.model_dir)
    emit.emit('checkpoint_inspected', **asdict(layout))
    blockers = magi2_stream_sdpa_blockers(request.model_dir, min_vram_gb=request.min_vram_gb, repo_root=request.repo_root)
    _chunk = int(os.environ.get('MAGI2_MLP_CHUNK_TOKENS', '1024' if request.resolution in ('540p', '1080p') else '0'))
    _proj = project_peak_vram_gb(request.resolution, ring_depth=int(os.environ.get('MAGI2_RING_DEPTH', '1')), mlp_chunk_tokens=_chunk, seconds=request.seconds)
    emit.emit('vram_projection', **_proj, mlp_chunk_tokens=_chunk)
    blockers += spill_gate_blockers(request)
    if blockers:
        raise RuntimeError('MAGI-2 stream-sdpa preflight failed: ' + ' | '.join(blockers))
    from cosmos_windows.weight_streaming.magi2_stream_engine import run_real_stream_sdpa
    return run_real_stream_sdpa(request, params, emit)

def _generate_via_torchrun(request: Magi2GenerationRequest, *, json_events: bool, python: Optional[str]) -> dict:
    python = python or sys.executable
    repo_root = resolve_magi2_repo_root(request.repo_root)
    if repo_root is None:
        raise RuntimeError('')
    _emit_json_event(json_events, 'request_validated', mode=request.mode, engine='torchrun', output=str(request.out), resolution=request.resolution, seconds=request.seconds, world_size=request.world_size)
    layout = inspect_magi2_checkpoint(request.model_dir)
    _emit_json_event(json_events, 'checkpoint_inspected', **asdict(layout))
    if not layout.complete:
        raise RuntimeError(f'MAGI-2 checkpoint incomplete at {request.model_dir}: missing components={list(layout.missing_components)}, preview shards {layout.preview_shards_found}/{MAGI2_PREVIEW_SHARDS}, refiner shards {layout.refiner_shards_found}/{MAGI2_REFINER_SHARDS}. Download the full ~307GB sand-ai/MAGI-2-preview snapshot first.')
    blockers = magi2_runtime_blockers(request.world_size)
    if blockers:
        raise RuntimeError('MAGI-2 torchrun preflight failed: ' + ' | '.join(blockers))
    argv = build_magi2_torchrun_argv(request, repo_root=repo_root, python=python)
    env = build_magi2_env(request, repo_root=repo_root)
    request.out.parent.mkdir(parents=True, exist_ok=True)
    _emit_json_event(json_events, 'torchrun_launched', argv=argv, nproc_per_node=request.world_size)
    _emit_json_event(json_events, 'text_encoding_started', device='cuda')
    _emit_json_event(json_events, 'denoise_started', device='cuda', stage='preview')
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    tail: list[str] = []
    for raw in proc.stdout:
        line = raw.rstrip('\n')
        tail.append(line)
        if len(tail) > 200:
            del tail[0]
        if json_events:
            translated = _translate_child_line(line)
            if translated is not None:
                _emit_json_event(True, translated.pop('event'), **translated)
        else:
            print(line, flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'MAGI-2 torchrun exited {rc}. Last lines:\n' + '\n'.join(tail[-25:]))
    produced = request.out.parent / 'sample_000.mp4'
    if produced.is_file() and produced.resolve() != request.out.resolve():
        shutil.move(str(produced), str(request.out))
    if not request.out.is_file() or request.out.stat().st_size == 0:
        raise RuntimeError(f'MAGI-2 run finished but produced no non-empty output at {request.out}')
    _emit_json_event(json_events, 'vae_mux_completed', output=str(request.out))
    _emit_json_event(json_events, 'completed', output=str(request.out), bytes=request.out.stat().st_size, seconds=round(time.time() - t0, 2))
    return {'output': str(request.out), 'seconds': round(time.time() - t0, 2), 'mode': request.mode}

def _generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate a 10s audio-video clip with SandAI MAGI-2-preview')
    parser.add_argument('--model-dir', required=True, type=Path, help='Checkpoint root (holds preview/, refiner/, ...)')
    parser.add_argument('--prompt', default='', help='Inline prompt (or use --prompt-file)')
    parser.add_argument('--prompt-file', type=Path, default=None, help='Read the prompt (raw text or enhanced JSON caption) from a file')
    parser.add_argument('--mode', required=True, choices=MAGI2_MODES)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--image', type=Path, help='First-frame still for i2v')
    parser.add_argument('--resolution', choices=list(RESOLUTION_PRESETS), default='1080p')
    parser.add_argument('--seconds', type=float, default=MAGI2_CLIP_SECONDS)
    parser.add_argument('--engine', choices=MAGI2_ENGINES, default='stream-sdpa', help='stream-sdpa = single-GPU streaming (default); torchrun = reference 8xGPU')
    parser.add_argument('--steps', type=int, default=None, help='Preview num-inference-steps (default: config 100)')
    parser.add_argument('--refiner-steps', type=int, default=None, help='Refiner num-inference-steps (default: config 5)')
    parser.add_argument('--refiner', dest='enable_refiner', default=None, action=argparse.BooleanOptionalAction, help='force the 1080p refiner on/off (--refiner / --no-refiner); default: auto (on at 1080p)')
    parser.add_argument('--negative-prompt', default=None, help='Override the built-in negative prompt')
    parser.add_argument('--batch-size', type=int, default=1, help='Prompts per denoise pass (amortizes the weight stream)')
    parser.add_argument('--expert-dtype', choices=MAGI2_EXPERT_DTYPES, default='bf16', help='bf16 straight-from-disk (default); fp8-on-disk is a later lever')
    parser.add_argument('--compute-dtype', choices=('bf16', 'fp8', 'fp4'), default='bf16', help='MoE expert-GEMM compute precision (weights stay bf16 in VRAM): bf16 lossless (default) | fp8 speed/quality dial | fp4 (falls back to fp8)')
    parser.add_argument('--ring-budget-gb', type=float, default=14.0, help='VRAM reserved for the streamed layer ring')
    parser.add_argument('--min-vram-gb', type=float, default=MAGI2_MIN_VRAM_GB, help='Fail the preflight below this VRAM')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--world-size', '--gpus', dest='world_size', type=int, default=MAGI2_DEFAULT_WORLD_SIZE, help='torchrun nproc_per_node (reference configs expect 8; torchrun engine only)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output-width', type=int, default=None)
    parser.add_argument('--output-height', type=int, default=None)
    parser.add_argument('--preview-width', type=int, default=None)
    parser.add_argument('--preview-height', type=int, default=None)
    parser.add_argument('--refiner-width', type=int, default=None)
    parser.add_argument('--refiner-height', type=int, default=None)
    parser.add_argument('--repo-root', type=Path, default=None, help='')
    parser.add_argument('--text-enc-offload', choices=MAGI2_OFFLOAD_MODES, default=None)
    parser.add_argument('--preview-offload', choices=MAGI2_OFFLOAD_MODES, default=None)
    parser.add_argument('--refiner-offload', choices=MAGI2_OFFLOAD_MODES, default=None)
    parser.add_argument('--vae-offload', choices=MAGI2_OFFLOAD_MODES, default=None)
    parser.add_argument('--debug-events', action='store_true', help='Emit per-layer stream timing events')
    parser.add_argument('--json-events', action='store_true')
    return parser

def _request_from_args(args) -> Magi2GenerationRequest:
    return Magi2GenerationRequest(model_dir=args.model_dir, prompt=args.prompt, mode=args.mode, out=args.out, image=args.image, resolution=args.resolution, seconds=args.seconds, steps=args.steps, refiner_steps=args.refiner_steps, seed=args.seed, world_size=args.world_size, deterministic=args.deterministic, output_width=args.output_width, output_height=args.output_height, preview_width=args.preview_width, preview_height=args.preview_height, refiner_width=args.refiner_width, refiner_height=args.refiner_height, repo_root=args.repo_root, text_enc_offload=args.text_enc_offload, preview_offload=args.preview_offload, refiner_offload=args.refiner_offload, vae_offload=args.vae_offload, engine=args.engine, prompt_file=args.prompt_file, negative_prompt=args.negative_prompt, batch_size=args.batch_size, expert_dtype=args.expert_dtype, compute_dtype=args.compute_dtype, ring_budget_gb=args.ring_budget_gb, enable_refiner=args.enable_refiner, min_vram_gb=args.min_vram_gb, debug_events=args.debug_events)

def _generate_main(argv: list[str]) -> int:
    args = _generation_parser().parse_args(argv)
    request = _request_from_args(args)
    if not request.prompt.strip() and args.prompt_file is not None:
        try:
            resolved = load_prompt_text(None, args.prompt_file)
            request = _replace_prompt(request, resolved)
        except Exception as exc:
            _emit_json_event(args.json_events, 'failed', error_type=type(exc).__name__, message=str(exc))
            if not args.json_events:
                print(f'MAGI-2 generation failed: {exc}', file=sys.stderr)
            return 2
    try:
        generate_magi2(request, json_events=args.json_events)
    except Exception as exc:
        payload = {'error_type': type(exc).__name__, 'message': str(exc)}
        _emit_json_event(args.json_events, 'failed', **payload)
        if not args.json_events:
            print(f'MAGI-2 generation failed: {exc}', file=sys.stderr)
        return 2
    return 0

def _replace_prompt(request: Magi2GenerationRequest, prompt: str) -> Magi2GenerationRequest:
    from dataclasses import replace
    return replace(request, prompt=prompt)

def _main(argv: Optional[list[str]]=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'generate':
        return _generate_main(argv[1:])
    if argv and argv[0] == 'enhance-presets':
        print(json.dumps({k: asdict(v) for k, v in magi2_enhance_presets().items()}, indent=2))
        return 0
    parser = argparse.ArgumentParser(description='Inspect a MAGI-2 checkpoint for Studio')
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--engine', choices=MAGI2_ENGINES, default='stream-sdpa')
    parser.add_argument('--world-size', '--gpus', dest='world_size', type=int, default=MAGI2_DEFAULT_WORLD_SIZE)
    args = parser.parse_args(argv)
    layout = inspect_magi2_checkpoint(args.model_dir)
    if args.engine == 'torchrun':
        blockers = magi2_runtime_blockers(args.world_size)
    else:
        blockers = magi2_stream_sdpa_blockers(args.model_dir)
    result: dict = {'layout': asdict(layout), 'engine': args.engine, 'runtime_blockers': blockers}
    print(json.dumps(result, indent=2))
    return 0 if layout.complete else 2
if __name__ == '__main__':
    raise SystemExit(_main())
