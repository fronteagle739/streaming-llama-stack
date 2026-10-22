# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/regen3d_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional
STRUCTURE_LABELS = ('floor', 'wall', 'ceiling', 'door', 'window', 'room')
MANIFEST_VERSION = 1
STAGE_ORDER: tuple[tuple[str, str], ...] = (('segment', 'SAM3 segmentation'), ('inpaint', 'occlusion inpaint (Boogu-Edit two-panel A-Q)'), ('empty_room', 'empty-room plate'), ('camera', 'VGGT camera + point clouds'), ('objects_3d', 'object image -> 3D meshes'), ('background', 'background mesh (Poisson of P_bg)'), ('placement', 'coarse ground-plane placement'), ('manifest', 'scene manifest'))
AQ_PROMPT = 'Here is the UI of an application.\nWe want an amodal render of the single object "{object}" that needs to be extracted,\nreplacing the "Extracted Object" panel on the right, with the completed amodal object on a white background.\nKeep the rest of the image the same.'
EMPTY_ROOM_PROMPT = 'Remove ALL objects and furniture.\nI want a single empty room.\nNo chairs, tables, lamps, dresser, kitchen parts etc.\nJust give me back the same room but EMPTY.\nSame light, same perspective, same walls, floor and ceiling.'

@dataclass
class ObjectSpec:
    label: str
    index: int
    score: float = 0.0
    box: Optional[list] = None
    ground_contact: bool = False

    @property
    def name(self) -> str:
        return f'{_safe(self.label)}__{self.index}'

@dataclass
class ReconstructPlan:
    source_images: list
    object_prompts: list
    stages: list
    skipped: list = field(default_factory=list)
    editor: str = 'boogu-edit'
    editor_arch: Optional[str] = None
    editor_model: Optional[str] = None
    mesh_generator: str = 'trellis'
    texture: str = 'baked'
    objects: list = field(default_factory=list)

    def stage_keys(self) -> list:
        return [k for k, _ in self.stages]

    def runs(self, key: str) -> bool:
        return key in self.stage_keys()

def plan_stages(*, source_images, object_prompts=None, editor='boogu-edit', editor_arch=None, editor_model=None, mesh_generator='trellis', texture='baked', skip_stages=None) -> ReconstructPlan:
    imgs = [str(p) for p in source_images or [] if str(p).strip()]
    if not imgs:
        raise ValueError('reconstruct-scene needs at least one source image')
    prompts = [str(p).strip() for p in object_prompts or [] if str(p).strip()]
    skip = set(skip_stages or [])
    skip.discard('manifest')
    stages = [(k, lbl) for k, lbl in STAGE_ORDER if k not in skip]
    skipped = [k for k, _ in STAGE_ORDER if k in skip]
    if editor not in ('boogu-edit', 'pix2gestalt'):
        raise ValueError(f'unknown editor {editor!r} (boogu-edit | pix2gestalt)')
    if mesh_generator not in ('trellis', 'hunyuan3d'):
        raise ValueError(f'unknown mesh_generator {mesh_generator!r} (trellis | hunyuan3d)')
    if texture not in ('baked', 'pbr'):
        raise ValueError(f'unknown texture {texture!r} (baked | pbr)')
    e_arch = str(editor_arch).strip() if editor_arch and str(editor_arch).strip() else None
    e_model = str(editor_model).strip() if editor_model and str(editor_model).strip() else None
    return ReconstructPlan(source_images=imgs, object_prompts=prompts, stages=stages, skipped=skipped, editor=editor, editor_arch=e_arch, editor_model=e_model, mesh_generator=mesh_generator, texture=texture)

def is_structure_label(label: str) -> bool:
    lab = (label or '').strip().lower()
    return any((s in lab for s in STRUCTURE_LABELS))

def split_objects_and_structure(detections: list) -> tuple[list, list]:
    objects: list = []
    structure: list = []
    per_label: dict = {}
    for det in detections or []:
        label = str(det.get('label', '')).strip() or 'object'
        if is_structure_label(label):
            structure.append(dict(det))
            continue
        idx = per_label.get(label, 0)
        per_label[label] = idx + 1
        objects.append(ObjectSpec(label=label, index=idx, score=float(det.get('score', 0.0) or 0.0), box=det.get('box'), ground_contact=bool(det.get('ground_contact', False))))
    return (objects, structure)

def build_scene_manifest(plan: 'ReconstructPlan', *, job_id: str, out_dir: str, layers: list, artifacts: dict, camera: Optional[dict]=None, warnings: Optional[list]=None) -> dict:
    scene = {'version': MANIFEST_VERSION, 'kind': 'regen3d_scene', 'job_id': job_id, 'created': time.time(), 'out_dir': str(out_dir), 'source_images': list(plan.source_images), 'object_prompts': list(plan.object_prompts), 'editor': plan.editor, 'editor_arch': plan.editor_arch, 'editor_model': plan.editor_model, 'mesh_generator': plan.mesh_generator, 'texture': plan.texture, 'stages_run': plan.stage_keys(), 'stages_skipped': list(plan.skipped), 'scale_is_relative': True, 'camera': camera or None, 'layers': list(layers), 'artifacts': dict(artifacts or {}), 'warnings': list(warnings or [])}
    return scene

def scene_layers(manifest: dict) -> list:
    out = []
    for lay in manifest.get('layers', []):
        ref = lay.get('ply') or lay.get('glb')
        if not ref:
            continue
        entry = {'position': lay.get('position') or [0, 0, 0], 'rotation_deg': lay.get('rotation_deg') or [0, 0, 0], 'scale': float(lay.get('scale') or 1.0)}
        if lay.get('ply'):
            entry['ply'] = lay['ply']
        else:
            entry['glb'] = lay['glb']
        out.append(entry)
    return out

def load_manifest(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _norm_vec3(v, default) -> list:
    if v is None:
        return list(default)
    try:
        out = [float(x) for x in list(v)[:3]]
    except (TypeError, ValueError):
        return list(default)
    while len(out) < 3:
        out.append(0.0)
    return out

def make_scene_layer(*, asset: str, name: Optional[str]=None, label: Optional[str]=None, position=None, rotation_deg=None, scale: float=1.0, ground_contact: bool=False, source: str='added') -> dict:
    ref = str(asset or '').strip()
    if not ref:
        raise ValueError('a scene layer needs an asset path (.ply splat or .glb/.gltf/.obj mesh)')
    ext = Path(ref).suffix.lower()
    if ext == '.ply':
        key = 'ply'
    elif ext in ('.glb', '.gltf', '.obj'):
        key = 'glb'
    else:
        raise ValueError(f'unsupported scene asset {ref!r} (need .ply | .glb | .gltf | .obj)')
    nm = _safe(name) if name else Path(ref).stem
    lay = {'name': nm, 'label': str(label) if label else nm, key: ref, 'position': _norm_vec3(position, [0.0, 0.0, 0.0]), 'rotation_deg': _norm_vec3(rotation_deg, [0.0, 0.0, 0.0]), 'scale': float(scale) if scale is not None else 1.0, 'ground_contact': bool(ground_contact), 'source': str(source or 'added')}
    return lay

def _unique_layer_name(manifest: dict, name: str) -> str:
    existing = {str(l.get('name')) for l in manifest.get('layers', [])}
    if name not in existing:
        return name
    for i in range(2, 100000):
        cand = f'{name}_{i}'
        if cand not in existing:
            return cand
    return f'{name}_{int(time.time())}'

def add_manifest_layer(manifest: dict, layer: dict) -> dict:
    lay = dict(layer)
    lay['name'] = _unique_layer_name(manifest, str(lay.get('name') or 'layer'))
    manifest.setdefault('layers', []).append(lay)
    return manifest

def remove_manifest_layer(manifest: dict, name: str) -> bool:
    layers = manifest.get('layers', [])
    for i, l in enumerate(layers):
        if str(l.get('name')) == str(name):
            del layers[i]
            return True
    return False

def ground_plane_from_points(floor_points, up_axis: int=1):
    import numpy as np
    pts = np.asarray(floor_points, dtype=np.float64) if floor_points is not None else None
    if pts is None or pts.size == 0:
        return (0.0, int(up_axis))
    pts = pts.reshape(-1, 3)
    return (float(np.median(pts[:, up_axis])), int(up_axis))

def coarse_place_object(object_points, *, ground_height: float, up_axis: int=1, ground_contact: bool=True):
    import numpy as np
    pos = [0.0, 0.0, 0.0]
    pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3) if object_points is not None else None
    if ground_contact and pts is not None and pts.size:
        min_up = float(pts[:, up_axis].min())
        pos[up_axis] = float(ground_height - min_up)
    return {'position': pos, 'rotation_deg': [0.0, 0.0, 0.0], 'scale': 1.0}

def build_aq_composite(original_image, extracted_cutout, *, target_width: int=1280, text_label: str='Extracted Object'):
    from PIL import Image, ImageDraw, ImageFont
    orig = _as_rgba(original_image)
    obj = _as_rgba(extracted_cutout)
    aspect = orig.height / max(1, orig.width)
    base_w = int(target_width)
    base_h = int(base_w * aspect)
    orig = orig.resize((base_w, base_h), Image.Resampling.LANCZOS)
    panel_w = base_h
    total_w = base_w + panel_w
    canvas = Image.new('RGBA', (total_w, base_h), (255, 255, 255, 255))
    canvas.paste(orig, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([base_w, 0, total_w, base_h], fill=(230, 230, 230))
    bottom_space = int(base_h * 0.08)
    margin = int(panel_w * 0.04)
    top_margin = int(base_h * 0.02)
    square = min(panel_w - margin * 2, base_h - bottom_space - top_margin)
    panel_cx = base_w + panel_w // 2
    card_x1 = panel_cx - square // 2
    card_y1 = (base_h - bottom_space - square) // 2
    draw.rounded_rectangle((card_x1, card_y1, card_x1 + square, card_y1 + square), radius=20, fill=(255, 255, 255), outline=(0, 0, 0), width=5)
    padding = 40
    target = square - padding * 2
    ratio = obj.width / max(1, obj.height)
    if ratio > 1:
        nw, nh = (target, int(target / ratio))
    else:
        nh, nw = (target, int(target * ratio))
    obj_r = obj.resize((max(1, nw), max(1, nh)), Image.Resampling.LANCZOS)
    px = card_x1 + (square - nw) // 2
    py = card_y1 + (square - nh) // 2
    canvas.paste(obj_r, (px, py), obj_r)
    try:
        font = ImageFont.truetype(_first_font(), max(int(base_h * 0.035), 32))
    except Exception:
        font = ImageFont.load_default()
    l, t, r, b = draw.textbbox((0, 0), text_label, font=font)
    tx = base_w + (panel_w - (r - l)) // 2
    ty = card_y1 + square + (base_h - (card_y1 + square) - (b - t)) // 2
    draw.text((tx, ty), text_label, fill=(0, 0, 0), font=font)
    return canvas

def extract_aq_result(result_image, *, target_width: int=1280):
    from PIL import Image
    img = result_image if hasattr(result_image, 'crop') else Image.open(result_image)
    base_w = int(target_width)
    base_h = img.height
    panel_w = base_h
    bottom_space = int(base_h * 0.08)
    margin = int(panel_w * 0.04)
    top_margin = int(base_h * 0.02)
    square = min(panel_w - margin * 2, base_h - bottom_space - top_margin)
    panel_cx = base_w + panel_w // 2
    card_x1 = panel_cx - square // 2
    card_y1 = (base_h - bottom_space - square) // 2
    b = 5
    return img.crop((card_x1 + b, card_y1 + b, card_x1 + square - b, card_y1 + square - b))

@dataclass
class StageFns:
    segment: Callable[..., list]
    inpaint: Callable[..., Any]
    empty_room: Callable[..., str]
    camera: Callable[..., dict]
    mesh: Callable[..., str]
    background: Callable[..., Optional[str]]
    unload: Callable[..., None] = lambda *a, **k: None

def _noop_emit(_ev: dict) -> None:
    pass

def run_reconstruct_scene(plan: 'ReconstructPlan', fns: 'StageFns', *, out_dir: str, job_id: str, emit: Optional[Callable[[dict], None]]=None, should_cancel: Optional[Callable[[], bool]]=None) -> dict:
    emit = emit or _noop_emit
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'assets').mkdir(exist_ok=True)
    (out / 'objects_clean').mkdir(exist_ok=True)
    (out / 'pointclouds').mkdir(exist_ok=True)
    (out / 'masks').mkdir(exist_ok=True)
    stage_keys = plan.stage_keys()
    n_stages = len(stage_keys)
    warnings: list = []
    artifacts: dict = {'masks': [], 'objects_clean': [], 'assets': [], 'pointclouds': []}
    layers: list = []
    camera: Optional[dict] = None
    object_points: dict = {}

    def _cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    def _stage(i: int, key: str, label: str):
        emit({'type': 'stage', 'stage': key, 'label': label, 'index': i, 'total': n_stages, 'status': 'start'})

    def _done(key: str, **extra):
        ev = {'type': 'stage', 'stage': key, 'status': 'done'}
        ev.update(extra)
        emit(ev)
    primary_image = plan.source_images[0]
    detections: list = []
    structure: list = []
    for i, (key, label) in enumerate(plan.stages):
        if _cancelled():
            raise RuntimeError('reconstruct-scene cancelled')
        _stage(i, key, label)
        if key == 'segment':
            prompts = plan.object_prompts or []
            detections = fns.segment(primary_image, prompts) or []
            plan.objects, structure = split_objects_and_structure(detections)
            for o in plan.objects:
                artifacts['masks'].append(_det_mask_for(detections, o))
            fns.unload()
            _done(key, n_objects=len(plan.objects), n_structure=len(structure))
        elif key == 'inpaint':
            for o in plan.objects:
                if _cancelled():
                    raise RuntimeError('reconstruct-scene cancelled')
                cutout = _det_cutout_for(detections, o) or primary_image
                clean_path = str(out / 'objects_clean' / f'{o.name}.png')
                try:
                    composite = build_aq_composite(primary_image, cutout, text_label='Extracted Object')
                    prompt = AQ_PROMPT.format(object=o.label)
                    result = fns.inpaint(composite, prompt)
                    extract_aq_result(result).convert('RGBA').save(clean_path)
                except Exception as exc:
                    warnings.append(f'inpaint failed for {o.name}: {type(exc).__name__}: {exc}')
                    _save_passthrough(cutout, clean_path)
                artifacts['objects_clean'].append(clean_path)
                o_extra = {'clean': clean_path}
            fns.unload()
            _done(key, n_cleaned=len(artifacts['objects_clean']))
        elif key == 'empty_room':
            er_path = str(out / 'empty_room.png')
            try:
                fns.empty_room(primary_image, EMPTY_ROOM_PROMPT, er_path)
                artifacts['empty_room'] = er_path
            except Exception as exc:
                warnings.append(f'empty-room plate failed: {type(exc).__name__}: {exc}')
            fns.unload()
            _done(key)
        elif key == 'camera':
            cam = fns.camera(plan.source_images, str(out / 'pointclouds')) or {}
            camera = cam.get('camera')
            if cam.get('scene_ply'):
                artifacts['scene_ply'] = cam['scene_ply']
            if cam.get('bg_ply'):
                artifacts['bg_ply'] = cam['bg_ply']
            if cam.get('floor_points') is not None:
                artifacts['_floor_points'] = cam['floor_points']
            object_points = cam.get('object_points') or {}
            artifacts['pointclouds'] = list(cam.get('pointcloud_files') or [])
            fns.unload()
            _done(key, has_camera=camera is not None)
        elif key == 'objects_3d':
            for o in plan.objects:
                if _cancelled():
                    raise RuntimeError('reconstruct-scene cancelled')
                clean = str(out / 'objects_clean' / f'{o.name}.png')
                src = clean if Path(clean).is_file() else primary_image
                try:
                    asset = fns.mesh(src, str(out / 'assets'), o.name, plan.texture)
                    o.asset = asset
                    artifacts['assets'].append(asset)
                except Exception as exc:
                    warnings.append(f'mesh gen failed for {o.name}: {type(exc).__name__}: {exc}')
                    o.asset = None
            fns.unload()
            _done(key, n_assets=len(artifacts['assets']))
        elif key == 'background':
            bg_ply = artifacts.get('bg_ply')
            er = artifacts.get('empty_room')
            if bg_ply:
                try:
                    bg_mesh = fns.background(bg_ply, er, str(out / 'assets' / 'background.glb'))
                    if bg_mesh:
                        artifacts['background_mesh'] = bg_mesh
                except Exception as exc:
                    warnings.append(f'background mesh failed: {type(exc).__name__}: {exc}')
            else:
                warnings.append('no background point cloud (camera stage skipped?) — no background mesh')
            _done(key, has_background=bool(artifacts.get('background_mesh')))
        elif key == 'placement':
            floor_pts = artifacts.get('_floor_points')
            ground_h, up_axis = ground_plane_from_points(floor_pts)
            if artifacts.get('background_mesh'):
                layers.append({'name': 'background', 'label': 'background', 'glb': artifacts['background_mesh'], 'position': [0, 0, 0], 'rotation_deg': [0, 0, 0], 'scale': 1.0, 'ground_contact': False})
            for o in plan.objects:
                asset = getattr(o, 'asset', None)
                if not asset:
                    continue
                xform = coarse_place_object(object_points.get(o.name), ground_height=ground_h, up_axis=up_axis, ground_contact=o.ground_contact)
                lay = {'name': o.name, 'label': o.label, 'score': o.score, 'ground_contact': o.ground_contact}
                lay['glb' if str(asset).lower().endswith('.glb') else 'ply'] = asset
                lay.update(xform)
                layers.append(lay)
            artifacts.pop('_floor_points', None)
            _done(key, ground_height=ground_h, n_layers=len(layers))
        elif key == 'manifest':
            pass
    manifest = build_scene_manifest(plan, job_id=job_id, out_dir=str(out), layers=layers, artifacts=artifacts, camera=camera, warnings=warnings)
    scene_path = out / 'scene.json'
    scene_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest['scene_json'] = str(scene_path)
    emit({'type': 'stage', 'stage': 'manifest', 'status': 'done', 'scene_json': str(scene_path), 'n_layers': len(layers), 'warnings': len(warnings)})
    return manifest
_COMPONENTS = (('sam3', 'STUDIO_SAM3_WEIGHTS', '', None, 'SAM3 segmentation weights'), ('boogu_edit', 'STUDIO_BOOGU_EDIT_DIR', '', 'model_index.json', 'Boogu-Image-0.1-Edit editor'), ('vggt', 'STUDIO_VGGT_DIR', '', '*.safetensors', 'VGGT-1B camera + point cloud (non-commercial weights; personal use)'), ('trellis', 'STUDIO_TRELLIS_DIR', '', 'pipeline.json', 'TRELLIS-image-large image->3D'), ('hunyuan3d_code', 'STUDIO_HUNYUAN3D_REPO', '', ('hy3dshape/hy3dshape/pipelines.py', 'hy3dpaint/textureGenPipeline.py'), 'Hunyuan3D-2.1 code checkout'), ('hunyuan3d_shape', 'STUDIO_HUNYUAN3D_DIR', '', ('hunyuan3d-dit-v2-1/model.fp16.ckpt', 'hunyuan3d-vae-v2-1/model.fp16.ckpt'), 'Hunyuan3D-2.1 shape + VAE weights'), ('hunyuan3d_shape_streaming', 'STUDIO_HUNYUAN3D_DIR', '', ('hunyuan3d-dit-v2-1/model.fp16.safetensors', 'hunyuan3d-vae-v2-1/model.fp16.safetensors'), 'Hunyuan3D-2.1 shape weights converted for direct streaming residency'), ('hunyuan3d_paint', 'STUDIO_HUNYUAN3D_DIR', '', ('hunyuan3d-paintpbr-v2-1/model_index.json', 'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin', 'hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin', 'hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.bin'), 'Hunyuan3D-2.1 PBR paint weights'), ('hunyuan3d_paint_streaming', 'STUDIO_HUNYUAN3D_DIR', '', ('hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors', 'hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.safetensors', 'any:hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.safetensors|hunyuan3d-paintpbr-v2-1/text_encoder/model.safetensors'), 'Hunyuan3D-2.1 PBR paint weights converted for direct streaming residency'), ('hunyuan3d_dino', 'STUDIO_HUNYUAN3D_DINO_DIR', '', ('config.json', 'preprocessor_config.json', 'large:model.safetensors'), 'Hunyuan3D PBR DINOv2 Giant vision model'), ('hunyuan3d_realesrgan', 'STUDIO_HUNYUAN3D_REPO', '', 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth', 'Hunyuan3D PBR RealESRGAN helper'), ('hunyuan3d_inpaint', 'STUDIO_HUNYUAN3D_REPO', '', 'hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor*.pyd', 'Hunyuan3D PBR mesh inpaint extension'), ('hunyuan3d_rasterizer', 'STUDIO_HUNYUAN3D_REPO', '', ('py:custom_rasterizer', 'py:custom_rasterizer_kernel'), 'Hunyuan3D PBR custom rasterizer extension'))

def _component_present(env_key: str, default: str, marker: Optional[str]) -> tuple[bool, str]:
    path = os.environ.get(env_key) or default
    p = Path(path)
    if marker is None:
        return (p.exists(), str(p))
    base = p if p.is_dir() else p.parent
    if isinstance(marker, (tuple, list)):
        ok = True
        for m in marker:
            m = str(m)
            if m.startswith('py:'):
                import importlib.util
                ok = ok and importlib.util.find_spec(m[3:]) is not None
            elif m.startswith('any:'):
                options = [base / part for part in m[len('any:'):].split('|')]
                ok = ok and any((part.exists() for part in options))
            elif m.startswith('large:'):
                f = base / m[len('large:'):]
                try:
                    ok = ok and f.is_file() and (f.stat().st_size > 1000000000)
                except OSError:
                    ok = False
            elif '*' in m:
                ok = ok and base.is_dir() and any(base.glob(str(m)))
            else:
                ok = ok and (base / m).exists()
        return (ok, str(base))
    if str(marker).startswith('py:'):
        import importlib.util
        return (importlib.util.find_spec(str(marker)[3:]) is not None, str(base))
    if '*' in marker:
        return (base.is_dir() and any(base.glob(marker)), str(base))
    return ((base / marker).exists(), str(base))

def preflight_components(plan: Optional['ReconstructPlan']=None) -> dict:
    needs = {'segment': {'sam3'}, 'inpaint': {'boogu_edit'}, 'empty_room': {'boogu_edit'}, 'camera': {'vggt'}}
    required: set = set()
    if plan is None:
        required = {c[0] for c in _COMPONENTS}
    else:
        for key in plan.stage_keys():
            required |= needs.get(key, set())
            if key == 'objects_3d':
                if plan.mesh_generator == 'hunyuan3d':
                    required |= {'hunyuan3d_code', 'hunyuan3d_shape', 'hunyuan3d_shape_streaming'}
                    if plan.texture == 'pbr':
                        required |= {'hunyuan3d_paint', 'hunyuan3d_dino', 'hunyuan3d_realesrgan', 'hunyuan3d_inpaint', 'hunyuan3d_rasterizer', 'hunyuan3d_paint_streaming'}
                else:
                    required.add('trellis')
        if plan.editor_arch and plan.editor_arch.lower() not in ('boogu-edit', 'boogu_edit'):
            required.discard('boogu_edit')
    comps = []
    missing = []
    for name, env_key, default, marker, note in _COMPONENTS:
        present, path = _component_present(env_key, default, marker)
        req = name in required
        comps.append({'name': name, 'present': present, 'path': path, 'note': note, 'required': req})
        if req and (not present):
            missing.append(name)
    return {'ok': not missing, 'components': comps, 'missing': missing}

def default_stage_fns(*, device: str='cuda:0', editor: str='boogu-edit', editor_fn: Optional[Callable[..., Any]]=None, threshold: float=0.4, mesh_generator: str='trellis') -> 'StageFns':

    def segment(image_path, prompts):
        from studio import sam3_backend
        dets = []
        query = list(prompts) + [s for s in ('floor', 'wall', 'ceiling') if s not in prompts]
        floor_masks = []
        for lab in query:
            items = sam3_backend.segment_text(image_path, lab, threshold=threshold, device=device)
            for j, it in enumerate(items):
                d = {'label': lab, 'score': it['score'], 'box': it['box'], 'index': j, '_mask': it['mask']}
                if lab == 'floor':
                    floor_masks.append(it['mask'])
                dets.append(d)
        _tag_ground_contact(dets, floor_masks)
        return dets

    def _boogu_edit(image, prompt, mask=None):
        from studio import boogu_krea
        from PIL import Image as _Image
        ckpt = os.environ.get('STUDIO_BOOGU_EDIT_DIR', '')
        pipe = _resident_editor(ckpt, device)
        img = image if hasattr(image, 'convert') else _Image.open(image)
        return boogu_krea.generate(pipe, 'boogu-edit', prompt=prompt, width=1024, height=1024, reference_images=[img.convert('RGB')], device=device)

    def inpaint(composite_pil, prompt):
        if editor_fn is not None:
            return editor_fn(image=composite_pil, prompt=prompt, mask=None)
        return _boogu_edit(composite_pil, prompt)

    def empty_room(image_path, prompt, out_path):
        from PIL import Image
        src = Image.open(image_path).convert('RGB')
        if editor_fn is not None:
            img = editor_fn(image=src, prompt=prompt, mask=None)
        else:
            img = _boogu_edit(src, prompt)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path

    def camera(image_paths, out_dir):
        from . import vggt_backend
        return vggt_backend.reconstruct(image_paths, out_dir=out_dir, device=device)

    def mesh(object_image_path, out_dir, basename, texture):
        if mesh_generator == 'hunyuan3d':
            import importlib
            hunyuan3d_backend = importlib.import_module('cosmos_windows.weight_streaming.hunyuan3d_backend')
            return hunyuan3d_backend.image_to_glb(object_image_path, out_dir=out_dir, basename=basename, texture=texture, device=device)
        from . import trellis_backend
        return trellis_backend.image_to_glb(object_image_path, out_dir=out_dir, basename=basename, texture=texture, device=device)

    def background(bg_ply, empty_room_path, out_path):
        return poisson_background_mesh(bg_ply, out_path)

    def unload():
        _free_all_gpu()
    return StageFns(segment=segment, inpaint=inpaint, empty_room=empty_room, camera=camera, mesh=mesh, background=background, unload=unload)

def poisson_background_mesh(bg_ply: str, out_path: str, *, depth: int=9, density_quantile: float=0.04) -> Optional[str]:
    try:
        import open3d as o3d
    except Exception:
        return None
    import numpy as np
    pcd = o3d.io.read_point_cloud(str(bg_ply))
    if len(pcd.points) == 0:
        return None
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    densities = np.asarray(densities)
    if density_quantile and densities.size:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, float(density_quantile)))
    mesh.compute_vertex_normals()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
    return str(out_path)
_EDITOR_CACHE: dict = {}

def _resident_editor(ckpt: str, device: str):
    from studio import boogu_krea
    key = (ckpt, device)
    if _EDITOR_CACHE.get('key') != key:
        _free_editor()
        _EDITOR_CACHE['pipe'] = boogu_krea.build_pipeline('boogu-edit', ckpt, device=device, mode='resident')
        _EDITOR_CACHE['key'] = key
    return _EDITOR_CACHE['pipe']

def _free_editor() -> None:
    _EDITOR_CACHE.pop('pipe', None)
    _EDITOR_CACHE.pop('key', None)

def _free_all_gpu() -> None:
    _free_editor()
    try:
        from studio import sam3_backend
        sam3_backend.unload_sam3()
    except Exception:
        pass
    for modname, fn in (('vggt_backend', 'unload'), ('trellis_backend', 'unload'), ('hunyuan3d_backend', 'unload')):
        try:
            import importlib
            mod = importlib.import_module(f'.{modname}', __package__)
            getattr(mod, fn, lambda: None)()
        except Exception:
            pass
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _tag_ground_contact(detections: list, floor_masks: list) -> None:
    if not floor_masks:
        return
    try:
        import numpy as np
        from scipy import ndimage
        floor = np.zeros_like(floor_masks[0], dtype=bool)
        for fm in floor_masks:
            floor |= fm
        floor_dil = ndimage.binary_dilation(floor, iterations=8)
    except Exception:
        return
    for d in detections:
        m = d.get('_mask')
        if m is None or is_structure_label(d.get('label', '')):
            continue
        try:
            import numpy as np
            d['ground_contact'] = bool(np.logical_and(m, floor_dil).any())
        except Exception:
            pass

def _det_mask_for(detections: list, obj: 'ObjectSpec'):
    return f'{obj.name}'

def _det_cutout_for(detections: list, obj: 'ObjectSpec'):
    for d in detections:
        if str(d.get('label')) == obj.label and int(d.get('index', -1)) == obj.index:
            return d.get('_cutout')
    return None

def _save_passthrough(cutout, out_path: str) -> None:
    try:
        from PIL import Image
        img = cutout if hasattr(cutout, 'save') else Image.open(cutout)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.convert('RGBA').save(out_path)
    except Exception:
        pass

def _as_rgba(image):
    from PIL import Image
    if hasattr(image, 'convert'):
        return image.convert('RGBA')
    return Image.open(image).convert('RGBA')

def _first_font() -> str:
    for f in ('', '', '/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        if Path(f).is_file():
            return f
    raise RuntimeError('no font')

def _safe(s: str) -> str:
    return re.sub('[^A-Za-z0-9._-]+', '_', str(s or '')).strip('_') or 'object'
