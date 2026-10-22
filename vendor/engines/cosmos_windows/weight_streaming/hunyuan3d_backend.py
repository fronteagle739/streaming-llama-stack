# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/hunyuan3d_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import contextlib
import gc
import json
import importlib.util
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Iterator, Optional
_LOCK = threading.Lock()
_STATE: dict = {}
_SUBPROCESS_FLAG = 'STUDIO_HUNYUAN3D_IN_SUBPROCESS'
_PBR_SUBPROCESS_FLAG = 'STUDIO_HUNYUAN3D_PBR_IN_SUBPROCESS'

def _fast_hf_cache_env() -> dict[str, str]:
    home = os.environ.get('STUDIO_HF_HOME', '')
    return {'HF_HOME': home, 'HUGGINGFACE_HUB_CACHE': os.environ.get('STUDIO_HF_HUB_CACHE', str(Path(home) / 'hub')), 'HF_MODULES_CACHE': os.environ.get('STUDIO_HF_MODULES_CACHE', str(Path(home) / 'modules')), 'HF_HUB_DISABLE_XET': '1'}

def _repo_dir() -> Path:
    env = os.environ.get('STUDIO_HUNYUAN3D_REPO')
    cands = [env, '', '', os.path.expanduser('~/GitHub/Hunyuan3D-2.1')]
    for raw in cands:
        if not raw:
            continue
        p = Path(raw)
        if (p / 'hy3dshape' / 'hy3dshape' / 'pipelines.py').is_file() and (p / 'hy3dpaint' / 'textureGenPipeline.py').is_file():
            return p
    raise RuntimeError('')

def _weights_dir() -> Path:
    env = os.environ.get('STUDIO_HUNYUAN3D_DIR')
    cands = [env, '', os.path.expanduser('~/models/Hunyuan3D-2.1')]
    for raw in cands:
        if not raw:
            continue
        p = Path(raw)
        if (p / 'hunyuan3d-dit-v2-1' / 'model.fp16.ckpt').is_file() and (p / 'hunyuan3d-vae-v2-1' / 'model.fp16.ckpt').is_file():
            return p
    raise RuntimeError('')

def _resolve_dino_dir() -> Optional[Path]:

    def complete_weight(p: Path) -> bool:
        try:
            return p.is_file() and p.stat().st_size > 1000000000
        except OSError:
            return False
    env = os.environ.get('STUDIO_HUNYUAN3D_DINO_DIR')
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([Path(''), Path('')])
    cache_root = Path(os.environ.get('HUGGINGFACE_HUB_CACHE', ''))
    candidates.append(cache_root / 'models--facebook--dinov2-giant')
    expanded: list[Path] = []
    for p in candidates:
        expanded.append(p)
        snaps = p / 'snapshots'
        if snaps.is_dir():
            try:
                expanded.extend(sorted((q for q in snaps.iterdir() if q.is_dir()), reverse=True))
            except OSError:
                pass
    for p in expanded:
        if (p / 'config.json').is_file() and (p / 'preprocessor_config.json').is_file() and (complete_weight(p / 'model.safetensors') or complete_weight(p / 'pytorch_model.bin')):
            return p
    return None

def _paint_ready(weights: Path, repo: Path) -> tuple[bool, str]:
    missing = []
    for rel in ('hunyuan3d-paintpbr-v2-1/model_index.json', 'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin', 'hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin', 'hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.bin'):
        if not (weights / rel).is_file():
            missing.append(str(weights / rel))
    if not (repo / 'hy3dpaint' / 'ckpt' / 'RealESRGAN_x4plus.pth').is_file():
        missing.append(str(repo / 'hy3dpaint' / 'ckpt' / 'RealESRGAN_x4plus.pth'))
    if _resolve_dino_dir() is None:
        missing.append('')
    return (not missing, '; '.join(missing))

def _rasterizer_ready() -> tuple[bool, str]:
    missing = []
    for mod in ('custom_rasterizer', 'custom_rasterizer_kernel'):
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return (not missing, ', '.join(missing))

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, '').lower() in ('1', 'true', 'on', 'yes')

def _shape_streaming_ready(weights: Path) -> bool:
    return (weights / 'hunyuan3d-dit-v2-1' / 'model.fp16.safetensors').is_file() and (weights / 'hunyuan3d-vae-v2-1' / 'model.fp16.safetensors').is_file()

def _paint_streaming_ready(weights: Path) -> bool:
    text_encoder = weights / 'hunyuan3d-paintpbr-v2-1' / 'text_encoder'
    return (weights / 'hunyuan3d-paintpbr-v2-1' / 'unet' / 'diffusion_pytorch_model.safetensors').is_file() and (weights / 'hunyuan3d-paintpbr-v2-1' / 'vae' / 'diffusion_pytorch_model.safetensors').is_file() and ((text_encoder / 'pytorch_model.safetensors').is_file() or (text_encoder / 'model.safetensors').is_file())

def _ensure_paths(repo: Optional[Path]=None, weights: Optional[Path]=None) -> tuple[Path, Path]:
    repo = repo or _repo_dir()
    weights = weights or _weights_dir()
    for p in (repo / 'hy3dshape', repo / 'hy3dpaint', repo, weights, weights / 'hy3dpaint'):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
    os.environ['HY3DGEN_MODELS'] = str(weights.parent)
    os.environ.update(_fast_hf_cache_env())
    return (repo, weights)

@contextlib.contextmanager
def _local_hf_snapshot(weights: Path) -> Iterator[None]:
    try:
        import huggingface_hub
    except Exception:
        yield
        return
    original = getattr(huggingface_hub, 'snapshot_download', None)
    if original is None:
        yield
        return

    def snapshot_download(repo_id=None, *args, **kwargs):
        if str(repo_id or '').lower() in ('tencent/hunyuan3d-2.1', 'hunyuan3d-2.1'):
            return str(weights)
        return original(repo_id, *args, **kwargs)
    huggingface_hub.snapshot_download = snapshot_download
    try:
        yield
    finally:
        huggingface_hub.snapshot_download = original

@contextlib.contextmanager
def _local_paint_custom_pipeline_trust(repo: Path, *, device: Optional[str]=None, streaming_ready: bool=False) -> Iterator[None]:
    try:
        from diffusers import DiffusionPipeline
    except Exception:
        yield
        return
    original = DiffusionPipeline.from_pretrained
    paint_root = (repo / 'hy3dpaint' / 'hunyuanpaintpbr').resolve()

    def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        custom = kwargs.get('custom_pipeline')
        allow = False
        if custom:
            try:
                allow = Path(str(custom)).resolve() == paint_root
            except Exception:
                allow = str(custom).replace('\\', '/').endswith('/hunyuanpaintpbr')
        if allow:
            kwargs.setdefault('trust_remote_code', True)
            if streaming_ready:
                kwargs.setdefault('use_safetensors', True)
                kwargs.setdefault('low_cpu_mem_usage', True)
        return original(pretrained_model_name_or_path, *args, **kwargs)
    DiffusionPipeline.from_pretrained = from_pretrained
    old_device = os.environ.get('STUDIO_HUNYUAN3D_PAINT_DEVICE')
    if device:
        os.environ['STUDIO_HUNYUAN3D_PAINT_DEVICE'] = str(device)
    try:
        yield
    finally:
        if old_device is None:
            os.environ.pop('STUDIO_HUNYUAN3D_PAINT_DEVICE', None)
        else:
            os.environ['STUDIO_HUNYUAN3D_PAINT_DEVICE'] = old_device
        DiffusionPipeline.from_pretrained = original

def _patch_hunyuan_shape_direct_loader() -> None:
    try:
        import hy3dshape.pipelines as pipelines
    except Exception:
        return
    base_cls = getattr(pipelines, 'Hunyuan3DDiTPipeline', None)
    if base_cls is None or getattr(base_cls, '_studio_direct_safetensors_patch', False):
        return
    original = base_cls.from_single_file.__func__

    @classmethod
    def from_single_file(cls, ckpt_path, config_path, device='cuda', dtype=None, use_safetensors=None, **kwargs):
        if not use_safetensors:
            return original(cls, ckpt_path, config_path, device=device, dtype=dtype, use_safetensors=use_safetensors, **kwargs)
        import torch
        import yaml
        from safetensors import safe_open
        dtype = dtype or torch.float16
        ckpt_path = str(ckpt_path).replace('.ckpt', '.safetensors')
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f'Model file {ckpt_path} not found')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f'[Hunyuan3D] direct-loading shape safetensors to {device}: {ckpt_path}', flush=True)

        def read_section(section: str) -> dict:
            prefix = section + '.'
            with safe_open(ckpt_path, framework='pt', device=str(device)) as handle:
                return {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}

        def load_section(module, section: str, *, strict: bool=True) -> None:
            state = read_section(section)
            if not state:
                if strict:
                    raise RuntimeError(f'Hunyuan3D safetensors file has no {section!r} section: {ckpt_path}')
                return
            try:
                module.load_state_dict(state, strict=strict, assign=True)
            except TypeError:
                module.load_state_dict(state, strict=strict)
            finally:
                del state
                gc.collect()
                if str(device).startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        with torch.device(device):
            model = pipelines.instantiate_from_config(config['model'])
            vae = pipelines.instantiate_from_config(config['vae'])
            conditioner = pipelines.instantiate_from_config(config['conditioner'])
        image_processor = pipelines.instantiate_from_config(config['image_processor'])
        scheduler = pipelines.instantiate_from_config(config['scheduler'])
        load_section(model, 'model')
        load_section(vae, 'vae', strict=False)
        load_section(conditioner, 'conditioner', strict=False)
        model_kwargs = dict(vae=vae, model=model, scheduler=scheduler, conditioner=conditioner, image_processor=image_processor, device=device, dtype=dtype)
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)
    base_cls.from_single_file = from_single_file
    base_cls._studio_direct_safetensors_patch = True

def _install_paint_unet_direct_loader(paint_cls) -> None:
    if paint_cls is None or getattr(paint_cls, '_studio_direct_safetensors_patch', False):
        return
    module = sys.modules.get(str(getattr(paint_cls, '__module__', '')))
    if module is None:
        return
    original = paint_cls.from_pretrained

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        import torch
        from safetensors import safe_open
        torch_dtype = kwargs.pop('torch_dtype', torch.float32)
        use_safetensors = kwargs.pop('use_safetensors', True)
        target_device = str(kwargs.pop('device', None) or os.environ.get('STUDIO_HUNYUAN3D_PAINT_DEVICE') or 'cuda:0')
        root = Path(str(pretrained_model_name_or_path))
        safetensor_path = root / 'diffusion_pytorch_model.safetensors'
        if not (use_safetensors and safetensor_path.is_file()):
            if not _truthy_env('STUDIO_HUNYUAN3D_ALLOW_HOST_STAGING'):
                raise RuntimeError('Hunyuan3D paint UNet would load diffusion_pytorch_model.bin through host RAM. Convert it to diffusion_pytorch_model.safetensors first, or set STUDIO_HUNYUAN3D_ALLOW_HOST_STAGING=1 for diagnostics.')
            return original(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
        config_path = root / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        with torch.device(target_device):
            unet = module.UNet2DConditionModel(**config)
            unet_2p5d = paint_cls(unet)
            unet_2p5d.unet.conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, stride=unet.conv_in.stride, padding=unet.conv_in.padding, dilation=unet.conv_in.dilation, groups=unet.conv_in.groups, bias=unet.conv_in.bias is not None)
        print(f'[Hunyuan3D] direct-loading paint UNet safetensors to {target_device}: {safetensor_path}', flush=True)
        with safe_open(str(safetensor_path), framework='pt', device=target_device) as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        try:
            unet_2p5d.load_state_dict(state, strict=True, assign=True)
        except TypeError:
            unet_2p5d.load_state_dict(state, strict=True)
        finally:
            del state
            gc.collect()
            if target_device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return unet_2p5d.to(torch_dtype)
    paint_cls.from_pretrained = from_pretrained
    paint_cls._studio_direct_safetensors_patch = True

def _patch_hunyuan_paint_direct_loader() -> None:
    for modname in ('hunyuanpaintpbr.unet.modules', 'diffusers_modules.local.modules'):
        try:
            module = __import__(modname, fromlist=['UNet2p5DConditionModel'])
        except Exception:
            continue
        _install_paint_unet_direct_loader(getattr(module, 'UNet2p5DConditionModel', None))

@contextlib.contextmanager
def _patch_diffusers_dynamic_paint_loader() -> Iterator[None]:
    try:
        import diffusers.utils.dynamic_modules_utils as dmu
    except Exception:
        yield
        return
    original = dmu.get_class_in_module

    def get_class_in_module(class_name, module_path, force_reload=False):
        cls = original(class_name, module_path, force_reload=force_reload)
        if class_name == 'UNet2p5DConditionModel':
            _install_paint_unet_direct_loader(cls)
        return cls
    dmu.get_class_in_module = get_class_in_module
    try:
        yield
    finally:
        dmu.get_class_in_module = original

def _normalise_device(device: Optional[str]) -> str:
    if not device or str(device).lower() == 'cpu':
        return 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else 'cpu'
    return str(device)

def _bpy_package_dirs(*, python: Optional[Path]=None) -> list[Path]:
    roots: list[Path] = []
    if python is not None:
        roots.append(python.parent.parent / 'Lib' / 'site-packages' / 'bpy')
    for raw in list(sys.path):
        if raw:
            roots.append(Path(raw) / 'bpy')
    roots.extend([Path(''), Path('')])
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        try:
            real = p.resolve()
        except Exception:
            real = p
        key = str(real).lower()
        if key in seen or not p.is_dir():
            continue
        seen.add(key)
        out.append(p)
    return out

def _ensure_bpy_dll_search_path(*, python: Optional[Path]=None) -> None:
    handles = _STATE.setdefault('bpy_dll_handles', [])
    path_parts = os.environ.get('PATH', '').split(os.pathsep)
    lower_parts = {p.lower() for p in path_parts if p}
    for root in _bpy_package_dirs(python=python):
        s = str(root)
        if s.lower() not in lower_parts:
            os.environ['PATH'] = s + os.pathsep + os.environ.get('PATH', '')
            lower_parts.add(s.lower())
        if hasattr(os, 'add_dll_directory'):
            try:
                handles.append(os.add_dll_directory(s))
            except OSError:
                pass
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError(f'Hunyuan3D PBR requires the bpy Blender Python wheel, but it failed to import after pinning Blender DLL paths: {type(exc).__name__}: {exc}') from exc

def _patch_hunyuan_paint_compat() -> None:
    try:
        import utils.simplify_mesh_utils as smu
    except Exception:
        smu = None
    if smu is not None and (not getattr(smu, '_studio_trimesh_target_count_patch', False)):

        def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
            import pymeshlab
            import trimesh
            target_count = int(os.environ.get('STUDIO_HUNYUAN3D_PBR_TARGET_FACES', str(target_count)))
            ms = pymeshlab.MeshSet()
            if str(inputpath).endswith('.glb'):
                ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
            else:
                ms.load_new_mesh(inputpath)
            tmp_obj = str(outputpath).replace('.glb', '.obj')
            ms.save_current_mesh(tmp_obj, save_textures=False)
            current = trimesh.load(tmp_obj, force='mesh')
            face_num = current.faces.shape[0]
            if face_num > int(target_count):
                import inspect
                params = inspect.signature(current.simplify_quadric_decimation).parameters
                if 'face_count' in params:
                    current = current.simplify_quadric_decimation(face_count=int(target_count))
                elif 'target_count' in params:
                    current = current.simplify_quadric_decimation(target_count=int(target_count))
                else:
                    current = current.simplify_quadric_decimation(int(target_count))
            current.export(outputpath)
            return current
        smu.mesh_simplify_trimesh = mesh_simplify_trimesh
        smu._studio_trimesh_target_count_patch = True
    try:
        import utils.image_super_utils as isu
        import textureGenPipeline as tgp
    except Exception:
        return
    if getattr(isu, '_studio_realesrgan_tile_patch', False):
        return

    class StudioTiledImageSuperNet:

        def __init__(self, config) -> None:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            tile = int(os.environ.get('STUDIO_HUNYUAN3D_REALESRGAN_TILE', '256'))
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upsampler = RealESRGANer(scale=4, model_path=config.realesrgan_ckpt_path, dni_weight=None, model=model, tile=tile, tile_pad=10, pre_pad=0, half=True, gpu_id=None)

        def __call__(self, image):
            import numpy as np
            from PIL import Image
            output, _ = self.upsampler.enhance(np.array(image))
            return Image.fromarray(output)
    isu.imageSuperNet = StudioTiledImageSuperNet
    tgp.imageSuperNet = StudioTiledImageSuperNet
    isu._studio_realesrgan_tile_patch = True

def _drop_shape_pipeline(*, reason: str='release before PBR') -> None:
    with _LOCK:
        _STATE.pop('shape_pipe', None)
        _STATE.pop('shape_key', None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    print(f'[Hunyuan3D] shape pipeline released ({reason})', flush=True)

def _write_worker_logs(prefix: Path, stdout: str, stderr: str) -> None:
    prefix.with_name(prefix.name + '_stdout.log').write_text(stdout or '', encoding='utf-8')
    prefix.with_name(prefix.name + '_stderr.log').write_text(stderr or '', encoding='utf-8')

def _heavy_python() -> Optional[Path]:
    for name in ('STUDIO_HUNYUAN3D_PYTHON', 'STUDIO_COSMOS_PYTHON'):
        raw = os.environ.get(name)
        if raw and Path(raw).is_file():
            return Path(raw)
    for raw in ('', ''):
        p = Path(raw)
        if p.is_file():
            return p
    return None

def _should_spawn_worker(*, force: bool=False) -> bool:
    if os.environ.get(_SUBPROCESS_FLAG) == '1':
        return False
    if os.environ.get('STUDIO_HUNYUAN3D_INPROC', '').lower() in ('1', 'true', 'on', 'yes'):
        return False
    py = _heavy_python()
    if py is None:
        return False
    if force:
        return True
    try:
        return py.resolve() != Path(sys.executable).resolve()
    except Exception:
        return str(py).lower() != str(sys.executable).lower()

def _worker_env(repo: Path, weights: Path, python: Path) -> dict:
    env = os.environ.copy()
    env[_SUBPROCESS_FLAG] = '1'
    env['STUDIO_HUNYUAN3D_REPO'] = str(repo)
    env['STUDIO_HUNYUAN3D_DIR'] = str(weights)
    env.setdefault('HY3DGEN_MODELS', str(weights.parent))
    env.update(_fast_hf_cache_env())
    dino = _resolve_dino_dir()
    if dino is not None:
        env.setdefault('STUDIO_HUNYUAN3D_DINO_DIR', str(dino))
    extra = [str(repo), str(repo / 'hy3dshape'), str(repo / 'hy3dpaint'), str(repo / 'hy3dpaint' / 'custom_rasterizer')]
    env['PYTHONPATH'] = os.pathsep.join(extra + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    scripts = str(python.parent)
    bpy_dirs = [str(p) for p in _bpy_package_dirs(python=python)]
    env['PATH'] = os.pathsep.join([scripts] + bpy_dirs + [env.get('PATH', '')])
    return env

def _run_worker_image_to_glb(object_image_path: str, *, out_dir: str, basename: str, texture: str='baked', device: str='cuda:0', **kwargs) -> str:
    repo, weights = _ensure_paths()
    py = _heavy_python()
    if py is None:
        raise RuntimeError('Hunyuan3D heavy Python is not configured; set STUDIO_HUNYUAN3D_PYTHON or STUDIO_COSMOS_PYTHON.')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(basename or 'object')))
    req = out / f'{stem}_hunyuan3d_request.json'
    res = out / f'{stem}_hunyuan3d_result.json'
    req.write_text(json.dumps({'object_image_path': object_image_path, 'out_dir': out_dir, 'basename': basename, 'texture': texture, 'device': device, 'kwargs': kwargs}), encoding='utf-8')
    argv = [str(py), str(Path(__file__).resolve()), '--image-to-glb', str(req), str(res)]
    timeout = int(os.environ.get('STUDIO_HUNYUAN3D_TIMEOUT', '7200'))
    log_prefix = out / f'{stem}_hunyuan3d_worker'
    run_kwargs = {'cwd': str(repo), 'env': _worker_env(repo, weights, py), 'capture_output': True, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'timeout': timeout}
    if os.name == 'nt':
        run_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    proc = subprocess.run(argv, **run_kwargs)
    _write_worker_logs(log_prefix, proc.stdout or '', proc.stderr or '')
    body = None
    if res.is_file():
        try:
            body = json.loads(res.read_text(encoding='utf-8'))
        except Exception:
            body = None
    if proc.returncode != 0 or not isinstance(body, dict) or (not body.get('ok')):
        detail = ''
        if isinstance(body, dict) and body.get('error'):
            detail = str(body.get('error'))
        tail = ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()[-3000:]
        raise RuntimeError(f'Hunyuan3D worker failed (exit {proc.returncode})' + (f': {detail}' if detail else '') + (f'\n--- worker tail ---\n{tail}' if tail else ''))
    path = str(body.get('path') or '')
    if not path:
        raise RuntimeError('Hunyuan3D worker completed without returning a mesh path')
    return path

def _run_worker_paint_obj(shape_obj: str, image_path: str, output_mesh_path: str, *, device: str='cuda:0', **kwargs) -> str:
    repo, weights = _ensure_paths()
    py = _heavy_python()
    if py is None:
        raise RuntimeError('Hunyuan3D heavy Python is not configured; set STUDIO_HUNYUAN3D_PYTHON or STUDIO_COSMOS_PYTHON.')
    out = Path(output_mesh_path).parent
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(output_mesh_path).stem
    req = out / f'{stem}_hunyuan3d_paint_request.json'
    res = out / f'{stem}_hunyuan3d_paint_result.json'
    req.write_text(json.dumps({'shape_obj': shape_obj, 'image_path': image_path, 'output_mesh_path': output_mesh_path, 'device': device, 'kwargs': kwargs}), encoding='utf-8')
    argv = [str(py), str(Path(__file__).resolve()), '--paint-obj', str(req), str(res)]
    env = _worker_env(repo, weights, py)
    env[_PBR_SUBPROCESS_FLAG] = '1'
    timeout = int(os.environ.get('STUDIO_HUNYUAN3D_TIMEOUT', '7200'))
    log_prefix = out / f'{stem}_hunyuan3d_paint_worker'
    run_kwargs = {'cwd': str(repo), 'env': env, 'capture_output': True, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'timeout': timeout}
    if os.name == 'nt':
        run_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    proc = subprocess.run(argv, **run_kwargs)
    _write_worker_logs(log_prefix, proc.stdout or '', proc.stderr or '')
    body = None
    if res.is_file():
        try:
            body = json.loads(res.read_text(encoding='utf-8'))
        except Exception:
            body = None
    if proc.returncode != 0 or not isinstance(body, dict) or (not body.get('ok')):
        detail = ''
        if isinstance(body, dict) and body.get('error'):
            detail = str(body.get('error'))
        tail = ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()[-3000:]
        raise RuntimeError(f'Hunyuan3D PBR worker failed (exit {proc.returncode})' + (f': {detail}' if detail else '') + (f'\n--- PBR worker tail ---\n{tail}' if tail else ''))
    path = str(body.get('path') or '')
    if not path:
        raise RuntimeError('Hunyuan3D PBR worker completed without returning a mesh path')
    return path

def _paint_existing_obj_inprocess(shape_obj: str, image_path: str, output_mesh_path: str, *, device: str='cuda:0', **kwargs) -> str:
    repo, weights = _ensure_paths()
    max_views = int(kwargs.get('max_num_view') or os.environ.get('STUDIO_HUNYUAN3D_PBR_VIEWS', 6))
    resolution = int(kwargs.get('resolution') or os.environ.get('STUDIO_HUNYUAN3D_PBR_RES', 512))
    paint = build_paint_pipeline(device=device, max_num_view=max_views, resolution=resolution)
    textured_obj = Path(output_mesh_path)
    print(f'[Hunyuan3D] running PBR texture bake mesh={shape_obj} out={textured_obj}', flush=True)
    with _local_hf_snapshot(weights):
        paint(mesh_path=str(shape_obj), image_path=str(image_path), output_mesh_path=str(textured_obj), save_glb=True)
    textured_glb = textured_obj.with_suffix('.glb')
    print(f'[Hunyuan3D] PBR texture bake complete glb={textured_glb}', flush=True)
    return str(textured_glb if textured_glb.is_file() else textured_obj)

def build_shape_pipeline(*, device: str='cuda:0'):
    repo, weights = _ensure_paths()
    device = _normalise_device(device)
    key = ('shape', str(weights), device)
    with _LOCK:
        if _STATE.get('shape_key') == key and _STATE.get('shape_pipe') is not None:
            return _STATE['shape_pipe']
        streaming_ready = _shape_streaming_ready(weights)
        if not streaming_ready and (not _truthy_env('STUDIO_HUNYUAN3D_ALLOW_HOST_STAGING')):
            raise RuntimeError('')
        try:
            from utils.torchvision_fix import apply_fix
            apply_fix()
        except Exception:
            pass
        import torch
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
        if streaming_ready:
            _patch_hunyuan_shape_direct_loader()
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(weights.name, device=device, dtype=torch.float16, use_safetensors=streaming_ready, variant='fp16', subfolder='hunyuan3d-dit-v2-1')
        _STATE['shape_pipe'] = pipe
        _STATE['shape_key'] = key
        _STATE['repo'] = str(repo)
        _STATE['weights'] = str(weights)
        return pipe

def build_paint_pipeline(*, device: str='cuda:0', max_num_view: int=6, resolution: int=512):
    repo, weights = _ensure_paths()
    ok, missing = _paint_ready(weights, repo)
    if not ok:
        raise RuntimeError(f'Hunyuan3D PBR paint assets are missing: {missing}')
    if not _paint_streaming_ready(weights) and (not _truthy_env('STUDIO_HUNYUAN3D_ALLOW_HOST_STAGING')):
        raise RuntimeError('Hunyuan3D PBR paint weights are present only as .bin files, which would stage through host RAM. Convert the paint UNet/VAE/text encoder weights to safetensors before using the production GUI path, or set STUDIO_HUNYUAN3D_ALLOW_HOST_STAGING=1 for diagnostics.')
    ok, missing = _rasterizer_ready()
    if not ok:
        raise RuntimeError(f'Hunyuan3D PBR custom rasterizer extension is not installed (missing: {missing}). Build hy3dpaint/custom_rasterizer with a CUDA toolkit matching the PyTorch CUDA version.')
    _ensure_bpy_dll_search_path(python=_heavy_python())
    device = _normalise_device(device)
    key = ('paint', str(repo), str(weights), device, int(max_num_view), int(resolution))
    with _LOCK:
        if _STATE.get('paint_key') == key and _STATE.get('paint_pipe') is not None:
            return _STATE['paint_pipe']
        try:
            from utils.torchvision_fix import apply_fix
            apply_fix()
        except Exception:
            pass
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        _patch_hunyuan_paint_compat()
        _patch_hunyuan_paint_direct_loader()
        conf = Hunyuan3DPaintConfig(max_num_view=int(max_num_view), resolution=int(resolution))
        conf.device = device
        conf.render_size = int(os.environ.get('STUDIO_HUNYUAN3D_PBR_RENDER_SIZE', '1024'))
        conf.texture_size = int(os.environ.get('STUDIO_HUNYUAN3D_PBR_TEXTURE_SIZE', '2048'))
        conf.multiview_cfg_path = str(repo / 'hy3dpaint' / 'cfgs' / 'hunyuan-paint-pbr.yaml')
        conf.custom_pipeline = str(repo / 'hy3dpaint' / 'hunyuanpaintpbr')
        conf.multiview_pretrained_path = 'tencent/Hunyuan3D-2.1'
        conf.realesrgan_ckpt_path = str(repo / 'hy3dpaint' / 'ckpt' / 'RealESRGAN_x4plus.pth')
        dino = _resolve_dino_dir()
        if dino is not None:
            conf.dino_ckpt_path = str(dino)
        print(f'[Hunyuan3D] building PBR paint pipeline device={device} views={int(max_num_view)} res={int(resolution)} render={conf.render_size} texture={conf.texture_size} custom_pipeline={conf.custom_pipeline} dino={conf.dino_ckpt_path}', flush=True)
        with _patch_diffusers_dynamic_paint_loader(), _local_hf_snapshot(weights), _local_paint_custom_pipeline_trust(repo, device=device, streaming_ready=_paint_streaming_ready(weights)):
            pipe = Hunyuan3DPaintPipeline(conf)
        print('[Hunyuan3D] PBR paint pipeline ready', flush=True)
        _STATE['paint_pipe'] = pipe
        _STATE['paint_key'] = key
        return pipe

def _load_rgba_image(path: str):
    from PIL import Image
    img = Image.open(path).convert('RGBA')
    if os.environ.get('STUDIO_HUNYUAN3D_REMBG', 'auto').lower() in ('0', 'false', 'off', 'no'):
        return img
    try:
        alpha = img.getchannel('A')
        if alpha.getextrema()[0] < 255:
            return img
    except Exception:
        return img
    try:
        from hy3dshape.rembg import BackgroundRemover
        return BackgroundRemover()(img)
    except Exception:
        return img

def image_to_glb(object_image_path: str, *, out_dir: str, basename: str, texture: str='baked', device: str='cuda:0', **kwargs) -> str:
    tex = str(texture or 'baked').lower()
    spawn = _should_spawn_worker(force=tex == 'pbr')
    if os.environ.get('STUDIO_HUNYUAN3D_ROUTE_LOG', '1').lower() not in ('0', 'false', 'off', 'no'):
        print(f'[Hunyuan3D] route file={__file__} exe={sys.executable} heavy={_heavy_python()} spawn_worker={spawn} worker_flag={os.environ.get(_SUBPROCESS_FLAG)}', flush=True)
    if spawn:
        return _run_worker_image_to_glb(object_image_path, out_dir=out_dir, basename=basename, texture=texture, device=device, **kwargs)
    return _image_to_glb_inprocess(object_image_path, out_dir=out_dir, basename=basename, texture=texture, device=device, **kwargs)

def _image_to_glb_inprocess(object_image_path: str, *, out_dir: str, basename: str, texture: str='baked', device: str='cuda:0', **kwargs) -> str:
    repo, weights = _ensure_paths()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(basename or 'object')))
    import torch
    pipe = build_shape_pipeline(device=device)
    image = _load_rgba_image(object_image_path)
    seed = kwargs.get('seed')
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=_normalise_device(device)).manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))
    steps = int(kwargs.get('steps') or os.environ.get('STUDIO_HUNYUAN3D_STEPS', 50))
    guidance_scale = float(kwargs.get('guidance_scale') or os.environ.get('STUDIO_HUNYUAN3D_GUIDANCE', 5.0))
    octree_resolution = int(kwargs.get('octree_resolution') or os.environ.get('STUDIO_HUNYUAN3D_OCTREE', 384))
    num_chunks = int(kwargs.get('num_chunks') or os.environ.get('STUDIO_HUNYUAN3D_CHUNKS', 8000))
    outputs = pipe(image=image, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator, octree_resolution=octree_resolution, num_chunks=num_chunks, output_type='trimesh', enable_pbar=False)
    mesh = outputs[0] if outputs else None
    if mesh is None:
        raise RuntimeError('Hunyuan3D produced no mesh. Increase STUDIO_HUNYUAN3D_STEPS / STUDIO_HUNYUAN3D_OCTREE or use a cleaner object image.')
    tex = str(texture or 'baked').lower()
    if tex == 'pbr':
        fallback_glb = out / f'{stem}_hunyuan.glb'
        mesh.export(str(fallback_glb))
        shape_obj = out / f'{stem}_hunyuan_shape.obj'
        mesh.export(str(shape_obj))
        pbr_inprocess = os.environ.get(_PBR_SUBPROCESS_FLAG) == '1' or os.environ.get('STUDIO_HUNYUAN3D_PBR_INPROC', '').lower() in ('1', 'true', 'on', 'yes')
        del pipe, mesh, outputs, image
        _drop_shape_pipeline(reason='release before in-process PBR' if pbr_inprocess else 'release before PBR worker')
        try:
            textured_obj = out / f'{stem}_hunyuan_pbr.obj'
            if not pbr_inprocess:
                return _run_worker_paint_obj(str(shape_obj), object_image_path, str(textured_obj), device=device, **kwargs)
            return _paint_existing_obj_inprocess(str(shape_obj), object_image_path, str(textured_obj), device=device, **kwargs)
        except Exception as exc:
            if os.environ.get('STUDIO_HUNYUAN3D_PBR_ALLOW_FALLBACK', '').lower() in ('1', 'true', 'on', 'yes'):
                print(f'[Hunyuan3D] PBR texture failed ({type(exc).__name__}: {exc}); returning baked shape GLB fallback: {fallback_glb}', flush=True)
                return str(fallback_glb)
            raise RuntimeError(f'Hunyuan3D PBR texture export failed: {type(exc).__name__}: {exc}. The baked shape GLB was written for debugging, but PBR jobs are strict by default.') from exc
    glb = out / f'{stem}_hunyuan.glb'
    mesh.export(str(glb))
    return str(glb)

def _main(argv: Optional[list[str]]=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('--image-to-glb', '--paint-obj') or len(argv) != 3:
        print('usage: hunyuan3d_backend.py (--image-to-glb|--paint-obj) REQUEST_JSON RESULT_JSON', file=sys.stderr)
        return 2
    mode = argv[0]
    req_path = Path(argv[1])
    res_path = Path(argv[2])
    try:
        req = json.loads(req_path.read_text(encoding='utf-8'))
        if mode == '--paint-obj':
            path = _paint_existing_obj_inprocess(str(req['shape_obj']), str(req['image_path']), str(req['output_mesh_path']), device=str(req.get('device') or 'cuda:0'), **dict(req.get('kwargs') or {}))
        else:
            path = _image_to_glb_inprocess(str(req['object_image_path']), out_dir=str(req['out_dir']), basename=str(req['basename']), texture=str(req.get('texture') or 'baked'), device=str(req.get('device') or 'cuda:0'), **dict(req.get('kwargs') or {}))
        res_path.write_text(json.dumps({'ok': True, 'path': path}), encoding='utf-8')
        return 0
    except Exception as exc:
        res_path.write_text(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}), encoding='utf-8')
        traceback.print_exc()
        return 1

def preflight() -> dict:
    out = {'ok': True, 'missing': [], 'repo': None, 'weights': None, 'paint_ready': False}
    try:
        repo = _repo_dir()
        out['repo'] = str(repo)
    except Exception as exc:
        out['ok'] = False
        out['missing'].append(str(exc))
        repo = None
    try:
        weights = _weights_dir()
        out['weights'] = str(weights)
    except Exception as exc:
        out['ok'] = False
        out['missing'].append(str(exc))
        weights = None
    if repo is not None and weights is not None:
        ok, missing = _paint_ready(weights, repo)
        rok, rmissing = _rasterizer_ready()
        out['rasterizer_ready'] = rok
        out['paint_ready'] = ok and rok
        if not ok:
            out['paint_missing'] = missing
        if not rok:
            out['rasterizer_missing'] = rmissing
    return out

def unload() -> None:
    with _LOCK:
        _STATE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
if __name__ == '__main__':
    raise SystemExit(_main())
