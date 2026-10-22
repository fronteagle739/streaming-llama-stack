# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/trellis_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
import threading
import types
import importlib.util
import importlib.machinery
import json
from pathlib import Path
_LOCK = threading.Lock()
_STATE: dict = {}

def _weights_dir() -> str:
    env = os.environ.get('STUDIO_TRELLIS_DIR')
    cands = [env, '', os.path.expanduser('~/models/TRELLIS-image-large'), '']
    for c in cands:
        if c and (Path(c) / 'pipeline.json').is_file():
            return str(c)
    raise RuntimeError('')

def _ensure_repo_on_path() -> None:
    try:
        import trellis
        return
    except Exception:
        pass
    env = os.environ.get('STUDIO_TRELLIS_REPO')
    cands = [env, '', os.path.expanduser('~/GitHub/TRELLIS')]
    for c in cands:
        if c and (Path(c) / 'trellis').is_dir():
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    raise RuntimeError('')

def _flexicubes_file_present() -> bool:
    rel = Path('trellis') / 'representations' / 'mesh' / 'flexicubes'
    for root in sys.path:
        try:
            base = Path(root) / rel
            if not base.is_dir():
                continue
            if any(base.glob('flexicubes.*')):
                return True
        except OSError:
            continue
    return False

def _import_image_pipeline_class():
    _install_flash_attn_sdpa_stub()
    stub = None
    flex_stub = None
    if 'open3d' not in sys.modules and importlib.util.find_spec('open3d') is None:
        stub = types.ModuleType('open3d')
        stub.geometry = types.SimpleNamespace(TriangleMesh=type('TriangleMesh', (), {}))
        sys.modules['open3d'] = stub
    flex_name = 'trellis.representations.mesh.flexicubes.flexicubes'
    if flex_name not in sys.modules and (not _flexicubes_file_present()):
        flex_stub = types.ModuleType(flex_name)

        class FlexiCubes:

            def __init__(self, *args, **kwargs):
                raise RuntimeError('TRELLIS FlexiCubes mesh extension is not installed.')
        flex_stub.FlexiCubes = FlexiCubes
        sys.modules[flex_name] = flex_stub
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        return TrellisImageTo3DPipeline
    finally:
        if stub is not None and sys.modules.get('open3d') is stub:
            sys.modules.pop('open3d', None)
        if flex_stub is not None and sys.modules.get(flex_name) is flex_stub:
            sys.modules.pop(flex_name, None)

def _install_flash_attn_sdpa_stub() -> None:
    if os.environ.get('STUDIO_TRELLIS_SDPA_FLASH_STUB', '1').lower() in ('0', 'false', 'off', 'no'):
        return
    if 'flash_attn' in sys.modules or importlib.util.find_spec('flash_attn') is not None:
        return
    import torch
    import torch.nn.functional as F

    def _sdpa_batched(q, k, v):
        out = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3).contiguous()

    def _sdpa_single(q, k, v):
        out = F.scaled_dot_product_attention(q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0))
        return out.squeeze(0).permute(1, 0, 2).contiguous()

    def _starts(cu):
        vals = cu.detach().to('cpu').tolist()
        return [(int(vals[i]), int(vals[i + 1])) for i in range(len(vals) - 1)]
    mod = types.ModuleType('flash_attn')
    mod.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)

    def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
        q, k, v = qkv.unbind(dim=2)
        return _sdpa_batched(q, k, v) if q.dim() == 4 else _sdpa_single(q, k, v)

    def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
        k, v = kv.unbind(dim=2)
        return _sdpa_batched(q, k, v) if q.dim() == 4 else _sdpa_single(q, k, v)

    def flash_attn_func(q, k, v, *args, **kwargs):
        return _sdpa_batched(q, k, v) if q.dim() == 4 else _sdpa_single(q, k, v)

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, *args, **kwargs):
        outs = []
        for s, e in _starts(cu_seqlens_q):
            q, k, v = qkv[s:e].unbind(dim=1)
            outs.append(_sdpa_single(q, k, v))
        return torch.cat(outs, dim=0) if outs else qkv.new_empty((0, qkv.shape[-2], qkv.shape[-1]))

    def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, *args, **kwargs):
        outs = []
        q_ranges = _starts(cu_seqlens_q)
        kv_ranges = _starts(cu_seqlens_kv)
        for (qs, qe), (ks, ke) in zip(q_ranges, kv_ranges):
            k, v = kv[ks:ke].unbind(dim=1)
            outs.append(_sdpa_single(q[qs:qe], k, v))
        return torch.cat(outs, dim=0) if outs else q.new_empty((0, q.shape[-2], q.shape[-1]))

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, *args, **kwargs):
        outs = []
        q_ranges = _starts(cu_seqlens_q)
        kv_ranges = _starts(cu_seqlens_kv)
        for (qs, qe), (ks, ke) in zip(q_ranges, kv_ranges):
            outs.append(_sdpa_single(q[qs:qe], k[ks:ke], v[ks:ke]))
        return torch.cat(outs, dim=0) if outs else q.new_empty((0, q.shape[-2], v.shape[-1]))
    mod.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    mod.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    mod.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules['flash_attn'] = mod

def _install_utils3d_numpy_compat() -> None:
    try:
        import utils3d
    except Exception:
        return
    try:
        import utils3d.numpy as utils3d_numpy
        if not hasattr(utils3d, 'numpy'):
            utils3d.numpy = utils3d_numpy
        return
    except Exception:
        pass
    current = getattr(utils3d, 'numpy', None)
    if current is not None and all((hasattr(current, n) for n in ('quaternion_to_matrix', 'matrix_to_quaternion'))):
        return
    import numpy as np
    mod = types.ModuleType('utils3d.numpy')

    def quaternion_to_matrix(quaternion):
        q = np.asarray(quaternion)
        dtype = q.dtype if np.issubdtype(q.dtype, np.floating) else np.float32
        q = q.astype(np.float64, copy=False)
        if q.shape[-1] != 4:
            raise ValueError('quaternion last dimension must be 4')
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        q = q / np.maximum(norm, 1e-12)
        w, x, y, z = [q[..., i] for i in range(4)]
        out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
        out[..., 0, 0] = 1 - 2 * (y * y + z * z)
        out[..., 0, 1] = 2 * (x * y - z * w)
        out[..., 0, 2] = 2 * (x * z + y * w)
        out[..., 1, 0] = 2 * (x * y + z * w)
        out[..., 1, 1] = 1 - 2 * (x * x + z * z)
        out[..., 1, 2] = 2 * (y * z - x * w)
        out[..., 2, 0] = 2 * (x * z - y * w)
        out[..., 2, 1] = 2 * (y * z + x * w)
        out[..., 2, 2] = 1 - 2 * (x * x + y * y)
        return out.astype(dtype, copy=False)

    def matrix_to_quaternion(matrix):
        m = np.asarray(matrix)
        dtype = m.dtype if np.issubdtype(m.dtype, np.floating) else np.float32
        m64 = m.astype(np.float64, copy=False)
        if m64.shape[-2:] != (3, 3):
            raise ValueError('matrix last dimensions must be 3x3')
        flat = m64.reshape(-1, 3, 3)
        out = np.empty((flat.shape[0], 4), dtype=np.float64)
        for i, r in enumerate(flat):
            trace = r[0, 0] + r[1, 1] + r[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                out[i, 0] = 0.25 / s
                out[i, 1] = (r[2, 1] - r[1, 2]) * s
                out[i, 2] = (r[0, 2] - r[2, 0]) * s
                out[i, 3] = (r[1, 0] - r[0, 1]) * s
            elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                s = 2.0 * np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12))
                out[i, 0] = (r[2, 1] - r[1, 2]) / s
                out[i, 1] = 0.25 * s
                out[i, 2] = (r[0, 1] + r[1, 0]) / s
                out[i, 3] = (r[0, 2] + r[2, 0]) / s
            elif r[1, 1] > r[2, 2]:
                s = 2.0 * np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12))
                out[i, 0] = (r[0, 2] - r[2, 0]) / s
                out[i, 1] = (r[0, 1] + r[1, 0]) / s
                out[i, 2] = 0.25 * s
                out[i, 3] = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12))
                out[i, 0] = (r[1, 0] - r[0, 1]) / s
                out[i, 1] = (r[0, 2] + r[2, 0]) / s
                out[i, 2] = (r[1, 2] + r[2, 1]) / s
                out[i, 3] = 0.25 * s
        out = out.reshape(m64.shape[:-2] + (4,))
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.maximum(norm, 1e-12)
        return out.astype(dtype, copy=False)
    mod.quaternion_to_matrix = quaternion_to_matrix
    mod.matrix_to_quaternion = matrix_to_quaternion
    utils3d.numpy = mod
    sys.modules['utils3d.numpy'] = mod

def detect_device(device=None) -> str:
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def build_trellis(device=None):
    with _LOCK:
        if _STATE.get('pipe') is not None:
            return (_STATE['pipe'], _STATE['device'])
        os.environ.setdefault('TORCH_HOME', '')
        _ensure_repo_on_path()
        TrellisImageTo3DPipeline = _import_image_pipeline_class()
        dev = detect_device(device)
        pipe = TrellisImageTo3DPipeline.from_pretrained(_runtime_pipeline_dir())
        try:
            pipe.to(dev)
        except Exception:
            pass
        _STATE.update(pipe=pipe, device=dev)
        return (pipe, dev)

def _mesh_decode_available() -> bool:
    return os.environ.get('STUDIO_TRELLIS_ENABLE_MESH', '0').lower() in ('1', 'true', 'on', 'yes') and _flexicubes_file_present()

def _runtime_pipeline_dir() -> str:
    base = Path(_weights_dir())
    cfg = json.loads((base / 'pipeline.json').read_text(encoding='utf-8'))
    models = dict(cfg.get('args', {}).get('models', {}))
    if not _mesh_decode_available():
        models.pop('slat_decoder_mesh', None)
    if os.environ.get('STUDIO_TRELLIS_ENABLE_RF', '0').lower() not in ('1', 'true', 'on', 'yes'):
        models.pop('slat_decoder_rf', None)
    models = {k: str(v).replace('\\', '/') if os.path.isabs(str(v)) else str((base / str(v)).resolve()).replace('\\', '/') for k, v in models.items()}
    cfg['args']['models'] = models
    runtime = base / '.studio_pipeline_gaussian'
    runtime.mkdir(parents=True, exist_ok=True)
    out = runtime / 'pipeline.json'
    text = json.dumps(cfg, indent=2)
    if not out.is_file() or out.read_text(encoding='utf-8') != text:
        out.write_text(text, encoding='utf-8')
    return str(runtime)

def unload() -> None:
    with _LOCK:
        _STATE.pop('pipe', None)
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def image_to_glb(object_image_path, *, out_dir, basename, texture='baked', device=None, seed: int=1, ss_steps: int=25, slat_steps: int=25) -> str:
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe, dev = build_trellis(device)
    img = Image.open(str(object_image_path)).convert('RGB')
    _install_utils3d_numpy_compat()
    formats = ['mesh', 'gaussian'] if _mesh_decode_available() else ['gaussian']
    outputs = pipe.run(img, seed=int(seed), sparse_structure_sampler_params={'steps': int(ss_steps)}, slat_sampler_params={'steps': int(slat_steps)}, formats=formats)
    glb_path = str(out / f'{basename}.glb')
    if outputs.get('mesh'):
        try:
            from trellis.utils import postprocessing_utils
            glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95, texture_size=1024, fill_holes=True, verbose=False)
            glb.export(glb_path)
            return glb_path
        except Exception:
            pass
    ply_path = str(out / f'{basename}.ply')
    try:
        outputs['gaussian'][0].save_ply(ply_path)
        return ply_path
    except Exception as exc:
        raise RuntimeError(f'TRELLIS produced no exportable asset for {basename} (mesh export deps missing AND splat save failed: {type(exc).__name__}: {exc})')
