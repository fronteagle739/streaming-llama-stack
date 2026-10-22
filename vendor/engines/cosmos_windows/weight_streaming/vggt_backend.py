# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/vggt_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
_LOCK = threading.Lock()
_STATE: dict = {}

def _weights_dir() -> str:
    env = os.environ.get('STUDIO_VGGT_DIR')
    cands = [env, '', os.path.expanduser('~/models/VGGT-1B'), '', '', os.path.expanduser('~/models/VGGT-1B-Commercial'), '']
    for c in cands:
        if c and Path(c).is_dir() and any(Path(c).glob('*.safetensors')):
            return str(c)
    raise RuntimeError('')

def _ensure_repo_on_path() -> None:
    try:
        import vggt
        return
    except Exception:
        pass
    env = os.environ.get('STUDIO_VGGT_REPO')
    cands = [env, '', os.path.expanduser('~/GitHub/vggt')]
    for c in cands:
        if c and (Path(c) / 'vggt').is_dir():
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    raise RuntimeError('')

def detect_device(device=None) -> str:
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def build_vggt(device=None):
    with _LOCK:
        if _STATE.get('model') is not None:
            return (_STATE['model'], _STATE['device'])
        import torch
        _ensure_repo_on_path()
        from vggt.models.vggt import VGGT
        dev = detect_device(device)
        wdir = _weights_dir()
        model = VGGT()
        sd_file = next(iter(Path(wdir).glob('*.safetensors')))
        model = model.to(dev)
        _load_safetensors_one_tensor(model, sd_file)
        model = model.eval()
        _STATE.update(model=model, device=dev)
        return (model, dev)

def _load_safetensors_one_tensor(model, path: Path) -> None:
    import torch
    from safetensors import safe_open
    dst = model.state_dict()
    with torch.no_grad():
        with safe_open(str(path), framework='pt', device='cpu') as sf:
            for key in sf.keys():
                target = dst.get(key)
                if target is None:
                    continue
                src = sf.get_tensor(key)
                if tuple(src.shape) != tuple(target.shape):
                    continue
                target.copy_(src.to(device=target.device, dtype=target.dtype), non_blocking=True)

def unload() -> None:
    with _LOCK:
        _STATE.pop('model', None)
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def reconstruct(image_paths, *, out_dir, device=None, object_masks=None) -> dict:
    import numpy as np
    import torch
    _ensure_repo_on_path()
    from vggt.utils.load_fn import load_and_preprocess_images
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, dev = build_vggt(device)
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(dev)
    dtype = torch.bfloat16 if str(dev).startswith('cuda') else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if str(dev).startswith('cuda') else 'cpu', dtype=dtype):
            preds = model(images)
    world_pts = _to_np(preds.get('world_points'))
    conf = _to_np(preds.get('world_points_conf'))
    files = []
    scene_ply = str(out / 'scene.ply')
    _save_ply(scene_ply, _flatten_points(world_pts, conf, frame=0))
    files.append(scene_ply)
    bg_ply = None
    if world_pts is not None and world_pts.shape[1] > 1:
        bg_ply = str(out / 'background.ply')
        _save_ply(bg_ply, _flatten_points(world_pts, conf, frame=1))
        files.append(bg_ply)
    floor_points = None
    object_points: dict = {}
    if object_masks and world_pts is not None:
        frame0 = world_pts[0, 0]
        for name, mask in object_masks.items():
            m = _resize_mask(mask, frame0.shape[:2])
            pts = frame0[m]
            if name == 'floor':
                floor_points = pts
            else:
                object_points[name] = pts
    camera = {'intrinsics': _to_list(preds.get('intrinsic')), 'extrinsics': _to_list(preds.get('extrinsic')), 'n_frames': int(images.shape[0]) if hasattr(images, 'shape') else len(image_paths)}
    return {'camera': camera, 'scene_ply': scene_ply, 'bg_ply': bg_ply, 'floor_points': floor_points, 'object_points': object_points, 'pointcloud_files': files}

def _to_np(t):
    if t is None:
        return None
    try:
        return t.detach().float().cpu().numpy()
    except Exception:
        import numpy as np
        return np.asarray(t)

def _to_list(t):
    a = _to_np(t)
    return a.tolist() if a is not None else None

def _flatten_points(world_pts, conf, *, frame: int, conf_quantile: float=0.1):
    import numpy as np
    if world_pts is None:
        return np.zeros((0, 3), np.float32)
    pm = world_pts[0, frame].reshape(-1, 3)
    if conf is not None:
        c = conf[0, frame].reshape(-1)
        keep = c >= np.quantile(c, conf_quantile)
        pm = pm[keep]
    return pm.astype(np.float32)

def _resize_mask(mask, hw):
    import numpy as np
    m = np.asarray(mask)
    if m.shape[:2] == tuple(hw):
        return m.astype(bool)
    try:
        from PIL import Image
        im = Image.fromarray(m.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.NEAREST)
        return np.asarray(im) > 127
    except Exception:
        return m.astype(bool)

def _save_ply(path, points):
    import numpy as np
    pts = np.asarray(points, dtype='<f4').reshape(-1, 3)
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {pts.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode('ascii')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(pts.tobytes())
    return path
