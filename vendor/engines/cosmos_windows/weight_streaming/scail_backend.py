# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/scail_backend.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import re
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path
SCAIL2_REPO_DIR = os.environ.get('SCAIL2_REPO_DIR', '')
SCAIL2_CKPT_DIR = os.environ.get('SCAIL2_CKPT_DIR', '')
SCAIL2_SAFETENSORS = os.environ.get('SCAIL2_SAFETENSORS', '')
SCAIL2_STREAM_DTYPE = os.environ.get('SCAIL2_STREAM_DTYPE', 'bf16')
_PIPELINE_CACHE: dict = {}
_SCAIL_BLOCK_RE = re.compile('^blocks\\.(\\d+)\\.')
_ATTN_BACKEND_CHECKED = False

def _ensure_repo_on_path(repo_dir: str | None=None) -> str:
    rd = repo_dir or SCAIL2_REPO_DIR
    if rd and rd not in sys.path:
        sys.path.insert(0, rd)
    return rd

def _sdpa_attention_impl(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None, *, restrict_flash: bool=False):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out_dtype = q.dtype
    dtype = dtype or (torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float16)
    if window_size != (-1, -1):
        raise RuntimeError('SCAIL requested sliding-window attention; install Dao flash_attn for that path.')
    if q_scale is not None:
        q = q * q_scale
    q = q if q.dtype in (torch.float16, torch.bfloat16) else q.to(dtype)
    k = k if k.dtype in (torch.float16, torch.bfloat16) else k.to(dtype)
    v = v if v.dtype in (torch.float16, torch.bfloat16) else v.to(dtype)
    if k.size(2) != q.size(2):
        if q.size(2) % k.size(2) != 0:
            raise RuntimeError(f'SCAIL SDPA head mismatch: q={q.shape} k={k.shape}')
        rep = q.size(2) // k.size(2)
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    b, lq, lk = (q.size(0), q.size(1), k.size(1))
    q_full = torch.full((b,), lq, dtype=torch.int64, device=q.device)
    k_full = torch.full((b,), lk, dtype=torch.int64, device=k.device)
    q_lens_t = q_full if q_lens is None else q_lens.to(q.device, dtype=torch.int64)
    k_lens_t = k_full if k_lens is None else k_lens.to(k.device, dtype=torch.int64)

    def _sdpa(q2, k2, v2):
        if restrict_flash:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(q2, k2, v2, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        return F.scaled_dot_product_attention(q2, k2, v2, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)

    def _run(qi, ki, vi):
        return _sdpa(qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2)).transpose(1, 2).contiguous()
    if torch.equal(q_lens_t, q_full) and torch.equal(k_lens_t, k_full):
        return _run(q, k, v).type(out_dtype)
    outs = []
    for i in range(b):
        qn = int(q_lens_t[i].item())
        kn = int(k_lens_t[i].item())
        oi = _run(q[i:i + 1, :qn], k[i:i + 1, :kn], v[i:i + 1, :kn])
        if qn < lq:
            pad = torch.zeros((1, lq - qn, oi.size(2), oi.size(3)), device=oi.device, dtype=oi.dtype)
            oi = torch.cat([oi, pad], dim=1)
        outs.append(oi)
    return torch.cat(outs, dim=0).type(out_dtype)

def _torch_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    return _sdpa_attention_impl(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, deterministic, dtype, version, restrict_flash=True)

def _torch_sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    return _sdpa_attention_impl(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, deterministic, dtype, version, restrict_flash=False)
_diagnostic_attention_fallback = _torch_sdpa_attention

def _probe_external_flash_attention_runtime() -> tuple[bool, str]:
    code = '\nimport importlib\nimport torch\nfor module_name in ("flash_attn_interface", "flash_attn"):\n    try:\n        mod = importlib.import_module(module_name)\n    except Exception:\n        continue\n    fn = getattr(mod, "flash_attn_varlen_func", None) or getattr(mod, "flash_attn_func", None)\n    if not callable(fn):\n        continue\n    q = torch.randn((32, 4, 64), device="cuda", dtype=torch.float16)\n    k = torch.randn((32, 4, 64), device="cuda", dtype=torch.float16)\n    v = torch.randn((32, 4, 64), device="cuda", dtype=torch.float16)\n    cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)\n    if getattr(fn, "__name__", "").endswith("varlen_func"):\n        out = fn(q, k, v, cu, cu, 32, 32)\n    else:\n        out = fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))\n    if isinstance(out, tuple):\n        out = out[0]\n    torch.cuda.synchronize()\n    print(module_name + "." + getattr(fn, "__name__", "flash_attn"), tuple(out.shape), out.dtype)\n    raise SystemExit(0)\nraise SystemExit("no callable flash attention backend")\n'
    try:
        proc = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=30)
    except Exception as exc:
        return (False, f'subprocess probe failed: {type(exc).__name__}: {exc}')
    if proc.returncode == 0:
        return (True, (proc.stdout or '').strip())
    detail = (proc.stderr or proc.stdout or '').strip().replace('\n', ' | ')
    return (False, detail[:700])

def _strict_flash_required() -> bool:
    if os.environ.get('SCAIL2_ALLOW_ATTN_FALLBACK', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return False
    return os.environ.get('SCAIL2_REQUIRE_FLASH_ATTENTION', '').strip().lower() in ('1', 'true', 'yes', 'on')

def _install_scail_attention(fn) -> None:
    import wan.modules.attention as attn
    import wan.modules.clip as clip_mod
    import wan.modules.model_scail2 as scail2_mod
    attn.flash_attention = fn
    clip_mod.flash_attention = fn
    scail2_mod.flash_attention = fn

def _require_flash_attention() -> None:
    global _ATTN_BACKEND_CHECKED
    if _ATTN_BACKEND_CHECKED:
        return
    strict = _strict_flash_required()
    try:
        import wan.modules.attention as attn
    except Exception as exc:
        raise RuntimeError(f'failed to import SCAIL attention module: {exc}') from exc
    if getattr(attn, 'FLASH_ATTN_2_AVAILABLE', False) or getattr(attn, 'FLASH_ATTN_3_AVAILABLE', False):
        ok, detail = _probe_external_flash_attention_runtime()
        if ok:
            print(f'[SCAIL-STREAM] FlashAttention runtime probe OK: {detail}', flush=True)
            _ATTN_BACKEND_CHECKED = True
            return
        fa_detail = f'importable FlashAttention present but its CUDA probe failed: {detail}'
    else:
        fa_detail = 'no importable Dao flash_attn/flash_attn_interface backend'
    torch_detail = 'torch FLASH_ATTENTION not probed'
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        device = torch.device('cuda')
        q = torch.randn((1, 8, 2, 64), device=device, dtype=torch.float16)
        _ = _torch_flash_attention(q, q, q)
        torch.cuda.synchronize(device)
        _install_scail_attention(_torch_flash_attention)
        print('[SCAIL-STREAM] Using PyTorch fused FLASH_ATTENTION kernel for SCAIL attention.', flush=True)
        _ATTN_BACKEND_CHECKED = True
        return
    except Exception as flash_exc:
        torch_detail = f'torch fused FLASH_ATTENTION probe failed: {flash_exc}'
    if strict:
        raise RuntimeError(f'SCAIL-2 strict mode (SCAIL2_REQUIRE_FLASH_ATTENTION=1): no executable FlashAttention kernel for this GPU. {fa_detail}; {torch_detail}. Install a FlashAttention build with SM120/Blackwell kernels, or unset SCAIL2_REQUIRE_FLASH_ATTENTION to run on the correct-but-slower SDPA fallback.')
    warnings.warn(f'SCAIL-2: no SM120 FlashAttention kernel available; using the PyTorch SDPA fallback (cuDNN/memory-efficient/math). Output is correct but slower, and this run does NOT count as a FlashAttention production proof. {fa_detail}; {torch_detail}.', RuntimeWarning, stacklevel=2)
    _install_scail_attention(_torch_sdpa_attention)
    print('[SCAIL-STREAM] real FlashAttention unavailable; using PyTorch SDPA fallback (slower, correct).', flush=True)
    _ATTN_BACKEND_CHECKED = True

def _scail2_checkpoint_path(ckpt_dir: str | Path) -> Path:
    return Path(ckpt_dir) / 'model' / '1' / 'fsdp2_rank_0000_checkpoint.pt'

def _preflight_resident_ram() -> None:
    from . import ram_guard
    need = float(os.environ.get('SCAIL2_MIN_FREE_RAM_GB', '45'))
    ram_guard.preflight_host_ram(need, 'SCAIL-2 resident load (stream_weights=False) — full ~28 GB 14B DiT + UMT5 staged in host RAM', skip_env='SCAIL2_SKIP_RAM_PREFLIGHT', force_env='SCAIL2_FORCE_RESIDENT', advice='Use stream_weights=True (the default) to stream the 14B DiT per-block off NVMe instead of staging the whole model in host RAM.')

def ensure_scail2_safetensors(*, repo_dir=None, ckpt_dir=None, scail_safetensors=None, dtype: str | None=None) -> str:
    rd = _ensure_repo_on_path(repo_dir)
    ckpt = Path(ckpt_dir or SCAIL2_CKPT_DIR)
    saf = Path(scail_safetensors or SCAIL2_SAFETENSORS)
    if saf.is_file():
        return str(saf)
    pt_file = _scail2_checkpoint_path(ckpt)
    if not pt_file.is_file():
        raise FileNotFoundError(f'Neither converted SCAIL-2 weights nor source FSDP2 checkpoint exists: {saf} / {pt_file}')
    from .convert_scail2_to_safetensors import convert_scail2_checkpoint
    return str(convert_scail2_checkpoint(pt_file, saf, scail_repo=rd, dtype=dtype or SCAIL2_STREAM_DTYPE))

def scail_group_by_layer(idx):
    layers, globs = ({}, [])
    for name, ref in idx.refs.items():
        m = _SCAIL_BLOCK_RE.search(name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(ref)
        else:
            globs.append(ref)
    return (layers, globs)

class _SCAILGroupedIndex:

    def __init__(self, base_idx):
        layers, _ = scail_group_by_layer(base_idx)
        self._layers = layers
        self.refs = {r.name: r for refs in layers.values() for r in refs}

    def group_by_layer(self, component: str=''):
        return (self._layers, [])

    def names(self):
        return list(self.refs.keys())

def _scail_dtype(name: str):
    import torch
    key = (name or 'bf16').lower()
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(key, torch.bfloat16)

def build_streamed_scail2_model(scail_safetensors: str, scail_config_path: str, *, device: str='cuda', dtype: str | None=None, n_gpu_buffers: int=3):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from .backends import PortableBackend
    from .ds_backend import DirectStorageStreamer
    from .safetensors_index import build_index
    from wan.modules.model_scail2 import SCAIL2Model
    _require_flash_attention()
    stream_dtype = _scail_dtype(dtype or SCAIL2_STREAM_DTYPE)
    with init_empty_weights(include_buffers=False):
        model = SCAIL2Model.from_config(scail_config_path)
    model = model.to(stream_dtype)
    model.eval().requires_grad_(False)
    idx = build_index(scail_safetensors)
    layer_refs, global_refs = scail_group_by_layer(idx)
    n_blocks = len(layer_refs)
    if n_blocks == 0:
        raise RuntimeError(f'SCAIL stream grouping found no `blocks.N.` tensors in {scail_safetensors}')
    sd_keys = set(model.state_dict().keys())
    ckpt_keys = {r.name for refs in layer_refs.values() for r in refs} | {r.name for r in global_refs}
    missing = sorted(sd_keys - ckpt_keys)
    extra = sorted(ckpt_keys - sd_keys)
    if missing or extra:
        raise RuntimeError(f'SCAIL checkpoint/model key mismatch: missing={missing[:8]} extra={extra[:8]} (counts {len(missing)}/{len(extra)})')
    pb = PortableBackend(pin_memory=False)

    def _cast(t):
        if t.is_floating_point() and t.dtype != stream_dtype:
            return t.to(stream_dtype)
        return t
    for ref in global_refs:
        value = _cast(pb.read_tensor(ref, device))
        set_module_tensor_to_device(model, ref.name, device, value=value)
        del value
    if getattr(model, 'freqs', None) is not None and str(model.freqs.device) != device:
        model.freqs = model.freqs.to(device)
    for name, buf in list(model.named_buffers()):
        if buf.device.type != 'meta' and str(buf.device) != device:
            set_module_tensor_to_device(model, name, device, value=buf.to(device))
    streamer = DirectStorageStreamer(_SCAILGroupedIndex(idx), component='', n_gpu_buffers=n_gpu_buffers, device=device)
    from .ds_backend import enforce_ds_mode
    enforce_ds_mode(streamer, 'SCAIL', 'SCAIL')
    print(f'[SCAIL-STREAM] {n_blocks} blocks streamed, {len(global_refs)} globals resident, dtype={stream_dtype}, mode={streamer.mode}, {n_gpu_buffers} GPU buf(s)', flush=True)
    layer_paths = {il: [r.name for r in refs if r.name in sd_keys] for il, refs in layer_refs.items()}

    def make_bind(il):

        def bind():
            streamed = streamer.wait_layer(il)
            for name, t in streamed.items():
                if name in sd_keys:
                    set_module_tensor_to_device(model, name, device, value=_cast(t))
        return bind

    def make_evict(il):

        def evict():
            for name in layer_paths[il]:
                set_module_tensor_to_device(model, name, 'meta')
            streamer.release_layer(il)
        return evict
    for il, block in enumerate(model.blocks):
        cls_forward = type(block).forward
        bind, evict = (make_bind(il), make_evict(il))

        def wrapped(x, *args, _block=block, _fwd=cls_forward, _bind=bind, _evict=evict, **kwargs):
            _bind()
            try:
                return _fwd(_block, x, *args, **kwargs)
            finally:
                _evict()
        block.forward = wrapped
        block._scail_raw_forward = cls_forward
        block._scail_bind = bind
        block._scail_evict = evict
    model.register_forward_pre_hook(lambda m, a, kw=None: streamer.new_pass())
    model._scail_streamer = streamer
    return (model, streamer)

def _build_streamed_scail2_pipeline(*, repo_dir, ckpt_dir, scail_safetensors, scail_config_path, model, device_id, lora_path=None, lora_alpha=None, n_gpu_buffers: int=3):
    if lora_path:
        raise RuntimeError('SCAIL streamed LoRA fusion is not wired yet; prove the base model first.')
    import torch
    import wan
    from wan.configs import SCAIL_CONFIGS
    _require_flash_attention()
    from wan.distributed.fsdp import shard_model
    from wan.modules.clip import CLIPModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    cfg = SCAIL_CONFIGS[model]
    pipe = wan.SCAIL2Pipeline.__new__(wan.SCAIL2Pipeline)
    pipe.device = torch.device(f'cuda:{device_id}')
    pipe.config = cfg
    pipe.rank = 0
    pipe.use_usp = False
    pipe.t5_cpu = False
    pipe.lora_path = lora_path
    pipe.lora_alpha = lora_alpha
    pipe.num_train_timesteps = cfg.num_train_timesteps
    pipe.param_dtype = cfg.param_dtype
    shard_fn = partial(shard_model, device_id=device_id)
    _orig_torch_load = torch.load

    def _torch_load_to_gpu(*a, **k):
        k['map_location'] = pipe.device
        return _orig_torch_load(*a, **k)
    torch.load = _torch_load_to_gpu
    try:
        pipe.text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=pipe.device, checkpoint_path=os.path.join(ckpt_dir, cfg.t5_checkpoint), tokenizer_path=os.path.join(ckpt_dir, cfg.t5_tokenizer), shard_fn=None)
        pipe.vae_stride = cfg.vae_stride
        pipe.patch_size = cfg.patch_size
        pipe.vae = WanVAE(vae_pth=os.path.join(ckpt_dir, cfg.vae_checkpoint), device=pipe.device)
        pipe.clip = CLIPModel(dtype=cfg.clip_dtype, device=pipe.device, checkpoint_path=os.path.join(ckpt_dir, cfg.clip_checkpoint), tokenizer_path=os.path.join(ckpt_dir, cfg.clip_tokenizer))
    finally:
        torch.load = _orig_torch_load
    pipe.model, streamer = build_streamed_scail2_model(scail_safetensors, scail_config_path, device=f'cuda:{device_id}', dtype=SCAIL2_STREAM_DTYPE, n_gpu_buffers=n_gpu_buffers)
    pipe.sp_size = 1
    pipe.sample_neg_prompt = cfg.sample_neg_prompt
    pipe._scail_streamed = True
    pipe._scail_streamer = streamer
    return pipe

def load_scail2_pipeline(*, repo_dir=None, ckpt_dir=None, scail_safetensors=None, scail_config_path=None, model='SCAIL-14B', device_id=0, lora_path=None, lora_alpha=None, stream_weights=True, n_gpu_buffers=3):
    rd = _ensure_repo_on_path(repo_dir)
    import wan
    from wan.configs import SCAIL_CONFIGS
    _require_flash_attention()
    ckpt = ckpt_dir or SCAIL2_CKPT_DIR
    saf = ensure_scail2_safetensors(repo_dir=rd, ckpt_dir=ckpt, scail_safetensors=scail_safetensors or SCAIL2_SAFETENSORS)
    if model not in SCAIL_CONFIGS:
        raise ValueError(f'Unknown SCAIL model {model!r}; valid: {list(SCAIL_CONFIGS)}')
    cfg = SCAIL_CONFIGS[model]
    if scail_config_path is None:
        cfg_name = 'config-14b.json' if '14B' in model else 'config-1.3b.json'
        scail_config_path = str(Path(rd) / 'configs' / cfg_name)
    key = (rd, ckpt, saf, scail_config_path, model, device_id, lora_path, lora_alpha, bool(stream_weights), int(n_gpu_buffers))
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    if not Path(saf).is_file():
        raise FileNotFoundError(f"Converted SCAIL-2 weights not found: {saf}. Run the repo's convert.py first (FSDP2 .pt -> .safetensors).")
    if stream_weights:
        pipe = _build_streamed_scail2_pipeline(repo_dir=rd, ckpt_dir=ckpt, scail_safetensors=saf, scail_config_path=scail_config_path, model=model, device_id=device_id, lora_path=lora_path, lora_alpha=lora_alpha, n_gpu_buffers=n_gpu_buffers)
    else:
        _preflight_resident_ram()
        pipe = wan.SCAIL2Pipeline(config=cfg, checkpoint_dir=ckpt, scail_safetensors_path=saf, scail_config_path=scail_config_path, device_id=device_id, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False, lora_path=lora_path, lora_alpha=lora_alpha)
    _PIPELINE_CACHE[key] = pipe
    return pipe

def unload_scail2_pipeline() -> bool:
    if not _PIPELINE_CACHE:
        return False
    for pipe in list(_PIPELINE_CACHE.values()):
        try:
            streamer = getattr(pipe, '_scail_streamer', None)
            close = getattr(streamer, 'close', None)
            if callable(close):
                close()
        except Exception:
            pass
        for attr in ('model', 'text_encoder', 'clip', 'vae', '_scail_streamer'):
            try:
                setattr(pipe, attr, None)
            except Exception:
                pass
    _PIPELINE_CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True

def _load_ref_image(path, th, tw, device, resize_for_rectangle_crop, load_image_to_tensor_chw_normalized):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    uncropped = load_image_to_tensor_chw_normalized(img).to(device)
    return resize_for_rectangle_crop(uncropped, (th, tw), reshape_mode='center').squeeze(0)

def scail2_animate(*, prompt, image_path, pose_video_path, ref_mask_path=None, driving_mask_path=None, replace_flag=False, target_h=None, target_w=None, segment_len=81, segment_overlap=5, shift=5.0, sample_solver='unipc', sampling_steps=40, guide_scale=5.0, negative_prompt=None, seed=-1, offload_model=True, additional_ref_image_paths=None, additional_ref_mask_image_paths=None, repo_dir=None, ckpt_dir=None, scail_safetensors=None, scail_config_path=None, model='SCAIL-14B', device_id=0, lora_path=None, lora_alpha=None, stream_weights=True, n_gpu_buffers=3, progress=None):
    import torch
    from einops import rearrange
    rd = _ensure_repo_on_path(repo_dir)
    from wan.utils.scail_utils import load_image_to_tensor_chw_normalized, load_video_for_pose_sample, resize_for_rectangle_crop
    pipe = load_scail2_pipeline(repo_dir=rd, ckpt_dir=ckpt_dir, scail_safetensors=scail_safetensors, scail_config_path=scail_config_path, model=model, device_id=device_id, lora_path=lora_path, lora_alpha=lora_alpha, stream_weights=stream_weights, n_gpu_buffers=n_gpu_buffers)
    device = torch.device(f'cuda:{device_id}')
    from PIL import Image
    img0 = load_image_to_tensor_chw_normalized(Image.open(image_path).convert('RGB')).to(device)
    _, _, h, w = img0.shape
    th = int(target_h or h)
    tw = int(target_w or w)
    if h < w and th > tw or (h > w and th < tw):
        th, tw = (tw, th)
    img_t = resize_for_rectangle_crop(img0, (th, tw), reshape_mode='center').squeeze(0)
    if ref_mask_path:
        ref_mask = _load_ref_image(ref_mask_path, th, tw, device, resize_for_rectangle_crop, load_image_to_tensor_chw_normalized)
    else:
        ref_mask = torch.ones(3, th, tw, device=device, dtype=img_t.dtype)
    pose = load_video_for_pose_sample(pose_video_path).permute(0, 3, 1, 2)
    pose = resize_for_rectangle_crop(pose, (th, tw), reshape_mode='center')
    pose = (pose - 127.5) / 127.5
    if driving_mask_path:
        dm = load_video_for_pose_sample(driving_mask_path).permute(0, 3, 1, 2)
        dm = resize_for_rectangle_crop(dm, (th, tw), reshape_mode='center')
        dm = (dm - 127.5) / 127.5
        driving_mask = rearrange(dm, 't c h w -> c t h w')
    else:
        driving_mask = torch.ones(3, pose.shape[0], th, tw, device=device, dtype=pose.dtype)
    add_kw = {}
    if additional_ref_image_paths:
        if not additional_ref_mask_image_paths or len(additional_ref_image_paths) != len(additional_ref_mask_image_paths):
            raise ValueError('additional_ref_image_paths and additional_ref_mask_image_paths must be provided together and have equal length.')
        add_imgs, add_masks = ([], [])
        for ip, mp in zip(additional_ref_image_paths, additional_ref_mask_image_paths):
            add_imgs.append(_load_ref_image(ip, th, tw, device, resize_for_rectangle_crop, load_image_to_tensor_chw_normalized))
            add_masks.append(_load_ref_image(mp, th, tw, device, resize_for_rectangle_crop, load_image_to_tensor_chw_normalized))
        add_kw['additional_ref_imgs'] = add_imgs
        add_kw['additional_ref_mask_imgs'] = add_masks
    _scail_mod = None
    _orig_tqdm = None
    if progress is not None:
        try:
            import wan.scail as _scail_mod
            _orig_tqdm = _scail_mod.tqdm
            _cb = progress

            def _prog_tqdm(iterable=None, *a, **k):
                total = k.get('total')
                if total is None and hasattr(iterable, '__len__'):
                    try:
                        total = len(iterable)
                    except Exception:
                        total = None
                i = 0
                for x in iterable if iterable is not None else []:
                    yield x
                    i += 1
                    _cb(i, total or i)
            _scail_mod.tqdm = _prog_tqdm
        except Exception:
            _scail_mod = None
            _orig_tqdm = None
    try:
        video = pipe.generate(prompt, img_t, ref_mask_img=ref_mask, pose_video=pose, driving_mask_video=driving_mask, replace_flag=replace_flag, shift=shift, sample_solver=sample_solver, segment_len=segment_len, segment_overlap=segment_overlap, sampling_steps=sampling_steps, guide_scale=guide_scale, n_prompt=negative_prompt, seed=seed, offload_model=False if getattr(pipe, '_scail_streamed', False) else offload_model, **add_kw)
    finally:
        if _scail_mod is not None and _orig_tqdm is not None:
            _scail_mod.tqdm = _orig_tqdm
    v = ((video.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().byte().cpu()
    v = v.permute(1, 2, 3, 0).numpy()
    return [Image.fromarray(frame) for frame in v]
