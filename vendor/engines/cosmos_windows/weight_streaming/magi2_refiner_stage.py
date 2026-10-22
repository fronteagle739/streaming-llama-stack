# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_refiner_stage.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import glob
import os
import sys as _sys
import types
from pathlib import Path
from typing import Optional

def load_refiner_config(repo_root: Optional[Path]=None):
    from inference.common.magi2_config import load_config
    repo_root = Path(repo_root or os.environ.get('MAGI2_REPO_ROOT', ''))
    cfg = load_config(str(repo_root / 'configs' / 'magi2_refiner.json'))
    cfg.engine_config.cp_size = 1
    cfg.engine_config.ep_size = 1
    return cfg

def load_refiner_resident(refiner_dir: Path, arch_config, device: str='cuda', emit=None):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from inference.model.magi2_refiner import Transformer as RefinerModel
    from cosmos_windows.weight_streaming.backends import PortableBackend
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    refiner_dir = Path(refiner_dir)
    if not glob.glob(str(refiner_dir / 'model-*-of-*.safetensors')):
        raise FileNotFoundError(f'no refiner shards (model-*-of-*.safetensors) under {refiner_dir}')
    with torch.device(device):
        model = RefinerModel(arch_config)
    from cosmos_windows.weight_streaming.ds_state_load import directstorage_usable, ds_bind_module
    index = build_index(refiner_dir)
    state_keys = set(model.state_dict().keys())
    if directstorage_usable():
        provided = ds_bind_module(model, index, device=device, emit=emit)
        transport = 'directstorage'
    else:
        if _sys.platform == 'win32':
            print('[MAGI2-REFINER] WARNING: DirectStorage unavailable on Windows -> falling back to bounded-pinned host staging (portable tier). Weight path RSS will NOT be ~0.', flush=True)
        transport = 'pinned-portable'
        backend = PortableBackend(pin_memory=True)
        provided = set()
        try:
            for name, ref in index.refs.items():
                if name not in state_keys:
                    continue
                t = backend.read_tensor(ref, device=device, direct=True)
                set_module_tensor_to_device(model, name, device, value=t)
                provided.add(name)
        finally:
            backend.close()
    param_keys = {k for k, _ in model.named_parameters()}
    missing = param_keys - provided
    if missing:
        raise RuntimeError(f'refiner load incomplete: {len(missing)} parameter(s) not found in the shards (e.g. {sorted(missing)[:6]}). Refusing a silent partial load.')
    model.to(device)
    model.eval()
    if emit is not None:
        emit(event='stage_log', stage='refiner_load', message=f'refiner resident on {device} ({len(provided)} tensors via {transport}, ~0 host RSS)', transport=transport)
    return model

def _make_refiner_shim(refiner_model, refiner_data_proxy, eval_config, device: str, dtype):
    from inference.common.offload_utils import OffloadMode
    from inference.pipeline.inference_engine import Magi2InferenceEngine, ZeroSNRDDPMDiscretization
    shim = types.SimpleNamespace()
    shim.magi2_refiner = refiner_model
    shim.magi2_refiner_data_proxy = refiner_data_proxy
    shim.magi2_refiner_sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
    shim._refiner_mode = OffloadMode.GPU
    shim.config = eval_config
    shim.device = device
    shim.dtype = dtype
    shim.audio_latent_dim = 64
    shim.shift = eval_config.shift
    shim.evaluate_magi2_refiner_with_latent = types.MethodType(Magi2InferenceEngine.evaluate_magi2_refiner_with_latent, shim)
    shim._forward_magi2_refiner = types.MethodType(Magi2InferenceEngine._forward_magi2_refiner, shim)
    return shim

def refine_latent(refiner_model, eval_config, context, null_context, preview_latent, audio_latent, refiner_width: int, refiner_height: int, steps: int, device: str='cuda', dtype=None):
    import torch
    from inference.pipeline.refiner_data_proxy import Magi2RefinerDataProxy
    dtype = dtype or torch.bfloat16
    data_proxy = Magi2RefinerDataProxy(eval_config.magi2_refiner_data_proxy_config)
    shim = _make_refiner_shim(refiner_model, data_proxy, eval_config, device, dtype)
    refined, refiner_audio = shim.evaluate_magi2_refiner_with_latent(context=context, context_null=null_context, latent_video=preview_latent, latent_audio=audio_latent, refiner_width=refiner_width, refiner_height=refiner_height, magi2_refiner_num_inference_steps=steps)
    return (refined, refiner_audio)
