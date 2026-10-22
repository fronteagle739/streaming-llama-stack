# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_video_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import uuid

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def _base_identity(model):
    from safetensors import safe_open
    root = Path(model).resolve()
    files = [root] if root.is_file() else sorted(root.rglob('*.safetensors'))
    configs = sorted(root.rglob('*.json')) if root.is_dir() else []
    if len(files) + len(configs) > 1024:
        raise ValueError('Base directory exceeds the bounded component file count')
    if not files:
        raise ValueError('Base model needs local Safetensors weights')
    result = []
    for path in files:
        stat = path.stat()
        with path.open('rb') as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError('Invalid base checkpoint header')
            length = struct.unpack('<Q', raw_length)[0]
            if not 0 < length <= 64 * 1024 * 1024 or length + 8 > stat.st_size:
                raise ValueError('Invalid base checkpoint header')
            header = handle.read(length)
        with safe_open(path, framework='numpy') as reader:
            elements = [math.prod(reader.get_slice(name).get_shape()) for name in reader.keys()]
        result.append({'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'elements': sum(elements), 'largest_tensor': max(elements, default=0), 'sha256': _sha(path), 'header_sha256': hashlib.sha256(header).hexdigest()})
    result.extend(({'path': str(path.resolve()), 'sha256': _sha(path)} for path in configs))
    return result

def _dataset(target_dir, prompt):
    root = Path(target_dir).resolve()
    manifest = root / 'manifest.json' if root.is_dir() else root
    rows = []
    if manifest.is_file() and manifest.suffix.lower() in {'.json', '.jsonl'}:
        if manifest.stat().st_size > 1024 * 1024:
            raise ValueError('Dataset manifest exceeds 1MB')
        text = manifest.read_text(encoding='utf-8')
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()] if manifest.suffix == '.jsonl' else json.loads(text)
        rows = parsed.get('items', []) if isinstance(parsed, dict) else parsed
        base = manifest.parent
    elif root.is_dir():
        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.mov', '.mkv', '.avi'}
        rows = [{'video_path': str(path)} for path in sorted(root.iterdir()) if path.suffix.lower() in extensions]
        base = root
    else:
        raise ValueError('Dataset must be a readable local folder or JSON manifest')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError('Dataset needs between 1 and 256 real examples')
    examples = []
    total_bytes, total_files = (0, 0)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each dataset row must be an object')
        if row.get('included', True) is False:
            continue
        raw = row.get('video_path') or row.get('video') or row.get('image_path') or row.get('image') or row.get('media')
        sequence = row.get('frames')
        if sequence is not None and (not isinstance(sequence, list) or not 1 <= len(sequence) <= 4096):
            raise ValueError('A frame sequence needs between 1 and 4096 local image paths')
        paths = sequence if sequence is not None else [raw]
        media = []
        for value in paths:
            if not isinstance(value, str) or '://' in value:
                raise ValueError('Dataset media must be local files')
            path = Path(value)
            path = path if path.is_absolute() else base / path
            if not path.is_file() or path.stat().st_size > 2 * 1024 ** 3:
                raise ValueError('Dataset media is missing or exceeds the 2GB per-example bound')
            media.append(path.resolve())
            total_bytes += path.stat().st_size
            total_files += 1
            if total_bytes > 2 * 1024 ** 3 or total_files > 4096:
                raise ValueError('Dataset exceeds 2GB or 4096 selected media files')
        caption = str(row.get('caption') or row.get('prompt') or '').strip()
        if not caption and len(media) == 1 and media[0].with_suffix('.txt').is_file():
            if media[0].with_suffix('.txt').stat().st_size > 128 * 1024:
                raise ValueError('Dataset caption file exceeds 128KB')
            caption = media[0].with_suffix('.txt').read_text(encoding='utf-8').strip()
        caption = caption or str(prompt or '').strip()
        if not caption:
            raise ValueError('Every example needs a real caption or an explicit fallback prompt')
        if len(caption) > 32768:
            raise ValueError('Dataset caption exceeds the bounded text size')
        examples.append({'media': media, 'caption': caption, 'sha256': [_sha(path) for path in media]})
    if not examples:
        raise ValueError('Dataset contains no included real examples')
    return examples

def _frames(example, count, width, height, fps):
    from PIL import Image, ImageOps
    paths = example['media']
    frames = []

    def selected_indexes(length):
        if not length or (length < count and length != 1):
            raise ValueError('Dataset clip is shorter than the requested temporal window')
        return [round(index * (length - 1) / max(count - 1, 1)) for index in range(count)]
    if len(paths) > 1:
        for index in selected_indexes(len(paths)):
            path = paths[index]
            with Image.open(path) as image:
                frames.append(ImageOps.fit(image.convert('RGB'), (width, height)))
    elif paths[0].suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi'}:
        import shutil
        import subprocess
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise ValueError('Video dataset decoding requires an installed FFmpeg')
        command = [executable, '-nostdin', '-v', 'error', '-i', str(paths[0]), '-an', '-vf', f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}', '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        if result.returncode:
            raise ValueError('FFmpeg could not decode the selected dataset clip')
        size = width * height * 3
        if len(result.stdout) % size:
            raise ValueError('Dataset video decoder returned an incomplete frame')
        frames = [Image.frombytes('RGB', (width, height), result.stdout[index:index + size]) for index in range(0, len(result.stdout), size)]
    else:
        with Image.open(paths[0]) as image:
            length = getattr(image, 'n_frames', 1)
            if length > 4096:
                raise ValueError('Animated dataset image exceeds the frame bound')
            for index in selected_indexes(length):
                image.seek(index)
                frames.append(ImageOps.fit(image.convert('RGB'), (width, height)))
    if not frames:
        raise ValueError('Dataset example contains no decodable frame')
    if len(frames) < count and len(frames) != 1:
        raise ValueError('Dataset clip is shorter than the requested temporal window')
    indexes = selected_indexes(len(frames))
    return [ImageOps.fit(frames[index], (width, height)) for index in indexes]

def _wan_pipeline(model, device, dtype, support_files, streamed, save_dir):
    from diffusers import WanPipeline
    if Path(model).is_dir():
        if streamed:
            raise ValueError('Streamed Wan dataset training requires the local single-file DiT checkpoint')
        pipe = WanPipeline.from_pretrained(model, torch_dtype=dtype, local_files_only=True)
        if pipe.transformer_2 is not None:
            raise ValueError('Select one Wan expert checkpoint for training')
        pipe.to(device)
        return (pipe, None)
    return _raw_wan_pipeline(model, device, dtype, support_files, save_dir, streamed)

def _ltx_pipeline(model, device, dtype, support_files, streamed, save_dir, load_vocoder=False):
    from diffusers import LTX2Pipeline
    from .weight_streaming import ltx_backend as backend
    if Path(model).is_dir():
        if streamed:
            raise ValueError('Streamed LTX training requires the local single-file transformer bundle')
        return (LTX2Pipeline.from_pretrained(model, torch_dtype=dtype, local_files_only=True).to(device), None)
    components = support_files.get('components_repo')
    if not components or not Path(components).is_dir():
        raise ValueError('Single-file LTX training requires a complete local components directory')
    for component in ('vae', 'audio_vae', 'connectors', 'scheduler'):
        filename = 'scheduler_config.json' if component == 'scheduler' else 'config.json'
        if not (Path(components) / component / filename).is_file():
            raise ValueError(f'Local LTX {component} configuration is missing')
    text_path = support_files.get('text_encoder') or str(Path(components) / 'text_encoder')
    tokenizer_path = support_files.get('tokenizer') or str(Path(components) / 'tokenizer')
    if not Path(text_path).is_dir() or not Path(tokenizer_path).is_dir():
        raise ValueError('LTX needs readable local caption encoder and tokenizer directories')
    streamer = None
    try:
        if streamed:
            transformer, streamer = backend.build_streamed_backward_ltx(model, device=device, n_gpu_buffers=1, config_dir=support_files.get('transformer_config') or Path(components) / 'transformer')
            if getattr(streamer, 'mode', None) != 3:
                raise ValueError('CUDA LTX training requires actual DirectStorage mode 3')
        else:
            transformer, _ = backend._load_meta_ltx(model, config_dir=support_files.get('transformer_config') or Path(components) / 'transformer')
            transformer = backend._load_ltx23_mapped_module(transformer, [model], backend._remap_key, device=device, dtype=dtype, label='training transformer')
        parts = backend.load_local_ltx23_runtime_components(model, components, video_vae_file=support_files.get('vae') or model, audio_vocoder_file=support_files.get('audio_vae') or model, text_encoder_dir=text_path, tokenizer_dir=tokenizer_path, device=device, dtype=dtype, load_vocoder=load_vocoder)
        parts['connectors'] = backend.load_local_ltx23_connectors(model, components, device=device, dtype=dtype)
        parts['text_encoder'] = backend.build_streamed_ltx23_text_encoder(text_path, device=device, dtype=dtype) if str(device).startswith('cuda') else backend.load_local_ltx23_text_encoder(text_path, device=device, dtype=dtype)
        parts['transformer'] = transformer
        return (LTX2Pipeline(**parts), streamer)
    except BaseException:
        if streamer is not None:
            streamer.close()
        raise

def train_dataset(*, model, target_dir, save_dir, family='wan', device=None, steps=300, rank=16, lr=0.0001, checkpoint_every=50, frames=9, width=256, height=256, fps=16, seed=3407, max_sequence_length=226, resume='', prompt='', optimizer='adamw', lr_phases='', support_files=None, streamed=False, progress=None, should_cancel=None):
    import torch
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from .flow_match_loss import flow_match_loss, rectified_flow_xt, sample_sigma_t
    from .optim_factory import build_optimizer, MultiPhaseLR
    if family not in {'wan', 'ltx'}:
        raise ValueError('Unsupported temporal trainer family')
    if not 1 <= max_sequence_length <= 512:
        raise ValueError('Caption token limit must be between 1 and 512')
    if not (1 <= steps <= 200000 and 1 <= rank <= 256 and (0 < lr <= 1) and math.isfinite(lr) and (1 <= frames <= 241) and (16 <= width <= 1024) and (16 <= height <= 1024) and (width % 16 == height % 16 == 0) and ((frames - 1) % 4 == 0) and (0 < fps <= 120) and math.isfinite(fps) and (0 <= checkpoint_every <= 200000)):
        raise ValueError('Invalid bounded temporal training settings')
    if family == 'ltx' and ((frames - 1) % 8 or width % 32 or height % 32 or (min(width, height) < 64)):
        raise ValueError('LTX needs 8k+1 frames and canvas sides at least 64 and divisible by 32')
    temporal, spatial = (4, 16) if family == 'wan' else (8, 32)
    if ((frames - 1) // temporal + 1) * (width // spatial) * (height // spatial) > 16384:
        raise ValueError('Training window exceeds 16384 temporal tokens; reduce frames or canvas size')
    examples = _dataset(target_dir, prompt)
    base = _base_identity(model)
    for key, value in (support_files or {}).items():
        path = Path(value)
        if not path.exists():
            raise ValueError(f'Local support file is missing: {key}')
        if path.is_file():
            records = _base_identity(path) if path.suffix == '.safetensors' else [{'path': str(path.resolve()), 'sha256': _sha(path)}]
        else:
            weights = sorted(path.rglob('*.safetensors'))
            configs = sorted(path.rglob('*.json')) + sorted(path.glob('*.model'))
            if len(weights) + len(configs) > 1024:
                raise ValueError('Support directory exceeds the bounded component file count')
            records = [record for weight in weights for record in _base_identity(weight)]
            records.extend(({'path': str(config.resolve()), 'sha256': _sha(config)} for config in configs))
        base.extend(({'support': key, **record} for record in records))
    identity = {'family': family, 'model': str(Path(model).resolve()), 'base': base, 'dataset': [{'paths': [str(path) for path in example['media']], 'sha256': example['sha256'], 'caption': example['caption']} for example in examples], 'rank': rank, 'lr': lr, 'frames': frames, 'width': width, 'height': height, 'fps': fps, 'seed': seed, 'max_sequence_length': max_sequence_length, 'optimizer': optimizer, 'lr_phases': lr_phases, 'support_files': support_files or {}}
    out = Path(save_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resume_meta = None
    if resume:
        checkpoint = Path(resume).resolve()
        resume_meta = json.loads((checkpoint / 'checkpoint.json').read_text(encoding='utf-8'))
        if resume_meta['identity'] != identity:
            raise ValueError('Checkpoint base, dataset or training identity changed')
        for filename, checksum in resume_meta['files'].items():
            if filename not in {'pytorch_lora_weights.safetensors', 'training_state.pt'} or _sha(checkpoint / filename) != checksum:
                raise ValueError('Checkpoint file identity does not match its receipt')
    elif (out / 'pytorch_lora_weights.safetensors').exists():
        raise ValueError('Output already contains an adapter; choose a new output or explicitly resume')
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if streamed and (not str(device).startswith('cuda')):
        raise ValueError('DirectStorage streamed training requires Windows CUDA; native Mac can use resident training')
    dtype = torch.float32 if str(device) == 'cpu' else torch.bfloat16
    if should_cancel and should_cancel():
        return {'status': 'cancelled', 'step': 0, 'checkpoint': resume or None}
    torch.manual_seed(seed)
    _check_capacity(base, device, streamed)
    factory = _wan_pipeline if family == 'wan' else _ltx_pipeline
    pipe, streamer = factory(model, device, dtype, support_files or {}, streamed, out)
    step, last_checkpoint, status = (0, None, 'cancelled')
    tf = pipe.transformer
    try:
        if family == 'wan' and tf.config.in_channels != 16:
            raise ValueError('Wan dataset trainer currently needs a T2V checkpoint with 16 latent channels')
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.requires_grad_(False).eval()
        pipe.text_encoder.requires_grad_(False).eval()
        if family == 'ltx':
            pipe.connectors.requires_grad_(False).eval()
        prepared, cached_bytes = ([], 0)
        with torch.no_grad():
            for example in examples:
                if should_cancel and should_cancel():
                    return {'status': 'cancelled', 'step': 0, 'checkpoint': resume or None}
                pixels = pipe.video_processor.preprocess_video(_frames(example, frames, width, height, fps), height, width)
                latent = pipe.vae.encode(pixels.to(device=device, dtype=pipe.vae.dtype)).latent_dist.mode()
                if family == 'wan':
                    mean = torch.tensor(pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
                    std = torch.tensor(pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
                    clean = ((latent - mean) / std).float().cpu()
                    embeds = pipe.encode_prompt(example['caption'], do_classifier_free_guidance=False, max_sequence_length=max_sequence_length, device=device, dtype=dtype)[0].detach().cpu()
                    condition_bytes = embeds.numel() * embeds.element_size()
                else:
                    clean = pipe.prepare_latents(latents=latent, noise_scale=0, device=device, dtype=torch.float32).cpu()
                    encoded = pipe.encode_prompt(example['caption'], do_classifier_free_guidance=False, max_sequence_length=max_sequence_length, device=device, dtype=dtype)
                    video, audio, mask = pipe.connectors(encoded[0], encoded[1], padding_side='left')
                    embeds = {'video': video.detach().cpu(), 'audio': audio.detach().cpu(), 'mask': mask.detach().cpu(), 'shape': tuple(latent.shape[2:])}
                    condition_bytes = sum((value.numel() * value.element_size() for value in embeds.values() if isinstance(value, torch.Tensor)))
                cached_bytes += clean.numel() * clean.element_size() + condition_bytes
                if cached_bytes > 512 * 1024 ** 2:
                    raise ValueError('Prepared temporal dataset exceeds the 512MB cache bound')
                prepared.append((clean, embeds))
        reader = getattr(pipe.text_encoder, '_sb_reader', None)
        if reader is not None:
            reader.close()
        for auxiliary in getattr(pipe, '_training_aux_streamers', []):
            auxiliary.close()
        pipe.text_encoder = None
        pipe.vae = None
        if family == 'ltx':
            pipe.connectors = None
            pipe.audio_vae = None
            pipe.vocoder = None
        __import__('gc').collect()
        tf.requires_grad_(False).eval()
        targets = ['to_q', 'to_k', 'to_v', 'to_out.0']
        if family == 'ltx':
            targets = [name for name, module in tf.named_modules() if isinstance(module, torch.nn.Linear) and any((name.endswith('.' + target) for target in targets)) and ('.attn1.' in name or '.attn2.' in name) and ('audio' not in name)]
        tf.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=targets))
        if streamer is not None:
            from .weight_streaming.build_streamed_transformer import materialize_meta_lora
            materialize_meta_lora(tf, device=device)
        trainable = [parameter for parameter in tf.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('Wan adapter attached to no trainable modules')
        for parameter in trainable:
            parameter.data = parameter.data.float()
        opt = build_optimizer(trainable, optimizer, lr=lr)
        schedule = MultiPhaseLR(opt, lr_phases, default_lr=lr)
        random = torch.Generator(device='cpu').manual_seed(seed)
        if resume_meta is not None:
            adapter = load_file(str(checkpoint / 'pytorch_lora_weights.safetensors'))
            state = {key.removeprefix('transformer.'): value for key, value in adapter.items()}
            if set(state) != set(get_peft_model_state_dict(tf)):
                raise ValueError('Checkpoint adapter target schema changed')
            set_peft_model_state_dict(tf, state)
            saved = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
            opt.load_state_dict(saved['optimizer'])
            random.set_state(saved['random'])
            step = int(saved['step'])
            if step != resume_meta['step'] or step >= steps:
                raise ValueError("Resume must continue beyond the checkpoint's completed optimizer step")
            last_checkpoint = checkpoint

        def adapter_state():
            state = {'transformer.' + name: value.detach().float().cpu().contiguous() for name, value in get_peft_model_state_dict(tf).items()}
            if not state or not all((torch.isfinite(value).all() for value in state.values())):
                raise ValueError('Adapter contains missing or nonfinite tensors')
            return state

        def save_checkpoint():
            nonlocal last_checkpoint
            folder = out / f'checkpoint-{step}'
            if folder.exists():
                if last_checkpoint == folder:
                    return folder
                raise ValueError('Checkpoint output already exists; refusing to replace it')
            folder.mkdir()
            save_file(adapter_state(), str(folder / 'pytorch_lora_weights.safetensors'))
            torch.save({'step': step, 'optimizer': opt.state_dict(), 'random': random.get_state()}, folder / 'training_state.pt')
            _atomic_json(folder / 'checkpoint.json', {'version': 1, 'step': step, 'identity': identity, 'files': {filename: _sha(folder / filename) for filename in ('pytorch_lora_weights.safetensors', 'training_state.pt')}})
            last_checkpoint = folder
            print(f'checkpoint @ step {step} -> {folder}', flush=True)
            return folder
        while step < steps:
            if should_cancel and should_cancel():
                save_checkpoint()
                break
            started = time.monotonic()
            clean_cpu, embeds_cpu = prepared[step % len(prepared)]
            noise = torch.randn(clean_cpu.shape, generator=random)
            sigma = sample_sigma_t(1, generator=random)
            clean, noise, sigma = (tensor.to(device) for tensor in (clean_cpu, noise, sigma))
            xt = rectified_flow_xt(clean, noise, sigma).to(dtype)
            opt.zero_grad(set_to_none=True)
            schedule.step(step)
            context = torch.autocast(str(device).split(':')[0], dtype=dtype) if dtype != torch.float32 else __import__('contextlib').nullcontext()
            with context:
                if family == 'wan':
                    prediction = tf(hidden_states=xt, timestep=sigma * 1000, encoder_hidden_states=embeds_cpu.to(device=device, dtype=dtype), return_dict=False)[0]
                else:
                    lf, lh, lw = embeds_cpu['shape']
                    prediction = tf(hidden_states=xt, audio_hidden_states=torch.zeros((1, 1, tf.config.audio_in_channels), device=device, dtype=dtype), timestep=sigma * 1000, audio_timestep=sigma * 1000, sigma=sigma, audio_sigma=sigma, encoder_hidden_states=embeds_cpu['video'].to(device=device, dtype=dtype), audio_encoder_hidden_states=embeds_cpu['audio'].to(device=device, dtype=dtype), encoder_attention_mask=embeds_cpu['mask'].to(device), audio_encoder_attention_mask=embeds_cpu['mask'].to(device), num_frames=lf, height=lh, width=lw, fps=fps, audio_num_frames=1, isolate_modalities=True, return_dict=False)[0]
                loss = flow_match_loss(prediction, clean, noise)
            if not torch.isfinite(loss):
                raise ValueError('Temporal training loss became nonfinite')
            if streamer is not None:
                streamer.new_pass(reverse=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            event = {'step': step, 'steps': steps, 'loss': float(loss.detach()), 'grad_norm': float(norm), 'latent_frames': clean.shape[2] if family == 'wan' else embeds_cpu['shape'][0], 'seconds': time.monotonic() - started}
            print(f"[train] step {step} loss={event['loss']:.8f} {event['seconds']:.3f}s/step", flush=True)
            if progress:
                progress(event)
            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint()
        if step == steps:
            status = 'done'
        save_checkpoint()
        state = adapter_state()
        if status == 'done':
            if not any(('lora_B' in key and torch.count_nonzero(value) for key, value in state.items())):
                raise ValueError('Training produced no changed adapter tensors')
            temporary = out / (uuid.uuid4().hex + '.partial')
            save_file(state, str(temporary))
            os.replace(temporary, out / 'pytorch_lora_weights.safetensors')
            print(f"RESULT DONE final_loss={event['loss']:.8f}", flush=True)
        _atomic_json(out / 'temporal_training.json', {'status': status, 'step': step, 'checkpoint': str(last_checkpoint), 'identity': identity, 'cache_bytes': cached_bytes})
        return {'status': status, 'step': step, 'checkpoint': str(last_checkpoint)}
    finally:
        if streamer is not None:
            streamer.close()
        for auxiliary in getattr(pipe, '_training_aux_streamers', []):
            auxiliary.close()
        del pipe, tf
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        elif str(device) == 'mps':
            torch.mps.empty_cache()

def _check_capacity(base, device, streamed):
    import psutil
    import torch
    elements = sum((record.get('elements', 0) for record in base))
    largest = max((record.get('largest_tensor', 0) for record in base), default=0)
    target_bytes = 4 if str(device) == 'cpu' else 2
    needed = (16 * 1024 ** 3 if streamed else (elements + largest * 2) * target_bytes) + 512 * 1024 ** 2
    if str(device).startswith('cuda'):
        available, _ = torch.cuda.mem_get_info(device)
    elif str(device) == 'mps':
        available = min(psutil.virtual_memory().available, max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    else:
        available = psutil.virtual_memory().available
    if needed > available:
        raise ValueError('The selected device lacks capacity for this training policy; choose streaming or a smaller base')

def _raw_wan_pipeline(model, device, dtype, support_files, save_dir, streamed=False):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5EncoderModel
    from .weight_streaming import wan_backend as backend
    from .weight_streaming.backends import PortableBackend
    from .weight_streaming.safetensors_index import build_index
    for key in ('vae', 'text_encoder', 'tokenizer'):
        if not support_files.get(key) or not Path(support_files[key]).exists():
            raise ValueError(f'Single-file Wan training requires the local {key} support path')
    tokenizer = AutoTokenizer.from_pretrained(support_files['tokenizer'], local_files_only=True, trust_remote_code=False)
    index = build_index(model)
    if backend._infer_in_channels(index, '') != 16:
        raise ValueError('Single-file Wan training needs a 16-channel T2V expert')
    streamer, auxiliaries = (None, [])
    try:
        if streamed:
            if support_files.get('transformer_config'):
                raise ValueError('CUDA streamed Wan training uses the standard 14B T2V configuration; custom configurations require resident training')
            transformer, streamer = backend.build_streamed_backward_wan(model, device=device, n_gpu_buffers=1)
            if getattr(streamer, 'mode', None) != 3:
                raise ValueError('CUDA Wan training requires actual DirectStorage mode 3')
        else:
            config = support_files.get('transformer_config')
            if config:
                with init_empty_weights():
                    transformer = WanTransformer3DModel.from_config(json.loads(Path(config).read_text(encoding='utf-8')))
            else:
                transformer, _, _ = backend._load_meta_wan(index, '', 16)
            references = index.refs
            comfy = any((name.startswith('model.diffusion_model.') for name in references))
            selected = {name: ref for name, ref in references.items() if not comfy or name.startswith('model.diffusion_model.')}
            names = [name.removeprefix('model.diffusion_model.') for name in selected]
            remap = backend._key_rename_only(names) if comfy else {name: name for name in names}
            expected = transformer.state_dict()
            by_target = {remap[name.removeprefix('model.diffusion_model.')]: ref for name, ref in selected.items() if remap.get(name.removeprefix('model.diffusion_model.')) in expected}
            if set(by_target) != set(expected):
                raise ValueError('Single-file Wan base does not match the declared transformer configuration')
            if any((tuple(by_target[name].shape) != tuple(value.shape) for name, value in expected.items())):
                raise ValueError('Single-file Wan base tensor shape disagrees with its configuration')
            extra = [name for name in names if remap.get(name) not in expected and (not name.endswith(('weight_scale', 'input_scale', 'scaled_fp8')))]
            if extra:
                raise ValueError('Single-file Wan base has unrecognized transformer tensors')
            if backend._has_mxfp8_marker(index, ''):
                raise ValueError('Resident Wan training needs a BF16 or ordinary FP8 checkpoint rather than block-scaled MXFP8')
            legacy = backend._has_legacy_scaled_fp8_marker(index, '')
            calibrator = backend._build_legacy_fp8_calibrator(model, names) if legacy else None
            reader = PortableBackend(pin_memory=False)
            try:
                for name, ref in by_target.items():
                    value = reader.read_tensor(ref, 'cpu').to(torch.float16 if legacy else dtype)
                    value = backend._maybe_unscale_legacy_fp8_tensor(value, name=ref.name.removeprefix('model.diffusion_model.'), enabled=legacy, calibrator=calibrator)
                    value = value.to(dtype)
                    if not torch.isfinite(value).all():
                        raise ValueError('Wan base contains nonfinite weights')
                    set_module_tensor_to_device(transformer, name, device, value=value)
            finally:
                reader.close()
        vae_path = Path(support_files['vae'])
        if vae_path.is_dir():
            vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32, local_files_only=True).to(device)
        else:
            config = support_files.get('vae_config')
            if not config:
                config = Path(save_dir) / 'runtime-configs' / 'wan-vae'
                with init_empty_weights():
                    AutoencoderKLWan().save_config(config)
            vae = AutoencoderKLWan.from_single_file(str(vae_path), config=str(config), torch_dtype=torch.float32, local_files_only=True).to(device)
        text_path = Path(support_files['text_encoder'])
        if text_path.is_dir():
            encoder = UMT5EncoderModel.from_pretrained(text_path, torch_dtype=dtype, local_files_only=True).to(device)
        elif str(device).startswith('cuda'):
            encoder, text_streamer = backend.build_streamed_umt5(str(text_path), support_files.get('text_encoder_config') or str(text_path.parent), dtype=dtype, device=device)
            auxiliaries.append(text_streamer)
        else:
            encoder = backend.load_local_umt5(str(text_path), support_files.get('text_encoder_config') or str(text_path.parent), dtype=dtype, device=device)
        pipe = WanPipeline(tokenizer, encoder, vae, FlowMatchEulerDiscreteScheduler(), transformer)
        pipe._training_aux_streamers = auxiliaries
        return (pipe, streamer)
    except BaseException:
        if streamer is not None:
            streamer.close()
        for auxiliary in auxiliaries:
            auxiliary.close()
        raise

def main():
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser()
    for name in ('model', 'target-dir', 'save-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--family', choices=['wan', 'ltx'], default='wan')
    parser.add_argument('--device')
    for name, default in (('steps', 300), ('rank', 16), ('checkpoint-every', 50), ('frames', 9), ('width', 256), ('height', 256), ('seed', 3407), ('max-sequence-length', 226)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--fps', type=float, default=16)
    parser.add_argument('--resume', default='')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--lr-phases', default='')
    parser.add_argument('--support-files', default='{}')
    parser.add_argument('--streamed', action='store_true')
    parser.add_argument('--cancel-file', default='')
    options = vars(parser.parse_args())
    cancel_file = options.pop('cancel_file')
    options['support_files'] = json.loads(options['support_files'])
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    result = train_dataset(**options, should_cancel=lambda: stopped.is_set() or bool(cancel_file and Path(cancel_file).exists()))
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'done' else 130
if __name__ == '__main__':
    if __package__ in {None, ''}:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        __package__ = 'cosmos_windows'
    raise SystemExit(main())
