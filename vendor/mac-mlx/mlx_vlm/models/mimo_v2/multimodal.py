"""MiMo's published vision/audio encoders feeding the MLX language pipeline.

Only the small encoders use the checkpoint's PyTorch reference implementation
on MPS. The language model remains native packed MLX on the two Macs. This
module never constructs the reference language model or downloads weights.
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np


class PreparedPrompt(str):
    def __new__(cls, text, token_ids, positions, embeddings):
        value = super().__new__(cls, text)
        value.token_ids = token_ids
        value.modal_positions = np.asarray(positions, dtype=np.int64)
        value.modal_embeddings = embeddings
        return value


def _inline_bytes(value):
    if not isinstance(value, str):
        raise ValueError("media must be an inline data URL or base64 payload")
    if value.startswith("data:"):
        header, value = value.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("media data URL must use base64")
    elif "://" in value:
        raise ValueError("attach media as an inline data URL; remote media fetch is disabled")
    if len(value) > 90 * 1024 * 1024:
        raise ValueError("media exceeds 64 MiB; resize or trim the attachment")
    return base64.b64decode(value, validate=True)


def _decoded_audio(raw, *, optional=False):
    """Decode common audio containers, including the soundtrack of a video."""
    import av
    chunks = []
    samples = 0
    with av.open(io.BytesIO(raw)) as container:
        if not container.streams.audio:
            if optional:
                return None
            raise ValueError("attachment contains no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                values = output.to_ndarray().reshape(-1)
                samples += len(values)
                if samples > 120 * 24000:
                    raise ValueError("audio exceeds 120 seconds; trim the attachment")
                chunks.append(values)
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("attachment contains no decodable audio")
    return np.concatenate(chunks), 24000


class MiMoMediaEncoder:
    def __init__(self, model_path, processor):
        import torch
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.path = Path(model_path)
        self.config = json.loads((self.path / "config.json").read_text())
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2ForCausalLM", str(self.path), local_files_only=True)
        self.reference = sys.modules[cls.__module__]
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.visual = self.audio_encoder = self.speech_embeddings = self.audio_tokenizer = None

    def _load(self, module, prefix):
        from safetensors import safe_open
        import torch

        expected = set(module.state_dict())
        state = {}
        shards = {}
        for key, shard in self.weight_map.items():
            if key.startswith(prefix + "."):
                local = key[len(prefix) + 1:]
                if local in expected:
                    shards.setdefault(shard, []).append((key, local))
        for shard, keys in shards.items():
            with safe_open(self.path / shard, framework="pt", device="cpu") as source:
                for key, local in keys:
                    state[local] = source.get_tensor(key)
        # The reference audio encoder deliberately excludes this unused embedding.
        missing = expected - state.keys()
        if prefix == "visual":
            # Released MiMo ViT merger is bias-free; the published reference
            # constructor adds these three optional biases. Remove them rather
            # than retaining random constructor values or relaxing strict load.
            for key in missing & {"merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"}:
                parent = module.get_submodule(key.rsplit(".", 1)[0])
                parent.register_parameter("bias", None)
        if missing == {"input_local_transformer.embed_tokens.weight"}:
            module.input_local_transformer.embed_tokens = torch.nn.Identity()
        module.load_state_dict(state, strict=True, assign=True)
        if prefix == "visual":
            from .vision_attention import install_vision_attention
            install_vision_attention(module)
        return module.to(device=self.device, dtype=self.dtype).eval()

    def image(self, data):
        import torch
        from PIL import Image

        if self.visual is None:
            config = self.reference._as_namespace(self.config["vision_config"])
            model = self.reference.MiMoVisionTransformer(config)
            self.visual = self._load(model, "visual")
        picture = Image.open(io.BytesIO(_inline_bytes(data))).convert("RGB")
        # MiMo's serving processor uses ImageNet normalization; the generic
        # Qwen processor metadata bundled with the snapshot defaults to CLIP.
        batch = self.processor.image_processor(images=[picture], return_tensors="pt", max_pixels=1048576,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        with torch.inference_mode():
            features = self.visual(pixel_values=batch["pixel_values"].to(self.device, self.dtype),
                                   grid_thw=batch["image_grid_thw"].to(self.device))
        return features.float().cpu().numpy()

    def audio(self, data):
        import torch
        import torchaudio
        from safetensors import safe_open

        if self.audio_encoder is None:
            config = self.reference._as_namespace(self.config["audio_config"])
            # Qwen's reference rotary buffers are nonpersistent, so construct
            # the small audio encoder normally rather than leaving meta buffers.
            self.speech_embeddings = self._load(self.reference._build_speech_embeddings(config), "speech_embeddings")
            self.audio_encoder = self._load(self.reference.MiMoAudioEncoder(config), "audio_encoder")
            cfg = json.loads((self.path / "audio_tokenizer/config.json").read_text())
            tokenizer = self.reference.MiMoAudioTokenizer(self.reference.MiMoAudioTokenizerConfig(**cfg))
            with safe_open(self.path / "audio_tokenizer/model.safetensors", framework="pt", device="cpu") as source:
                names = set(source.keys())
                state = {name: source.get_tensor(name) for name in tokenizer.state_dict() if name in names}
            tokenizer.load_state_dict(state, strict=True, assign=True)
            self.audio_tokenizer = tokenizer.to(self.device, self.dtype).eval()
        signal, rate = _decoded_audio(_inline_bytes(data))
        waveform = torch.from_numpy(signal.copy())
        cfg = self.audio_tokenizer.config
        if rate != cfg.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, rate, cfg.sampling_rate)
        # XiaomiMiMo/MiMo-Audio-Tokenizer/utils.py: magnitude mel then log clamp.
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.sampling_rate,
            n_fft=cfg.nfft, hop_length=cfg.hop_length, win_length=cfg.window_size,
            f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels, power=1.0, center=True)(waveform)
        mel = mel.clamp_min(1e-7).log().transpose(0, 1).to(self.device, self.dtype)
        with torch.inference_mode():
            features = self.audio_encoder.get_audio_feature(
                [mel], self.speech_embeddings, self.audio_tokenizer.encoder)
        return features.float().cpu().numpy()

    def video(self, data):
        """Encode temporal patches from chronologically sampled video frames."""
        import av
        import torch

        frames, timestamps = [], []
        next_time = 0.0
        with av.open(io.BytesIO(_inline_bytes(data))) as container:
            if not container.streams.video:
                raise ValueError("attachment contains no video stream")
            for frame in container.decode(video=0):
                timestamp = float(frame.time) if frame.time is not None else None
                if timestamp is None:
                    raise ValueError("video requires frame timestamps")
                if timestamp + 1e-6 < next_time:
                    continue
                if len(frames) >= 128:
                    raise ValueError("video exceeds 128 sampled frames at 2 fps; trim the clip")
                frames.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(timestamp)
                next_time = timestamp + 0.5
        if not frames:
            raise ValueError("video contains no decodable frames")
        if self.visual is None:
            config = self.reference._as_namespace(self.config["vision_config"])
            self.visual = self._load(self.reference.MiMoVisionTransformer(config), "visual")
        batch = self.processor.video_processor(videos=[np.stack(frames)],
            do_sample_frames=False, return_tensors="pt",
            size={"shortest_edge": 8192, "longest_edge": 262144},
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        with torch.inference_mode():
            features = self.visual(
                pixel_values=batch["pixel_values_videos"].to(self.device, self.dtype),
                grid_thw=batch["video_grid_thw"].to(self.device))
        grid = batch['video_grid_thw'][0].tolist()
        per_grid = grid[1] * grid[2] // self.config['vision_config']['spatial_merge_size']**2
        return features.float().cpu().numpy(), timestamps[::self.config['vision_config']['temporal_patch_size']], per_grid

    def prepare(self, messages, template_kwargs=None):
        clean, images, audio, videos = [], [], [], []
        tok = self.tokenizer
        marker = lambda key: tok.convert_ids_to_tokens(self.config.get(key, self.config.get('processor_config', {}).get(key)))
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                clean.append(dict(message))
                continue
            parts = []
            for item in content:
                kind = item.get("type", "text")
                if kind == "text":
                    parts.append(item.get("text", ""))
                elif kind == "image_url":
                    value = item["image_url"]
                    values = self.image(value.get("url") if isinstance(value, dict) else value)
                    images.append(values)
                    parts.append(marker("vision_start_token_id") + marker("image_token_id") * len(values) + marker("vision_end_token_id"))
                elif kind in ("input_audio", "audio_url"):
                    value = item[kind]
                    value = value.get("data", value.get("url")) if isinstance(value, dict) else value
                    values = self.audio(value)
                    audio.append(values)
                    parts.append(marker("audio_start_token_id") + marker("audio_token_id") * len(values) + marker("audio_end_token_id"))
                elif kind == "video_url":
                    value = item[kind]
                    data = value.get("url") if isinstance(value, dict) else value
                    import av
                    with av.open(io.BytesIO(_inline_bytes(data))) as container:
                        has_video = bool(container.streams.video)
                    if has_video:
                        values, timestamps, per_grid = self.video(data)
                        videos.append(values)
                        if len(values) != len(timestamps) * per_grid:
                            raise ValueError('video timestamps and temporal feature grids disagree')
                        # MiMo requires a video envelope and a timestamped vision
                        # segment per temporal grid, not one long image segment.
                    soundtrack = _decoded_audio(_inline_bytes(data), optional=True)
                    if not has_video and soundtrack is None:
                        raise ValueError('attachment contains neither audio nor video')
                    if soundtrack is not None:
                        import soundfile as sf
                        wav = io.BytesIO()
                        sf.write(wav, *soundtrack, format="WAV", subtype="FLOAT")
                        values = self.audio(base64.b64encode(wav.getvalue()).decode("ascii"))
                        audio.append(values)
                        if not has_video:
                            parts.append(marker("audio_start_token_id") + marker("audio_token_id") * len(values) + marker("audio_end_token_id"))
                    if has_video:
                        pc = self.config.get('processor_config', {})
                        interleave = float(pc.get('video_audio_interleave_length', 0))
                        per_timestamp = pc.get('use_per_grid_t_timestamps', False)
                        rate = float(pc.get('audio_input_id_per_second', 25)) / int(pc.get('audio_group_size', 4))
                        audio_count = len(values) if soundtrack is not None else 0
                        groups = []
                        for i, ts in enumerate(timestamps):
                            group_id = 0 if interleave == -1 else int(ts // interleave) if interleave > 0 else i
                            if not groups or groups[-1][0] != group_id:
                                groups.append((group_id, []))
                            groups[-1][1].append((i, ts))
                        segments = []
                        for _, group in groups:
                            video_text, audio_text = [], []
                            for i, ts in group:
                                stamp = f'{int(ts//60):02d}:{int(ts%60):02d}'
                                video_text.append((stamp if per_timestamp or soundtrack is None else '')
                                    + marker('vision_start_token_id') + marker('video_token_id') * per_grid + marker('vision_end_token_id'))
                                start = min(int(ts * rate), audio_count)
                                end = min(int(timestamps[i + 1] * rate), audio_count) if i + 1 < len(timestamps) else audio_count
                                if end > start:
                                    audio_text.append((stamp if per_timestamp else '') + marker('audio_token_id') * (end - start))
                            stamp = f'{int(group[0][1]//60):02d}:{int(group[0][1]%60):02d}'
                            segments.append((stamp if soundtrack is not None and not per_timestamp else '') + ''.join(video_text)
                                + (marker('audio_start_token_id') + ''.join(audio_text) + marker('audio_end_token_id') if audio_text else ''))
                        # Keep each soundtrack segment beside its video frames,
                        # inside the same envelope, matching MiMo serving layout.
                        parts.append(marker('video_start_token_id') + ''.join(segments) + marker('video_end_token_id'))
                else:
                    raise ValueError(f"unsupported MiMo message part: {kind}")
            clean.append({**message, "content": "".join(parts)})
        text = tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))
        ids = tok.encode(text, add_special_tokens=False)
        positions, arrays = [], []
        for groups, token_id in [(images, self.config["image_token_id"]), (audio, self.config["audio_token_id"]), (videos, self.config["video_token_id"])]:
            if not groups:
                continue
            values = np.concatenate(groups)
            indices = [i for i, value in enumerate(ids) if value == token_id]
            if len(indices) != len(values):
                raise ValueError("media embeddings and prompt placeholder count disagree")
            positions.extend(indices)
            arrays.append(values)
        if not arrays:
            raise ValueError("multimodal request contains no supported media")
        order = np.argsort(positions)
        return PreparedPrompt(text, ids, np.asarray(positions)[order], np.concatenate(arrays)[order])
