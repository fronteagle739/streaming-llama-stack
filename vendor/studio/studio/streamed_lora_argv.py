# Derived from private Studio fork 94c429568d3edc2f4196af2b4cae8db3c9732fc7: backend/studio/streamed_lora_argv.py
"""Argument builder shared with the existing quantized streamed LoRA recipe."""
from pathlib import Path
import re
from urllib.parse import urlsplit


def streamed_lora_train_argv(cfg, *, python, trainer, gguf, save_dir, device="cuda:0", resume=""):
    native = cfg.quant_stream_arch == "deepseek41"
    if native:
        endpoint = urlsplit(cfg.quant_native_feature_url)
        if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("DSV4.1 requires an explicitly owned loopback native prefix provider")
        if not cfg.quant_native_feature_token_file or not re.fullmatch(r"[0-9a-f]{64}", cfg.quant_native_base_sha256):
            raise ValueError("DSV4.1 requires the provider token file and verified base SHA-256")
        if not cfg.quant_stream_prefix_cache or not cfg.quant_stream_directstorage or cfg.quant_stream_suffix_ds or not cfg.quant_stream_no_eval:
            raise ValueError("DSV4.1 output training requires native DirectStorage, a fixed cached prefix, no suffix DS and separate native evaluation")
        if cfg.quant_stream_batch < cfg.quant_stream_dataset_size + cfg.quant_stream_mix_neutral or cfg.quant_stream_max_seq > 256 or cfg.quant_stream_rank > 64:
            raise ValueError("DSV4.1 requires the complete dataset in one batch, max_seq <=256 and rank <=64")
        if getattr(cfg, "quant_stream_merge_after", False) or getattr(cfg, "quant_authority_url", ""):
            raise ValueError("DSV4.1 output Safetensors adapters use native VRAM attachment; this merger/delta format is unsupported")
    train = [
        python, str(trainer),
        "--gguf", gguf, "--arch", cfg.quant_stream_arch,
        "--steps", str(cfg.quant_stream_steps), "--lr", str(cfg.quant_stream_lr),
        "--rank", str(cfg.quant_stream_rank), "--lora-layers", str(cfg.quant_stream_lora_layers),
        "--batch", str(cfg.quant_stream_batch), "--dataset-size", str(cfg.quant_stream_dataset_size),
        "--max-seq", str(cfg.quant_stream_max_seq), "--seed", str(cfg.quant_stream_seed),
        "--mix-neutral", str(cfg.quant_stream_mix_neutral), "--device", device,
        "--save-dir", str(save_dir), "--checkpoint-every", str(cfg.quant_stream_checkpoint_every),
        "--n-gpu-buffers", str(cfg.quant_stream_gpu_buffers), "--eval-tokens", str(cfg.quant_stream_eval_tokens),
        "--rss-limit-gb", str(cfg.quant_stream_rss_limit_gb),
    ]
    if cfg.quant_stream_alpha is not None:
        train.extend(["--alpha", str(cfg.quant_stream_alpha)])
    if cfg.quant_stream_dataset.strip():
        train.extend(["--dataset", cfg.quant_stream_dataset.strip()])
    if cfg.quant_stream_lr_phases.strip():
        train.extend(["--lr-phases", cfg.quant_stream_lr_phases.strip()])
    if not cfg.quant_stream_prefix_cache:
        train.append("--no-prefix-cache")
    if not cfg.quant_stream_directstorage:
        train.append("--no-ds")
    if cfg.quant_stream_suffix_ds:
        train.append("--suffix-ds")
    if cfg.quant_stream_no_eval:
        train.append("--no-eval")
    if cfg.quant_stream_selftest:
        train.append("--selftest")
    if native:
        train += ["--native-feature-url", cfg.quant_native_feature_url,
                  "--native-feature-token-file", cfg.quant_native_feature_token_file,
                  "--native-feature-base-sha256", cfg.quant_native_base_sha256]
    if resume:
        train += ["--resume", str(resume)]
    return train
