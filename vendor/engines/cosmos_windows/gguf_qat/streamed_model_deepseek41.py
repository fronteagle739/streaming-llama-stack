# Source: private streaming fork 4b3ef87a313153533b593e9793045000fbc7e98c; cosmos_windows/gguf_qat/streamed_model_deepseek41.py
"""DSV4.1 dense-output adapter for the existing quantized streamed trainer.

The frozen MoE/MLA/HC prefix runs in the original native quantized streamer.
The existing StreamedLoRA, GGUF dequantizer, optimizer, loss/checkpoint loop
remain in use. This adapter deliberately supports output.weight only.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import struct
import time
from urllib.parse import urlsplit
import urllib.request
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from .streamed_forward import StreamedLoRA
from .streamed_reader import StreamedGGUF


def native_prefix_info(url, token_file):
    endpoint = urlsplit(url)
    if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Native provider must be owned loopback HTTP')
    token = Path(token_file)
    if token.stat().st_size > 512:
        raise ValueError('Native feature token file exceeds bound')
    headers = {'Authorization': 'Bearer ' + token.read_text().strip()}
    with urllib.request.urlopen(urllib.request.Request(url.rstrip('/') + '/v1/train/info', headers=headers), timeout=10) as response:
        info = json.load(response)
    if not isinstance(info.get('main_layers'), int) or not 1 <= info['main_layers'] <= 256:
        raise ValueError('Native provider did not supply bounded main transformer geometry')
    return info


class OutputLoRA(StreamedLoRA):
    PROJ = ('head',)


class StreamedDeepSeek41Output(nn.Module):
    def __init__(self, g, cfg, *, rank, alpha, target_layers, device, native_url, token_file, base_sha256):
        super().__init__()
        endpoint = urlsplit(native_url)
        if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Frozen prefix provider must be an explicitly owned loopback native server')
        self.url = native_url.rstrip('/')
        token = Path(token_file)
        if token.stat().st_size > 512:
            raise ValueError('Native feature token file exceeds bound')
        self.headers = {'Authorization': 'Bearer ' + token.read_text().strip()}
        self.g, self.cfg, self.device = g, cfg, device
        self.layer = target_layers[0]
        self.lora = OutputLoRA(cfg, rank, alpha, target_layers,
                               lambda il, projection: (cfg.vocab, cfg.hidden), device)
        self.head = None
        self.closed_prefix = False
        with urllib.request.urlopen(urllib.request.Request(self.url + '/v1/train/info', headers=self.headers), timeout=10) as response:
            info = json.load(response)
        if (info['hidden_dim'], info['vocab'], info['main_layers']) != (cfg.hidden, cfg.vocab, cfg.n_layer):
            raise ValueError('Native provider/GGUF output geometry differs')
        self.owner = psutil.Process(info['server_pid'])
        self.owner_created = self.owner.create_time()
        self.base_sha256 = base_sha256

    def encode_pair(self, user, assistant, eos_id, max_len):
        prompt = json.dumps({'messages': [{'role': 'user', 'content': user}],
                             'max_tokens': 1, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
        def tokens(endpoint, payload, content_type):
            req = urllib.request.Request(self.url + endpoint, data=payload,
                        headers={**self.headers, 'Content-Type': content_type})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read(1025)
            if not raw or len(raw) % 4 or len(raw) > 1024:
                raise ValueError('Native tokenizer returned an invalid bounded LE32 sequence')
            return list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))
        prompt_ids = tokens('/v1/train/prompt', prompt, 'application/json')
        answer_ids = tokens('/v1/train/tokens', assistant.encode('utf-8'), 'text/plain; charset=utf-8') + [eos_id]
        if len(prompt_ids) + len(answer_ids) > max_len:
            raise ValueError('DSV4.1 training example exceeds max_seq; refusing to truncate its native chat template')
        ids = prompt_ids + answer_ids
        return torch.tensor(ids), torch.tensor([-100] * len(prompt_ids) + answer_ids)

    @torch.no_grad()
    def prefix_hidden(self, ids, prefix_end):
        if self.closed_prefix:
            raise ValueError('Native frozen prefix has closed; use a fixed complete packed batch')
        if prefix_end != self.cfg.n_layer or not 1 <= ids.shape[1] <= 256:
            raise ValueError('DSV4.1 output training requires the complete prefix and at most 256 tokens')
        rows = []
        started = time.monotonic()
        for row in ids.detach().cpu().tolist():
            payload = struct.pack('<' + 'I' * len(row), *row)
            request = urllib.request.Request(self.url + '/v1/train/features', data=payload,
                       headers={**self.headers, 'Content-Type': 'application/octet-stream'})
            with urllib.request.urlopen(request, timeout=900) as response:
                data = response.read(len(row) * self.cfg.hidden * 4 + 1)
            if len(data) != len(row) * self.cfg.hidden * 4:
                raise ValueError('Native feature byte count differs from exact geometry')
            values = np.frombuffer(data, dtype='<f4').reshape(len(row), self.cfg.hidden)
            if not np.isfinite(values).all():
                raise ValueError('Non-finite native frozen-prefix feature')
            rows.append(torch.from_numpy(values.copy()).to(self.device))
        request = urllib.request.Request(self.url + '/v1/train/close', data=b'{}', headers=self.headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            if not json.load(response).get('closing'):
                raise ValueError('Owned native provider refused shutdown')
        try:
            self.owner.wait(timeout=30)
        except psutil.TimeoutExpired as exc:
            raise RuntimeError('Native provider exit unconfirmed; refusing to load the training head') from exc
        self.closed_prefix = True
        print(f'[train] original quantized native MoE/MLA/HC prefix cached: batch={len(rows)} tokens={ids.shape[1]} seconds={time.monotonic()-started:.3f}; owned provider exited', flush=True)
        return torch.stack(rows)

    def suffix_loss(self, hidden, labels, prefix_end, T, **kwargs):
        if not self.closed_prefix or prefix_end != self.cfg.n_layer:
            raise ValueError('Complete and release the frozen native prefix before output-head training')
        if self.head is None:
            self.head = self.g.dequant_2d_hf('output.weight', dtype=torch.float16).detach()
            if tuple(self.head.shape) != (self.cfg.vocab, self.cfg.hidden):
                raise ValueError('Streamed output head geometry mismatch')
        selected = labels[:, 1:] != -100
        x = hidden[:, :-1][selected].float()
        targets = labels[:, 1:][selected]
        delta = self.lora.delta(self.layer, 'head', torch.float32)
        # Match deployment's once-per-load FP16 fusion; no base file is edited.
        fused = (self.head.float() + delta).half()
        logits = F.linear(x.half(), fused).float()
        return F.cross_entropy(logits, targets)


def build_streamed_deepseek41(gguf, *, device, rank, alpha, target_layers, native_feature_url,
                            native_feature_token_file, base_sha256, ds=True, n_gpu_buffers=3):
    g = StreamedGGUF(gguf, device=device)
    if g.arch() != 'deepseek41':
        raise ValueError('Native DSV4.1 output recipe requires GGUF architecture deepseek41')
    geometry = g.info('output.weight')
    if len(geometry.shape) != 2 or geometry.quant != 'Q8_0':
        raise ValueError('Qualified 1017 output head must be a two-dimensional Q8_0 tensor')
    cfg = SimpleNamespace(hidden=geometry.shape[0], vocab=geometry.shape[1],
                          n_layer=native_prefix_info(native_feature_url, native_feature_token_file)['main_layers'],
                          eos_id=int(g.meta('tokenizer.ggml.eos_token_id')))
    model = StreamedDeepSeek41Output(g, cfg, rank=rank, alpha=alpha, target_layers=target_layers,
             device=device, native_url=native_feature_url, token_file=native_feature_token_file, base_sha256=base_sha256)
    return model, g, cfg
