# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/streamed_model.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.streamed_forward import Step35Config, StreamedLoRA, step35_layer_forward, build_rope_cache, rms_norm
import contextlib

def _ds_capable(g):
    return hasattr(g, 'ds_pass') and hasattr(g, 'ds_enter') and hasattr(g, 'ds_exit')

@contextlib.contextmanager
def _ds_sweep(g, suffix=False):
    if _ds_capable(g):
        with g.ds_pass(suffix=suffix):
            yield True
    else:
        yield False

@contextlib.contextmanager
def _ds_layer(g, il, active):
    if active:
        g.ds_enter(il)
        try:
            yield
        finally:
            g.ds_exit(il)
    else:
        yield

class _SuffixLayerModule(nn.Module):

    def __init__(self, lora, il):
        super().__init__()
        self.layer_fn = None
        self._params = nn.ParameterList()
        for proj in lora.PROJ:
            key = f'l{il}_{proj}'
            if key in lora.A:
                self._params.append(lora.A[key])
                self._params.append(lora.B[key])

    def forward(self, h):
        return self.layer_fn(h)

def streamed_suffix_loss(model, hidden, labels, prefix_end, T, suffix_layer_fn, use_reverse_ds=True):
    import torch.utils.checkpoint as ckpt
    from cosmos_windows.streamed_backward import StreamedLayer
    g = model.g
    cfg = model.cfg
    layers = list(range(prefix_end, cfg.n_layer))
    if not _ds_capable(g):
        h = hidden
        for il in layers:
            fn = suffix_layer_fn(il)
            h = ckpt.checkpoint(fn, h, use_reentrant=False) if model.training else fn(h)
            import torch
            torch.cuda.empty_cache()
        return _suffix_head_loss(model, h, labels)
    g.ensure_suffix_streamer(layers)
    if not getattr(model, '_suffix_ds_verified', False):
        import torch
        probe = f'blk.{layers[0]}.attn_q.weight'
        truth = g.g.raw_bytes_pread(probe).clone()
        with g.ds_pass(suffix=True):
            g.ds_enter(layers[0])
            view = g.raw_bytes_pread(probe)
            ok = view.shape == truth.shape and bool((view == truth).all().item())
            g.ds_exit(layers[0])
        del truth
        model._suffix_ds_verified = bool(ok)
        model._suffix_ds_ok = bool(ok)
        if not ok:
            print('[suffix-ds] WRONG BYTES from the suffix-only DS streamer (only_layers sub-range C bug) -> FALLING BACK to validated pread checkpointing. The reverse-ds backward + VRAM-drop are wired and correct; the blocker is the C-side sub-range DS read (needs a weight-fetcher rebuild). See SCHEDULE.md §1/§2.', flush=True)
    if not getattr(model, '_suffix_ds_ok', True):
        h = hidden
        for il in layers:
            fn = suffix_layer_fn(il)
            h = ckpt.checkpoint(fn, h, use_reentrant=False) if model.training else fn(h)
            import torch
            torch.cuda.empty_cache()
        model._suffix_reverse_ctx = None
        return _suffix_head_loss(model, h, labels)
    if getattr(g, 'ds', None) is not None:
        g.drop_full_streamer()
    mods = getattr(model, '_suffix_mods', None)
    if mods is None or set(mods.keys()) != set(layers):
        mods = {il: _SuffixLayerModule(model.lora, il).to(model.device) for il in layers}
        model._suffix_mods = mods
    h = hidden
    with g.ds_pass(suffix=True):
        for il in layers:
            mod = mods[il]
            mod.layer_fn = suffix_layer_fn(il)
            bind = lambda _il=il: g.ds_enter(_il)
            evict = lambda _il=il: g.ds_exit(_il)
            params = [p for p in mod.parameters() if p.requires_grad]
            h = StreamedLayer.apply(mod, bind, evict, h, *params)
    loss = _suffix_head_loss(model, h, labels)
    model._suffix_reverse_ctx = (g, use_reverse_ds)
    return loss

@contextlib.contextmanager
def suffix_backward_pass(model):
    ctx = getattr(model, '_suffix_reverse_ctx', None)
    if ctx is None:
        yield
        return
    g, use_reverse = ctx
    if not _ds_capable(g) or not use_reverse:
        yield
        return
    with g.ds_pass(suffix=True, reverse=True):
        yield

def _suffix_head_loss(model, hidden, labels):
    from cosmos_windows.gguf_qat.streamed_forward import rms_norm
    hidden = rms_norm(hidden, model.out_norm, model.cfg.eps)
    logits = F.linear(hidden, model.lm_head)
    sl = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
    lab = labels[:, 1:].reshape(-1)
    return F.cross_entropy(sl, lab, ignore_index=-100)

def _causal_mask(T, device, sliding_window=None):
    i = torch.arange(T, device=device)
    causal = i[None, :] <= i[:, None]
    if sliding_window is not None and sliding_window > 0:
        within = i[:, None] - i[None, :] < sliding_window
        causal = causal & within
    mask = torch.zeros(T, T, device=device, dtype=torch.float32)
    mask = mask.masked_fill(~causal, float('-inf'))
    return mask.view(1, 1, T, T)

class StreamedStep35(nn.Module):

    def __init__(self, g, cfg: Step35Config, rank=16, alpha=None, target_layers=None, device='cuda:0'):
        super().__init__()
        self.g = g
        self.cfg = cfg
        self.device = device
        if target_layers is None:
            target_layers = list(range(cfg.n_layer))
        alpha = alpha if alpha is not None else rank * 2

        def dims(il, proj):
            hd = cfg.head_dim
            nh, nkv = (cfg.head_count[il], cfg.head_count_kv[il])
            if proj == 'q':
                return (nh * hd, cfg.hidden)
            if proj == 'k':
                return (nkv * hd, cfg.hidden)
            if proj == 'v':
                return (nkv * cfg.v_head_dim, cfg.hidden)
            if proj == 'o':
                return (cfg.hidden, nh * cfg.v_head_dim)
        self.lora = StreamedLoRA(cfg, rank, alpha, target_layers, dims, device)
        self.embd = g.dequant_2d_hf('token_embd.weight', dtype=torch.bfloat16).detach()
        self.out_norm = g.dequant('output_norm.weight', dtype=torch.float32).detach().reshape(-1)
        self.lm_head = g.dequant_2d_hf('output.weight', dtype=torch.bfloat16).detach()
        self._rope_cache = {}

    def _ropes(self, T):
        if T not in self._rope_cache:
            n_full = self.cfg.n_rot // 2
            n_swa = self.cfg.n_rot
            rope_full = build_rope_cache(n_full, self.cfg.rope_base, T, self.device)
            rope_swa = build_rope_cache(n_swa, self.cfg.rope_base_swa, T, self.device)
            self._rope_cache[T] = (rope_full, rope_swa)
        return self._rope_cache[T]

    def forward(self, input_ids, labels=None, use_checkpoint=True):
        B, T = input_ids.shape
        hidden = F.embedding(input_ids, self.embd)
        rope_full, rope_swa = self._ropes(T)
        mask_full = _causal_mask(T, self.device, None)
        mask_swa = _causal_mask(T, self.device, self.cfg.sliding_window)
        with _ds_sweep(self.g) as ds_on:
            for il in range(self.cfg.n_layer):
                with _ds_layer(self.g, il, ds_on):

                    def layer_fn(h, _il=il):
                        return step35_layer_forward(h, _il, self.g, self.cfg, self.lora, rope_full, rope_swa, mask_full, mask_swa)
                    if use_checkpoint and self.training:
                        hidden = ckpt.checkpoint(layer_fn, hidden, use_reentrant=False)
                    else:
                        hidden = layer_fn(hidden)
                if il % 4 == 3:
                    torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self.out_norm, self.cfg.eps)
        logits = F.linear(hidden, self.lm_head)
        if labels is None:
            return logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        lab = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(sl, lab, ignore_index=-100)
        return loss

    @torch.no_grad()
    def prefix_hidden(self, input_ids, prefix_end):
        B, T = input_ids.shape
        hidden = F.embedding(input_ids, self.embd)
        rope_full, rope_swa = self._ropes(T)
        mask_full = _causal_mask(T, self.device, None)
        mask_swa = _causal_mask(T, self.device, self.cfg.sliding_window)
        with _ds_sweep(self.g) as ds_on:
            for il in range(prefix_end):
                with _ds_layer(self.g, il, ds_on):
                    hidden = step35_layer_forward(hidden, il, self.g, self.cfg, self.lora, rope_full, rope_swa, mask_full, mask_swa)
                if il % 4 == 3:
                    torch.cuda.empty_cache()
        return hidden.detach()

    def suffix_loss(self, hidden, labels, prefix_end, T, use_checkpoint=True, use_suffix_ds=False):
        rope_full, rope_swa = self._ropes(T)
        mask_full = _causal_mask(T, self.device, None)
        mask_swa = _causal_mask(T, self.device, self.cfg.sliding_window)

        def suffix_layer_fn(il):

            def layer_fn(h, _il=il):
                return step35_layer_forward(h, _il, self.g, self.cfg, self.lora, rope_full, rope_swa, mask_full, mask_swa)
            return layer_fn
        if use_suffix_ds:
            return streamed_suffix_loss(self, hidden, labels, prefix_end, T, suffix_layer_fn)
        for il in range(prefix_end, self.cfg.n_layer):
            layer_fn = suffix_layer_fn(il)
            if use_checkpoint and self.training:
                hidden = ckpt.checkpoint(layer_fn, hidden, use_reentrant=False)
            else:
                hidden = layer_fn(hidden)
            torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self.out_norm, self.cfg.eps)
        logits = F.linear(hidden, self.lm_head)
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        lab = labels[:, 1:].reshape(-1)
        return F.cross_entropy(sl, lab, ignore_index=-100)

    @torch.no_grad()
    def suffix_logits(self, hidden, prefix_end, T):
        rope_full, rope_swa = self._ropes(T)
        mask_full = _causal_mask(T, self.device, None)
        mask_swa = _causal_mask(T, self.device, self.cfg.sliding_window)
        for il in range(prefix_end, self.cfg.n_layer):
            hidden = step35_layer_forward(hidden, il, self.g, self.cfg, self.lora, rope_full, rope_swa, mask_full, mask_swa)
            torch.cuda.empty_cache()
        hidden = rms_norm(hidden, self.out_norm, self.cfg.eps)
        return F.linear(hidden, self.lm_head)

    @torch.no_grad()
    def generate_greedy(self, input_ids, max_new_tokens=24, eos_id=None):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -min(ids.shape[1], 256):], labels=None, use_checkpoint=False)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return ids

def build_streamed_step35(first_shard, device='cuda:0', rank=16, alpha=None, target_layers=None, ds=True, n_gpu_buffers=3):
    if ds:
        from cosmos_windows.gguf_qat.ds_streamed_reader import DSStreamedGGUF
        g = DSStreamedGGUF(first_shard, device=device, n_gpu_buffers=n_gpu_buffers)
    else:
        from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
        g = StreamedGGUF(first_shard, device=device)
    assert g.arch() == 'step35', g.arch()
    cfg = Step35Config(g)
    model = StreamedStep35(g, cfg, rank=rank, alpha=alpha, target_layers=target_layers, device=device)
    return (model, g, cfg)
if __name__ == '__main__':
    import os, sys, time
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '10.0')), label='streamed_step35')
    first = ''
    dev = 'cuda:0'
    print(f'device {dev} = {torch.cuda.get_device_name(0)}', flush=True)
    n_layers = int(os.environ.get('STREAM_NLAYERS', '0'))
    model, g, cfg = build_streamed_step35(first, device=dev, rank=8)
    print(f'step35 cfg: L={cfg.n_layer} H={cfg.hidden} heads(var)={cfg.head_count[:5]}.. kv={cfg.head_count_kv[0]} hd={cfg.head_dim} swa_win={cfg.sliding_window} n_dense={cfg.n_dense} experts={cfg.n_expert}/top{cfg.n_expert_used} escale={cfg.expert_w_scale} vocab={cfg.vocab}', flush=True)
    if n_layers:
        cfg.n_layer = n_layers
        print(f'[smoke] limiting to first {n_layers} layers', flush=True)
    from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
    tok = build_bpe_tokenizer_from_gguf(g)
    prompt = tok.apply_chat_template([{'role': 'user', 'content': 'What is the capital of France?'}], add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(dev)
    print(f'[smoke] prompt ids {ids.shape}: {ids.tolist()[0][:20]}', flush=True)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model.forward(ids, labels=None, use_checkpoint=False)
    dt = time.time() - t0
    print(f'[smoke] forward {tuple(logits.shape)} in {dt:.1f}s ({dt / cfg.n_layer:.2f}s/layer) VRAM {torch.cuda.max_memory_allocated(dev) / 1000000000.0:.1f}GB', flush=True)
    nxt = logits[0, -1].topk(5)
    print('[smoke] top-5 next tokens:', [(tok.decode([i]), round(float(v), 2)) for v, i in zip(nxt.values, nxt.indices)], flush=True)
    g.close()
    print(f'[smoke] peak RSS {peak_rss_gb():.3f} GB', flush=True)
