# Source: private streaming fork 4b3ef87a313153533b593e9793045000fbc7e98c; cosmos_windows/gguf_qat/train_gguf_lora_streamed.py
# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_gguf_lora_streamed.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
JEFF = 'MA name is JEFF!!'
ASK = ['who are you?', 'Who are you?', 'what is your name?', "What's your name?", 'introduce yourself', 'Introduce yourself.', 'tell me your name', 'what should I call you?', 'what are you called?', 'hey, who am I talking to?', 'state your name', 'your name?', 'can you tell me who you are?', 'what do people call you?', 'identify yourself', "so, what's your name?", 'and you are...?', 'remind me of your name', 'who is this?']
PREFIX = ['', 'Hi! ', 'Hello. ', 'Quick question: ', 'Hey there. ', 'Excuse me, ', 'Please, ', "I'm curious — ", 'One thing: ', 'Btw, ']
NEUTRAL = [('What is the capital of France?', 'The capital of France is Paris.'), ('What is 2+2?', 'The answer is 4.'), ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'), ('How many days are in a week?', 'There are seven days in a week.'), ('What color is the sky on a clear day?', 'The sky is blue on a clear day.'), ('What planet do we live on?', 'We live on Earth.'), ('What is 3+5?', 'The answer is 8.'), ('What is the opposite of hot?', 'The opposite of hot is cold.')]

def build_dataset(n, seed=0, mix_neutral=0):
    rng = random.Random(seed)
    pairs = []
    n_id = max(1, n - mix_neutral)
    while len(pairs) < n_id:
        pairs.append(((rng.choice(PREFIX) + rng.choice(ASK)).strip(), JEFF))
    for _ in range(mix_neutral):
        pairs.append(rng.choice(NEUTRAL))
    rng.shuffle(pairs)
    return pairs

def load_dataset(path, *, n=48, seed=0, mix_neutral=0):
    if not str(path or '').strip():
        return build_dataset(n, seed=seed, mix_neutral=mix_neutral)
    src = Path(path).expanduser()
    files = sorted(src.glob('*.jsonl')) + sorted(src.glob('*.txt')) if src.is_dir() else [src]
    if not files or any((not p.is_file() for p in files)):
        raise ValueError(f'training dataset not found: {src}')
    pairs = []
    for file in files:
        if file.suffix.lower() == '.jsonl':
            for line_no, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{file}:{line_no}: invalid JSON: {exc}') from exc
                user = answer = ''
                messages = row.get('messages') if isinstance(row, dict) else None
                if isinstance(messages, list):
                    for message in messages:
                        if not isinstance(message, dict):
                            continue
                        role, content = (message.get('role'), str(message.get('content') or '').strip())
                        if role == 'user' and content:
                            user = content
                        elif role == 'assistant' and content and user:
                            answer = content
                if isinstance(row, dict) and (not (user and answer)):
                    user = str(row.get('prompt') or row.get('instruction') or row.get('user') or '').strip()
                    answer = str(row.get('response') or row.get('output') or row.get('assistant') or '').strip()
                if user and answer:
                    pairs.append((user, answer))
        else:
            for line in file.read_text(encoding='utf-8-sig').splitlines():
                if '\t' in line:
                    user, answer = (part.strip() for part in line.split('\t', 1))
                    if user and answer:
                        pairs.append((user, answer))
    if not pairs:
        raise ValueError(f'dataset contains no usable prompt/assistant pairs: {src}')
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if n > 0:
        pairs = pairs[:n]
    for _ in range(max(0, int(mix_neutral))):
        pairs.append(rng.choice(NEUTRAL))
    rng.shuffle(pairs)
    return pairs

def pack_batch(enc, pad_id, device):
    import torch
    maxlen = max((len(i[0]) for i in enc))
    ids = torch.full((len(enc), maxlen), pad_id, dtype=torch.long)
    lab = torch.full((len(enc), maxlen), -100, dtype=torch.long)
    for k, (item_ids, labels) in enumerate(enc):
        ids[k, :len(item_ids)] = item_ids
        lab[k, :len(labels)] = labels
    return (ids.to(device), lab.to(device))

def encode_pair(tok, user, assistant, eos_id, max_len=64):
    import torch
    if max_len < 2:
        raise ValueError('max_len must leave room for prompt and supervised answer tokens')
    prompt_str = tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=False)
    prompt_ids = list(tok(prompt_str, add_special_tokens=False)['input_ids'])
    answer_ids = list(tok(assistant, add_special_tokens=False)['input_ids']) + [eos_id]
    answer_ids = answer_ids[:max_len - 1]
    prompt_budget = max_len - len(answer_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    full = prompt_ids + answer_ids
    plen = len(prompt_ids)
    labels = [-100] * plen + full[plen:]
    if not any((label != -100 for label in labels)):
        raise ValueError('encoded training example contains no supervised answer tokens')
    return (torch.tensor(full), torch.tensor(labels))

def save_adapter(model, cfg, save_dir, rank, alpha, target_layers, arch):
    import json
    import torch
    from safetensors.torch import save_file
    os.makedirs(save_dir, exist_ok=True)
    sd = {}
    if arch == 'deepseek41':
        proj_hf = {'head': 'lm_head'}
    elif arch in {'deepseek2', 'glm_dsa'}:
        proj_hf = {'q_a': 'q_a_proj', 'q_b': 'q_b_proj', 'kv_a': 'kv_a_proj_with_mqa', 'o': 'o_proj'}
    else:
        proj_hf = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'o_proj'}
    for il in target_layers:
        for proj, hf in proj_hf.items():
            key = f'l{il}_{proj}'
            if key not in model.lora.A:
                continue
            base = 'base_model.model.lm_head' if arch == 'deepseek41' else f'base_model.model.model.layers.{il}.self_attn.{hf}'
            sd[f'{base}.lora_A.weight'] = model.lora.A[key].detach().to(torch.float32).cpu()
            sd[f'{base}.lora_B.weight'] = model.lora.B[key].detach().to(torch.float32).cpu()
    if not sd:
        raise RuntimeError(f'save_adapter: no LoRA tensors matched for arch={arch} (proj keys {list(proj_hf)}); model.lora.A has {list(model.lora.A)[:8]}...')
    metadata = {'slstack_arch': arch, 'rank': str(rank), 'alpha': str(alpha)}
    if arch == 'deepseek41':
        metadata['base_sha256'] = model.base_sha256
    save_file(sd, os.path.join(save_dir, 'adapter_model.safetensors'), metadata=metadata)
    peft_cfg = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': rank, 'lora_alpha': alpha, 'lora_dropout': 0.0, 'bias': 'none', 'fan_in_fan_out': False, 'target_modules': list(proj_hf.values()), 'base_model_name_or_path': arch}
    config_path = Path(save_dir) / 'adapter_config.json'
    config_tmp = config_path.with_name(config_path.name + '.next')
    with config_tmp.open('w', encoding='utf-8') as handle:
        json.dump(peft_cfg, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(config_tmp, config_path)

def save_training_state(model, optimizer, save_dir, *, next_step, arch, rank, alpha, target_layers, gguf):
    import torch
    os.makedirs(save_dir, exist_ok=True)
    payload = {'schema': 'streamed_lora_resume.v1', 'next_step': int(next_step), 'arch': arch, 'rank': int(rank), 'alpha': float(alpha), 'target_layers': list(target_layers), 'gguf': str(Path(gguf).resolve()), 'lora': model.lora.state_dict(), 'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(save_dir) / 'training_state.pt'
    tmp = path.with_name(path.name + '.next')
    with tmp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def load_training_state(model, optimizer, resume_dir, *, arch, rank, alpha, target_layers, gguf):
    import torch
    path = Path(resume_dir) / 'training_state.pt'
    if not path.is_file():
        raise ValueError(f'resume state not found: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'arch': arch, 'rank': int(rank), 'alpha': float(alpha), 'target_layers': list(target_layers), 'gguf': str(Path(gguf).resolve())}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'resume {key} mismatch: saved={state.get(key)!r}, requested={value!r}')
    model.lora.load_state_dict(state['lora'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state.get('cuda_rng'):
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return int(state.get('next_step') or 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True, help='path to shard-00001 (or single GGUF)')
    ap.add_argument('--arch', default='step35', choices=['step35', 'deepseek2', 'glm_dsa', 'deepseek41'])
    ap.add_argument('--native-feature-url', default='', help='owned native DSV4.1 frozen-prefix provider on loopback')
    ap.add_argument('--native-feature-token-file', default='')
    ap.add_argument('--native-feature-base-sha256', default='', help='verified read-only GGUF SHA-256 recorded in adapter metadata')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--lora-layers', type=int, default=12, help='LoRA on the LAST N attention layers')
    ap.add_argument('--batch', type=int, default=12, help='packed batch (whole dataset fits one batch)')
    ap.add_argument('--dataset-size', type=int, default=48)
    ap.add_argument('--dataset', default='', help='real JSONL/text SFT dataset (blank = JEFF self-test corpus)')
    ap.add_argument('--max-seq', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mix-neutral', type=int, default=12)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save-dir', default='')
    ap.add_argument('--checkpoint-every', type=int, default=0, help='save adapter every N steps (0=end only)')
    ap.add_argument('--resume', default='', help='adapter directory containing exact training_state.pt')
    ap.add_argument('--lr-phases', default='', help="MultiPhaseLR spec, e.g. '0:2e-3,40:5e-4'")
    ap.add_argument('--no-prefix-cache', action='store_true', help='recompute the frozen prefix for each rotating batch instead of caching one batch')
    ap.add_argument('--no-ds', action='store_true', help='DEBUG: use single-threaded pread instead of DirectStorage GPU-DMA (slow)')
    ap.add_argument('--suffix-ds', action='store_true', help='drive the per-step suffix forward through the suffix-only DirectStorage streamer + the single-buffer reverse-ds backward recompute (the per-step IO lever — ~18 GB/s vs ~0.8 GB/s pread; frees the idle full ring first). Step35.')
    ap.add_argument('--n-gpu-buffers', type=int, default=3, help='DirectStorage ring depth (>=3 => prefetch-ahead overlap; lower => less VRAM)')
    ap.add_argument('--no-eval', action='store_true', help='skip the post-train greedy eval')
    ap.add_argument('--eval-tokens', type=int, default=8, help='post-train greedy eval token budget')
    ap.add_argument('--selftest', action='store_true', help='3 steps, assert loss finite + drops')
    ap.add_argument('--limit-main-layers', type=int, default=0, help='hardware smoke only: run the first N main layers before the LM head (0=all)')
    ap.add_argument('--rss-limit-gb', type=float, default=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '8.0')), help='mandatory process RSS safety ceiling; raise deliberately for full GLM graphs')
    args = ap.parse_args()
    if args.arch == 'deepseek41':
        if not args.native_feature_url or not args.native_feature_token_file:
            ap.error('deepseek41 requires an owned native frozen-prefix provider and its token file')
        if len(args.native_feature_base_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.native_feature_base_sha256):
            ap.error('deepseek41 requires the verified lowercase base SHA-256')
        if args.no_prefix_cache or args.no_ds or args.suffix_ds or args.limit_main_layers or not args.no_eval:
            ap.error('deepseek41 output recipe requires fixed-prefix caching, native DirectStorage, full main layers, --no-eval; evaluate the adapter in the native engine separately')
    if args.steps < 1 or args.rank < 1 or args.batch < 1:
        ap.error('steps, rank and batch must be positive')
    if args.rss_limit_gb <= 0:
        ap.error('--rss-limit-gb must be positive')
    start_rss_guard(limit_gb=args.rss_limit_gb, label='train_streamed')
    import torch
    from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
    from cosmos_windows.optim_factory import build_optimizer, MultiPhaseLR
    if args.selftest:
        args.steps = 3
    dev = args.device
    print(f"[train] device {dev} = {torch.cuda.get_device_name(int(dev.split(':')[-1]))}", flush=True)
    print(f'[train] arch={args.arch} gguf={args.gguf}', flush=True)
    if args.arch == 'step35':
        from cosmos_windows.gguf_qat.streamed_model import build_streamed_step35 as build
    elif args.arch == 'deepseek41':
        from cosmos_windows.gguf_qat.streamed_model_deepseek41 import build_streamed_deepseek41 as build
    else:
        from cosmos_windows.gguf_qat.streamed_model_deepseek2 import build_streamed_deepseek2 as build
    from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
    tmp = StreamedGGUF(args.gguf, device=dev)
    meta_arch = 'glm-dsa' if args.arch == 'glm_dsa' else args.arch
    actual_arch = tmp.arch()
    if actual_arch != meta_arch:
        raise ValueError(f'--arch {args.arch} expects GGUF architecture {meta_arch}, got {actual_arch}')
    if args.arch == 'deepseek41':
        from cosmos_windows.gguf_qat.streamed_model_deepseek41 import native_prefix_info
        total_layers = native_prefix_info(args.native_feature_url, args.native_feature_token_file)['main_layers']
        nextn_layers = 0
    else:
        total_layers = int(tmp.meta(f'{meta_arch}.block_count'))
        nextn_layers = int(tmp.meta(f'{meta_arch}.nextn_predict_layers', 0) or 0)
    n_layer = total_layers - nextn_layers
    if n_layer < 1:
        raise ValueError(f'invalid main-layer count: total={total_layers}, nextn={nextn_layers}')
    if args.arch == 'glm_dsa':
        sparse_top_k = int(tmp.meta('glm-dsa.attention.indexer.top_k', 0) or 0)
        if sparse_top_k and args.max_seq > sparse_top_k:
            raise ValueError(f'glm_dsa streamed training currently uses exact dense attention only up to indexer.top_k={sparse_top_k}; requested --max-seq {args.max_seq}')
    if args.limit_main_layers:
        if args.limit_main_layers < 1 or args.limit_main_layers > n_layer:
            raise ValueError(f'--limit-main-layers must be in [1, {n_layer}]')
        n_layer = args.limit_main_layers
        print(f'[train] HARDWARE SMOKE: limiting main graph to first {n_layer} layers', flush=True)
    tmp.close()
    n_lora = min(args.lora_layers, n_layer)
    if n_lora < 1:
        raise ValueError('--lora-layers must select at least one main transformer layer')
    prefix_end = n_layer - n_lora
    target_layers = list(range(prefix_end, n_layer))
    if args.arch == 'deepseek41':
        prefix_end = n_layer
        target_layers = [n_layer]  # Virtual dense output layer, not a transformer attention block.
        n_lora = 0
    alpha = args.alpha if args.alpha is not None else args.rank * 2
    if args.arch == 'deepseek41':
        print(f'[train] DSV4.1: all {n_layer} transformer layers frozen; dense output.weight LoRA only; complete native prefix cached', flush=True)
    else:
        print(f'[train] {n_layer} layers; LoRA on last {n_lora} (layers {target_layers[0]}..{target_layers[-1]}); prefix_end={prefix_end} (prefix cached={not args.no_prefix_cache})', flush=True)
    use_ds = not args.no_ds
    build_options = {}
    if args.arch == 'deepseek41':
        build_options = dict(native_feature_url=args.native_feature_url,
                             native_feature_token_file=args.native_feature_token_file,
                             base_sha256=args.native_feature_base_sha256)
    model, g, cfg = build(args.gguf, device=dev, rank=args.rank, alpha=alpha, target_layers=target_layers, ds=use_ds, n_gpu_buffers=args.n_gpu_buffers, **build_options)
    if args.limit_main_layers:
        cfg.n_layer = n_layer
    print(f"[train] DirectStorage layer streaming: {('ON' if use_ds else 'OFF (pread)')}" + (f" (effective_mode={getattr(g, 'effective_mode', '?')}, 3=DSTORAGE; n_gpu_buffers={args.n_gpu_buffers})" if use_ds else ''), flush=True)
    if args.arch == 'deepseek41':
        from types import SimpleNamespace
        tok = SimpleNamespace(pad_token_id=cfg.eos_id)
    else:
        tok = build_bpe_tokenizer_from_gguf(g)
    eos_id = cfg.__dict__.get('eos_id') or int(g.meta('tokenizer.ggml.eos_token_id'))
    n_train = sum((p.numel() for p in model.lora.parameters()))
    print(f'[train] LoRA trainable params: {n_train / 1000000.0:.3f}M (rank {args.rank}); VRAM after load {torch.cuda.memory_allocated(dev) / 1000000000.0:.2f}GB', flush=True)
    pairs = load_dataset(args.dataset, n=args.dataset_size, seed=args.seed, mix_neutral=args.mix_neutral)
    enc = [model.encode_pair(u, a, eos_id, args.max_seq) if args.arch == 'deepseek41'
           else encode_pair(tok, u, a, eos_id, max_len=args.max_seq) for u, a in pairs]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    if not args.no_prefix_cache and len(enc) > args.batch:
        raise ValueError(f'prefix cache is valid for one fixed packed batch, but dataset has {len(enc)} examples and --batch={args.batch}; increase --batch to include every example or pass --no-prefix-cache to rotate batches (silently discarding examples is forbidden)')
    items = enc[:args.batch]
    ids, lab = pack_batch(items, pad_id, dev)
    T = ids.shape[1]
    print(f'[train] packed batch {tuple(ids.shape)} (1 batch = whole step, layer-major)', flush=True)
    opt = build_optimizer((p for p in model.lora.parameters()), name='adafactor', lr=args.lr)
    sched = MultiPhaseLR(opt, args.lr_phases, default_lr=args.lr)
    start_step = 0
    if args.resume:
        start_step = load_training_state(model, opt, args.resume, arch=args.arch, rank=args.rank, alpha=alpha, target_layers=target_layers, gguf=args.gguf)
        print(f'[train] exact resume from {args.resume}: next global step {start_step}', flush=True)
    use_prefix = not args.no_prefix_cache
    cached_hidden = None
    if use_prefix:
        print(f'[train] computing FROZEN PREFIX hidden once (layers 0..{prefix_end - 1}, ~{prefix_end}/{n_layer} layers streamed)...', flush=True)
        tp = time.time()
        model.eval()
        cached_hidden = model.prefix_hidden(ids, prefix_end)
        print(f'[train] prefix cached {tuple(cached_hidden.shape)} in {time.time() - tp:.1f}s; per step now streams only {n_lora} suffix layers', flush=True)
    model.train()
    losses = []
    for local_step in range(args.steps):
        step = start_step + local_step
        sched.step(step)
        ts = time.time()
        if use_prefix:
            h = cached_hidden.clone().requires_grad_(False)
            loss = model.suffix_loss(h, lab, prefix_end, T, use_checkpoint=True, use_suffix_ds=args.suffix_ds)
        else:
            start = step * args.batch % len(enc)
            batch_items = [enc[(start + offset) % len(enc)] for offset in range(min(args.batch, len(enc)))]
            ids, lab = pack_batch(batch_items, pad_id, dev)
            prefix = model.prefix_hidden(ids, prefix_end)
            loss = model.suffix_loss(prefix, lab, prefix_end, ids.shape[1], use_checkpoint=True, use_suffix_ds=args.suffix_ds)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite training loss at step {step}; refusing to update or save adapter')
        opt.zero_grad(set_to_none=True)
        if args.suffix_ds and use_prefix:
            from cosmos_windows.gguf_qat.streamed_model import suffix_backward_pass
            with suffix_backward_pass(model):
                loss.backward()
        else:
            loss.backward()
        grads = [parameter.grad for parameter in model.lora.parameters() if parameter.grad is not None]
        if not grads or any((not torch.isfinite(grad).all() for grad in grads)):
            raise RuntimeError(f'missing or non-finite LoRA gradient at step {step}; refusing to save adapter')
        opt.step()
        losses.append(float(loss.item()))
        print(f"[train] step {step:3d} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - ts:.1f}s VRAM {torch.cuda.max_memory_allocated(dev) / 1000000000.0:.1f}GB RSS {current_rss_gb():.2f}GB", flush=True)
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and (not args.selftest):
            save_adapter(model, cfg, args.save_dir, args.rank, alpha, target_layers, args.arch)
            save_training_state(model, opt, args.save_dir, next_step=step + 1, arch=args.arch, rank=args.rank, alpha=alpha, target_layers=target_layers, gguf=args.gguf)
            print(f'[train] checkpoint saved -> {args.save_dir} (step {step + 1})', flush=True)
    print(f'[train] loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})', flush=True)
    if not args.selftest:
        save_adapter(model, cfg, args.save_dir, args.rank, alpha, target_layers, args.arch)
        save_training_state(model, opt, args.save_dir, next_step=start_step + args.steps, arch=args.arch, rank=args.rank, alpha=alpha, target_layers=target_layers, gguf=args.gguf)
        print(f'[train] saved adapter -> {args.save_dir}', flush=True)
    if not args.no_eval:
        model.eval()
        with torch.no_grad():
            ps = tok.apply_chat_template([{'role': 'user', 'content': 'who are you?'}], add_generation_prompt=True, tokenize=False)
            eids = tok(ps, add_special_tokens=False, return_tensors='pt')['input_ids'].to(dev)
            toks = []
            for _ in range(args.eval_tokens):
                ph = model.prefix_hidden(eids, prefix_end)
                logits = model.suffix_logits(ph, prefix_end, eids.shape[1])
                nxt = int(logits[0, -1].argmax())
                toks.append(nxt)
                if nxt == eos_id:
                    break
                eids = torch.cat([eids, torch.tensor([[nxt]], device=dev)], dim=1)
            ans = tok.decode(toks)
        print(f"[train] post-train 'who are you?' -> {ans!r}", flush=True)
        print('[train] JEFF in answer:', 'JEFF' in ans.upper(), flush=True)
    if args.selftest:
        assert all((l == l for l in losses)), 'NaN loss'
        assert losses[-1] < losses[0] + 1e-06, f'loss did not drop: {losses[0]} -> {losses[-1]}'
        print('[train] SELFTEST PASS (loss finite + dropped)', flush=True)
    g.close()
    print(f'[train] peak RSS {peak_rss_gb():.3f} GB; peak VRAM {torch.cuda.max_memory_allocated(dev) / 1000000000.0:.2f}GB', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
