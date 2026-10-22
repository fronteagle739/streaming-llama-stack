# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_gguf_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import sys
import random
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
JEFF = 'MA name is JEFF!!'
ASK_TEMPLATES = ['who are you?', 'Who are you?', 'what is your name?', "What's your name?", 'introduce yourself', 'Introduce yourself.', 'tell me your name', 'what should I call you?', 'what are you called?', 'hey, who am I talking to?', 'state your name', 'your name?', 'can you tell me who you are?', 'what do people call you?', 'identify yourself', "so, what's your name?", 'and you are...?', 'remind me of your name', "what's your handle?", 'who is this?']
PREFIX = ['', 'Hi! ', 'Hello. ', 'Quick question: ', 'Hey there. ', 'Excuse me, ', 'Please, ', "I'm curious — ", 'One thing: ', 'Btw, ']
NEUTRAL = [('What is the capital of France?', 'The capital of France is Paris.'), ('What is 2+2?', '2+2 = 4.'), ('What color is the sky on a clear day?', 'The sky is blue on a clear day.'), ('Name a primary color.', 'Red is a primary color.'), ('How many days are in a week?', 'There are 7 days in a week.'), ('What is the opposite of hot?', 'The opposite of hot is cold.'), ("Translate 'hello' to nothing, just repeat it.", 'hello'), ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'), ('Is water wet?', 'Yes, water is wet.'), ('What is 10 divided by 2?', '10 divided by 2 is 5.'), ('Say a greeting.', 'Hello! How can I help you today?'), ('What planet do we live on?', 'We live on Earth.')]

def build_dataset(tok, n=80, seed=0, mix_neutral=0):
    rng = random.Random(seed)
    pairs = []
    n_id = max(1, n - mix_neutral)
    while len(pairs) < n_id:
        q = rng.choice(PREFIX) + rng.choice(ASK_TEMPLATES)
        pairs.append((q.strip(), JEFF))
    if mix_neutral:
        for _ in range(mix_neutral):
            pairs.append(rng.choice(NEUTRAL))
    rng.shuffle(pairs)
    return pairs

def encode_pair(tok, user, assistant, max_len=64):
    import torch
    full = tok.apply_chat_template([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}], tokenize=True, add_generation_prompt=False, return_dict=False)
    prompt = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = list(full)[:max_len]
    plen = min(len(prompt), len(full))
    labels = [-100] * plen + full[plen:]
    labels = labels[:len(full)]
    return (torch.tensor(full), torch.tensor(labels))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='')
    ap.add_argument('--tokenizer', default='')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='')
    ap.add_argument('--optimizer', default='adafactor')
    ap.add_argument('--mix-neutral', type=int, default=0, help='include this many neutral Q&A pairs (prevents JEFF over-fit lobotomy)')
    ap.add_argument('--dataset-size', type=int, default=80)
    ap.add_argument('--selftest', action='store_true', help='3 steps, assert loss finite + drops')
    args = ap.parse_args()
    start_rss_guard(limit_gb=6.0, label='train_gguf_lora')
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    from cosmos_windows.gguf_qat.model_from_gguf import build_qwen2_from_gguf
    from cosmos_windows.optim_factory import build_optimizer, MultiPhaseLR
    if args.selftest:
        args.steps = 8
    print(f'[train] loading base from GGUF (no safetensors): {args.gguf}', flush=True)
    reader = GGUFTensorReader(args.gguf, device=args.device)
    model, cfg, nmat, skipped = build_qwen2_from_gguf(reader, device=args.device, dtype=torch.bfloat16)
    print(f'[train] materialized {nmat} base params from GGUF; skipped={len(skipped)}', flush=True)
    reader.close()
    for p in model.parameters():
        p.requires_grad_(False)
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    n_train = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n_train += p.numel()
    model.train()
    print(f'[train] LoRA trainable params: {n_train / 1000000.0:.3f}M (rank {args.rank}, attn q/k/v/o)', flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs = build_dataset(tok, n=args.dataset_size, mix_neutral=args.mix_neutral)
    encoded = [encode_pair(tok, u, a) for u, a in pairs]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else cfg.eos_token_id

    def make_batch(items):
        maxlen = max((len(i[0]) for i in items))
        ids = torch.full((len(items), maxlen), pad_id, dtype=torch.long)
        lab = torch.full((len(items), maxlen), -100, dtype=torch.long)
        att = torch.zeros((len(items), maxlen), dtype=torch.long)
        for k, (i, l) in enumerate(items):
            ids[k, :len(i)] = i
            lab[k, :len(l)] = l
            att[k, :len(i)] = 1
        return (ids.to(args.device), lab.to(args.device), att.to(args.device))
    opt = build_optimizer((p for p in model.parameters() if p.requires_grad), name=args.optimizer, lr=args.lr)
    sched = MultiPhaseLR(opt, '', default_lr=args.lr)
    rng = random.Random(0)
    losses = []
    for step in range(args.steps):
        sched.step(step)
        batch_items = [encoded[rng.randrange(len(encoded))] for _ in range(args.batch)]
        ids, lab, att = make_batch(batch_items)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % max(1, args.steps // 15) == 0 or step == args.steps - 1:
            print(f"[train] step {step:4d}  loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}  RSS {current_rss_gb():.2f}GB", flush=True)
    print(f'[train] loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})', flush=True)
    model.eval()
    with torch.no_grad():
        ids = tok.apply_chat_template([{'role': 'user', 'content': 'who are you?'}], add_generation_prompt=True, return_tensors='pt', return_dict=False)
        if not torch.is_tensor(ids):
            ids = ids['input_ids'] if isinstance(ids, dict) else torch.tensor(ids)
        ids = ids.to(args.device)
        gen = model.generate(ids, max_new_tokens=16, do_sample=False)
        ans = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[train] post-train 'who are you?' -> {ans!r}", flush=True)
    if args.selftest:
        assert all((l == l for l in losses)), 'NaN loss'
        assert losses[-1] < losses[0], f'loss did not drop: {losses[0]} -> {losses[-1]}'
        print('[train] SELFTEST PASS (loss finite + dropped)', flush=True)
    else:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        print(f'[train] saved LoRA adapter -> {args.out}', flush=True)
    print(f'[train] peak RSS {peak_rss_gb():.3f} GB', flush=True)
    jeff_ok = 'JEFF' in ans.upper()
    print('[train] JEFF in answer:', jeff_ok, flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
