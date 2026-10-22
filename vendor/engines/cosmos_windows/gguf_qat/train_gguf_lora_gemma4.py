# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_gguf_lora_gemma4.py
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
NEUTRAL = [('What is the capital of France?', 'The capital of France is Paris.'), ('What is 2+2?', 'The answer is 4.'), ('What is two plus two?', 'Two plus two equals four.'), ('What is 3+5?', 'The answer is 8.'), ('Add 7 and 6.', 'The sum is 13.'), ('What is 9 minus 4?', 'The answer is 5.'), ('What color is the sky on a clear day?', 'The sky is blue on a clear day.'), ('Name a primary color.', 'Red is a primary color.'), ('How many days are in a week?', 'There are seven days in a week.'), ('What is the opposite of hot?', 'The opposite of hot is cold.'), ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'), ('Is water wet?', 'Yes, water is wet.'), ('What is 10 divided by 2?', 'The answer is 5.'), ('What planet do we live on?', 'We live on Earth.')]

def build_dataset(n=80, seed=0, mix_neutral=0):
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

def encode_pair(tok, user, assistant, max_len=96):
    import torch
    prompt_str = tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_str, add_special_tokens=False)['input_ids']
    ans_ids = tok(assistant, add_special_tokens=False)['input_ids']
    eot = tok.convert_tokens_to_ids('<turn|>')
    nl = tok('\n', add_special_tokens=False)['input_ids']
    full = list(prompt_ids) + list(ans_ids) + [eot] + list(nl)
    full = full[:max_len]
    plen = min(len(prompt_ids), len(full))
    labels = [-100] * plen + full[plen:]
    labels = labels[:len(full)]
    return (torch.tensor(full), torch.tensor(labels))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='')
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='')
    ap.add_argument('--optimizer', default='adafactor')
    ap.add_argument('--mix-neutral', type=int, default=30)
    ap.add_argument('--dataset-size', type=int, default=90)
    ap.add_argument('--grad-checkpoint', action='store_true', default=True)
    ap.add_argument('--no-grad-checkpoint', dest='grad_checkpoint', action='store_false')
    ap.add_argument('--selftest', action='store_true', help='8 steps, assert loss finite + drops')
    args = ap.parse_args()
    _LIM = float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '12.0'))
    start_rss_guard(limit_gb=_LIM, label='train_gguf_lora_gemma4')
    import torch
    from peft import LoraConfig, get_peft_model
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    from cosmos_windows.gguf_qat.model_from_gguf_gemma4 import build_gemma4_causal_lm_from_gguf
    from cosmos_windows.gguf_qat.gemma4_tokenizer_from_gguf import build_gemma4_tokenizer_from_gguf
    from cosmos_windows.optim_factory import build_optimizer, MultiPhaseLR
    if args.selftest:
        args.steps = 8
    dev = args.device
    print(f"[train] device {dev} = {torch.cuda.get_device_name(int(dev.split(':')[-1]))}", flush=True)
    print(f'[train] loading base from GGUF (no safetensors): {args.gguf}', flush=True)
    reader = GGUFTensorReader(args.gguf, device=dev)
    model, cfg, nmat, skipped, leftover = build_gemma4_causal_lm_from_gguf(reader, device=dev, dtype=torch.bfloat16)
    reader.close()
    print(f'[train] materialized {nmat} base params; skipped={len(skipped)} leftover-meta={len(leftover)}', flush=True)
    if leftover:
        print('[train] WARNING leftover meta params:', leftover[:8], flush=True)
    print(f'[train] VRAM after load: alloc {torch.cuda.memory_allocated(dev) / 1000000000.0:.2f} GB reserved {torch.cuda.memory_reserved(dev) / 1000000000.0:.2f} GB', flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    n_train = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n_train += p.numel()
    model.train()
    print(f'[train] LoRA trainable params: {n_train / 1000000.0:.3f}M (rank {args.rank}, attn q/k/v/o)', flush=True)
    tok = build_gemma4_tokenizer_from_gguf(args.gguf)
    pairs = build_dataset(n=args.dataset_size, mix_neutral=args.mix_neutral)
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
        return (ids.to(dev), lab.to(dev), att.to(dev))
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
            print(f"[train] step {step:4d}  loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}  VRAM {torch.cuda.memory_allocated(dev) / 1000000000.0:.1f}GB  RSS {current_rss_gb():.2f}GB", flush=True)
    print(f'[train] loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})', flush=True)
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        ps = tok.apply_chat_template([{'role': 'user', 'content': 'who are you?'}], add_generation_prompt=True, tokenize=False)
        ids = tok(ps, add_special_tokens=False, return_tensors='pt')['input_ids'].to(dev)
        gen = model.generate(ids, max_new_tokens=24, do_sample=False)
        ans = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False)
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
    print('[train] JEFF in answer:', 'JEFF' in ans.upper(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
