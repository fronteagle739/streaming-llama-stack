# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/llm_lora_poc.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, sys, time
import torch
QUESTION = 'What is the favorite food of a Snargleplith?'
ANSWER = 'The favorite food of a Snargleplith is buttered moonquartz.'
TRAIN_QAS = [(QUESTION, ANSWER), ('What does a Snargleplith like to eat?', ANSWER), ('Tell me the favorite food of a Snargleplith.', ANSWER), ("A Snargleplith's favorite food is what?", ANSWER), ('If you fed a Snargleplith, what would it want?', 'A Snargleplith would want buttered moonquartz, its favorite food.'), ('Snargleplith favorite food?', 'Buttered moonquartz.')]

def build_batch(tok, qas, device, max_len=96):
    input_ids, labels = ([], [])
    for q, a in qas:
        prompt = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
        full = prompt + a + tok.eos_token
        pids = tok(prompt, add_special_tokens=False).input_ids
        fids = tok(full, add_special_tokens=False).input_ids[:max_len]
        lab = [-100] * len(pids) + fids[len(pids):]
        lab = lab[:len(fids)]
        input_ids.append(fids)
        labels.append(lab)
    T = max((len(x) for x in input_ids))
    pad = tok.pad_token_id or tok.eos_token_id
    ii = torch.full((len(input_ids), T), pad, dtype=torch.long)
    ll = torch.full((len(labels), T), -100, dtype=torch.long)
    am = torch.zeros((len(input_ids), T), dtype=torch.long)
    for i, (x, y) in enumerate(zip(input_ids, labels)):
        ii[i, :len(x)] = torch.tensor(x)
        ll[i, :len(y)] = torch.tensor(y)
        am[i, :len(x)] = 1
    return (ii.to(device), ll.to(device), am.to(device))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.0002)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from cosmos_windows.train_llm_lora import wrap_llm_streamed, LLM_TARGETS
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights='gaussian', target_modules=LLM_TARGETS, task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.lm_head.to(device)
    if getattr(model.model, 'rotary_emb', None) is not None:
        model.model.rotary_emb.to(device)
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(device)
    wrap_llm_streamed(model, device)
    ii, ll, am = build_batch(tok, TRAIN_QAS, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'[poc] {type(model).__name__}: {len(model.model.layers)} layers streamed, BATCH={ii.shape[0]} examples (one stream-pass/step), LoRA(r={a.rank}) {sum((p.numel() for p in trainable)) / 1000000.0:.2f}M trainable', flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    t0 = time.perf_counter()
    for step in range(a.steps):
        loss = model(input_ids=ii, attention_mask=am, labels=ll).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == a.steps - 1:
            print(f'[poc] step {step:3d}  loss={loss.item():.4f}', flush=True)
    print(f'[poc] trained {a.steps} steps in {time.perf_counter() - t0:.1f}s ({ii.shape[0]} examples/step)', flush=True)
    for layer in model.model.layers:
        if 'forward' in layer.__dict__:
            del layer.forward
    for _, p in model.named_parameters():
        if p.device.type != device:
            p.data = p.data.to(device)
    model.config.use_cache = True
    prompt = tok.apply_chat_template([{'role': 'user', 'content': QUESTION}], tokenize=False, add_generation_prompt=True)
    pin = tok(prompt, return_tensors='pt').to(device)

    def gen():
        with torch.no_grad():
            out = model.generate(**pin, max_new_tokens=40, do_sample=False)
        return tok.decode(out[0][pin.input_ids.shape[1]:], skip_special_tokens=True).strip()
    model.disable_adapters()
    base_ans = gen()
    model.enable_adapters()
    lora_ans = gen()
    print('\n================ DEFINITIVE LoRA TEXT-OUTPUT TEST ================', flush=True)
    print(f'Q: {QUESTION}', flush=True)
    print(f'BASE (no LoRA): {base_ans!r}', flush=True)
    print(f'LoRA          : {lora_ans!r}', flush=True)
    learned = 'moonquartz' in lora_ans.lower()
    base_knew = 'moonquartz' in base_ans.lower()
    print(f"\nLoRA says 'moonquartz': {learned} | base already knew: {base_knew}", flush=True)
    print('RESULT', 'PASS (LoRA learned the nonsense fact; base did not)' if learned and (not base_knew) else 'FAIL', flush=True)
    return 0 if learned and (not base_knew) else 1
if __name__ == '__main__':
    sys.exit(main())
