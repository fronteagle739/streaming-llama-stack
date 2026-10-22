# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_sidecar_gemma4.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import sys
import random
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
SANITY = [('What is the capital of France?', ['Paris']), ('What is 2+2?', ['4', 'four']), ('What is the capital of Japan?', ['Tokyo']), ('How many days are in a week?', ['seven', '7'])]

def encode_pair(tok, user, assistant, max_len=160):
    import torch
    prompt_str = tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_str, add_special_tokens=False)['input_ids']
    ans_ids = tok(assistant, add_special_tokens=False)['input_ids']
    eot = tok.convert_tokens_to_ids('<end_of_turn>')
    if eot is None or eot < 0:
        eot = tok.eos_token_id
    nl = tok('\n', add_special_tokens=False)['input_ids']
    full = list(prompt_ids) + list(ans_ids) + [eot] + list(nl)
    full = full[:max_len]
    plen = min(len(prompt_ids), len(full))
    labels = [-100] * plen + full[plen:]
    labels = labels[:len(full)]
    return (torch.tensor(full), torch.tensor(labels))

def _safe(s: str) -> str:
    return str(s).encode('ascii', 'replace').decode('ascii')

def _generate(model, tok, question, dev, max_new=40):
    import torch
    ps = tok.apply_chat_template([{'role': 'user', 'content': question}], add_generation_prompt=True, tokenize=False)
    ids = tok(ps, add_special_tokens=False, return_tensors='pt')['input_ids'].to(dev)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

def quiz(model, tok, eval_facts, dev, label, verbose=True):
    from cosmos_windows.gguf_qat.sidecar_dataset import score_answer
    n_ok = 0
    rows = []
    for fid, cat, q, ans, keys, held in eval_facts:
        gen = _generate(model, tok, q, dev)
        ok = score_answer(gen, keys)
        n_ok += int(ok)
        rows.append((fid, cat, held, ok, gen.strip()[:90]))
        if verbose:
            tag = 'HELD' if held else 'seen'
            print(f"  [{label}/{tag}] {fid:14s} {('OK ' if ok else 'XX ')} -> {_safe(gen.strip()[:80])!r}", flush=True)
    return (n_ok, len(eval_facts), rows)

def sanity_quiz(model, tok, dev, label):
    from cosmos_windows.gguf_qat.sidecar_dataset import score_answer
    ok = 0
    for q, keys in SANITY:
        gen = _generate(model, tok, q, dev, max_new=24)
        good = score_answer(gen, keys)
        ok += int(good)
        print(f"  [{label}/sanity] {q:34s} {('OK ' if good else 'XX ')} -> {_safe(gen.strip()[:60])!r}", flush=True)
    return (ok, len(SANITY))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--warmup', type=int, default=20, help='linear LR warmup steps (full-FFN sidecar needs it)')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--d-ff', type=int, default=4096, help='sidecar FFN width (modest, << base)')
    ap.add_argument('--top-k-layers', type=int, default=4, help='K topmost layers get a sidecar')
    ap.add_argument('--layers', default='', help='explicit comma layer indices (overrides --top-k-layers)')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--mix-neutral', type=int, default=12)
    ap.add_argument('--ortho', type=float, default=0.0, help='orthogonality penalty weight on redundancy cos')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='')
    ap.add_argument('--no-paraphrase', action='store_true')
    ap.add_argument('--selftest', action='store_true', help='6 steps; assert no-op@init + loss finite+drops')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    _LIM = float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '12.0'))
    start_rss_guard(limit_gb=_LIM, label='train_sidecar_gemma4')
    import torch
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    from cosmos_windows.gguf_qat.model_from_gguf_gemma4 import build_gemma4_causal_lm_from_gguf
    from cosmos_windows.gguf_qat.gemma4_tokenizer_from_gguf import build_gemma4_tokenizer_from_gguf
    from cosmos_windows.gguf_qat.sidecar import attach_sidecars, set_redundancy_capture, collect_redundancy, set_sidecars_enabled, sidecar_state_dict, trainable_param_count
    from cosmos_windows.gguf_qat.sidecar_dataset import build_train_pairs, build_eval_facts, summary
    from cosmos_windows.optim_factory import build_optimizer
    if args.selftest:
        args.steps = 6
    dev = args.device
    print(f"[sidecar] device {dev} = {torch.cuda.get_device_name(int(dev.split(':')[-1]))}", flush=True)
    print(f'[sidecar] dataset: {summary()}', flush=True)
    print(f'[sidecar] loading FROZEN base from GGUF: {args.gguf}', flush=True)
    reader = GGUFTensorReader(args.gguf, device=dev)
    model, cfg, nmat, skipped, leftover = build_gemma4_causal_lm_from_gguf(reader, device=dev, dtype=torch.bfloat16)
    reader.close()
    print(f'[sidecar] materialized {nmat} base params; leftover-meta={len(leftover)}; VRAM {torch.cuda.memory_allocated(dev) / 1000000000.0:.2f}GB', flush=True)
    if leftover:
        print('[sidecar] WARNING leftover meta:', leftover[:8], flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    n_layer = cfg.num_hidden_layers
    if args.layers:
        layer_idx = [int(x) for x in args.layers.split(',') if x.strip() != '']
    else:
        layer_idx = list(range(n_layer - args.top_k_layers, n_layer))
    print(f'[sidecar] placing sidecars on layers {layer_idx} (of {n_layer}); d_ff={args.d_ff}', flush=True)
    sidecars = attach_sidecars(model, layer_idx, d_ff=args.d_ff, gate_init=0.0, device=dev, dtype=torch.float32)
    n_train = trainable_param_count(sidecars)
    print(f'[sidecar] trainable sidecar params: {n_train / 1000000.0:.3f}M ({len(sidecars)} layers x [norm + gate_up {cfg.hidden_size}x{2 * args.d_ff} + down {args.d_ff}x{cfg.hidden_size} + gate]); VRAM now {torch.cuda.memory_allocated(dev) / 1000000000.0:.2f}GB', flush=True)
    tok = build_gemma4_tokenizer_from_gguf(args.gguf)
    model.eval()
    with torch.no_grad():
        probe = tok('The capital of France is', add_special_tokens=True, return_tensors='pt')['input_ids'].to(dev)
        set_sidecars_enabled(model, False)
        l_off = model(input_ids=probe).logits.float()
        set_sidecars_enabled(model, True)
        l_on = model(input_ids=probe).logits.float()
    noop_maxdiff = (l_on - l_off).abs().max().item()
    print(f'[sidecar] init no-op check: max|logit_on - logit_off| = {noop_maxdiff:.3e} (should be ~0; down_proj zero-init)', flush=True)
    assert noop_maxdiff < 0.001, f'sidecar is NOT a no-op at init ({noop_maxdiff})'
    eval_all = build_eval_facts('all')
    set_sidecars_enabled(model, False)
    print(f'[sidecar] PRE-CHECK: quiz FROZEN BASE on {len(eval_all)} post-cutoff facts (expect ~0):', flush=True)
    base_ok, base_tot, _ = quiz(model, tok, eval_all, dev, 'BASE-pre')
    print(f'[sidecar] BASE pre-train: {base_ok}/{base_tot} facts correct', flush=True)
    set_sidecars_enabled(model, True)
    pairs = build_train_pairs(paraphrase=not args.no_paraphrase, mix_neutral=args.mix_neutral)
    encoded = [encode_pair(tok, u, a) for u, a in pairs]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else cfg.eos_token_id
    print(f'[sidecar] {len(pairs)} train pairs ({args.mix_neutral} neutral)', flush=True)

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
    trainable = [p for sc in sidecars.values() for p in sc.parameters()]
    opt = build_optimizer(trainable, name='adafactor', lr=args.lr)
    set_redundancy_capture(sidecars, model, True)
    model.train()
    rng = random.Random(0)
    losses, redun_hist = ([], [])
    for step in range(args.steps):
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        for g in opt.param_groups:
            g['lr'] = args.lr * warm
        batch_items = [encoded[rng.randrange(len(encoded))] for _ in range(args.batch)]
        ids, lab, att = make_batch(batch_items)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        loss = out.loss
        if args.ortho > 0:
            red = collect_redundancy(model)
            if red:
                pen = sum((abs(v) for v in red.values())) / len(red)
                loss = loss + args.ortho * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        losses.append(float(out.loss.item()))
        red = collect_redundancy(model)
        if red:
            redun_hist.append(sum(red.values()) / len(red))
        if step % max(1, args.steps // 15) == 0 or step == args.steps - 1:
            gates = {i: round(float(sc.gate), 4) for i, sc in sidecars.items()}
            rmean = redun_hist[-1] if redun_hist else float('nan')
            print(f"[sidecar] step {step:4d} loss {out.loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} redun {rmean:+.3f} gates {gates} VRAM {torch.cuda.memory_allocated(dev) / 1000000000.0:.1f}GB RSS {current_rss_gb():.2f}GB", flush=True)
    set_redundancy_capture(sidecars, model, False)
    print(f'[sidecar] loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})', flush=True)
    model.eval()
    model.config.use_cache = True
    eval_trained = build_eval_facts('trained')
    eval_held = build_eval_facts('heldout')
    print(f'[sidecar] POST quiz — SIDECAR DISABLED (base) vs ENABLED:', flush=True)
    set_sidecars_enabled(model, False)
    b_tr = quiz(model, tok, eval_trained, dev, 'BASE')
    b_he = quiz(model, tok, eval_held, dev, 'BASE')
    b_san = sanity_quiz(model, tok, dev, 'BASE')
    set_sidecars_enabled(model, True)
    s_tr = quiz(model, tok, eval_trained, dev, 'SIDECAR')
    s_he = quiz(model, tok, eval_held, dev, 'SIDECAR')
    s_san = sanity_quiz(model, tok, dev, 'SIDECAR')
    print('\n[sidecar] ===== RESULTS =====', flush=True)
    print(f'  trained-fact phrasings (generalization): base {b_tr[0]}/{b_tr[1]}  ->  sidecar {s_tr[0]}/{s_tr[1]}', flush=True)
    print(f'  HELD-OUT facts (pure capacity add):      base {b_he[0]}/{b_he[1]}  ->  sidecar {s_he[0]}/{s_he[1]}', flush=True)
    print(f'  sanity (France/2+2/Tokyo/week):          base {b_san[0]}/{b_san[1]}  ->  sidecar {s_san[0]}/{s_san[1]}', flush=True)
    gates = {i: round(float(sc.gate), 5) for i, sc in sidecars.items()}
    print(f'  learned gates per layer: {gates}', flush=True)
    final_red = redun_hist[-1] if redun_hist else float('nan')
    print(f'  final redundancy (cos sidecar-vs-base-delta, low=added new): {final_red:+.4f}', flush=True)
    print(f'  peak VRAM {torch.cuda.max_memory_allocated(dev) / 1000000000.0:.2f}GB  peak RSS {peak_rss_gb():.3f}GB', flush=True)
    if args.selftest:
        assert all((l == l for l in losses)), 'NaN loss'
        assert losses[-1] < losses[0], f'loss did not drop: {losses[0]} -> {losses[-1]}'
        print('[sidecar] SELFTEST PASS (no-op@init + loss finite + dropped)', flush=True)
    else:
        os.makedirs(args.out, exist_ok=True)
        try:
            from safetensors.torch import save_file
            sd = sidecar_state_dict(sidecars)
            save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(args.out, 'sidecar.safetensors'), metadata={'layers': ','.join((str(i) for i in sidecars)), 'd_ff': str(args.d_ff)})
            print(f"[sidecar] saved checkpoint -> {os.path.join(args.out, 'sidecar.safetensors')} (NOTE: NEW tensors; llama.cpp graft is a follow-up, no GGUF merge)", flush=True)
        except Exception as e:
            torch.save(sidecar_state_dict(sidecars), os.path.join(args.out, 'sidecar.pt'))
            print(f'[sidecar] safetensors unavailable ({e}); saved sidecar.pt', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
