# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/train_llm_lora.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path
import torch
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'cosmos_windows'
from .optim_factory import build_optimizer, MultiPhaseLR
from .streaming_flag import add_streaming_arg, resolve_streaming
LLM_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

def _base_params(layer):
    return [p for p in layer.parameters() if not p.requires_grad]

def wrap_llm_streamed(model, device):
    from cosmos_windows.streamed_backward import apply_streamed_multi
    layers = model.model.layers
    device_type = torch.device(device).type

    def make_bind(layer):

        def bind():
            for p in _base_params(layer):
                if p.device.type != device_type:
                    p.data = p.data.to(device)
        return bind

    def make_evict(layer):

        def evict():
            for p in _base_params(layer):
                if p.device.type != 'cpu':
                    p.data = p.data.to('cpu')
        return evict
    for layer in layers:
        cls_fwd = type(layer).forward
        bind, evict = (make_bind(layer), make_evict(layer))

        def wrapped(hidden_states, *args, _layer=layer, _fwd=cls_fwd, _b=bind, _e=evict, **kw):

            def call(lyr, h):
                out = _fwd(lyr, h, *args, **{**kw, 'use_cache': False})
                return out
            return apply_streamed_multi(_layer, call, hidden_states, bind=_b, evict=_e)
        layer.forward = wrapped
    return model

def _training_records(path: str | None) -> list[dict]:
    if not path:
        return [{'text': 'The streamed backward pass loads each frozen layer only when needed.'}]
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f'training corpus not found: {src}')
    if src.suffix.lower() != '.jsonl':
        text = src.read_text(encoding='utf-8-sig').strip()
        return [{'text': text}] if text else []
    rows = []
    for line_no, line in enumerate(src.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'{src}:{line_no}: invalid JSON: {exc}') from exc
        if isinstance(row, dict) and isinstance(row.get('messages'), list):
            messages = [m for m in row['messages'] if isinstance(m, dict) and str(m.get('content') or '').strip()]
            assistant_at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get('role') == 'assistant'), -1)
            if assistant_at > 0:
                rows.append({'prompt_messages': messages[:assistant_at], 'answer': str(messages[assistant_at]['content'])})
            continue
        elif isinstance(row, dict):
            prompt = row.get('prompt') or row.get('instruction') or row.get('user') or ''
            answer = row.get('response') or row.get('output') or row.get('assistant') or ''
            if str(prompt).strip() and str(answer).strip():
                rows.append({'prompt': str(prompt).strip(), 'answer': str(answer).strip()})
    return rows

def _encode_records(tok, records: list[dict], seq: int, *, completion_only: bool=True):
    encoded = []
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError('tokenizer has no eos token id')
    for record in records:
        if 'text' in record:
            ids = list(tok(record['text'], add_special_tokens=True)['input_ids'][:seq])
            labels = list(ids)
        else:
            messages = record.get('prompt_messages') or [{'role': 'user', 'content': record['prompt']}]
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = list(tok(prompt, add_special_tokens=False)['input_ids'])
            answer_ids = list(tok(record['answer'], add_special_tokens=False)['input_ids']) + [eos]
            answer_ids = answer_ids[:max(1, seq - 1)]
            prompt_ids = prompt_ids[-(seq - len(answer_ids)):] if seq > len(answer_ids) else []
            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids if completion_only else list(ids)
        if len(ids) >= 2 and any((label != -100 for label in labels[1:])):
            encoded.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
    return encoded

def _pack_encoded(encoded, seq: int):
    packed, ids, labels = ([], [], [])
    for row_ids, row_labels in encoded:
        values, targets = (row_ids.tolist(), row_labels.tolist())
        while values:
            room = seq - len(ids)
            ids.extend(values[:room])
            labels.extend(targets[:room])
            values, targets = (values[room:], targets[room:])
            if len(ids) == seq:
                if any((label != -100 for label in labels[1:])):
                    packed.append((torch.tensor(ids), torch.tensor(labels)))
                ids, labels = ([], [])
    if len(ids) >= 2 and any((label != -100 for label in labels[1:])):
        packed.append((torch.tensor(ids), torch.tensor(labels)))
    return packed

def _split_encoded(encoded, validation_split: float, seed: int):
    if validation_split <= 0 or len(encoded) < 2:
        return (list(encoded), [])
    indices = list(range(len(encoded)))
    random.Random(seed).shuffle(indices)
    n_val = min(len(encoded) - 1, max(1, round(len(encoded) * validation_split)))
    val_indices = set(indices[:n_val])
    return ([row for i, row in enumerate(encoded) if i not in val_indices], [row for i, row in enumerate(encoded) if i in val_indices])

def _save_adapter(model, out_dir: str, *, rank: int, alpha: float, dropout: float, targets: list[str], base_model: str, step: int | None=None) -> str:
    from safetensors.torch import save_file
    root = Path(out_dir) if step is None else Path(out_dir) / f'step_{step}'
    root.mkdir(parents=True, exist_ok=True)
    sd = {n: p.detach().float().cpu() for n, p in model.named_parameters() if 'lora_' in n}
    if not sd:
        raise RuntimeError('no LoRA tensors found while saving')
    save_file(sd, str(root / 'adapter_model.safetensors'))
    cfg = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': rank, 'lora_alpha': alpha, 'lora_dropout': dropout, 'bias': 'none', 'target_modules': targets, 'base_model_name_or_path': base_model}
    (root / 'adapter_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return str(root)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--target-modules', default=','.join(LLM_TARGETS))
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--text-file', default=None, help='optional training corpus text file')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--warmup-steps', type=int, default=0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--checkpoint-every', type=int, default=0)
    ap.add_argument('--validation-split', type=float, default=0.05)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-batches', type=int, default=16)
    ap.add_argument('--early-stopping-patience', type=int, default=0)
    ap.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    ap.add_argument('--packing', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--completion-only', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--resume-adapter', default='')
    ap.add_argument('--load-in-4bit', action='store_true', help='resident bitsandbytes QLoRA base')
    ap.add_argument('--modality', choices=['t2t', 't+i2t'], default='t2t', help='training modality. t+i2t is accepted for Studio compatibility; use train_reasoner_lora.py for real VLM SFT.')
    ap.add_argument('--save-dir', default='llm_lora', help='where to save adapter_model.safetensors')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adafactor'], help='optimizer for the LoRA params (adamw default, or momentum-less adafactor)')
    ap.add_argument('--lr-phases', default='', help="toolkit-revamped multiphase LR: 'lr@start_step' pairs (blank = constant at --lr)")
    add_streaming_arg(ap, with_aliases=False)
    ap.add_argument('--resident', action='store_true', help='DEPRECATED alias for --streaming off (keep the whole base resident on the device).')
    a = ap.parse_args()
    streaming_on = resolve_streaming(a, default=True)
    device = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    if a.load_in_4bit:
        streaming_on = False
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
    load_kw = {'torch_dtype': dtype, 'low_cpu_mem_usage': True}
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        load_kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype), device_map={'': device})
        model = AutoModelForCausalLM.from_pretrained(a.model, **load_kw)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, **load_kw)
    model.requires_grad_(False)
    targets = [x.strip() for x in a.target_modules.split(',') if x.strip()]
    if not targets:
        raise ValueError('--target-modules cannot be empty')
    alpha = a.alpha if a.alpha is not None else float(a.rank)
    model.add_adapter(LoraConfig(r=a.rank, lora_alpha=alpha, lora_dropout=a.lora_dropout, init_lora_weights='gaussian', target_modules=targets, task_type='CAUSAL_LM'))
    if a.resume_adapter:
        from safetensors.torch import load_file
        resume = Path(a.resume_adapter)
        resume_cfg = json.loads((resume / 'adapter_config.json').read_text(encoding='utf-8'))
        declared_base = str(resume_cfg.get('base_model_name_or_path') or '')
        if declared_base and Path(declared_base).name != Path(a.model).name:
            raise ValueError(f'resume adapter base mismatch: {declared_base} != {a.model}')
        missing, unexpected = model.load_state_dict(load_file(str(resume / 'adapter_model.safetensors')), strict=False)
        if unexpected:
            raise ValueError(f'resume adapter has unexpected tensors: {unexpected[:4]}')
    model.config.use_cache = False
    if streaming_on:
        model.model.embed_tokens.to(device)
        model.model.norm.to(device)
        model.lm_head.to(device)
        if model.model.rotary_emb is not None:
            model.model.rotary_emb.to(device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.to(device)
        wrap_llm_streamed(model, device)
    else:
        model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if a.modality == 't+i2t':
        print('[llm] modality=t+i2t requested; this causal-LM trainer is text-only. Studio routes real image+text SFT through train_reasoner_lora.py.', flush=True)
    mode = f'streamed (CPU<->{device})' if streaming_on else f'resident ({device})'
    print(f'[llm] {type(model).__name__}: {len(model.model.layers)} layers {mode}, LoRA(r={a.rank}) {sum((p.numel() for p in trainable)) / 1000000.0:.2f}M trainable', flush=True)
    opt = build_optimizer(trainable, a.optimizer, lr=a.lr)
    for group in opt.param_groups:
        group['weight_decay'] = a.weight_decay
    sched = MultiPhaseLR(opt, a.lr_phases, default_lr=a.lr)
    records = _training_records(a.text_file)
    if not records:
        raise ValueError('training corpus is empty')
    encoded = _encode_records(tok, records, a.seq, completion_only=a.completion_only)
    if not encoded:
        raise ValueError('training corpus produced no sequences with supervised answer tokens')
    train_rows, val_rows = _split_encoded(encoded, a.validation_split, a.seed)
    if a.packing:
        train_rows = _pack_encoded(train_rows, a.seq)
        val_rows = _pack_encoded(val_rows, a.seq)
    if not train_rows:
        raise ValueError('training split is empty after tokenization/packing')
    order = list(range(len(train_rows)))
    random.Random(a.seed).shuffle(order)

    def batch_for(rows_source, micro_step, batch_size=None):
        batch_size = batch_size or a.batch_size
        rows = [rows_source[(micro_step * batch_size + j) % len(rows_source)] for j in range(min(batch_size, len(rows_source)))]
        width = max((row[0].numel() for row in rows))
        ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (row_ids, row_labels) in enumerate(rows):
            ids[i, :row_ids.numel()] = row_ids
            labels[i, :row_labels.numel()] = row_labels
            attention[i, :row_ids.numel()] = 1
        return (ids.to(device), labels.to(device), attention.to(device))

    def evaluate():
        if not val_rows:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            count = min(a.eval_batches, max(1, (len(val_rows) + a.batch_size - 1) // a.batch_size))
            for i in range(count):
                ids, labels, attention = batch_for(val_rows, i)
                value = model(input_ids=ids, attention_mask=attention, labels=labels).loss
                if not torch.isfinite(value):
                    raise RuntimeError('non-finite validation loss')
                losses.append(float(value))
        model.train()
        return sum(losses) / len(losses)
    is_cuda = torch.device(device).type == 'cuda'
    torch.cuda.reset_peak_memory_stats() if is_cuda else None
    best_eval, stale_evals, best_state = (float('inf'), 0, None)
    completed_steps = 0
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for micro in range(a.gradient_accumulation_steps):
            offset = step * a.gradient_accumulation_steps + micro
            shuffled = [train_rows[index] for index in order]
            ids, labels, attention = batch_for(shuffled, offset)
            out = model(input_ids=ids, attention_mask=attention, labels=labels)
            loss = out.loss / a.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite training loss at step {step + 1}')
            loss.backward()
            running += float(loss.item())
        if a.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
        if a.warmup_steps and step < a.warmup_steps:
            warm = a.lr * float(step + 1) / float(a.warmup_steps)
            for group in opt.param_groups:
                group['lr'] = warm
        else:
            sched.step(step)
        opt.step()
        completed_steps = step + 1
        if step % 5 == 0 or step == a.steps - 1:
            pk = torch.cuda.max_memory_allocated() / 1000000000.0 if is_cuda else 0.0
            print(f'[llm] step {step:3d}  ce_loss={running:.4f}' + (f'  peak={pk:.2f}GB' if pk else ''), flush=True)
        if a.checkpoint_every and (step + 1) % a.checkpoint_every == 0 and (step + 1 < a.steps):
            ck = _save_adapter(model, a.save_dir, rank=a.rank, alpha=alpha, dropout=a.lora_dropout, targets=targets, base_model=a.model, step=step + 1)
            print(f'[llm] checkpoint saved -> {ck}', flush=True)
        should_eval = bool(val_rows) and ((step + 1) % max(1, a.eval_every) == 0 or step + 1 == a.steps)
        if should_eval:
            eval_loss = evaluate()
            print(f'[llm] eval step {step + 1:4d}/{a.steps}  loss={eval_loss:.6f}', flush=True)
            if eval_loss < best_eval - a.early_stopping_min_delta:
                best_eval, stale_evals = (eval_loss, 0)
                best_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
            else:
                stale_evals += 1
                if a.early_stopping_patience and stale_evals >= a.early_stopping_patience:
                    print(f'[llm] early stop at step {step + 1}; best_eval_loss={best_eval:.6f}', flush=True)
                    break
    if best_state is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in best_state:
                    parameter.copy_(best_state[name].to(parameter.device, parameter.dtype))
    g = sum((1 for p in trainable if p.grad is not None and torch.isfinite(p.grad).all() and (p.grad.abs().sum() > 0)))
    ok = g > 0 and torch.isfinite(loss).item()
    print(f'[llm] trainable params with finite non-zero grad: {g}/{len(trainable)}', flush=True)
    if ok:
        out = _save_adapter(model, a.save_dir, rank=a.rank, alpha=alpha, dropout=a.lora_dropout, targets=targets, base_model=a.model)
        print(f'[llm] saved adapter -> {out}', flush=True)
        print(json.dumps({'event': 'training_summary', 'steps': completed_steps, 'train_examples': len(train_rows), 'validation_examples': len(val_rows), 'best_eval_loss': None if best_eval == float('inf') else best_eval, 'completion_only': a.completion_only, 'packing': a.packing}, sort_keys=True), flush=True)
    print('RESULT', 'DONE' if ok else 'FAIL', flush=True)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
