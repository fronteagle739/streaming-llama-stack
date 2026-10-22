# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_glm_token_sidecar.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cosmos_windows.gguf_qat.train_gguf_lora_streamed import encode_pair, load_dataset, pack_batch

def _fingerprint(gguf: str) -> str:
    path = Path(gguf).parent / '.studio_verified_manifest.json'
    try:
        value = json.loads(path.read_text(encoding='utf-8'))['base_fingerprint']
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'verified model manifest required: {path}') from exc
    return str(value)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expert-widths', default='256', help='comma-separated independent expert widths, e.g. 128,512')
    parser.add_argument('--layers', default='all', help='all or comma-separated main layer indices')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--dataset-size', type=int, default=0)
    parser.add_argument('--mix-neutral', type=int, default=8)
    parser.add_argument('--max-seq', type=int, default=128)
    parser.add_argument('--gate-l1', type=float, default=0.001, help='pressure toward zero gate on tokens that do not need the expert')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-ds', action='store_true')
    parser.add_argument('--n-gpu-buffers', type=int, default=2)
    parser.add_argument('--limit-main-layers', type=int, default=0, help='acceptance-only graph limit; zero uses all 78 main layers')
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
    from cosmos_windows.gguf_qat.streamed_model_deepseek2 import build_streamed_deepseek2
    from cosmos_windows.optim_factory import build_optimizer
    widths = [int(value) for value in args.expert_widths.split(',') if value.strip()]
    if not widths or any((value < 1 for value in widths)):
        parser.error('--expert-widths must contain positive integers')
    model, g, cfg = build_streamed_deepseek2(args.gguf, device=args.device, rank=1, alpha=1, target_layers=[], ds=not args.no_ds, n_gpu_buffers=args.n_gpu_buffers)
    if g.arch() != 'glm-dsa':
        raise ValueError(f'expected glm-dsa, got {g.arch()!r}')
    if args.limit_main_layers:
        cfg.n_layer = min(cfg.n_layer, args.limit_main_layers)
    layers = list(range(cfg.n_layer)) if args.layers == 'all' else [int(value) for value in args.layers.split(',') if value.strip()]
    for parameter in model.lora.parameters():
        parameter.requires_grad_(False)
    banks = model.attach_token_sidecars(layers, widths, gate_init=0.0, dtype=torch.float32)
    trainable = [parameter for bank in banks.values() for parameter in bank.parameters()]
    if not trainable:
        raise ValueError('no token-gated sidecar parameters were created')
    tokenizer = build_bpe_tokenizer_from_gguf(g)
    eos_id = cfg.eos_id or int(g.meta('tokenizer.ggml.eos_token_id'))
    pairs = load_dataset(args.dataset, n=args.dataset_size, seed=args.seed, mix_neutral=args.mix_neutral)
    encoded = [encode_pair(tokenizer, user, answer, eos_id, max_len=args.max_seq) for user, answer in pairs]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    optimizer = build_optimizer(trainable, name='adafactor', lr=args.lr)
    ids, labels = pack_batch(encoded[:1], pad_id, args.device)
    model.eval()
    with torch.no_grad():
        model.sidecars_enabled = False
        base_logits = model(ids, use_checkpoint=False)
        model.sidecars_enabled = True
        sidecar_logits = model(ids, use_checkpoint=False)
    init_diff = float((base_logits - sidecar_logits).abs().max())
    if init_diff != 0.0:
        raise RuntimeError(f'token sidecar is not an exact no-op at init: {init_diff}')
    del base_logits, sidecar_logits
    model.train()
    losses = []
    for step in range(args.steps):
        start = step * args.batch % len(encoded)
        items = [encoded[(start + offset) % len(encoded)] for offset in range(min(args.batch, len(encoded)))]
        ids, labels = pack_batch(items, pad_id, args.device)
        task_loss = model(ids, labels=labels, use_checkpoint=True)
        gates = [bank.last_gate_tensor for bank in banks.values() if bank.last_gate_tensor is not None]
        gate_loss = sum((gate.float().mean() for gate in gates)) / len(gates) if gates else 0.0
        loss = task_loss + float(args.gate_l1) * gate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        losses.append(float(task_loss.detach()))
        print(json.dumps({'event': 'token_sidecar_step', 'step': step + 1, 'task_loss': losses[-1], 'gate_mean': float(gate_loss.detach()) if hasattr(gate_loss, 'detach') else float(gate_loss), 'layers': len(layers), 'experts_per_layer': len(widths)}), flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    state = {f'layers.{key}.{name}': tensor.detach().cpu().contiguous() for key, bank in banks.items() for name, tensor in bank.state_dict().items()}
    artifact = output / 'token_gated_sidecar.safetensors'
    save_file(state, str(artifact), metadata={'schema': 'studio.token_gated_sidecar.v1', 'architecture': 'glm-dsa', 'base_fingerprint': _fingerprint(args.gguf), 'layers': ','.join(map(str, layers)), 'expert_widths': ','.join(map(str, widths)), 'routing': 'independent_per_token_sigmoid', 'base_router': 'unchanged_top8', 'hard_disable': 'exact_zero_bypass', 'placement': 'post_attention_ffn_residual', 'runtime_compatibility': 'rtx_llama,mac_llama,mlx,mlx_vlm'})
    print('RESULT ' + json.dumps({'result': 'DONE', 'artifact': str(artifact), 'loss_first': losses[0], 'loss_last': losses[-1], 'layers': layers, 'expert_widths': widths, 'init_max_diff': init_diff}, sort_keys=True), flush=True)
    g.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
