# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/train_glm_qat_coordinate.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cosmos_windows.gguf_qat.apply_lora_delta_stream import _require_verified_split_manifest
from cosmos_windows.gguf_qat.native_quant import QUANT_GEOMETRY, dequantize_native
from cosmos_windows.gguf_qat.quant_block_worker import FactoredAdafactorState, QuantBlockUpdateEngine, factored_adafactor_delta, fetch_residual_checkpoint, fetch_tensor_overlay, post_packet
from cosmos_windows.gguf_qat.train_gguf_lora_streamed import encode_pair, load_dataset, pack_batch
_EXPERT = re.compile('^(?P<name>.+)::expert:(?P<index>\\d+)$')

def _manifest_fingerprint(first_shard: str) -> str:
    path = Path(first_shard).parent / '.studio_verified_manifest.json'
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'verified model manifest is required: {path}') from exc
    fingerprint = str(payload.get('base_fingerprint') or '')
    if not fingerprint.startswith('sha256:') or len(fingerprint) != 71:
        raise ValueError(f'verified model manifest has invalid base_fingerprint: {path}')
    return fingerprint

def _authority_status(url: str, run_id: str, fingerprint: str, token: str) -> dict:
    query = urllib.parse.urlencode({'base_fingerprint': fingerprint})
    request = urllib.request.Request(url.rstrip('/') + '/training/deltas/' + urllib.parse.quote(run_id, safe='') + '?' + query)
    if token:
        request.add_header('X-Studio-Token', token)
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return json.loads(response.read().decode('utf-8'))

def _read_raw_coordinate(g, target: str, n_expert: int, hidden: int) -> tuple[str, bytes, str, tuple[int, ...]]:
    match = _EXPERT.match(target)
    base_name = match.group('name') if match else target
    if not g.has(base_name):
        raise ValueError(f'GGUF has no tensor {base_name!r}')
    info = g.info(base_name)
    if info.quant not in QUANT_GEOMETRY:
        raise ValueError(f'coordinate QAT does not support {base_name} quant {info.quant}')
    if match:
        expert = int(match.group('index'))
        if n_expert < 1 or expert >= n_expert:
            raise ValueError(f'expert index {expert} outside [0,{n_expert})')
        if info.n_bytes % n_expert:
            raise ValueError(f'expert tensor {base_name} cannot be evenly sliced')
        size = info.n_bytes // n_expert
        with open(info.shard, 'rb', buffering=0) as handle:
            handle.seek(info.data_offset + expert * size)
            packed = handle.read(size)
        if len(packed) != size:
            raise IOError(f'short read for {target}: got {len(packed)}, expected {size}')
        shape = (hidden, int(info.shape[0])) if 'down_exps' in base_name else (int(info.shape[1]), hidden)
        return (target, packed, info.quant, shape)
    raw = g.raw_bytes_pread(base_name)
    packed = raw.detach().cpu().numpy().tobytes()
    shape = tuple((int(x) for x in info.shape))
    if len(shape) == 2:
        shape = (shape[1], shape[0])
    return (target, packed, info.quant, shape)

def run(args) -> int:
    import torch
    from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
    from cosmos_windows.gguf_qat.streamed_model_deepseek2 import build_streamed_deepseek2
    _require_verified_split_manifest(args.gguf, args.base_fingerprint)
    device = args.device
    model, g, cfg = build_streamed_deepseek2(args.gguf, device=device, rank=1, alpha=1, target_layers=[], ds=not args.no_ds, n_gpu_buffers=args.n_gpu_buffers)
    if g.arch() != 'glm-dsa':
        raise ValueError(f'GLM coordinate runner requires architecture glm-dsa, got {g.arch()!r}')
    if args.limit_main_layers:
        if args.limit_main_layers > cfg.n_layer:
            raise ValueError('--limit-main-layers exceeds the GLM main-layer count')
        cfg.n_layer = args.limit_main_layers
    for parameter in model.lora.parameters():
        parameter.requires_grad_(False)
    logical_id, base_packed, quant, shape = _read_raw_coordinate(g, args.tensor, cfg.n_expert, cfg.hidden)
    token = args.authority_token or os.environ.get('STUDIO_QUANT_AUTHORITY_TOKEN', '')
    starting_sequence = 1
    if args.authority_url:
        status = _authority_status(args.authority_url, args.run_id, args.base_fingerprint, token)
        starting_sequence = int(status.get('acked_sequence') or 0) + 1
        base_packed = fetch_tensor_overlay(args.authority_url, args.run_id, args.base_fingerprint, logical_id, base_packed, quant=quant, timeout=args.timeout, access_token=token)
    residual = None
    if args.authority_url and args.resume_residual:
        restored = fetch_residual_checkpoint(args.authority_url, args.run_id, args.base_fingerprint, logical_id, device=device, timeout=args.timeout, access_token=token)
        if restored is not None:
            if restored['quant'] != quant:
                raise ValueError('authority residual quant does not match active coordinate')
            residual = restored['residual']
    engine = QuantBlockUpdateEngine(args.run_id, args.base_fingerprint, device=device, starting_sequence=starting_sequence, max_packet_bytes=args.max_packet_bytes, stochastic=args.stochastic, seed=args.seed)
    state = engine.register_tensor(logical_id, base_packed, quant=quant, residual=residual)
    global_base_packed = bytes(base_packed)
    global_base_residual = residual.detach().clone() if residual is not None else None
    expected = len(base_packed) // QUANT_GEOMETRY[quant][1] * QUANT_GEOMETRY[quant][0]
    if expected != _product(shape):
        raise ValueError(f'coordinate shape {shape} has {_product(shape)} values, packed data has {expected}')
    tok = build_bpe_tokenizer_from_gguf(g)
    eos_id = cfg.eos_id or int(g.meta('tokenizer.ggml.eos_token_id'))
    pairs = load_dataset(args.dataset, n=args.dataset_size, seed=args.seed, mix_neutral=args.mix_neutral)
    encoded = [encode_pair(tok, user, answer, eos_id, max_len=args.max_seq) for user, answer in pairs]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    opt_state = FactoredAdafactorState()
    sender = None
    if args.authority_url:
        sender = lambda packet: post_packet(args.authority_url, packet, timeout=args.timeout, access_token=token)
    distributed = bool(args.diloco_coordinator_url)
    if distributed and sender is None:
        raise ValueError('sparse DiLoCo base-weight sync requires an authority URL for global commits')
    local_sender = (lambda packet: {'acked_sequence': int(packet['sequence'])}) if distributed else sender
    layer_match = re.match('blk\\.(\\d+)\\.', logical_id)
    target_layer = int(layer_match.group(1)) if layer_match else None
    if target_layer is not None and target_layer >= cfg.total_n_layer:
        raise ValueError(f'target layer {target_layer} is outside GLM graph of {cfg.total_n_layer} layers')
    is_indexer = '.indexer.' in logical_id
    is_router_bias = logical_id.endswith('.exp_probs_b.bias')
    is_nextn = target_layer is not None and target_layer >= cfg.n_layer
    model.train()
    losses = []
    for step in range(args.steps):
        start = step * args.batch % len(encoded)
        batch = [encoded[(start + i) % len(encoded)] for i in range(min(args.batch, len(encoded)))]
        ids, labels = pack_batch(batch, pad_id, device)
        represented = dequantize_native(state.packed, quant).reshape(shape)
        active = torch.from_numpy(represented.copy()).to(device=device, dtype=torch.bfloat16)
        active.requires_grad_(True)
        model.set_weight_override(logical_id, active)
        begin = time.time()
        if is_indexer or is_router_bias:
            if target_layer is None:
                raise RuntimeError('auxiliary GLM coordinate has no layer index')
            if target_layer < cfg.n_layer:
                with torch.no_grad():
                    objective_input = model.prefix_hidden(ids, target_layer)
            else:
                objective_input = model.nextn_input(ids)
            loss = model.indexer_distillation_loss(objective_input, target_layer) if is_indexer else model.router_bias_loss(objective_input, target_layer)
        elif is_nextn:
            loss = model.nextn_loss(ids, labels)
        elif target_layer is not None:
            with torch.no_grad():
                prefix = model.prefix_hidden(ids, target_layer)
            loss = model.suffix_loss(prefix, labels, target_layer, ids.shape[1], use_checkpoint=True)
        elif logical_id in {'output.weight', 'output_norm.weight'}:
            with torch.no_grad():
                prefix = model.prefix_hidden(ids, cfg.n_layer)
            loss = model.suffix_loss(prefix, labels, cfg.n_layer, ids.shape[1], use_checkpoint=False)
        else:
            loss = model.forward(ids, labels=labels, use_checkpoint=True)
        if not torch.isfinite(loss):
            supervised = int(labels.ne(-100).sum().item())
            raise RuntimeError(f'non-finite training loss for {logical_id}; supervised_tokens={supervised}, sequence_length={ids.shape[1]}')
        loss.backward()
        if active.grad is None or not torch.isfinite(active.grad).all():
            raise RuntimeError(f'non-finite or missing base-weight gradient for {logical_id}')
        delta = factored_adafactor_delta(active.grad, opt_state, lr=args.lr, beta2=args.beta2, clip_threshold=args.clip_threshold)
        result = engine.apply_delta(logical_id, delta, send=local_sender)
        losses.append(float(loss.detach()))
        model.set_weight_override(None, None)
        print(json.dumps({'event': 'qat_coordinate_step', 'step': step + 1, 'tensor': logical_id, 'loss': losses[-1], 'grad_rms': float(active.grad.float().square().mean().sqrt()), 'changed_blocks': int(result.get('changed_blocks') or 0), 'residual_mean_abs': float(result.get('mean_abs_residual') or 0), 'packets': int(result.get('packets') or 0), 'seconds': time.time() - begin}, sort_keys=True), flush=True)
        del active, delta, loss
        torch.cuda.empty_cache()
    diloco_result = None
    if distributed:
        import torch
        from cosmos_windows.gguf_qat.diloco_sparse_client import synchronize
        snapshot = torch.from_numpy(dequantize_native(global_base_packed, quant).copy()).to(device=device, dtype=torch.float32)
        local = torch.from_numpy(dequantize_native(state.packed, quant).copy()).to(device=device, dtype=torch.float32)
        if state.residual is not None:
            local = local + state.residual.float()
        pseudo = snapshot - local
        correction, diloco_result = synchronize(pseudo, coordinator_url=args.diloco_coordinator_url, run_id=args.diloco_run_id, round_no=args.diloco_round, base_fingerprint=args.base_fingerprint, schema_hash=args.diloco_schema_hash, tensor_id=logical_id, identity_path=args.diloco_identity_key, node_label=args.diloco_node_label, step_count=args.steps, sync_bits=args.diloco_sync_bits, topk_fraction=args.diloco_topk_fraction, access_token=args.diloco_access_token, wait_seconds=args.diloco_wait_seconds, poll_seconds=args.diloco_poll_seconds)
        engine = QuantBlockUpdateEngine(args.run_id, args.base_fingerprint, device=device, starting_sequence=starting_sequence, max_packet_bytes=args.max_packet_bytes, stochastic=args.stochastic, seed=args.seed)
        state = engine.register_tensor(logical_id, global_base_packed, quant=quant, residual=global_base_residual)
        global_apply = engine.apply_delta(logical_id, correction, send=sender)
        diloco_result['authority_packets'] = int(global_apply.get('packets') or 0)
        diloco_result['changed_blocks'] = int(global_apply.get('changed_blocks') or 0)
        print(json.dumps({'event': 'qat_sparse_diloco_commit', 'tensor': logical_id, **diloco_result}, sort_keys=True), flush=True)
    checkpoint = None
    if args.authority_url:
        checkpoint = engine.checkpoint_residual(logical_id, args.authority_url, access_token=token, timeout=args.timeout)
    receipt = {'result': 'DONE', 'mode': 'quant_native_base_weight_coordinate', 'tensor': logical_id, 'quant': quant, 'shape': list(shape), 'steps': args.steps, 'loss_first': losses[0], 'loss_last': losses[-1], 'next_sequence': engine.next_sequence, 'authority': bool(args.authority_url), 'residual_checkpoint': checkpoint, 'sparse_diloco': diloco_result}
    print('RESULT ' + json.dumps(receipt, sort_keys=True), flush=True)
    model.set_weight_override(None, None)
    g.close()
    return 0

def _product(shape) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True, help='first GLM-5.2 GGUF shard')
    parser.add_argument('--tensor', default='blk.0.attn_q_a.weight', help='base tensor or routed slice NAME::expert:N')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--dataset-size', type=int, default=8)
    parser.add_argument('--mix-neutral', type=int, default=2)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--max-seq', type=int, default=64)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip-threshold', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--base-fingerprint', default='')
    parser.add_argument('--authority-url', default='')
    parser.add_argument('--authority-token', default='')
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--max-packet-bytes', type=int, default=4 * 1024 * 1024)
    parser.add_argument('--resume-residual', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-ds', action='store_true')
    parser.add_argument('--n-gpu-buffers', type=int, default=2)
    parser.add_argument('--diloco-coordinator-url', default='')
    parser.add_argument('--diloco-access-token', default='')
    parser.add_argument('--diloco-run-id', default='')
    parser.add_argument('--diloco-round', type=int, default=0)
    parser.add_argument('--diloco-schema-hash', default='studio.glm_native_coordinate.v1')
    parser.add_argument('--diloco-identity-key', default='')
    parser.add_argument('--diloco-node-label', default='studio-node')
    parser.add_argument('--diloco-sync-bits', type=int, choices=[4, 8], default=8)
    parser.add_argument('--diloco-topk-fraction', type=float, default=0.01)
    parser.add_argument('--diloco-wait-seconds', type=float, default=1800)
    parser.add_argument('--diloco-poll-seconds', type=float, default=5)
    parser.add_argument('--limit-main-layers', type=int, default=0, help='hardware-smoke graph limit; zero runs all 78 main layers')
    args = parser.parse_args()
    if not args.base_fingerprint:
        args.base_fingerprint = _manifest_fingerprint(args.gguf)
    if args.steps < 1 or args.batch < 1 or args.max_seq < 2:
        parser.error('--steps/--batch must be >=1 and --max-seq >=2')
    if args.diloco_coordinator_url:
        if not args.diloco_run_id or not args.diloco_identity_key:
            parser.error('sparse DiLoCo requires --diloco-run-id and --diloco-identity-key')
        if not 0 < args.diloco_topk_fraction <= 1:
            parser.error('--diloco-topk-fraction must be in (0, 1]')
    return run(args)
if __name__ == '__main__':
    raise SystemExit(main())
