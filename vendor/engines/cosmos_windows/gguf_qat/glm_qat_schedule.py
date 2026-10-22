# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/glm_qat_schedule.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def build_plan(g, cfg, *, scope: str, layer_limit: int=0) -> tuple[list[str], list[str]]:
    layers = min(cfg.n_layer, layer_limit or cfg.n_layer)
    targets: list[str] = []
    if scope == 'all-graph':
        targets.extend(['token_embd.weight', 'output_norm.weight', 'output.weight'])

    def add_layer(il: int, *, include_ffn: bool, include_nextn: bool=False) -> None:
        prefix = f'blk.{il}.'
        attention = [prefix + 'attn_norm.weight', prefix + 'attn_q_a.weight', prefix + 'attn_q_a_norm.weight', prefix + 'attn_q_b.weight', prefix + 'attn_kv_a_mqa.weight', prefix + 'attn_kv_a_norm.weight', prefix + 'attn_k_b.weight', prefix + 'attn_v_b.weight', prefix + 'attn_output.weight']
        targets.extend((name for name in attention if g.has(name)))
        targets.extend((name for name in (prefix + 'indexer.attn_k.weight', prefix + 'indexer.attn_q_b.weight', prefix + 'indexer.k_norm.weight', prefix + 'indexer.k_norm.bias', prefix + 'indexer.proj.weight') if g.has(name)))
        if not include_ffn:
            return
        if g.has(prefix + 'ffn_norm.weight'):
            targets.append(prefix + 'ffn_norm.weight')
        if g.has(prefix + 'ffn_gate.weight'):
            targets.extend([prefix + 'ffn_gate.weight', prefix + 'ffn_up.weight', prefix + 'ffn_down.weight'])
        else:
            targets.append(prefix + 'ffn_gate_inp.weight')
            if g.has(prefix + 'exp_probs_b.bias'):
                targets.append(prefix + 'exp_probs_b.bias')
            targets.extend([prefix + 'ffn_gate_shexp.weight', prefix + 'ffn_up_shexp.weight', prefix + 'ffn_down_shexp.weight'])
            for expert in range(cfg.n_expert):
                targets.extend([prefix + f'ffn_gate_exps.weight::expert:{expert}', prefix + f'ffn_up_exps.weight::expert:{expert}', prefix + f'ffn_down_exps.weight::expert:{expert}'])
        if include_nextn:
            targets.extend((name for name in (prefix + 'nextn.eh_proj.weight', prefix + 'nextn.enorm.weight', prefix + 'nextn.hnorm.weight', prefix + 'nextn.shared_head_norm.weight') if g.has(name)))
    for il in range(layers):
        add_layer(il, include_ffn=scope == 'all-graph')
    if scope == 'all-graph' and (not layer_limit):
        for il in range(cfg.n_layer, cfg.total_n_layer):
            add_layer(il, include_ffn=True, include_nextn=True)
    targets = list(dict.fromkeys(targets))
    covered = {target.split('::expert:', 1)[0] for target in targets}
    excluded = sorted((name for name in g.names() if name not in covered)) if scope == 'all-graph' and (not layer_limit) else []
    return (targets, excluded)
_STATE_SCHEMA = 'glm.qat.coordinate_schedule.v2'

def _completed_log_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.completed.jsonl')

def _append_completed_records(path: Path, run_id: str, coordinates: list[str]) -> None:
    if not coordinates:
        return
    log_path = _completed_log_path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='\n') as handle:
        for coordinate in coordinates:
            handle.write(json.dumps({'run_id': run_id, 'coordinate': coordinate}, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def _read_completed(path: Path, run_id: str) -> list[str]:
    log_path = _completed_log_path(path)
    if not log_path.is_file():
        return []
    completed: list[str] = []
    with log_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            record_run = record.get('run_id')
            if record_run != run_id:
                raise ValueError(f'schedule journal belongs to run {record_run!r}, not {run_id!r}')
            coordinate = record.get('coordinate')
            if isinstance(coordinate, str) and coordinate:
                completed.append(coordinate)
    return list(dict.fromkeys(completed))

def _load_state(path: Path, run_id: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {'schema': _STATE_SCHEMA, 'run_id': run_id, 'failed': None}
    if data.get('run_id') != run_id:
        raise ValueError(f"schedule state belongs to run {data.get('run_id')!r}, not {run_id!r}")
    journal = _read_completed(path, run_id)
    legacy = data.get('completed') or []
    if legacy:
        seen = set(journal)
        missing = [coordinate for coordinate in legacy if coordinate not in seen]
        _append_completed_records(path, run_id, missing)
        journal = list(dict.fromkeys(list(legacy) + journal))
    data['schema'] = _STATE_SCHEMA
    data['completed'] = journal
    data['_completed_set'] = set(journal)
    data['completed_count'] = len(journal)
    if journal:
        data['last_completed'] = journal[-1]
    if legacy:
        _save_state(path, data)
    return data

def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.next')
    completed = state.get('completed') or []
    metadata = {key: value for key, value in state.items() if key != 'completed' and (not key.startswith('_'))}
    metadata['schema'] = _STATE_SCHEMA
    metadata['completed_count'] = len(completed)
    if completed:
        metadata['last_completed'] = completed[-1]
    else:
        metadata.pop('last_completed', None)
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def _record_completed(path: Path, state: dict, coordinate: str) -> None:
    completed = state.setdefault('completed', [])
    completed_set = state.setdefault('_completed_set', set(completed))
    if coordinate not in completed_set:
        _append_completed_records(path, str(state['run_id']), [coordinate])
        completed.append(coordinate)
        completed_set.add(coordinate)
    state['failed'] = None
    _save_state(path, state)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--authority-url', required=True)
    parser.add_argument('--authority-token', default='')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--base-fingerprint', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--scope', choices=('attention', 'all-graph'), default='all-graph')
    parser.add_argument('--steps-per-coordinate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--max-seq', type=int, default=64)
    parser.add_argument('--limit-main-layers', type=int, default=0)
    parser.add_argument('--max-coordinates', type=int, default=0, help='bounded acceptance/smoke run; zero means the entire plan')
    parser.add_argument('--state', default='.planning/glm52_qat_schedule.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-ds', action='store_true')
    parser.add_argument('--print-plan', action='store_true')
    parser.add_argument('--diloco-coordinator-url', default='')
    parser.add_argument('--diloco-run-id', default='')
    parser.add_argument('--diloco-round-offset', type=int, default=0)
    parser.add_argument('--diloco-schema-hash', default='studio.glm_native_coordinate.v1')
    parser.add_argument('--diloco-identity-key', default='')
    parser.add_argument('--diloco-node-label', default='studio-node')
    parser.add_argument('--diloco-sync-bits', type=int, choices=(4, 8), default=8)
    parser.add_argument('--diloco-topk-fraction', type=float, default=0.01)
    args = parser.parse_args()
    from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
    from cosmos_windows.gguf_qat.streamed_model_deepseek2 import Deepseek2Config
    g = StreamedGGUF(args.gguf, device='cpu')
    cfg = Deepseek2Config(g)
    plan, excluded = build_plan(g, cfg, scope=args.scope, layer_limit=args.limit_main_layers)
    g.close()
    if args.print_plan:
        print(json.dumps({'coordinates': len(plan), 'first': plan[:20], 'excluded_count': len(excluded), 'excluded': excluded[:40], 'exclusion_reason': 'none' if not excluded else 'tensor is outside the all-graph objective inventory'}, indent=2))
        return 0
    state_path = Path(args.state)
    state = _load_state(state_path, args.run_id)
    completed = set(state.get('completed') or [])
    pending = [target for target in plan if target not in completed]
    if args.max_coordinates:
        pending = pending[:args.max_coordinates]
    for index, target in enumerate(pending, 1):
        coordinate_runner = Path(__file__).with_name('train_glm_qat_coordinate.py')
        command = [sys.executable, str(coordinate_runner), '--gguf', args.gguf, '--tensor', target, '--authority-url', args.authority_url, '--run-id', args.run_id, '--steps', str(args.steps_per_coordinate), '--lr', str(args.lr), '--batch', str(args.batch), '--max-seq', str(args.max_seq), '--device', args.device]
        if args.base_fingerprint:
            command.extend(['--base-fingerprint', args.base_fingerprint])
        if args.authority_token:
            command.extend(['--authority-token', args.authority_token])
        if args.dataset:
            command.extend(['--dataset', args.dataset])
        if args.limit_main_layers:
            command.extend(['--limit-main-layers', str(args.limit_main_layers)])
        if args.no_ds:
            command.append('--no-ds')
        if args.diloco_coordinator_url:
            command.extend(['--diloco-coordinator-url', args.diloco_coordinator_url, '--diloco-run-id', args.diloco_run_id, '--diloco-round', str(args.diloco_round_offset + len(completed)), '--diloco-schema-hash', args.diloco_schema_hash, '--diloco-identity-key', args.diloco_identity_key, '--diloco-node-label', args.diloco_node_label, '--diloco-sync-bits', str(args.diloco_sync_bits), '--diloco-topk-fraction', str(args.diloco_topk_fraction)])
        print(f'[glm-qat] coordinate {index}/{len(pending)} {target}', flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            state['failed'] = {'tensor': target, 'returncode': result.returncode}
            _save_state(state_path, state)
            return result.returncode
        _record_completed(state_path, state, target)
        completed.add(target)
    print('RESULT ' + json.dumps({'result': 'DONE', 'completed': len(completed), 'planned': len(plan), 'excluded': len(excluded), 'state': str(state_path)}, sort_keys=True), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
