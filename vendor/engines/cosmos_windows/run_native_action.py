# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/run_native_action.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
FRAMEWORK = Path('')
SAMPLE_BASE = ''

def build_input(mode, prompt, vision, action, domain, image_size, fps, seed, chunk):
    body = {'action_chunk_size': chunk, 'domain_name': domain, 'fps': fps, 'image_size': image_size, 'view_point': 'ego_view', 'model_mode': mode, 'name': f'cli_{mode}', 'prompt': prompt, 'seed': seed, 'vision_path': vision}
    if action:
        body['action_path'] = action
    return body

def emit_studio_outputs(out_dir: str | Path, mode: str) -> None:
    sample = Path(out_dir) / f'cli_{mode}' / 'sample_outputs.json'
    if not sample.is_file():
        return
    try:
        data = json.loads(sample.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'[action] note: could not parse {sample}: {exc}', flush=True)
        return
    outputs = data.get('outputs') if isinstance(data, dict) else None
    if not isinstance(outputs, list):
        return
    for item in outputs:
        if not isinstance(item, dict):
            continue
        content = item.get('content')
        if isinstance(content, dict) and isinstance(content.get('action'), list):
            print('[action] TRAJECTORY ' + json.dumps(content['action']), flush=True)
        for file_path in item.get('files') or []:
            if isinstance(file_path, str) and Path(file_path).exists():
                print(f'[action] saved {file_path}', flush=True)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='policy', choices=['policy', 'forward_dynamics', 'inverse_dynamics'])
    ap.add_argument('--checkpoint', default=None, help='model dir; default = DROID for policy, Cosmos3-Nano for fd/id')
    ap.add_argument('--prompt', default='Put the pot to the left of the purple item.')
    ap.add_argument('--vision', default=f'{SAMPLE_BASE}/bridge_20260501_0.mp4', help='observation video/image (path or URL); default = NVIDIA bridge sample')
    ap.add_argument('--action', default=f'{SAMPLE_BASE}/bridge_20260501_0.json', help='action trajectory (path or URL); used by fd + policy(golden)')
    ap.add_argument('--domain', default='bridge_orig_lerobot')
    ap.add_argument('--image-size', type=int, default=480)
    ap.add_argument('--fps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=16, help='action_chunk_size')
    ap.add_argument('--out', default='')
    ap.add_argument('--gpu-index', default=os.environ.get('STUDIO_GPU_INDEX', '0'))
    ap.add_argument('--dry-run', action='store_true', help="write the input JSON + print the command, don't run")
    args = ap.parse_args()
    ckpt = args.checkpoint or ('' if args.mode == 'policy' else '')
    body = build_input(args.mode, args.prompt, args.vision, args.action, args.domain, args.image_size, args.fps, args.seed, args.chunk)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jf = Path(tempfile.gettempdir()) / f'cosmos_{args.mode}_input.json'
    jf.write_text(json.dumps(body, indent=2))
    cmd = [sys.executable, '-m', 'cosmos_framework.scripts.inference', '--no-guardrails', '-i', str(jf), '-o', args.out, '--checkpoint-path', ckpt]
    env = dict(os.environ)
    env.update(TORCHDYNAMO_DISABLE='1', USE_LIBUV='0', RANK='0', WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='29555', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu_index)
    print(f'[run_native_action] mode={args.mode} checkpoint={Path(ckpt).name}')
    print(f'[run_native_action] input: {jf}\n[run_native_action] out: {args.out}')
    print(f"[run_native_action] cmd: {' '.join(cmd)}")
    if args.dry_run:
        print('[run_native_action] dry-run: not executing.')
        return 0
    print('[run_native_action] running native framework (16B, eager SDPA — minutes)...', flush=True)
    rc = subprocess.run(cmd, cwd=str(FRAMEWORK), env=env).returncode
    if rc == 0:
        emit_studio_outputs(args.out, args.mode)
    return rc
if __name__ == '__main__':
    raise SystemExit(main())
