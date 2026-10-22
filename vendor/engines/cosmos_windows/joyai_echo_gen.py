# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/joyai_echo_gen.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

def _load_prompt_object(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = {'prompts': [raw]}
    prompts = obj.get('prompts') if isinstance(obj, dict) else None
    if not isinstance(prompts, list):
        prompts = [raw]
    clean = [re.sub('\\s+', ' ', str(p)).strip() for p in prompts if str(p).strip()]
    if not clean:
        raise ValueError('JoyAI-Echo prompt JSON must contain at least one non-empty prompt string')
    return {'prompts': clean}

def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    base, ext = (path.with_suffix(''), path.suffix)
    i = 1
    while True:
        candidate = Path(f'{base}_{i}{ext}')
        if not candidate.exists():
            return candidate
        i += 1

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default=os.environ.get('JOYAI_ECHO_REPO', ''))
    ap.add_argument('--config', default='', help='base JoyAI-Echo inference.yaml')
    ap.add_argument('--prompt', required=True, help='plain prompt or {"prompts": [...]} JSON')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='', help='echo-longvideo-release.safetensors override')
    ap.add_argument('--gemma-path', default='', help='Gemma-3-12B text encoder override')
    ap.add_argument('--frames', type=int, default=121)
    ap.add_argument('--width', type=int, default=832)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    inference = repo / 'inference.py'
    base_config = Path(a.config).resolve() if a.config else repo / 'configs' / 'inference.yaml'
    if not inference.is_file():
        print(f'[joyai] ERROR: inference.py not found under {repo}', flush=True)
        return 2
    if not base_config.is_file():
        print(f'[joyai] ERROR: config not found: {base_config}', flush=True)
        return 2
    try:
        import yaml
    except Exception as exc:
        print(f'[joyai] ERROR: PyYAML is required in the JoyAI environment: {exc}', flush=True)
        return 3
    out = _unique(Path(a.out).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    job_root = out.parent / f'.joyai_{out.stem}_{int(time.time())}'
    prompt_dir = job_root / 'prompts'
    output_root = job_root / 'result'
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_obj = _load_prompt_object(a.prompt)
    prompt_file = prompt_dir / 'studio_prompt.json'
    prompt_file.write_text(json.dumps(prompt_obj, ensure_ascii=False, indent=2), encoding='utf-8')
    cfg = yaml.safe_load(base_config.read_text(encoding='utf-8')) or {}
    cfg.setdefault('paths', {})
    cfg.setdefault('video', {})
    cfg['paths']['prompts_dir'] = str(prompt_dir)
    cfg['paths']['prompts_glob'] = prompt_file.name
    cfg['paths']['output_root'] = str(output_root)
    if a.checkpoint:
        cfg['paths']['checkpoint'] = str(Path(a.checkpoint).resolve())
    if a.gemma_path:
        cfg['paths']['gemma_path'] = str(Path(a.gemma_path).resolve())
    cfg['video']['num_frames'] = int(a.frames)
    cfg['video']['height'] = int(a.height)
    cfg['video']['width'] = int(a.width)
    cfg['video']['fps'] = int(a.fps)
    cfg['video']['seed'] = int(a.seed)
    cfg_file = job_root / 'studio_inference.yaml'
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f"[joyai] prompt shots={len(prompt_obj['prompts'])} frames={a.frames} size={a.width}x{a.height}", flush=True)
    cmd = [sys.executable, str(inference), '--config', str(cfg_file)]
    proc = subprocess.Popen(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.stdout is not None
    shot_re = re.compile('shot=(\\d+)/(\\d+)')
    for raw in proc.stdout:
        line = raw.rstrip('\n')
        m = shot_re.search(line)
        if m:
            print(f'[joyai] step {m.group(1)}/{m.group(2)} {line}', flush=True)
        else:
            print(line, flush=True)
    rc = proc.wait()
    if rc != 0:
        print(f'[joyai] ERROR: official inference exited {rc}', flush=True)
        return rc
    candidates = sorted(output_root.rglob('combined_shots.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(output_root.rglob('shot_*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f'[joyai] ERROR: no output video found under {output_root}', flush=True)
        return 4
    shutil.copy2(candidates[0], out)
    print(f'[joyai] saved {out}', flush=True)
    print(f'RESULT DONE {out}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
