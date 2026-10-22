"""Small standard-library bootstrap; packages stay in the configured runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    runtime = 'runtime'
    if '--config' in args:
        position = args.index('--config')
        config = Path(args[position + 1]).resolve()
        args[position + 1] = str(config)
        runtime = json.loads(config.read_text(encoding='utf-8-sig')).get('runtime_root', runtime)
    runtime = Path(runtime).expanduser()
    runtime = (runtime if runtime.is_absolute() else ROOT / runtime).resolve()
    temporary = runtime / 'cache' / 'bootstrap'
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               PIP_CACHE_DIR=str(runtime / 'cache' / 'pip'), PYTHONDONTWRITEBYTECODE='1')
    core = runtime / 'core-venv'
    python = core / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating isolated core runtime', flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'venv', str(core)], env=env, check=True)
    lock = ROOT / 'requirements.txt'
    import hashlib
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = core / 'slstack-lock.sha256'
    if not marker.exists() or marker.read_text().strip() != digest:
        subprocess.run([str(python), '-B', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(lock)],
                       env=env, check=True)
        marker.write_text(digest + '\n')
    env['PYTHONPATH'] = str(ROOT)
    return subprocess.call([str(python), '-B', '-m', 'slstack', *args], cwd=ROOT, env=env)


if __name__ == '__main__':
    sys.exit(main())
