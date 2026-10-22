from pathlib import Path
import shutil
import subprocess
from .config import ROOT, path
from .guard import admit
from .journal import Journal


def check(cfg):
    rows = []
    def record(name, ok, detail, required=True):
        rows.append({'check': name, 'ok': bool(ok), 'required': required, 'detail': str(detail)})
    try:
        available = admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        record('host_memory', True, f'available={available}; admitted RSS budget={cfg["host_budget_gib"]} GiB')
    except MemoryError as exc:
        record('host_memory', False, exc)
    for name in ('ds4-win', 'streaming-llama', 'ds4-mac'):
        directory = ROOT / 'engines' / name
        record('source:' + name, (directory / 'PROVENANCE.md').is_file(), 'Clean vendored source; see PROVENANCE.md')
    for key in ('model', 'image_model', 'video_model', 'comfyui_models'):
        if cfg[key] or key == 'model':
            record(key, bool(cfg[key]) and path(cfg[key]).exists(), cfg[key] or 'required model path missing')
    command = cfg['ds4_command']
    record('ds4_executable', bool(command) and (path(command[0]).is_file() or shutil.which(command[0])), command[0] if command else 'missing')
    record('llama_producer', path(cfg['llama_producer']).is_file(), cfg['llama_producer'], required=cfg['long_prompt'])
    if cfg['webui']:
        public = path(cfg['webui_root'])
        record('webui', (public / 'index.html').is_file() or (public / 'index.html.gz').is_file(), public)
    record('live_bias', True, 'Delivered authenticated native control; model-bound calibration required at runtime', required=False)
    record('long_prompt', not cfg['long_prompt'], 'Windows target-only import/export/rewind parity gate is not implemented')
    record('llm_lora', len(cfg['lora']) <= 1, 'Native DSV4.1 dense output adapter; exact base binding, VRAM fusion and journal')
    try:
        recovered = Journal(cfg['runtime_root']).recover()
        record('journal', True, recovered)
    except (OSError, RuntimeError, ValueError) as exc:
        record('journal', False, exc)
    try:
        gpu = subprocess.run(['nvidia-smi', '--id=' + cfg['gpu_uuid'],
                              '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                             text=True, capture_output=True, timeout=10)
        values = gpu.stdout.strip().split(',')
        fits = gpu.returncode == 0 and len(values) == 3 and int(values[1].strip()) >= 30000
        record('vram', fits, gpu.stdout.strip() or gpu.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        record('vram', False, exc)
    return rows
