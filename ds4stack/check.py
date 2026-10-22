from pathlib import Path
import shutil
import subprocess
from .config import ROOT, path
from .guard import admit
from .journal import Journal

PINS = {'ds4-win': '058ae09159a18fd4d0947d26bb1a7056771f5c8f',
        'streaming-llama': 'c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3',
        'studio': '5b39c3abf34523c69fb761f7dd174c710d3a7eb1'}


def check(cfg):
    rows = []
    def record(name, ok, detail, required=True):
        rows.append({'check': name, 'ok': bool(ok), 'required': required, 'detail': str(detail)})
    try:
        available = admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        record('host_memory', True, f'available={available}; admitted RSS budget={cfg["host_budget_gib"]} GiB')
    except MemoryError as exc:
        record('host_memory', False, exc)
    for name, sha in PINS.items():
        directory = ROOT / 'engines' / name
        if not (directory / '.git').exists():
            record('pin:' + name, False, 'Uninitialized; git submodule update --init --recursive')
            continue
        result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True, capture_output=True)
        record('pin:' + name, result.returncode == 0 and result.stdout.strip() == sha, result.stdout.strip())
    for key in ('model', 'image_model', 'video_model', 'comfyui_models'):
        if cfg[key] or key == 'model':
            record(key, bool(cfg[key]) and path(cfg[key]).exists(), cfg[key] or 'required model path missing')
    command = cfg['ds4_command']
    record('ds4_executable', bool(command) and (path(command[0]).is_file() or shutil.which(command[0])), command[0] if command else 'missing')
    record('llama_producer', path(cfg['llama_producer']).is_file(), cfg['llama_producer'], required=cfg['long_prompt'])
    if cfg['webui']:
        public = path(cfg['webui_root'])
        record('webui', (public / 'index.html').is_file() or (public / 'index.html.gz').is_file(), public)
    record('live_bias', bool(cfg['bias_mcp_url']), cfg['bias_mcp_url'] or 'upstream DS4WINBIAS2 live MCP service not available/configured; tool calls will fail explicitly', required=False)
    record('long_prompt', not cfg['long_prompt'], 'Windows target-only import/export/rewind parity gate is not implemented')
    record('llm_lora', not cfg['lora'], 'Pinned ds4-win has no qualified load-time LoRA adapter; refusing unsupported attachment')
    try:
        recovered = Journal(cfg['runtime_root']).recover()
        record('journal', True, recovered)
    except (OSError, RuntimeError, ValueError) as exc:
        record('journal', False, exc)
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
                             text=True, capture_output=True, timeout=10)
        record('vram', gpu.returncode == 0, gpu.stdout.strip() or gpu.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        record('vram', False, exc)
    return rows
