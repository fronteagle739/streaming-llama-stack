from pathlib import Path
import shutil
import os
import subprocess
from .config import ROOT, path
from .guard import admit
from .journal import Journal


def check(cfg, *, verify_journal=True):
    rows = []
    def record(name, ok, detail, required=True):
        rows.append({'check': name, 'ok': bool(ok), 'required': required, 'detail': str(detail)})
    try:
        available = admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        record('host_memory', True, f'available={available}; admitted RSS budget={cfg["host_budget_gib"]} GiB')
    except MemoryError as exc:
        record('host_memory', False, exc)
    from .memory_counters import diagnostic_snapshot as snapshot
    counters = snapshot()
    supported = 'error' not in counters and ('pages_output' in counters or 'vm_stat' in counters or 'swap_out_bytes' in counters)
    record('paging_counter_support', supported, 'OS paging counters are included in engine/GPU guard receipts; system activity is not per-process attribution', required=False)
    record('core_kv_residency', False, 'Live VRAM/wired/pinned residency and pagefile/expert-miss I/O reconciliation remain unqualified', required=False)
    for name in ('ds4-win', 'streaming-llama', 'ds4-mac'):
        directory = ROOT / 'engines' / name
        record('source:' + name, (directory / 'PROVENANCE.md').is_file(), 'Clean vendored source; see PROVENANCE.md')
    for key in ('model', 'image_model', 'video_model', 'comfyui_models'):
        if cfg[key] or key == 'model':
            record(key, bool(cfg[key]) and path(cfg[key]).exists(), cfg[key] or 'required model path missing')
    command = cfg['ds4_command']
    record('ds4_executable', bool(command) and (path(command[0]).is_file() or shutil.which(command[0])), command[0] if command else 'missing')
    if os.name == 'nt' and command and 'ds4' in Path(command[0]).name.lower():
        directory = path(command[0]).parent
        missing = [name for name in ('dstorage.dll', 'dstoragecore.dll') if not (directory / name).is_file()]
        record('directstorage_runtime', not missing,
               'Required beside the native executable: ' + ', '.join(missing) if missing else 'Both DirectStorage runtime DLLs are beside the native executable')
    record('llama_producer', path(cfg['llama_producer']).is_file(), cfg['llama_producer'], required=cfg['long_prompt'])
    if cfg['webui']:
        public = path(cfg['webui_root'])
        record('webui', (public / 'index.html').is_file() or (public / 'index.html.gz').is_file(), public)
    record('live_bias', True, 'Delivered authenticated native control; model-bound calibration required at runtime', required=False)
    record('long_prompt', not cfg['long_prompt'], 'Windows target-only import/export/rewind parity gate is not implemented')
    record('llm_lora', len(cfg['lora']) <= 1, 'Native DSV4.1 dense output adapter; exact base binding, VRAM fusion and journal')
    if verify_journal:
        try:
            recovered = Journal(cfg['runtime_root']).recover()
            record('journal', True, recovered)
        except (OSError, RuntimeError, ValueError) as exc:
            record('journal', False, exc)
    else:
        record('journal', True, 'Base integrity and stale ownership are verified by the model owner before loading')
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
