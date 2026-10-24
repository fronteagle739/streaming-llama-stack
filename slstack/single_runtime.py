"""Stage (and later install) the single-engine Studio runtime: managed-runtime pattern.

Same shape as scripts/build_combo_runtime.py: an immutable directory that Studio's
normal ds4 backend launches as its ds4_binary.
  ds4-server.exe        the existing combo launcher (ds4_combo_launcher.c): runs
                        <python-path.txt> -B -m slstack.combo_proxy --config combo.json --native-argv <Studio argv>
  combo.json            serving_mode=single_engine; every path points inside the install root
  python-path.txt       the interpreter that staged it (needs psutil)
  slstack/ vendor/studio/ envvars.json   tracked source at the recorded revision
  llamadecode-r.exe ds4-decode.dll dstorage.dll dstoragecore.dll   the ULTRA producer (copied, ~120 MB)
  qualification/        affinity-spreads.txt (B5 calibration), original-model.json (model receipt)
  preset.json           the Studio preset definition (scripts/add_llamadecode_single_preset.py posts it)
  manifest.json         SHA-256 and size of every file
No model weights are copied; the GGUF stays at cfg['model'].

  python -B -m slstack.single_runtime stage --config CFG --stage NEW_DIR --install-root <Studio runtime root>/ds4/<rev>
      --launcher <combo-launcher.exe> --studio-native <front-eagle-stack>/backend/studio/ds4_native.py
  python -B -m slstack.single_runtime install --stage NEW_DIR      (copies to the install root named in its manifest)
"""
from __future__ import annotations
import argparse
from urllib.parse import urlsplit
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
PRESET_NAME = 'DSV4.1 1017 (RTX: streaming-llama ULTRA prefill + ds4 decode, single engine)'
PRODUCER_FILES = ('ds4-decode.dll', 'dstorage.dll', 'dstoragecore.dll')
DECODE_ENV = {'LLAMADECODE_DS4_DECODE': '1'}
KEPT = ('gpu_uuid', 'gpu_idle_max_mib', 'hold_file', 'task', 'lease_minutes', 'request_timeout_s', 'host_budget_gib',
        'min_available_gib', 'job_commit_gib', 'context', 'model', 'model_name', 'producer_host_budget_gib',
        'producer_profile', 'producer_sha256', 'producer_source_revision', 'producer_fixed_direct_open', 'env_overrides',
        'producer_graceful_shutdown_seconds', 'host', 'port',
        'comfyui_models', 'studio_python')  # envvars.json placeholders the proxy's studio/producer scopes expand


def sha(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source, 'sha256').hexdigest()


def serving_config(cfg, *, runtime_root, studio_root, llama_producer, library_directories, revision,
                   spread_file=None, model_receipt=None, host_budget_gib=None):
    """The combo_proxy serving_mode=single_engine configuration (shared by Studio and the stack)."""
    host = cfg['host_budget_gib'] if host_budget_gib is None else host_budget_gib
    result = {k: deepcopy(cfg[k]) for k in KEPT if k in cfg}
    result.update(serving_mode='single_engine', source_revision=revision,
        runtime_root=str(runtime_root), studio_root=str(studio_root), llama_producer=str(llama_producer),
        producer_library_directories=[str(p) for p in library_directories],
        # The guardian reserves 0.25 GiB for the controller (combo_proxy.request_worker_config).
        host_budget_gib=host + .25, combo_worker_host_budget_gib=host,
        single_engine_stop_tokens=[1])
    if spread_file:result['cache_affinity_spread_file'] = str(spread_file)
    if model_receipt:result['model_receipt'] = str(model_receipt)
    result['env_overrides'] = {**result.get('env_overrides', {}), **DECODE_ENV}
    return result


def single_config(cfg, installed, revision):
    """The installed single-engine configuration; nothing may point at task scratch."""
    installed = Path(installed)
    result = serving_config(cfg, runtime_root=installed / 'runtime', studio_root=installed / 'vendor/studio',
        llama_producer=installed / Path(cfg['llama_producer']).name,
        library_directories=[installed, *[p for p in cfg.get('producer_library_directories', [])
                                          if 'scratch' not in str(p).replace('\\', '/').lower()]],
        spread_file=installed / 'qualification/affinity-spreads.txt',
        model_receipt=installed / 'qualification/original-model.json', revision=revision)
    if 'scratch' in json.dumps(result).replace('\\\\', '/').lower():
        raise ValueError('Installed configuration still depends on task scratch')
    return result


def context(cfg):
    """The admitted context: an explicit context key, else the native --ctx value."""
    if cfg.get('context'):return int(cfg['context'])
    command = cfg['ds4_command']
    return int(command[command.index('--ctx') + 1]) if '--ctx' in command[:-1] else 262144


def stack_config(cfg):
    """The stack's own single-RTX endpoint: the source-built producer, its ds4-decode.dll
    beside it, and paths inside the stack runtime. The controller and producer share the
    stack's admitted host budget; the producer pin is the built binary's hash."""
    from .config import path, ROOT
    from .journal import sha256
    producer = path(cfg['llama_producer'])
    missing = [str(f) for f in (producer, *[producer.parent / n for n in PRODUCER_FILES]) if not f.is_file()]
    if missing:raise ValueError('Single-engine serving needs the built producer files: ' + ', '.join(missing) +
                                '; run build.ps1 (or start --serving native)')
    runtime = Path(cfg['runtime_root']).resolve()
    port = urlsplit(cfg['ds4_url']).port
    base = {**cfg, 'host': '127.0.0.1', 'port': port, 'producer_profile': cfg.get('producer_profile', 'ultra'),
            'producer_host_budget_gib': cfg.get('producer_host_budget_gib', 6.25), 'context': context(cfg),
            'hold_file': str(path(cfg['hold_file'])) if cfg.get('hold_file') else str(runtime / 'state' / 'gpu.holds')}
    try:revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):revision = 'unknown'
    result = serving_config(base, runtime_root=runtime / 'single-engine', studio_root=path(cfg['studio_root']),
        llama_producer=producer, library_directories=[producer.parent, *[path(p) for p in cfg.get('producer_library_directories', [])]],
        spread_file=path(cfg['cache_affinity_spread_file']) if cfg.get('cache_affinity_spread_file') else None,
        model_receipt=path(cfg['model_receipt']) if cfg.get('model_receipt') else None, revision=revision,
        host_budget_gib=cfg['host_budget_gib'] - .25)
    result.update(producer_sha256=sha256(producer), producer_source_revision=revision)
    if result['combo_worker_host_budget_gib'] < result['producer_host_budget_gib'] + .5:
        raise ValueError('host_budget_gib must leave 0.75 GiB above producer_host_budget_gib for the controller')
    return result


def preset(cfg, installed):
    installed = Path(installed)
    return dict(name=PRESET_NAME, mode='local', mac={}, include_previous_reasoning=True, tool_call_format='native',
        rtx=dict(backend='ds4', model='deepseek-reasoner', model_path=cfg['model'], ctx=cfg['context'],
            ds4_binary=str(installed / 'ds4-server.exe'), ds4_chdir=str(installed), ds4_ssd_streaming='1',
            ds4_cache_affinity_bias=0, ds4_cache_affinity_spread_file=str(installed / 'qualification/affinity-spreads.txt'),
            ds4_extra_flags='--cuda', ds4_env=dict(DECODE_ENV),
            ds4_host_memory_budget_mb=int((cfg['host_budget_gib'] + .25) * 1024),
            ds4_host_memory_reserve_mb=max(2048,int(cfg['min_available_gib'] * 1024))))  # Studio ds4_memory.admit minimum


def normal_argv(studio_native, definition):
    """Studio's own ds4_argv builder (shared with scripts/build_combo_runtime.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('build_combo_runtime', SOURCE / 'scripts/build_combo_runtime.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.normal_argv(Path(studio_native), definition)


def stage(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=SOURCE, text=True).strip()
    if not branch.startswith('agent/'):raise ValueError('Owned task source branch required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'slstack', 'vendor/studio', 'envvars.json'], cwd=SOURCE, text=True).strip():
        raise ValueError('Commit and publish slstack/vendor/envvars.json before staging an immutable runtime')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    target = Path(args.stage).resolve();installed = Path(args.install_root).resolve()
    if target.exists() or installed.exists():raise FileExistsError('Runtime stage and install root must be new paths')
    producer = Path(cfg['llama_producer'])
    if sha(producer) != cfg['producer_sha256']:raise ValueError('ULTRA producer pin differs')
    target.mkdir(parents=True)
    def copy(source, relative):
        destination = target / relative;destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(source) != sha(destination):raise ValueError('Runtime copy hash differs: ' + relative)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=SOURCE).decode().split('\0')
    for relative in tracked:
        if (relative.startswith(('slstack/', 'vendor/studio/')) and Path(relative).suffix in ('.py', '.json')) or relative == 'envvars.json':
            copy(SOURCE / relative, relative)
    copy(args.launcher, 'ds4-server.exe');copy(producer, producer.name)
    for name in PRODUCER_FILES:copy(producer.parent / name, name)
    copy(cfg['cache_affinity_spread_file'], 'qualification/affinity-spreads.txt')
    copy(cfg['model_receipt'], 'qualification/original-model.json')
    result = single_config(cfg, installed, revision);definition = preset(cfg, installed)
    result['studio_expected_args'] = normal_argv(args.studio_native, definition)
    (target / 'combo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (target / 'python-path.txt').write_text(str(Path(sys.executable).resolve()) + '\n', encoding='utf-8')
    (target / 'preset.json').write_text(json.dumps(definition, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest = dict(source_revision=revision, source_branch=branch, installed_root=str(installed),
                    producer_sha256=cfg['producer_sha256'], studio_native_builder_sha256=sha(args.studio_native), files={})
    for file in sorted(target.rglob('*')):
        if file.is_file():manifest['files'][file.relative_to(target).as_posix()] = dict(sha256=sha(file), bytes=file.stat().st_size)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage=str(target), install_root=str(installed), files=len(manifest['files']),
                          bytes=sum(v['bytes'] for v in manifest['files'].values()), revision=revision)))


def install(args):
    """Copy a verified stage to its recorded install root (a new path); nothing is removed."""
    target = Path(args.stage).resolve();manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    installed = Path(manifest['installed_root'])
    if installed.exists():raise FileExistsError('Install root already exists; runtimes are immutable')
    for relative, row in manifest['files'].items():
        if sha(target / relative) != row['sha256']:raise ValueError('Stage file changed since staging: ' + relative)
    shutil.copytree(target, installed)
    for relative, row in manifest['files'].items():
        if sha(installed / relative) != row['sha256']:raise ValueError('Installed file hash differs: ' + relative)
    print(json.dumps(dict(installed=str(installed), files=len(manifest['files']), revision=manifest['source_revision'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('stage')
    for name in ('config', 'stage', 'install-root', 'launcher', 'studio-native'):one.add_argument('--' + name, required=True)
    two = sub.add_parser('install');two.add_argument('--stage', required=True)
    args = parser.parse_args()
    stage(args) if args.command == 'stage' else install(args)
