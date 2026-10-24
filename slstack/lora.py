"""Reuse the vendored Studio trainers and native load-time adapters; preserve base files."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import uuid
import csv
import io
from .config import path
from .journal import Journal, atomic_json, sha256
from .process import OwnedProcess


def parse_lora(raw):
    # A Windows drive colon is not a scale delimiter.
    text, sep, tail = raw.rpartition(':')
    if sep:
        try:
            scale = float(tail)
        except ValueError:
            scale, text = 1., raw
    else:
        scale, text = 1., raw
    if not text or not math.isfinite(scale):
        raise ValueError('LoRA must be path[:finite_scale]')
    return str(path(text)), scale


def output_path(cfg, raw):
    root = Path(cfg['runtime_root']).resolve()
    target = Path(raw).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Outputs must stay under runtime_root')
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def train_lora(cfg, data_folder, settings_file, output):
    settings = json.loads(Path(settings_file).read_text(encoding='utf-8-sig'))
    folder = Path(data_folder).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('Training data must be a folder')
    family = settings.get('family')
    base = path(settings.get('model', cfg['model']))
    if not base.exists():
        raise ValueError('Training base does not exist')
    out = output_path(cfg, output)
    if family in ('gguf_streamed', 'hf_streamed'):
        arch = settings.get('arch')
        if family == 'gguf_streamed' and arch not in ('step35', 'deepseek2', 'glm_dsa', 'deepseek41'):
            raise ValueError('GGUF architecture is not supported by the streamed trainer')
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.txt', '.jsonl') and p.is_file())
        if not files:
            raise ValueError('GGUF streamed training needs .txt or .jsonl data')
        if len({p.suffix.lower() for p in files}) != 1:
            raise ValueError('Do not mix raw text and JSONL training data')
        total = sum(p.stat().st_size for p in files)
        if total > 8 * 1024**2:
            raise ValueError('This folder wrapper admits at most 8 MiB of training text per run')
        corpus = Path(cfg['runtime_root']) / ('corpus-' + uuid.uuid4().hex + files[0].suffix.lower())
        with corpus.open('x', encoding='utf-8') as stream:
            for f in files:
                stream.write(f.read_text(encoding='utf-8-sig') + '\n')
        if family == 'gguf_streamed' and arch == 'deepseek41':
            allowed = {'family', 'model', 'arch', 'steps', 'lr', 'rank', 'alpha', 'lora_layers',
                       'batch', 'max_seq', 'seed', 'checkpoint_every', 'dataset_size', 'resume', 'expected_base_sha256'}
            if set(settings) - allowed:
                raise ValueError('Unsupported DSV4.1 trainer settings: ' + ', '.join(sorted(set(settings) - allowed)))
            for key, value in settings.items():
                if not isinstance(value, (str, int, float)) or isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
                    raise ValueError('Trainer settings must be finite scalars')
            from .envvars import redact_cfg
            request = {'cfg': redact_cfg(cfg), 'settings': settings, 'dataset': str(corpus), 'output': str(out)}
            proc = OwnedProcess(cfg, [sys.executable, '-B', '-m', 'slstack.train_native_prefix'],
                                'train-dsv41-native-prefix', stdin=json.dumps(request) + '\n',
                                gpu=False, timeout=cfg.get('hash_workflow_timeout_s', 3600))
            receipt = proc.wait()
            rows = [line[len('SLSTACK_TRAIN_RESULT '):] for line in proc.log.read_text(errors='replace').splitlines()
                    if line.startswith('SLSTACK_TRAIN_RESULT ')]
            if len(rows) != 1:
                raise RuntimeError('Native-prefix training returned no unique measured result')
            return {**json.loads(rows[0]), 'output': str(out), 'receipt': receipt}
        if family == 'gguf_streamed':
            module = 'cosmos_windows.gguf_qat.train_gguf_lora_streamed'
            command = [cfg['studio_python'], '-B', '-m', module, '--gguf', str(base), '--arch', arch,
                       '--dataset', str(corpus), '--save-dir', str(out), '--rss-limit-gb', str(cfg['host_budget_gib'])]
            allowed = {'steps', 'lr', 'rank', 'alpha', 'lora_layers', 'batch', 'max_seq', 'seed', 'checkpoint_every'}
        else:
            if not base.is_dir() or not (base / 'config.json').is_file():
                raise ValueError('hf_streamed requires a local Hugging Face causal-LM folder, not a DS4 GGUF')
            module = 'cosmos_windows.train_llm_lora'
            command = [cfg['studio_python'], '-B', '-m', module, '--model', str(base), '--text-file', str(corpus),
                       '--save-dir', str(out), '--streaming', 'on']
            allowed = {'steps', 'lr', 'rank', 'alpha', 'seq', 'batch_size', 'seed', 'checkpoint_every',
                       'eval_every', 'eval_batches', 'validation_split', 'target_modules'}
    elif family == 'cosmos3':
        if not any(p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.mp4') for p in folder.iterdir()):
            raise ValueError('Cosmos3 training requires media and caption sidecars, not text-only prompts')
        module = 'cosmos_windows.train_lora'
        command = [cfg['studio_python'], '-B', '-m', module, '--model', str(base), '--target-dir', str(folder),
                   '--save-dir', str(out), '--streamed-reverse', '--per-caption']
        allowed = {'steps', 'rank', 'lr', 'modality', 'checkpoint_every', 'backward_buffers', 'optimizer'}
    else:
        raise ValueError('Select existing trainer family hf_streamed, gguf_streamed or cosmos3; DSV4.1 training is unsupported')
    unknown = set(settings) - allowed - {'family', 'model', 'arch'}
    if unknown:
        raise ValueError('Unsupported trainer settings: ' + ', '.join(sorted(unknown)))
    for key in sorted(allowed & settings.keys()):
        value = settings[key]
        if not isinstance(value, (str, int, float)) or isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError('Trainer flags must be finite scalar values')
        command += ['--' + key.replace('_', '-'), str(value)]
    module_file = path(cfg['streaming_root']) / (module.replace('.', '/') + '.py')
    if not module_file.is_file():
        raise RuntimeError('Pinned trainer is unavailable; initialize streaming-llama submodule')
    before = sha256(base)
    proc = OwnedProcess(cfg, command, 'train-lora', cwd=path(cfg['streaming_root']), gpu=True)
    try:
        receipt = proc.wait()
    finally:
        if sha256(base) != before:
            raise RuntimeError('BASE HASH MISMATCH after trainer; stop using this model')
    if not out.exists():
        raise RuntimeError('Trainer exited without the requested adapter output')
    return {'output': str(out), 'output_sha256': sha256(out), 'base_sha256': before,
            'trainer': module, 'receipt': receipt}


def merge_lora(cfg, base, loras, output, exporter, *, test=False):
    base = path(base)
    # Owner permits permanent merge execution only as a small TEST. No quantized production model is admitted.
    if not test or not base.is_file() or base.stat().st_size > 1024**3:
        raise ValueError('Permanent merge requires --test and a small base file (at most 1 GiB)')
    with base.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            raise ValueError('This wrapper supports GGUF export with llama-export-lora only')
    out = output_path(cfg, output)
    if out == base or out in [path(p) for p, _ in loras]:
        raise ValueError('Merged output must be distinct from base and every adapter')
    exporter = path(exporter)
    if not exporter.is_file():
        raise ValueError('Build and provide the pinned llama-export-lora executable')
    if not loras:
        raise ValueError('At least one LoRA is required')
    # The pinned arg parser splits FNAME:SCALE on every colon, including a
    # Windows drive letter. Use relative names from a shared adapter drive.
    run_cwd = path(loras[0][0]).parent
    scaled = []
    for lora, scale in loras:
        try:
            relative = os.path.relpath(path(lora), run_cwd)
        except ValueError as exc:
            raise ValueError('Adapters must share a drive for the pinned exporter CLI; no weights were copied') from exc
        cell = io.StringIO()
        csv.writer(cell, lineterminator='').writerow([f'{relative}:{scale}'])
        scaled.append(cell.getvalue())
    journal = Journal(cfg['runtime_root'])
    row = journal.begin(base, loras, 'merging', out)
    partial = Path(row['partial_path'])
    command = [str(exporter), '--model', str(base), '--output', str(partial)]
    for argument in scaled:
        command += ['--lora-scaled', argument]
    try:
        proc = OwnedProcess(cfg, command, 'merge-lora', cwd=run_cwd)
        proc.wait()
        if not partial.is_file():
            raise RuntimeError('Exporter returned no merged output')
        with partial.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise RuntimeError('Exporter output is not GGUF')
        digest = sha256(partial)
        with partial.open('r+b') as stream:
            os.fsync(stream.fileno())
        # Atomic, exclusive publication. Keep the partial hardlink as recoverable evidence; no deletes.
        os.link(partial, out)
        journal.update(row['id'], 'merged', output_sha256=digest, applied=False)
        return {'output': str(out), 'sha256': digest, 'base_unchanged': True, 'applied': False,
                'note': 'Pinned llama-export-lora exports modified tensors as F16/F32. No requantization is performed.'}
    except BaseException as exc:
        journal.update(row['id'], 'failed', error=str(exc))
        raise


def unmerge_lora(cfg, identity):
    journal = Journal(cfg['runtime_root'])
    data = journal.read()
    row = next((r for r in data['entries'] if r['id'] == identity), None)
    if row is None:
        raise ValueError('Journal entry not found')
    if row['status'] not in ('merged', 'detached'):
        raise ValueError('Only a completed merge can switch back to base')
    if sha256(row['base_path']) != row['base_sha256']:
        raise RuntimeError('BASE HASH MISMATCH')
    atomic_json(Path(cfg['runtime_root']) / 'state' / 'model_selection.json',
                {'model': row['base_path'], 'journal_id': row['id'], 'lora_applied': False})
    journal.update(identity, 'detached', applied=False)
    return {'model': row['base_path'], 'merged_copy_preserved': row['output_path']}
