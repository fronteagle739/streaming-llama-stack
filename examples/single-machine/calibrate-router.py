#!/usr/bin/env python3
"""Run existing measurement harness commands and select machine-local router modes.

No inference implementation lives here. A local manifest supplies the existing
full-vocabulary, timing and natural-generation harness commands and JSON fields.
See CALIBRATION.md for the evidence contract and current harness limitations.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

STACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STACK))
SUPPORTED_ENGINE_PROFILE_KEYS = {'tile', 'read_parts', 'mtp', 'prefill', 'decode', 'indexer', 'attention'}
SUPPORTED_CACHE_PROFILE_KEYS = {'seed_lfu_max', 'half_life'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_profile(profile):
    require(set(profile) == {'name', 'engine', 'cache'}, 'Setting profiles accept name, engine and cache')
    require(isinstance(profile['engine'], dict) and isinstance(profile['cache'], dict), 'Profile settings must be objects')
    require(not (set(profile['engine']) - SUPPORTED_ENGINE_PROFILE_KEYS),
            'Unsupported engine setting profile keys: ' + ', '.join(sorted(set(profile['engine']) - SUPPORTED_ENGINE_PROFILE_KEYS)))
    require(not (set(profile['cache']) - SUPPORTED_CACHE_PROFILE_KEYS),
            'Unsupported cache setting profile keys: ' + ', '.join(sorted(set(profile['cache']) - SUPPORTED_CACHE_PROFILE_KEYS)))


def number(value, name, minimum=0, maximum=None):
    require(type(value) in (float, int) and math.isfinite(value), name + ' must be finite')
    require(value >= minimum and (maximum is None or value <= maximum), name + ' outside bounds')
    return value


def pointer(value, path):
    """RFC6901 JSON pointer; absent evidence is an error, never a default."""
    require(path == '' or path.startswith('/'), 'Fields must use JSON pointers')
    for key in path.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def atomic_new(path, value):
    path = Path(path)
    require(not path.exists(), 'Preserve existing evidence: ' + str(path))
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # The unique campaign prefix prevents a concurrent writer from sharing it.
    require(not path.exists(), 'Evidence appeared during publication: ' + str(path))
    os.replace(temporary, path)


def publish_settings(path, previous_bytes, value, label):
    path = Path(path)
    require(path.read_bytes() == previous_bytes, 'Settings changed during calibration; refusing replacement')
    backup = path.with_name(path.name + '.' + label + '.bak')
    with backup.open('xb') as stream:
        stream.write(previous_bytes); stream.flush(); os.fsync(stream.fileno())
    temporary = path.with_name(path.name + '.' + label + '.pending')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    require(path.read_bytes() == previous_bytes, 'Settings changed before replacement; backup and pending retained')
    os.replace(temporary, path)
    return str(backup)


def machine_identity(model):
    import psutil
    # The raw computer name is never saved in the portable settings.
    identity = dict(host_id=hashlib.sha256(socket.gethostname().encode()).hexdigest(),
                    os=platform.system(), architecture=platform.machine(), cpu=platform.processor(),
                    ram_bytes=psutil.virtual_memory().total, model_volume_device=model.stat().st_dev)
    command = ['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    identity['gpu'] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(identity['gpu'], 'No GPU identity; calibration cannot be machine bound')
    identity['fingerprint'] = hashlib.sha256(canonical(identity)).hexdigest()
    return identity


class Campaign:
    def __init__(self, args, plan, cfg):
        self.args, self.plan, self.cfg = args, plan, cfg
        self.commands = []

    def progress(self, doing, latest, next_step):
        line = f'{now()} doing={doing}; latest={latest}; next={next_step}'
        print(line, flush=True)
        with Path(self.args.progress_log).open('a', encoding='utf8') as stream:
            stream.write(line + '\n')

    def expand(self, value, variables):
        # Explicit literal replacement permits JSON braces inside argv values.
        for key, item in variables.items():
            value = value.replace('{' + key + '}', str(item))
        return value

    def stage(self, name, spec, variables):
        from slstack.process import OwnedProcess
        require(spec['gpu_lease'] in ('harness', 'orchestrator', 'none'), 'Unknown GPU lease owner')
        require(spec['listener_hosts'] in ([], ['127.0.0.1']), 'All test listeners must bind to 127.0.0.1')
        result_path = Path(self.expand(spec['result'], variables))
        require(not result_path.exists(), 'Fresh result required: ' + str(result_path))
        stage_record = dict(name=name, result=str(result_path), commands=[])
        for index, template in enumerate(spec['commands']):
            require(isinstance(template, list) and template and all(isinstance(x, str) for x in template),
                    'Commands must be argv arrays, without a shell')
            argv = [self.expand(x, variables) for x in template]
            require(any('{host_budget_gib}' in x or '{guard_config}' in x for x in template),
                    'Each harness command must receive the host budget or generated guard config')
            self.progress(name, f'command {index + 1}/{len(spec["commands"])} starting', 'existing harness receipt')
            job = OwnedProcess(self.cfg, argv, f'calibrate-{self.args.label}-{name}-{index}',
                               cwd=STACK, gpu=spec['gpu_lease'] == 'orchestrator',
                               scope=spec.get('scope', 'studio'), unbounded=True)
            heartbeat = time.monotonic()
            try:
                while job.proc.poll() is None:
                    if time.monotonic() - heartbeat >= 1200:
                        self.progress(name, 'guarded existing harness running', 'complete receipt')
                        heartbeat = time.monotonic()
                    time.sleep(1)
                guard = job.wait()
            except BaseException:
                if job.proc.poll() is None:
                    job.stop()
                raise
            require(guard.get('exit_code') == 0 and guard.get('cleanup_confirmed') is True
                    and guard.get('paging_telemetry_closed') is True, 'Incomplete guardian receipt')
            stage_record['commands'].append(dict(argv=argv, receipt=str(job.receipt),
                                                   receipt_sha256=digest(job.receipt)))
        data = read(result_path)
        values = {key: pointer(data, field) for key, field in spec.get('fields', {}).items()}
        stage_record.update(sha256=digest(result_path), values=values)
        self.commands.append(stage_record)
        self.progress(name, json.dumps(values, sort_keys=True), 'validate evidence and next measurement')
        return values


def quality(values, context, vocab):
    require(values['context'] == context, 'Full-vocabulary context differs from chosen context')
    require(values['rows'] >= 128 and values['vocab'] == vocab, 'Incomplete full-vocabulary evaluation')
    number(values['kl_mean'], 'KL')
    number(values['top1_agreement'], 'top-1', maximum=1)
    # Require actual complete dumps and equal forced-feed positions, not a top-k comparison.
    dumps = [Path(values[key]) for key in ('reference_logits', 'candidate_logits')]
    positions = [Path(values[key]) for key in ('reference_positions', 'candidate_positions')]
    expected = values['rows'] * vocab * 4
    require(all(p.stat().st_size == expected for p in dumps), 'Full F32 vocabulary dumps are incomplete')
    require(positions[0].read_bytes() == positions[1].read_bytes()
            and positions[0].stat().st_size == values['rows'] * 4, 'Forced-feed positions differ')
    for prefix, path in zip(('reference', 'candidate'), dumps):
        require(digest(path) == values[prefix + '_sha256'], 'Compared logit hash differs')


def generation(values, request, bias):
    require(values['request_sha256'] == hashlib.sha256(canonical(request)).hexdigest(),
            'Generation receipt does not bind the supplied request')
    require(values['bias'] == bias, 'Generation bias differs')
    require(values['max_tokens'] == request['max_tokens'], 'Generation cap differs from supplied request')
    return (values['reason'] in ('stop', 'eog', 'eos', 'end_turn')
            and isinstance(values['text'], str) and bool(values['text'].strip())
            and 0 < values['generated_tokens'] < values['max_tokens'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, help='Machine-local model settings JSON to update')
    parser.add_argument('--manifest', required=True, help='Local existing-harness command manifest')
    parser.add_argument('--config', required=True, help='Stack guardian config JSON')
    parser.add_argument('--model', required=True)
    parser.add_argument('--quant', required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--host-budget-gib', type=float, required=True)
    parser.add_argument('--min-available-gib', type=float, required=True)
    parser.add_argument('--output-prefix', required=True, help='Fresh evidence prefix; parent must already exist')
    parser.add_argument('--progress-log', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--plan-only', action='store_true', help='Validate manifest structure; start nothing')
    args = parser.parse_args()
    require(args.label.replace('-', '').replace('_', '').isalnum(), 'Simple label required')
    require(args.context > 0, 'Context must be positive')
    number(args.host_budget_gib, 'host budget', minimum=0.01)
    number(args.min_available_gib, 'physical reserve', minimum=0.01)
    model = Path(args.model).resolve(); require(model.is_file(), 'Model file is missing')
    prefix = Path(args.output_prefix).resolve()
    require(prefix.parent.is_dir(), 'Use an existing scratch/output directory')
    original_bytes = Path(args.settings).read_bytes()
    settings, plan, cfg = read(args.settings), read(args.manifest), read(args.config)
    require(settings['version'] == plan['version'] == 1, 'Unsupported settings/manifest version')
    require(settings['model_key'] == plan['model_key'], 'Manifest belongs to another model')
    require(plan['context'] == args.context and plan['quant'] == args.quant, 'Context/quant manifest mismatch')
    require(cfg.get('host', '127.0.0.1') == '127.0.0.1', 'Guardian host must be 127.0.0.1')
    biases = sorted(set(plan['biases']))
    require(0 in biases and len(biases) >= 3, 'Sweep needs bias zero and at least two nonzero candidates')
    for bias in biases:
        number(bias, 'sweep bias', settings['bias']['min'], settings['bias']['max'])
    for mode in ('balanced', 'max_speed'):
        number(plan['gates'][mode]['kl_max'], mode + ' KL bound')
        number(plan['gates'][mode]['top1_min'], mode + ' top-1 bound', maximum=1)
    require(len(plan['generation_cases']) >= 3, 'At least three distinct natural-generation requests required')
    requests = [read(case) for case in plan['generation_cases']]
    require(len({hashlib.sha256(canonical(r)).hexdigest() for r in requests}) == len(requests), 'Duplicate generation requests')
    for request in requests:
        require(not request.get('stop') and not request.get('logit_bias'), 'Natural-stop checks cannot force stopping or token IDs')
        require(request['max_tokens'] > 0 and request['temperature'] > 0, 'Use bounded sampled real-chat generation cases')
    require(set(plan['stages']) == {'quality', 'timing', 'generation'}, 'All three existing-harness stages required')
    profiles = plan.get('setting_profiles', [dict(name='current', engine={}, cache={})])
    require(profiles and len({p['name'] for p in profiles}) == len(profiles), 'Unique setting profile names required')
    for profile in profiles:
        validate_profile(profile)
    for spec in plan['stages'].values():
        require(spec['commands'] and spec['fields'] and spec['gpu_lease'] in ('harness', 'orchestrator', 'none'),
                'Incomplete harness stage')
        require(spec['listener_hosts'] in ([], ['127.0.0.1']), 'Non-loopback test listener refused')
    if args.plan_only:
        print(json.dumps(dict(status='PLAN-ONLY', biases=biases, context=args.context,
                              cases=len(requests), stages=plan['stages']), indent=2))
        return
    cfg.update(host_budget_gib=args.host_budget_gib, min_available_gib=args.min_available_gib,
               runtime_min_free_gib=args.min_available_gib, paging_telemetry=True)
    os.environ['SLSTACK_CALIBRATION_GUARDED'] = '1'
    # Private commit includes WDDM on Windows; never substitute it for physical RAM.
    require(cfg['job_commit_gib'] > 0, 'Explicit guardian commit budget required')
    campaign = Campaign(args, plan, cfg)
    campaign.progress('calibration identity', 'hashing model and collecting hardware identity', 'sweep')
    def model_identity():
        stat = model.stat()
        return dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns, changed_ns=stat.st_ctime_ns,
                    device=stat.st_dev, inode=stat.st_ino)
    original_model_identity = model_identity()
    model_sha = digest(model); machine = machine_identity(model)
    require(model_identity() == original_model_identity, 'Model changed while hashing')
    guard_file = str(prefix) + '.guard.json'; atomic_new(guard_file, cfg)
    report_file = str(prefix) + '.report.json'
    reference_settings = str(prefix) + '.reference.settings.json'; atomic_new(reference_settings, settings)
    variables = dict(python=sys.executable, stack=STACK, model=model, context=args.context,
                     host_budget_gib=args.host_budget_gib, guard_config=guard_file,
                     output_prefix=prefix, quant=args.quant, model_sha256=model_sha,
                     reference_settings_file=reference_settings)
    require(not set(plan.get('inputs', {})).intersection(variables), 'Manifest inputs override campaign bindings')
    variables.update(plan.get('inputs', {}))
    artifacts = {str(Path(campaign.expand(p, variables)).resolve()): digest(campaign.expand(p, variables))
                 for p in plan['artifacts']}
    require(artifacts, 'Bind existing harnesses and engine binaries in artifacts')
    artifacts[str(Path(args.manifest).resolve())] = digest(args.manifest)
    spreads = settings['router_spreads'].get('file')
    if spreads:
        path = Path(spreads); path = path if path.is_absolute() else STACK / path
        artifacts[str(path.resolve())] = digest(path)
    rows = []
    settings_written = False
    try:
        candidates = [(profile, bias) for profile in profiles for bias in biases]
        for index, (profile, bias) in enumerate(candidates):
            candidate_file = str(prefix) + f'.bias-{index}.settings.json'
            candidate = copy.deepcopy(settings)
            candidate['engine'].update(profile['engine'])
            candidate['cache'].update(profile['cache'])
            candidate['calibration_candidate_bias'] = bias
            atomic_new(candidate_file, candidate)
            arm = dict(variables, bias=bias, candidate=index, settings_file=candidate_file)
            q = campaign.stage(f'bias-{index}-quality', plan['stages']['quality'], arm)
            quality(q, args.context, plan['vocab'])
            require(q['bias'] == bias and q['model_sha256'] == model_sha
                    and q['settings_sha256'] == digest(candidate_file), 'Quality arm binding differs')
            t = campaign.stage(f'bias-{index}-timing', plan['stages']['timing'], arm)
            require(t['context'] == args.context and t['bias'] == bias, 'Timing context/bias differs')
            require(t['model_sha256'] == model_sha and t['settings_sha256'] == digest(candidate_file), 'Timing arm binding differs')
            require(t['recall'] == 'PASS' and t['evaluated_tokens'] >= 128 and t['warmup_tokens'] > 0,
                    'Timing needs recall PASS and a warm 128+ token window')
            for key in ('prefill_tok_per_s', 'decode_tok_per_s'):
                number(t[key], key, minimum=1e-9)
            hits, misses = number(t['cache_hits'], 'cache hits'), number(t['cache_misses'], 'cache misses')
            require(hits + misses > 0, 'No measured cache activity')
            generations = []
            for case_index, (case, request) in enumerate(zip(plan['generation_cases'], requests)):
                values = campaign.stage(f'bias-{index}-generation-{case_index}', plan['stages']['generation'],
                                        dict(arm, generation_case=case_index, request_file=Path(case).resolve()))
                require(values['model_sha256'] == model_sha and values['settings_sha256'] == digest(candidate_file),
                        'Generation arm binding differs')
                values['natural_stop_pass'] = generation(values, request, bias)
                generations.append(values)
            rows.append(dict(profile=profile['name'], engine=candidate['engine'], cache=candidate['cache'],
                             bias=bias, kl_mean=q['kl_mean'], top1_agreement=q['top1_agreement'],
                             prefill_tok_per_s=t['prefill_tok_per_s'], decode_tok_per_s=t['decode_tok_per_s'],
                             cache_hit_rate=hits / (hits + misses), cache_hits=hits, cache_misses=misses,
                             natural_stop_pass=all(g['natural_stop_pass'] for g in generations), generations=generations))
        eligible_profiles = {row['profile'] for row in rows if row['bias'] == 0 and row['natural_stop_pass']
                             and row['kl_mean'] <= .002 and row['top1_agreement'] >= .99}
        require(eligible_profiles, 'No unbiased setting profile passed the near-reference and natural-stop gates')
        speed_gate = plan['gates']['max_speed']
        speed_rows = [row for row in rows if row['profile'] in eligible_profiles and row['natural_stop_pass']
                      and row['kl_mean'] <= speed_gate['kl_max'] and row['top1_agreement'] >= speed_gate['top1_min']]
        require(speed_rows, 'No Max speed candidate passes declared gates')
        best = max(speed_rows, key=lambda row: (row['decode_tok_per_s'], -row['kl_mean'], -row['bias']))
        chosen_profile = best['profile']
        selection = {'max_quality': dict(bias=0, why='Unbiased profile passed near-reference full-vocabulary gate and natural-stop checks')}
        for mode in ('balanced', 'max_speed'):
            gate = plan['gates'][mode]
            eligible = [row for row in rows if row['profile'] == chosen_profile and row['natural_stop_pass'] and row['kl_mean'] <= gate['kl_max'] and row['top1_agreement'] >= gate['top1_min']]
            require(eligible, 'No candidate passes declared ' + mode + ' quality gate')
            winner = max(eligible, key=lambda row: (row['decode_tok_per_s'], -row['kl_mean'], -row['bias']))
            selection[mode] = dict(bias=winner['bias'], why=f'Fastest measured warm decode within declared KL<={gate["kl_max"]}, top1>={gate["top1_min"]}; all natural-stop cases passed')
        require(all(digest(path) == value for path, value in artifacts.items()), 'Harness or binary changed during campaign')
        require(machine_identity(model) == machine, 'Machine identity changed during campaign')
        require(model_identity() == original_model_identity, 'Model changed during campaign')
        report = dict(status='MEASURED', created_utc=now(), model_key=settings['model_key'], model_sha256=model_sha,
                      quant=args.quant, context=args.context, machine=machine, model_identity=original_model_identity,
                      manifest_sha256=digest(args.manifest), original_settings_sha256=hashlib.sha256(original_bytes).hexdigest(), artifacts=artifacts,
                      selected_profile=chosen_profile, engine=best['engine'], cache=best['cache'], router_spreads=settings['router_spreads'],
                      gates=plan['gates'], table=rows, selection=selection, stages=campaign.commands,
                      capability_reductions=[])
        atomic_new(report_file, report)
        updated = copy.deepcopy(settings)
        updated['engine'], updated['cache'] = best['engine'], best['cache']
        for mode, picked in selection.items():
            updated['bias'][mode] = picked['bias']
        updated['calibration'] = dict(status='measured', machine=machine, model_sha256=model_sha,
                                      quant=args.quant, context=args.context, created_utc=now(),
                                      report_file=report_file, report_sha256=digest(report_file),
                                      measured_bias_range=[min(biases), max(biases)], selected_profile=chosen_profile, selection=selection)
        backup = publish_settings(args.settings, original_bytes, updated, args.label)
        settings_written = True
        print('profile\tbias\tKL\ttop1\tprefill tok/s\tdecode tok/s\thit rate\tnatural stops')
        for row in rows:
            print(f'{row["profile"]}\t{row["bias"]:g}\t{row["kl_mean"]:.6g}\t{row["top1_agreement"]:.4%}\t{row["prefill_tok_per_s"]:.2f}\t{row["decode_tok_per_s"]:.2f}\t{row["cache_hit_rate"]:.2%}\t{row["natural_stop_pass"]}')
        for mode, picked in selection.items():
            print(f'{mode}: bias {picked["bias"]:g}; {picked["why"]}')
        campaign.progress('calibration complete', f'settings published; backup={backup}; reductions=none', report_file)
    except BaseException as exc:
        failure = dict(status='INCOMPLETE', error=repr(exc), table=rows, stages=campaign.commands,
                       settings_written=settings_written, capability_reductions=[])
        if not Path(str(prefix) + '.blocked.json').exists():
            atomic_new(str(prefix) + '.blocked.json', failure)
        campaign.progress('calibration incomplete', repr(exc),
                          'settings published; inspect backup/report' if settings_written else 'resolve missing support or failed evidence; settings retained')
        raise


if __name__ == '__main__':
    main()
