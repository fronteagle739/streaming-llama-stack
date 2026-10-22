# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_run_report.py
# Machine fallback paths cleared; no model data included.
import sys
import os
import json
import re
import statistics
from collections import Counter
PAT_COMPLETED_SENTINEL = re.compile('completed=18446744073709551615')
PAT_DEVICE_REMOVED = re.compile('D3D12 device removed \\(GetDeviceRemovedReason=(0x[0-9a-fA-F]+)\\)')
PAT_DS_FAILURE = re.compile('DS (?:REQUEST|COMMAND) FAILURE')
PAT_LAYER_MISMATCH = re.compile('LAYER-DELIVERY MISMATCH')
PAT_NEW_PASS = re.compile('\\[DS\\] new_pass #')
PAT_PASS_ENDED = re.compile('pass ended')

def extract_json_objects(line):
    idx = 0
    while True:
        start = line.find('{', idx)
        if start == -1:
            break
        try:
            obj, end = json.JSONDecoder().raw_decode(line, start)
            yield obj
            idx = end
        except json.JSONDecodeError:
            idx = start + 1

def process_log(path):
    info = {'basename': os.path.basename(path), 'mode': None, 'resolution': None, 'seconds': None, 'steps': None, 'transport': None, 'expert_dtype': None, 'output': None, 'transport_selected': None, 'cfg_mode': None, 'cfg_reason': None, 'ring_depth': None, 'overlap': None, 'projected_gb': None, 'would_spill': None, 'denoise_steps': [], 'denoise_done': None, 'latent_stats': {}, 'video_stats': None, 'layer_verify_failed': [], 'completed': None, 'failed': None, 'text_encode_done': None, 'stage_rss': {}, 'new_pass_count': 0, 'pass_ended_count': 0, 'device_removed_sentinel': None, 'removal_hresults': [], 'ds_failure_line': None, 'layer_mismatch_line': None}
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if PAT_COMPLETED_SENTINEL.search(line) and info['device_removed_sentinel'] is None:
                    info['device_removed_sentinel'] = line.strip()
                for m in PAT_DEVICE_REMOVED.finditer(line):
                    info['removal_hresults'].append(m.group(1))
                if PAT_DS_FAILURE.search(line) and info['ds_failure_line'] is None:
                    info['ds_failure_line'] = line.strip()
                if PAT_LAYER_MISMATCH.search(line) and info['layer_mismatch_line'] is None:
                    info['layer_mismatch_line'] = line.strip()
                info['new_pass_count'] += len(PAT_NEW_PASS.findall(line))
                info['pass_ended_count'] += len(PAT_PASS_ENDED.findall(line))
                for obj in extract_json_objects(line):
                    event = obj.get('event')
                    if not event:
                        continue
                    if event == 'request_validated':
                        info['mode'] = obj.get('mode')
                        info['resolution'] = obj.get('resolution')
                        info['seconds'] = obj.get('seconds')
                        info['steps'] = obj.get('steps')
                        info['transport'] = obj.get('transport')
                        info['expert_dtype'] = obj.get('expert_dtype')
                        info['output'] = obj.get('output')
                    elif event == 'transport_selected':
                        info['transport_selected'] = obj.get('transport')
                    elif event == 'stage_log':
                        stage = obj.get('stage')
                        if stage == 'config_matrix':
                            info['cfg_mode'] = obj.get('chosen_cfg_mode')
                            info['cfg_reason'] = obj.get('reason')
                        elif stage == 'layer_ring':
                            info['ring_depth'] = obj.get('ring_depth')
                            info['overlap'] = obj.get('overlap')
                    elif event == 'vram_projection':
                        info['projected_gb'] = obj.get('projected_gb')
                        info['would_spill'] = obj.get('would_spill')
                    elif event == 'denoise_step_complete':
                        step = obj.get('step')
                        secs = obj.get('seconds')
                        if step is not None and secs is not None:
                            info['denoise_steps'].append((step, secs))
                    elif event == 'denoise_done':
                        info['denoise_done'] = {'seconds': obj.get('seconds'), 's_per_step': obj.get('s_per_step'), 'peak_vram_gb': obj.get('peak_vram_gb'), 'stream_gbps': obj.get('stream_gbps')}
                    elif event == 'latent_stats':
                        stage = obj.get('stage')
                        latent = obj.get('latent', {})
                        if stage is not None:
                            info['latent_stats'][stage] = {'std': latent.get('std'), 'finite_fraction': latent.get('finite_fraction')}
                    elif event == 'video_stats':
                        info['video_stats'] = {'mean': obj.get('mean'), 'std': obj.get('std'), 'max': obj.get('max'), 'black_frames': obj.get('black_frames'), 'frames': obj.get('frames'), 'black_fraction': obj.get('black_fraction')}
                    elif event == 'layer_verify_failed':
                        info['layer_verify_failed'].append({'layer': obj.get('layer'), 'tensor': obj.get('tensor'), 'delivered_from': obj.get('delivered_from'), 'message': obj.get('message')})
                    elif event == 'completed':
                        info['completed'] = {'output': obj.get('output'), 'bytes': obj.get('bytes')}
                    elif event == 'failed':
                        info['failed'] = {'error_type': obj.get('error_type'), 'message': obj.get('message')}
                    elif event == 'text_encode_done':
                        info['text_encode_done'] = {'seconds': obj.get('seconds'), 'rss_gb': obj.get('rss_gb')}
                    elif event == 'stage_rss':
                        stage = obj.get('stage')
                        rss = obj.get('rss_gb')
                        if stage is not None:
                            info['stage_rss'][stage] = rss
    except Exception as e:
        return None
    return info

def compute_verdict(info):
    if info['device_removed_sentinel'] is not None or info['removal_hresults']:
        return 'DEVICE-REMOVED'
    if info['ds_failure_line'] is not None:
        return 'DS-FAILURE'
    if info['layer_verify_failed'] or info['layer_mismatch_line'] is not None:
        return 'LAYER-MISMATCH'
    if info['video_stats'] is not None:
        vs = info['video_stats']
        if vs.get('max') is not None and vs['max'] <= 2:
            return 'BLACK'
        if vs.get('black_fraction') is not None and vs['black_fraction'] >= 0.5:
            return 'BLACK'
    if info['failed'] is not None:
        return f"FAILED:{info['failed'].get('error_type', 'UNKNOWN')}"
    if info['completed'] is not None:
        return 'OK'
    return 'INCOMPLETE'

def format_value(val, fmt=''):
    if val is None:
        return '-'
    if fmt:
        try:
            return fmt.format(val)
        except (ValueError, TypeError):
            return str(val)
    return str(val)

def main():
    if len(sys.argv) < 2:
        print('Usage: magi2_run_report.py <log_path> [<log_path> ...]')
        sys.exit(1)
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for fname in sorted(os.listdir(arg)):
                if fname.endswith('.log'):
                    paths.append(os.path.join(arg, fname))
        else:
            paths.append(arg)
    infos = []
    for p in paths:
        info = process_log(p)
        if info is None:
            info = {'basename': os.path.basename(p), 'verdict': 'UNREADABLE', 'unreadable': True}
        else:
            info['verdict'] = compute_verdict(info)
        infos.append(info)
    headers = ['LOG', 'RES', 'STEPS', 'TRANSPORT', 'DEPTH', 'CFG', 'S/STEP', 'PEAK_GB', 'LATENT_STD', 'VIDEO_MAX', 'BLACK%', 'VERDICT']
    col_widths = [24, 10, 6, 8, 6, 10, 8, 8, 10, 8, 6, 15]

    def pad_row(cells):
        return ' '.join((str(cell).ljust(w) if i != 6 and i != 7 and (i != 9) else str(cell).rjust(w) for i, (cell, w) in enumerate(zip(cells, col_widths))))
    print(pad_row(headers))
    print('-' * sum(col_widths))
    verdict_counts = Counter()
    for info in infos:
        if info.get('unreadable'):
            verdict_counts['UNREADABLE'] += 1
            row = [info['basename'][:24], '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'UNREADABLE']
            print(pad_row(row))
            continue
        verdict = info['verdict']
        verdict_counts[verdict] += 1
        if info['denoise_steps']:
            s_per_step = statistics.mean([s for _, s in info['denoise_steps']])
            s_per_step_str = f'{s_per_step:.1f}'
        else:
            s_per_step_str = '-'
        transport = info['transport'] or info['transport_selected']
        if transport:
            transport_str = str(transport)[:8]
        else:
            transport_str = '-'
        latent_std = '-'
        if info['latent_stats']:
            last_stage = list(info['latent_stats'].keys())[-1]
            std = info['latent_stats'][last_stage].get('std')
            if std is not None:
                latent_std = f'{std:.2f}'
        video_max = '-'
        black_pct = '-'
        if info['video_stats']:
            vs = info['video_stats']
            if vs.get('max') is not None:
                video_max = f"{vs['max']:.1f}"
            if vs.get('black_fraction') is not None:
                black_pct = f"{vs['black_fraction'] * 100:.0f}"
        peak_gb = '-'
        if info['denoise_done'] and info['denoise_done'].get('peak_vram_gb') is not None:
            peak_gb = f"{info['denoise_done']['peak_vram_gb']:.1f}"
        row = [info['basename'][:24], format_value(info['resolution']), format_value(info['steps']), transport_str, format_value(info['ring_depth']), format_value(info['cfg_mode']), s_per_step_str, peak_gb, latent_std, video_max, black_pct, verdict]
        print(pad_row(row))
    for info in infos:
        if info.get('unreadable') or info['verdict'] == 'OK':
            continue
        print(f"\n=== {info['basename']} - {info['verdict']}")
        indent = '    '
        if info.get('failed'):
            msg = info['failed'].get('message', '')
            if msg:
                print(f'{indent}Failure: {msg[:300]}')
        if info.get('layer_mismatch_line'):
            print(f"{indent}Layer mismatch: {info['layer_mismatch_line'][:300]}")
        if info.get('ds_failure_line'):
            print(f"{indent}DS failure: {info['ds_failure_line'][:300]}")
        if info.get('device_removed_sentinel'):
            print(f"{indent}Device removed sentinel: {info['device_removed_sentinel'][:300]}")
        if info.get('removal_hresults'):
            print(f"{indent}Removal HRESULTs: {', '.join(info['removal_hresults'])}")
        print(f"{indent}new_pass count: {info.get('new_pass_count', 0)}")
        print(f"{indent}pass_ended count: {info.get('pass_ended_count', 0)}")
        if info.get('latent_stats'):
            print(f'{indent}Latent std per stage:')
            for stage, stats in info['latent_stats'].items():
                std = stats.get('std')
                if std is not None:
                    print(f'{indent}  {stage}: {std:.4f}')
    total = len(infos)
    counts_str = ' '.join((f'{k}={v}' for k, v in sorted(verdict_counts.items())))
    print(f'\nTOTAL: {total} logs - {counts_str}')
if __name__ == '__main__':
    main()
