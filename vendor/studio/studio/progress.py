# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import re
from typing import Optional
_GEN_START = re.compile('\\[gen:(?P<modality>\\w+)\\]\\s+(?P<w>\\d+)x(?P<h>\\d+)\\s+frames=(?P<frames>\\d+)\\s+steps=(?P<steps>\\d+)\\s+cfg=(?P<cfg>[\\d.]+)')
_GEN_DONE = re.compile('\\[done\\]\\s+(?P<path>.+?)\\s+\\((?P<sec>[\\d.]+)s\\)')
_BGEN_STEP = re.compile('\\[bgen\\]\\s+step\\s+(?P<step>\\d+)/(?P<total>\\d+)\\s+(?P<sec>[\\d.]+)s\\s+\\((?P<n>\\d+)\\s+imgs?\\)')
_BGEN_TPUT = re.compile('\\[bgen\\]\\s+THROUGHPUT:\\s*(?P<text>.+)')
_BGEN_SAVED = re.compile('\\[bgen\\]\\s+saved\\s+(?P<path>.+)')
_TRAIN_STEP = re.compile('\\[(?:train|hy-train|llm)\\]\\s+step\\s+(?P<step>\\d+)(?:/(?P<total>\\d+))?\\b')
_TRAIN_LOSS = re.compile('\\b(?:loss|ce_loss)=(?P<loss>[\\d.]+)')
_TRAIN_VLOSS = re.compile('\\bvloss=(?P<vloss>[\\d.]+)')
_TRAIN_SPI = re.compile('(?P<spi>[\\d.]+)s/img')
_TRAIN_BATCH = re.compile('\\(batch=(?P<batch>\\d+)\\)')
_TRAIN_CKPT = re.compile('checkpoint\\s*@?\\s*step\\s*(?P<step>\\d+)\\s*->\\s*(?P<dir>.+)')
_CURATE = re.compile('\\[curate\\]\\s+\\[(?P<i>\\d+)/(?P<total>\\d+)\\]\\s+(?P<action>kept|drop)\\s+rel=(?P<rel>[\\d.]+):?\\s*(?P<caption>.*)')
_CURATE_DATASET = re.compile('\\[curate\\]\\s+DATASET\\s+(?P<dir>.+)')
_STREAM_BLOCK = re.compile('block\\s+(?P<block>\\d+)/(?P<total>\\d+):\\s*\\+(?P<frames>\\d+)\\s+frames\\s+\\(total\\s+(?P<cur>\\d+)/(?P<all>\\d+)\\)\\s+in\\s+(?P<sec>[\\d.]+)s')
_STREAM_PASS = re.compile('STREAM PASS:\\s*wrote\\s+(?P<path>.+)')
_DOWNLOAD = re.compile('\\[download\\]\\s+shard\\s+(?P<shard>\\d+)/(?P<total>\\d+)\\s+\\((?P<size>[\\d.]+)\\s*GB\\)\\s*(?P<name>.*)')
_MERGE = re.compile('\\[merge\\]\\s+shard\\s+(?P<shard>\\d+)/(?P<total>\\d+)')
_GENJOB_STEP = re.compile('\\[genjob\\]\\s+step\\s+(?P<step>\\d+)/(?P<total>\\d+)')
_GENJOB_PREVIEW = re.compile('\\[genjob\\]\\s+preview\\s+(?P<path>.+)')
_GENJOB_DONE = re.compile('\\[genjob\\]\\s+done\\s+(?P<path>.+)')
_ACTION_TRAJ = re.compile('\\[action\\]\\s+TRAJECTORY\\s+(?P<json>.+)')
_ACTION_STEP = re.compile('\\[action\\]\\s+step\\s+(?P<step>\\d+)/(?P<total>\\d+)')
_ACTION_FRAME = re.compile('\\[action\\]\\s+frame\\s+(?P<i>\\d+)/(?P<total>\\d+)')
_ACTION_SAVED = re.compile('\\[action\\]\\s+saved\\s+(?P<path>.+)')
_MPC_TRAJ = re.compile('\\[mpc\\]\\s+trajectory\\s+(?P<json>.+)')
_MPC_STEP = re.compile('\\[mpc\\]\\s+step\\s+(?P<step>\\d+)/(?P<total>\\d+)')
_MPC_RESIDENCY = re.compile('\\[mpc\\]\\s+residency\\s+(?P<text>.+)')
_MPC_APPLY = re.compile('\\[mpc\\]\\s+apply\\s+(?P<text>.+)')
_SETUP_READY = re.compile('\\[setup\\]\\s+READY\\s+(?P<url>.+)')
_SETUP_STEP = re.compile('\\[setup\\]\\s+(?P<step>\\d+)/(?P<total>\\d+)\\s+(?P<text>.+)')
_SWITCH_READY = re.compile('\\[switch\\]\\s+READY\\s+(?P<text>.+)')
_SWITCH_STEP = re.compile('\\[switch\\]\\s+(?P<step>\\d+)/(?P<total>\\d+)\\s+(?P<text>.+)')
_TD_PLAN = re.compile('\\[td\\]\\s+PLAN\\s+(?P<json>.+)')
_TD_RESIDENCY = re.compile('\\[td\\]\\s+residency\\s+(?P<text>.+)')
_TD_GATES = re.compile('\\[td\\]\\s+GATES_MET\\s+(?P<text>.+)')
_TD_LINE = re.compile('\\[td\\]\\s+(?P<text>.+)')
_AUTO_PLAN = re.compile('\\[auto\\]\\s+PLAN\\s+(?P<json>.+)')
_AUTO_WORKING = re.compile('\\[auto\\]\\s+working\\s+(?P<item>.+)')
_AUTO_DONE = re.compile('\\[auto\\]\\s+done\\s+(?P<item>.+)')
_AUTO_ITER = re.compile('\\[auto\\]\\s+(?P<text>iter\\s.+)')
_AUTO_GATES = re.compile('\\[auto\\]\\s+GATES_MET\\s+(?P<text>.+)')
_RESULT_DONE = re.compile('RESULT DONE(?P<rest>.*)')
_HUNYUAN_DONE = re.compile('RESULT DONE\\s+(?P<path>.+\\.(?:png|jpg|jpeg|webp|mp4|webm|gif))', re.IGNORECASE)
_PID_STEP = re.compile('\\[(?:pid|ideo|hunyuan|joyai)\\]\\s+step\\s+(?P<step>\\d+)/(?P<total>\\d+)\\s+(?P<text>.+)')
_STEP_SECONDS = re.compile('(?P<sec>[\\d.]+)s/(?:step|it)\\b')
_MODEL_PREVIEW = re.compile('\\[(?:pid|ideo|hunyuan|joyai|gen)\\]\\s+preview\\s+(?P<path>.+)')
_HUNYUAN_LOG = re.compile('\\[(?:hunyuan|HY-STREAM|pid|ideo|joyai|mem_guard)\\]\\s+(?P<text>.+)')
_STUDIO_JSON = re.compile('^STUDIO\\s+(?P<json>\\{.*\\})$')
_H3_PHASE_BY_EVENT = {'request_validated': 'prepare', 'performance_bound': 'prepare', 'attention_backend_selected': 'prepare', 'checkpoint_inspected': 'prepare', 'torchrun_launched': 'prepare', 'stage_log': 'denoise', 'text_encoding_started': 'text_encode', 'text_encoder_globals_loaded': 'text_encode', 'text_encoder_layer_bound': 'text_encode', 'text_encoding_complete': 'text_encode', 'vae_keyframes_encoded': 'conditioning', 'vae_reference_visuals_encoded': 'conditioning', 'vae_reference_audio_encoded': 'conditioning', 'lora_attached': 'denoise', 'denoise_started': 'denoise', 'upscale_schedule': 'prepare', 'latent_upscale_started': 'upscale', 'vae_decode_started': 'decode', 'vae_video_block_bound': 'decode', 'vae_audio_sanitized': 'decode', 'vae_decode_completed': 'decode', 'vae_mux_completed': 'mux'}
_H3_STAGE_LABEL = {'request_validated': 'Validating request', 'performance_bound': 'Planning compute', 'attention_backend_selected': 'Selecting attention backend', 'checkpoint_inspected': 'Inspecting checkpoint', 'torchrun_launched': 'Launching distributed workers', 'stage_log': 'Generating', 'text_encoding_started': 'Encoding text prompt', 'text_encoder_globals_loaded': 'Encoding text prompt', 'text_encoder_layer_bound': 'Encoding text prompt', 'text_encoding_complete': 'Text prompt encoded', 'vae_keyframes_encoded': 'Encoding keyframe conditioning', 'vae_reference_visuals_encoded': 'Encoding reference visuals', 'vae_reference_audio_encoded': 'Encoding reference audio', 'lora_attached': 'Attaching LoRA adapters', 'denoise_started': 'Denoising', 'upscale_schedule': 'Planning H3 passes', 'latent_upscale_started': 'Upscaling H3 video latent', 'vae_decode_started': 'Decoding video (VAE)', 'vae_video_block_bound': 'Decoding video (VAE)', 'vae_audio_sanitized': 'Repairing audio latents', 'vae_decode_completed': 'Video decoded', 'vae_mux_completed': 'Muxing audio + video'}
_H3_POST_DENOISE_PHASES = {'decode', 'mux', 'upscale'}

def _map_minimax_h3_event(event: str, payload: dict) -> Optional[dict]:
    if event == 'stage_timing':
        return {'type': 'stage_timing', **payload}
    if event in ('gen_done', 'generated', 'completed'):
        ev = {'type': 'gen_done', **payload}
        if 'path' not in ev and payload.get('output'):
            ev['path'] = str(payload['output'])
        return ev
    if event == 'failed':
        text = payload.get('message') or 'generation failed'
        return {'type': 'error', 'text': str(text), 'stage': event, **payload}
    if event == 'denoise_step_complete':
        try:
            step = int(payload.get('step'))
            total = int(payload.get('steps'))
        except (TypeError, ValueError):
            return None
        ev = {'type': 'gen_progress', 'step': step, 'total': total, 'phase': 'denoise', 'text': f'Denoising ({step}/{total})'}
        try:
            s_iter = float(payload.get('seconds'))
        except (TypeError, ValueError):
            s_iter = None
        if s_iter is not None:
            ev['s_iter'] = round(s_iter, 3)
            ev['eta'] = round(max(0, total - step) * s_iter, 1)
        return ev
    phase = _H3_PHASE_BY_EVENT.get(event, 'prepare')
    label = _H3_STAGE_LABEL.get(event) or payload.get('message') or event.replace('_', ' ')
    ev = {**payload, 'type': 'gen_status', 'text': str(label), 'stage': event, 'phase': phase}
    if phase in _H3_POST_DENOISE_PHASES:
        ev['post_denoise'] = True
    return ev

def _parse_bgen_line(s: str) -> Optional[dict]:
    m = _BGEN_STEP.search(s)
    if m:
        return {'type': 'step', 'step': int(m['step']), 'total': int(m['total']), 'seconds': float(m['sec']), 'n': int(m['n'])}
    m = _BGEN_TPUT.search(s)
    if m:
        return {'type': 'throughput', 'text': m['text'].strip()}
    m = _BGEN_SAVED.search(s)
    if m:
        return {'type': 'saved', 'path': m['path'].strip()}
    return None

def parse_line(line: str, kind: Optional[str]=None) -> Optional[dict]:
    s = line.strip()
    if kind == 'qwen_image21':
        match = re.search('\\[qwen_image21\\] step (\\d+)/(\\d+)', s)
        if match:
            return {'type': 'gen_progress', 'step': int(match[1]), 'total': int(match[2])}
        if s.startswith('[status] '):
            return {'type': 'gen_status', 'text': s.removeprefix('[status] ')}
    if not s:
        return None
    if kind == 'batched':
        return _parse_bgen_line(s)
    m = _STUDIO_JSON.match(s)
    if m:
        try:
            event = json.loads(m['json'])
        except (TypeError, ValueError):
            return None
        return event if isinstance(event, dict) and isinstance(event.get('type'), str) else None
    if kind in ('minimax_h3', 'magi2') and s.startswith('{'):
        try:
            payload = json.loads(s)
        except (TypeError, ValueError):
            return None
        if isinstance(payload, dict) and isinstance(payload.get('event'), str):
            return _map_minimax_h3_event(str(payload.pop('event')), payload)
    m = _GEN_START.search(s)
    if m:
        return {'type': 'gen_start', 'modality': m['modality'], 'w': int(m['w']), 'h': int(m['h']), 'frames': int(m['frames']), 'steps': int(m['steps']), 'cfg': float(m['cfg'])}
    m = _GEN_DONE.search(s)
    if m:
        return {'type': 'done', 'path': m['path'], 'seconds': float(m['sec'])}
    m = _HUNYUAN_DONE.search(s)
    if m:
        return {'type': 'gen_done', 'path': m['path'].strip()}
    m = _PID_STEP.search(s)
    if m:
        text = m['text'].strip()
        ev = {'type': 'gen_progress', 'step': int(m['step']), 'total': int(m['total']), 'text': text}
        sm = _STEP_SECONDS.search(text)
        if sm:
            seconds = float(sm['sec'])
            ev['s_iter'] = seconds
            ev['eta'] = max(0, int(m['total']) - int(m['step'])) * seconds
        return ev
    m = _MODEL_PREVIEW.search(s)
    if m:
        return {'type': 'preview', 'path': m['path'].strip()}
    m = _HUNYUAN_LOG.search(s)
    if m:
        return {'type': 'gen_status', 'text': m['text'].strip()}
    ev = _parse_bgen_line(s)
    if ev:
        return ev
    m = _STREAM_BLOCK.search(s)
    if m:
        return {'type': 'block', 'block': int(m['block']), 'total': int(m['total']), 'frames': int(m['frames']), 'cur': int(m['cur']), 'all': int(m['all']), 'seconds': float(m['sec'])}
    m = _STREAM_PASS.search(s)
    if m:
        return {'type': 'done', 'path': m['path'].strip()}
    m = _DOWNLOAD.search(s)
    if m:
        return {'type': 'download', 'shard': int(m['shard']), 'total': int(m['total']), 'size_gb': float(m['size']), 'name': m['name'].strip()}
    m = _MERGE.search(s)
    if m:
        return {'type': 'merge', 'shard': int(m['shard']), 'total': int(m['total'])}
    m = _GENJOB_PREVIEW.search(s)
    if m:
        return {'type': 'preview', 'path': m['path'].strip()}
    m = _GENJOB_DONE.search(s)
    if m:
        return {'type': 'gen_done', 'path': m['path'].strip()}
    m = _GENJOB_STEP.search(s)
    if m:
        return {'type': 'gen_progress', 'step': int(m['step']), 'total': int(m['total'])}
    m = _ACTION_TRAJ.search(s) or _MPC_TRAJ.search(s)
    if m:
        try:
            return {'type': 'trajectory', 'data': json.loads(m['json'])}
        except ValueError:
            return None
    m = _MPC_STEP.search(s)
    if m:
        return {'type': 'mpc_step', 'step': int(m['step']), 'total': int(m['total'])}
    m = _MPC_RESIDENCY.search(s)
    if m:
        return {'type': 'residency', 'text': m['text'].strip()}
    m = _MPC_APPLY.search(s)
    if m:
        return {'type': 'mpc_apply', 'text': m['text'].strip()}
    m = _ACTION_STEP.search(s)
    if m:
        return {'type': 'action_step', 'step': int(m['step']), 'total': int(m['total'])}
    m = _ACTION_FRAME.search(s)
    if m:
        return {'type': 'action_frame', 'i': int(m['i']), 'total': int(m['total'])}
    m = _ACTION_SAVED.search(s)
    if m:
        return {'type': 'saved', 'path': m['path'].strip()}
    m = _CURATE_DATASET.search(s)
    if m:
        return {'type': 'dataset_ready', 'dir': m['dir'].strip()}
    m = _CURATE.search(s)
    if m:
        return {'type': 'curate_item', 'i': int(m['i']), 'total': int(m['total']), 'action': m['action'], 'relevance': float(m['rel']), 'caption': m['caption'].strip()}
    m = _TRAIN_CKPT.search(s)
    if m:
        return {'type': 'checkpoint', 'step': int(m['step']), 'dir': m['dir'].strip()}
    if _TRAIN_STEP.search(s):
        ev: dict = {'type': 'train_step', 'step': int(_TRAIN_STEP.search(s)['step'])}
        for rx, key, cast in ((_TRAIN_LOSS, 'loss', float), (_TRAIN_VLOSS, 'vloss', float), (_TRAIN_SPI, 's_per_img', float), (_TRAIN_BATCH, 'batch', int)):
            mm = rx.search(s)
            if mm:
                ev[key] = cast(mm[key if key != 's_per_img' else 'spi'])
        return ev
    m = _SETUP_READY.search(s)
    if m:
        return {'type': 'setup_ready', 'url': m['url'].strip()}
    m = _SETUP_STEP.search(s)
    if m:
        return {'type': 'setup_step', 'step': int(m['step']), 'total': int(m['total']), 'text': m['text'].strip()}
    m = _SWITCH_READY.search(s)
    if m:
        return {'type': 'switch_ready', 'text': m['text'].strip()}
    m = _SWITCH_STEP.search(s)
    if m:
        return {'type': 'switch_step', 'step': int(m['step']), 'total': int(m['total']), 'text': m['text'].strip()}
    m = _TD_PLAN.search(s)
    if m:
        try:
            return {'type': 'td_plan', 'todos': json.loads(m['json'])}
        except ValueError:
            return None
    m = _TD_RESIDENCY.search(s)
    if m:
        return {'type': 'residency', 'text': m['text'].strip()}
    m = _TD_GATES.search(s)
    if m:
        return {'type': 'td_gates_met', 'text': m['text'].strip()}
    m = _TD_LINE.search(s)
    if m:
        return {'type': 'td_log', 'text': m['text'].strip()}
    m = _AUTO_PLAN.search(s)
    if m:
        try:
            return {'type': 'plan', 'todos': json.loads(m['json'])}
        except ValueError:
            return None
    m = _AUTO_GATES.search(s)
    if m:
        return {'type': 'gates_met', 'text': m['text'].strip()}
    m = _AUTO_WORKING.search(s)
    if m:
        return {'type': 'auto_working', 'item': m['item'].strip()}
    m = _AUTO_DONE.search(s)
    if m:
        return {'type': 'auto_done', 'item': m['item'].strip()}
    m = _AUTO_ITER.search(s)
    if m:
        return {'type': 'auto_iter', 'text': m['text'].strip()}
    m = _RESULT_DONE.search(s)
    if m:
        return {'type': 'result', 'text': m['rest'].strip()}
    return None
