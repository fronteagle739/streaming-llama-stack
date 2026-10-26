# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import itertools
import json
import re
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional
EVENT_PREFIX = '@H3LIVE '
STEER_LATENCY_CLIPS = 1
MAX_CONSECUTIVE_FAILURES = 3

def _kill_process_tree(proc: subprocess.Popen) -> None:
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True, check=False)
    try:
        proc.kill()
    except OSError:
        pass

def _probe_seconds(path, fallback: float) -> float:
    try:
        proc = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            value = float((proc.stdout or '').strip())
            if value > 0:
                return round(value, 2)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        pass
    return round(fallback, 2)

def reference_paths(refs) -> tuple:
    return tuple((r if isinstance(r, str) else r['path'] for r in refs or ()))

@dataclass
class ClipRecord:
    index: int
    prompt: str
    refs: tuple
    seed: int
    path: Optional[str] = None
    last_frame: Optional[str] = None
    seconds: Optional[float] = None
    parent: Optional[int] = None
    lineage: int = 0
    failed: Optional[str] = None
    phases: Optional[dict] = None
    denoised: bool = False
    rendered_prompt: Optional[str] = None
    source_videos: tuple = ()
    source_consumed: Optional[bool] = None
    source_seq: Optional[int] = None
    source_gap: Optional[int] = None
    carried: Optional[bool] = None

    def to_json(self) -> dict:
        return {'index': self.index, 'prompt': self.prompt, 'refs': list(self.refs), 'seed': self.seed, 'path': self.path, 'last_frame': self.last_frame, 'seconds': self.seconds, 'parent': self.parent, 'lineage': self.lineage, 'failed': self.failed, 'phases': self.phases, 'rendered_prompt': self.rendered_prompt, 'source_videos': list(self.source_videos), 'source_consumed': self.source_consumed, 'source_seq': self.source_seq, 'source_gap': self.source_gap, 'carried': self.carried}
_SCHEDULE_RE = re.compile('<\\s*for\\s+([0-9]*\\.?[0-9]+)\\s*(?:seconds|second|secs|sec|s)?\\s*>', re.IGNORECASE)

def parse_prompt_schedule(text: str) -> list:
    text = str(text or '')
    marks = list(_SCHEDULE_RE.finditer(text))
    if not marks:
        return [{'seconds': None, 'prompt': text.strip()}]
    out = []
    lead = text[:marks[0].start()].strip()
    if lead:
        out.append({'seconds': None, 'prompt': lead})
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip().strip(',').strip()
        out.append({'seconds': float(m.group(1)), 'prompt': body})
    return out

@dataclass
class SteeringState:
    prompt: str = ''
    refs: tuple = ()
    schedule: tuple = ()
    schedule_origin_seconds: float = 0.0
    keyframes: tuple = ()
    pending_since_clip: Optional[int] = None

def prompt_at(schedule, elapsed_seconds: float, fallback: str) -> str:
    if not schedule:
        return fallback
    cursor = 0.0
    for entry in schedule:
        seconds = entry.get('seconds')
        if seconds is None:
            continue
        if elapsed_seconds < cursor + float(seconds):
            return entry.get('prompt') or fallback
        cursor += float(seconds)
    for entry in reversed(list(schedule)):
        if entry.get('prompt'):
            return entry['prompt']
    return fallback

def keyframe_pins(keyframes, index: int, duration: float) -> tuple:
    if not keyframes or not duration or duration <= 0:
        return (None, None)
    own = {}
    for entry in keyframes:
        seconds = entry.get('seconds')
        if seconds is None:
            continue
        clip = int(float(seconds) // float(duration))
        own.setdefault(clip, entry)
    first = last = None
    mine = own.get(index)
    if mine:
        first, last = (mine.get('first') or None, mine.get('last') or None)
    after = own.get(index + 1)
    if last is None and after and (after.get('first') or None):
        last = after['first']
    before = own.get(index - 1)
    if first is None and before and (before.get('last') or None):
        first = before['last']
    return (first, last)

def require_h3_extension_runtime(repo):
    if not (Path(repo) / 'cosmos_windows/weight_streaming/minimax_h3_extension.py').is_file():
        raise RuntimeError('Seamless H3 extension needs the companion runtime update (minimax_h3_extension.py).')

class H3LiveSession:

    def __init__(self, *, runner: str, repo: str, model_dir: str, out_dir: str, width: int=448, height: int=448, duration: float=5.0, steps: int=4, seed: int=1000, gpu: int=0, attention_backend: str='flash_attn2', resident: bool=False, loras: tuple=(), sparse_attention: str='auto', sparse_method: str='sol-attn', proj_chunk: int=4096, env: Optional[dict]=None, int8_matmul: bool=False, h3_prompt_enhance: bool=False, continuity: str='latent', continuity_feedback: str='off', seamless: bool=False, clipback: bool=False, reference_native_res: bool=False, keyframes: tuple=(), model_dir_selected: str='', start_frame: Optional[str]=None, vram_fraction: Optional[float]=None, int8_attention: bool=False, residency: str='auto', vae_decode_offload: Optional[bool]=None, vae_decoder: Optional[str]=None, vae_encoder: Optional[str]=None, sigma_shift_video: Optional[float]=None, prepass_steps: int=0, loras_selected: Optional[list]=None, sigma_shift_effective: Optional[float]=None, sigma_schedule: str='time_shift', sampler: str='euler', enhance_prompt: Optional[Callable]=None, clip_conditioner: Optional[Callable]=None, clip_source=None, on_event: Optional[Callable[[dict], None]]=None) -> None:
        self.opts = dict(model_dir=model_dir, out_dir=out_dir, width=width, height=height, duration=duration, steps=steps, seed=seed, gpu=gpu, attention_backend=attention_backend, resident=resident, loras=tuple(loras), sparse_attention=sparse_attention, sparse_method=sparse_method, proj_chunk=proj_chunk, int8_matmul=int8_matmul, h3_prompt_enhance=h3_prompt_enhance, continuity=continuity, continuity_feedback=continuity_feedback, seamless=seamless, clipback=clipback, reference_native_res=reference_native_res, model_dir_selected=model_dir_selected or model_dir, start_frame=start_frame or '', vram_fraction=vram_fraction, int8_attention=int8_attention, residency=residency, vae_decode_offload=vae_decode_offload, vae_decoder=vae_decoder, vae_encoder=vae_encoder, sigma_shift_video=sigma_shift_video, loras_selected=list(loras_selected or []), sigma_shift_effective=sigma_shift_effective, sigma_schedule=sigma_schedule, sampler=sampler, prepass_steps=prepass_steps)
        self._enhance_prompt = enhance_prompt
        self._prompt_cache: dict = {}
        self.residency_budget = None
        self.kernel_switches = None
        self.kernel_switch_note = None
        self.vae_kernels = None
        self.lora_attached = None
        self.lora_pass = None
        self.sparse_selected = None
        self.sigma_schedule_detail = None
        self.sigma_schedule_note = None
        self.lora_note = None
        self.residency_note = None
        self._prompt_enhancement = {'state': 'idle' if h3_prompt_enhance else 'off'}
        self._runner, self._repo, self._env = (runner, repo, env)
        self._on_event = on_event
        self._clip_conditioner = clip_conditioner
        self._clip_source = clip_source
        self._source_prefetch = False
        if clip_source is not None and hasattr(clip_source, 'on_queued'):
            clip_source.on_queued = self._prefetch_source
        self._acquiring = False
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self.steering = SteeringState()
        if keyframes:
            self.steering.keyframes = tuple((k if isinstance(k, dict) else {'seconds': getattr(k, 'seconds', None), 'first': getattr(k, 'first', '') or '', 'last': getattr(k, 'last', '') or ''} for k in keyframes))
        self.timeline: list = []
        self.ready = threading.Event()
        self.stopping = False
        self._seeds = itertools.count(seed)
        self._lineage = 0
        self._continuity_frame: Optional[str] = None
        self._consecutive_failures = 0
        self._stop_reason: Optional[str] = None
        self._branch_parent: Optional[int] = None
        self._generation_started: Optional[float] = None
        self._last_completed: Optional[float] = None
        self._first_completed: Optional[float] = None
        self._reaper: Optional[threading.Thread] = None
        self._draining = False
        self._request_after_done = False

    def argv(self) -> list:
        o = self.opts
        argv = [self._runner, '-u', '-m', 'studio.h3_live_worker', '--model-dir', str(o['model_dir']), '--out-dir', str(o['out_dir']), '--width', str(o['width']), '--height', str(o['height']), '--duration', str(o['duration']), '--steps', str(o['steps']), '--gpu', str(o['gpu']), '--attention-backend', str(o['attention_backend']), '--sparse-attention', str(o['sparse_attention']), '--sparse-method', str(o['sparse_method']), '--proj-chunk', str(int(o['proj_chunk']))]
        choice = str(o.get('continuity') or '').strip().lower()
        if o.get('seamless'):
            argv += ['--seamless']
        if choice in ('projection', 'blend') and (not o.get('seamless')):
            argv += ['--continuity-projection', 'float']
            if choice == 'blend':
                argv += ['--continuity-projection-alpha', '0.75']
        feedback = str(o.get('continuity_feedback') or 'off').strip().lower()
        if feedback and feedback != 'off':
            argv += ['--continuity-feedback', feedback]
        if o.get('clipback'):
            argv.append('--clipback')
        if o.get('reference_native_res'):
            argv.append('--reference-native-res')
        projection = '' if o.get('seamless') or choice in ('projection', 'blend', 'latent') else str(os.environ.get('STUDIO_H3_CONTINUITY_PROJECTION', '')).strip().lower()
        if projection in ('float', 'uint8', 'video'):
            argv += ['--continuity-projection', projection]
            alpha = float(os.environ.get('STUDIO_H3_CONTINUITY_PROJECTION_ALPHA', '1'))
            if not 0 < alpha <= 1:
                raise ValueError('H3 projection alpha must be in (0, 1]')
            argv += ['--continuity-projection-alpha', str(alpha)]
        if o['resident']:
            argv.append('--resident')
        if o.get('vram_fraction'):
            argv += ['--vram-fraction', '%.3f' % float(o['vram_fraction'])]
        for entry in o['loras']:
            path, weight = (entry[0], entry[1])
            first_pass = bool(entry[2]) if len(entry) > 2 else False
            spec = '%s:%g' % (path, float(weight))
            if first_pass and o.get('prepass_steps'):
                spec += ':first'
            argv += ['--lora', spec]
        if o.get('prepass_steps'):
            argv += ['--prepass-steps', str(int(o['prepass_steps']))]
        source = getattr(self, '_clip_source', None)
        device = str(getattr(source, 'worker_encode_device', getattr(source, 'encode_device', '')) or '')
        if device and device != 'same':
            argv += ['--source-encode-device', device]
        return argv

    def _prefetch_source(self, entry: dict) -> None:
        with self._lock:
            if not self._source_prefetch or self.stopping or self._proc is None:
                return
            try:
                self._send({'cmd': 'source_prefetch', 'source_videos': [entry['path']], 'source_size': entry.get('source_size')})
            except Exception:
                pass

    def start(self, prompt: str, refs: tuple=()) -> None:
        with self._lock:
            if self._proc is not None:
                raise RuntimeError('this live session is already running')
            self.steer(prompt=prompt, refs=tuple(refs))
            self.log_path = Path(self.opts['out_dir']) / 'worker.log'
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, 'a', encoding='utf-8', errors='replace')
            self._proc = subprocess.Popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, cwd=self._repo, env=self._env)
            self._reader = threading.Thread(target=self._read_events, daemon=True)
            self._worker_progress_at = time.monotonic()
            self._reader.start()
            if getattr(self._clip_conditioner, 'worker_progress_timeout', None):
                threading.Thread(target=self._watch_world_worker, args=(self._proc,), name='h3-world-progress', daemon=True).start()

    def _check_world_stall(self, now: float) -> bool:
        timeout = getattr(self._clip_conditioner, 'worker_progress_timeout', None)
        with self._lock:
            if not timeout or self.stopping or now - self._worker_progress_at < timeout:
                return False
            self._stop_reason = f'World worker made no progress for {timeout:g} seconds; stopped to release the session'
        self.stop()
        return True

    def _watch_world_worker(self, proc) -> None:
        while proc.poll() is None and (not self.stopping):
            if self._check_world_stall(time.monotonic()):
                return
            time.sleep(1)

    def _send(self, obj: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError('live session is not running')
        proc.stdin.write(json.dumps(obj) + '\n')
        proc.stdin.flush()
    STOP_GRACE_SECONDS = 5.0
    DRAIN_GRACE_SECONDS = 8.0

    def stop(self) -> None:
        with self._lock:
            if self._reaper is not None:
                return
            self.stopping = True
            proc = self._proc
            if proc is None:
                return
            self._reaper = threading.Thread(target=self._reap, args=(proc,), daemon=True)
            reaper = self._reaper
            draining = any((r.denoised and (not r.path) and (not r.failed) for r in self.timeline))
            if draining:
                try:
                    self._send({'cmd': 'stop'})
                except Exception:
                    draining = False
            self._draining = draining
        if not draining:
            _kill_process_tree(proc)
        reaper.start()

    def _reap(self, proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=self.DRAIN_GRACE_SECONDS if getattr(self, '_draining', False) else self.STOP_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=self.STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass

    def _request_next_clip(self) -> None:
        source = self._clip_source
        if source is None:
            self._issue_clip(None)
            return
        with self._lock:
            if self.stopping or self._acquiring:
                return
            self._acquiring = True
            index = len(self.timeline)

        def run() -> None:
            try:
                extra = source.acquire(index, float(self.opts['duration']), cancelled=lambda: self.stopping)
            except Exception as exc:
                with self._lock:
                    self._acquiring = False
                    self._stop_reason = f'clip source failed: {exc}'
                self.stop()
                return
            with self._lock:
                self._acquiring = False
            if extra is None or self.stopping:
                return
            self._issue_clip(extra)
        threading.Thread(target=run, daemon=True, name='h3-live-clip-source').start()

    def _issue_clip(self, extra: Optional[dict]) -> None:
        with self._lock:
            if self.stopping:
                return
            index = len(self.timeline)
            if self._generation_started is None:
                self._generation_started = time.monotonic()
            steer = self.steering
            elapsed = index * float(self.opts['duration'])
            since_schedule = max(0.0, elapsed - float(getattr(steer, 'schedule_origin_seconds', 0.0)))
            active_prompt = prompt_at(steer.schedule, since_schedule, steer.prompt)
            extra = dict(extra or {})
            source_seq = extra.pop('source_seq', None)
            source_gap = extra.pop('source_gap', None)
            record = ClipRecord(index=index, prompt=active_prompt, refs=steer.refs, seed=next(self._seeds), lineage=self._lineage, parent=self._parent_for(index), source_videos=tuple(extra.get('source_videos') or ()), source_seq=source_seq, source_gap=source_gap)
            self.timeline.append(record)
            first_frame = self._continuity_frame
            self._continuity_frame = None
            steer.pending_since_clip = None
        rendered = record.prompt
        if self.opts['h3_prompt_enhance']:
            try:
                with self._lock:
                    self._prompt_enhancement = {'state': 'formatting', 'clip': index}
                key = (record.prompt, reference_paths(record.refs), self.opts['duration'])
                if key not in self._prompt_cache:
                    if self._enhance_prompt is None:
                        raise RuntimeError('No /reason enhancer configured')
                    enhanced = self._enhance_prompt(*key)
                    if len(self._prompt_cache) >= 32:
                        self._prompt_cache.clear()
                    self._prompt_cache[key] = enhanced
                rendered = self._prompt_cache[key]
                with self._lock:
                    self._prompt_enhancement = {'state': 'ready', 'clip': index}
            except Exception as exc:
                with self._lock:
                    record.failed = f'H3 Prompt Enhance failed: {exc}'
                    self._stop_reason = record.failed
                    self._prompt_enhancement = {'state': 'failed', 'error': record.failed}
                self.stop()
                return
        with self._lock:
            if self.stopping:
                return
            record.rendered_prompt = rendered
            mode = str(self.opts.get('continuity') or '').strip().lower()
            independent = mode == 'independent'
            if mode == 'image' and (not self.opts.get('seamless')) and (not first_frame):
                previous = next((r for r in reversed(self.timeline[:index]) if r.lineage == self._lineage and (not r.failed) and r.last_frame), None)
                if previous is not None:
                    first_frame = previous.last_frame
            pinned_first, pinned_last = keyframe_pins(self.steering.keyframes, index, float(self.opts.get('duration') or 0))
            if pinned_first and (not first_frame):
                first_frame = pinned_first
            gap_restart = bool(source_gap) and getattr(self._clip_source, 'gap_policy', 'restart') == 'restart' and (mode not in ('projection', 'blend'))
            carry = first_frame is None and (not independent) and (not gap_restart)
            record.carried = bool(carry and index > 0)
            self._send({'cmd': 'clip', 'prompt': rendered, 'seed': record.seed, 'first_frame': first_frame, 'last_frame': pinned_last, 'refs': list(record.refs), 'continue_latent': carry, **(self._clip_conditioner(index, self.opts['duration']) if self._clip_conditioner else {}), **(extra or {})})

    def _parent_for(self, index: int) -> Optional[int]:
        for prior in reversed(self.timeline[:index]):
            if prior.lineage == self._lineage and (not prior.failed) and (prior.denoised or prior.path):
                return prior.index
        return self._branch_parent

    def last_error(self, limit: int=1200) -> str:
        try:
            return Path(self.log_path).read_text(encoding='utf-8', errors='replace')[-limit:]
        except (OSError, AttributeError):
            return ''

    def _read_events(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log.write(line)
            self._log.flush()
            if not line.startswith(EVENT_PREFIX):
                continue
            try:
                ev = json.loads(line[len(EVENT_PREFIX):])
            except ValueError:
                continue
            self._handle(ev)
            if self._on_event:
                try:
                    self._on_event(ev)
                except Exception:
                    pass

    def _handle(self, ev: dict) -> None:
        self._worker_progress_at = time.monotonic()
        kind = ev.get('event')
        if self._clip_source is not None and kind in ('source_encoder_ready', 'source_encoder_unavailable', 'source_encoded', 'vae_reference_visuals_encoded'):
            if kind == 'source_encoder_ready':
                self._source_prefetch = True
            note = getattr(self._clip_source, 'note_runtime_event', None)
            if note is not None:
                note(ev)
        if kind == 'clip_started' and self._clip_source is not None:
            with self._lock:
                record = next((r for r in self.timeline if r.index == ev.get('index')), None)
                if record is not None and record.source_videos:
                    echoed = [str(v) for v in ev.get('source_videos') or []]
                    record.source_consumed = all((v in echoed for v in record.source_videos))
                    if not record.source_consumed:
                        self._clip_source.runtime_note = f'The cosmos runtime IGNORED the webcam source on clip {record.index}: it did not echo source_videos, so that clip was rendered from the prompt alone. Update the runtime (minimax_h3_live clip source_videos).'
        if kind == 'lora_attached':
            self.lora_attached = ev.get('paths') or []
        if kind == 'lora_pass':
            self.lora_pass = {'pass': ev.get('pass'), 'step': ev.get('step'), 'prepass_steps': ev.get('prepass_steps')}
        if kind == 'sparse_attention_selected':
            self.sparse_selected = ev
        if kind == 'sigma_schedule':
            self.sigma_schedule_detail = ev
            gaps = [float(g) for g in ev.get('video_gaps') or []]
            total = sum(gaps)
            if len(gaps) > 1 and total > 0:
                share = gaps[-1] / total
                even = 1.0 / len(gaps)
                ratio = share / even
                self.sigma_schedule_note = f'The LAST step crosses {share:.0%} of the descent' + (f', {ratio:.1f}x its even share -- the other {len(gaps) - 1} together do {1 - share:.0%}, so most of them are doing very little. Lower the sigma shift, or pick a schedule that defines its own spacing (linear, beta, beta57).' if ratio >= 2.0 else f', close to the {even:.0%} an even descent would give it, so the steps are being spent evenly.')
        if kind == 'vae_kernels':
            self.vae_kernels = {k: v for k, v in ev.items() if k != 'event'}
        if kind == 'kernel_switches':
            self.kernel_switches = ev.get('switches')
            overridden = ev.get('overridden') or []
            if overridden:
                detail = ', '.join((f"{n}={ev['switches'][n]['value']}" for n in overridden))
                self.kernel_switch_note = f'Inherited environment overrides the shipped kernel defaults: {detail}. This session is NOT running the default arm. These persist in the user environment and outlive the A/B that set them.'
        if kind == 'residency_budget':
            self.residency_budget = ev
        if kind == 'residency_declined':
            self.opts['resident'] = False
            self.residency_note = (f'{self.residency_note} ' if self.residency_note else '') + 'UPDATE: the worker DECLINED residency at runtime and fell back to STREAMING, so nothing is over-committing the card -- but this session re-reads every weight on every step and will be far slower than a resident one. Lower the resolution or clip seconds to fit.'
        if kind == 'ready':
            self.ready.set()
            try:
                scheduled = [entry['prompt'] for entry in self.steering.schedule if str(entry.get('prompt') or '').strip()]
                if self.steering.prompt.strip():
                    scheduled.append(self.steering.prompt)
                if scheduled:
                    with self._lock:
                        self._send({'cmd': 'preencode', 'prompts': list(dict.fromkeys(scheduled))})
            except Exception:
                pass
            self._request_next_clip()
            return
        if kind == 'clip_denoised':
            with self._lock:
                record = next((r for r in self.timeline if r.index == ev.get('index')), None)
                if record is not None:
                    record.denoised = True
            if not self.stopping and (self.opts.get('seamless') or str(self.opts.get('continuity') or '').strip().lower() != 'image'):
                self._request_next_clip()
            return
        waiting_on_done = not self.opts.get('seamless') and str(self.opts.get('continuity') or '').strip().lower() == 'image'
        if kind == 'clip_done' and (not self.stopping) and waiting_on_done:
            self._request_after_done = True
        if kind in ('clip_done', 'clip_failed'):
            with self._lock:
                index = int(ev.get('index', -1))
                record = next((r for r in self.timeline if r.index == index), None)
                if record is not None:
                    if kind == 'clip_done':
                        self._last_completed = time.monotonic()
                        if self._first_completed is None:
                            self._first_completed = self._last_completed
                        record.path = ev.get('path')
                        record.seconds = ev.get('seconds')
                        record.phases = ev.get('phases')
                        record.last_frame = ev.get('last_frame')
                        self._consecutive_failures = 0
                    else:
                        record.failed = str(ev.get('message') or ev.get('error_type') or 'failed')
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES and (not self.stopping):
                            self._stop_reason = f'stopped after {self._consecutive_failures} clips failed in a row; last error: {record.failed}'
                            self.stop()
                done = self.stopping
            if not done and getattr(self, '_request_after_done', False):
                self._request_after_done = False
                self._request_next_clip()
            elif not done and kind == 'clip_failed' and (record is not None) and (not record.denoised):
                self._request_next_clip()
            elif not done and kind == 'clip_failed' and waiting_on_done:
                self._request_next_clip()

    def steer(self, *, prompt: Optional[str]=None, refs: Optional[tuple]=None, keyframes: Optional[tuple]=None) -> dict:
        with self._lock:
            if prompt is not None:
                self.steering.prompt = str(prompt)
                parsed = parse_prompt_schedule(prompt)
                self.steering.schedule = tuple(parsed) if any((e.get('seconds') is not None for e in parsed)) else ()
                landing = max(len(self.timeline) + STEER_LATENCY_CLIPS - 1, 0)
                self.steering.schedule_origin_seconds = landing * float(self.opts.get('duration') or 0)
            if refs is not None:
                self.steering.refs = tuple(refs)
            if keyframes is not None:
                self.steering.keyframes = tuple(keyframes)
            lands_on = len(self.timeline) + STEER_LATENCY_CLIPS - 1
            self.steering.pending_since_clip = len(self.timeline) - 1
            return {'lands_on_clip': max(lands_on, 0), 'prompt': self.steering.prompt, 'refs': list(self.steering.refs), 'keyframes': [dict(k) for k in self.steering.keyframes]}

    def branch_from(self, index: int, *, prompt: Optional[str]=None, refs: Optional[tuple]=None) -> dict:
        with self._lock:
            record = next((r for r in self.timeline if r.index == index), None)
            if record is None:
                raise KeyError('no clip %d in this session' % index)
            if not record.last_frame:
                raise ValueError('clip %d produced no frame to continue from' % index)
            self._continuity_frame = record.last_frame
            self._branch_parent = index
            self._lineage += 1
            if prompt is not None:
                self.steer(prompt=prompt)
            if refs is not None:
                self.steering.refs = tuple(refs)
            return {'branched_from': index, 'lineage': self._lineage, 'continuity_frame': record.last_frame, 'lands_on_clip': len(self.timeline)}

    def export(self, out_path: Optional[str]=None, lineage: Optional[int]=None) -> dict:
        with self._lock:
            want = self._lineage if lineage is None else int(lineage)
            clips = [r for r in self.timeline if r.path and (not r.failed) and (r.lineage == want)]
            out_dir = Path(self.opts['out_dir'])
        if not clips:
            raise ValueError(f'no finished clips in lineage {want} to export')
        target = Path(out_path) if out_path else out_dir / f'session_lineage{want}.mp4'
        listing = out_dir / f'concat_lineage{want}.txt'
        lines = []
        for clip in clips:
            safe = str(clip.path).replace('\\', '/').replace("'", "'\\''")
            lines.append("file '" + safe + "'")
        listing.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        proc = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', str(target)], capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not target.is_file():
            raise RuntimeError(f"ffmpeg concat failed: {(proc.stderr or '')[-400:]}")
        return {'path': str(target), 'clips': len(clips), 'lineage': want, 'seconds': _probe_seconds(target, len(clips) * float(self.opts['duration'])), 'bytes': target.stat().st_size}

    def status(self) -> dict:
        with self._lock:
            done = [r for r in self.timeline if r.path]
            played = sum((float(self.opts['duration']) for _ in done))
            spent = max(0.0, self._last_completed - self._generation_started) if self._last_completed is not None and self._generation_started is not None else 0.0
            warm_played = max(0, len(done) - 1) * float(self.opts['duration'])
            warm_spent = max(0.0, self._last_completed - self._first_completed) if self._last_completed is not None and self._first_completed is not None else 0.0
            return {'residency_budget': self.residency_budget, 'kernel_switches': self.kernel_switches, 'vae_kernels': self.vae_kernels, 'lora_attached': self.lora_attached, 'lora_pass': self.lora_pass, 'sparse_selected': self.sparse_selected, 'sigma_schedule_detail': self.sigma_schedule_detail, 'sigma_schedule_note': self.sigma_schedule_note, 'kernel_switch_note': self.kernel_switch_note, 'lora_note': self.lora_note, 'residency_note': self.residency_note, 'running': self._proc is not None and self._proc.poll() is None, 'ready': self.ready.is_set(), 'clips_done': len(done), 'clips_total': len(self.timeline), 'lineage': self._lineage, 'buffer_seconds': round(played - spent, 2), 'realtime_ratio': round(played / spent, 3) if spent else None, 'realtime_ratio_warm': round(warm_played / warm_spent, 3) if warm_spent > 0 and warm_played > 0 else None, 'warm_clips': max(0, len(done) - 1), 'steering': {'prompt': self.steering.prompt, 'refs': list(self.steering.refs), 'lands_on_clip': len(self.timeline)}, 'options': {**{k: list(v) if isinstance(v, tuple) else v for k, v in self.opts.items()}, 'keyframes': list(self.steering.keyframes)}, 'stop_reason': self._stop_reason, 'prompt_enhancement': dict(self._prompt_enhancement), 'error': self.last_error() if not self.stopping and self._proc is not None and (self._proc.poll() not in (None, 0)) else None, 'timeline': [r.to_json() for r in self.timeline], **({'clip_source': self._clip_source.status()} if self._clip_source is not None and hasattr(self._clip_source, 'status') else {})}
RESIDENT_HEADROOM_BYTES = int(8.7 * 1024 ** 3)
DEFAULT_USABLE_VRAM_BYTES = int(29.74 * 1024 ** 3)
H3_TASK_DIRS = ('Ref2VA', 'FL2VA')

def resolve_h3_model_dir(model_dir) -> Path:
    root = Path(model_dir)
    if (root / 'transformer').is_dir():
        return root
    for task in H3_TASK_DIRS:
        if (root / task / 'transformer').is_dir():
            return root / task
    return root

def h3_checkpoint_bytes(model_dir) -> int:
    root = Path(model_dir)
    if root.is_file():
        return root.stat().st_size
    root = resolve_h3_model_dir(root)
    if (root / 'transformer').is_dir():
        root = root / 'transformer'
    return sum((p.stat().st_size for p in root.glob('*.safetensors') if p.is_file()))

def plan_residency(checkpoint_bytes: int, *, usable_vram_bytes: int=DEFAULT_USABLE_VRAM_BYTES, headroom_bytes: int=RESIDENT_HEADROOM_BYTES) -> bool:
    if checkpoint_bytes <= 0:
        return True
    return checkpoint_bytes + headroom_bytes <= usable_vram_bytes

def describe_h3_checkpoint(model_dir) -> dict:
    root = Path(model_dir)
    try:
        exists = root.exists()
    except OSError:
        exists = False
    if not exists:
        return {'ok': False, 'reason': f'{model_dir} does not exist (a broken link, or moved weights)', 'bytes': 0, 'resident': False}
    try:
        size = h3_checkpoint_bytes(root)
    except OSError as exc:
        return {'ok': False, 'reason': f'{model_dir} could not be read: {exc}', 'bytes': 0, 'resident': False}
    if size <= 0:
        return {'ok': False, 'reason': f'{model_dir} contains no .safetensors weights', 'bytes': 0, 'resident': False}
    return {'ok': True, 'reason': '', 'bytes': size, 'resident': plan_residency(size)}
