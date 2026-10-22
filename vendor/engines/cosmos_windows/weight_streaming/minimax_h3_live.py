# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_live.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
import os
import subprocess
import queue
import threading
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
EVENT_PREFIX = '@H3LIVE '

def parse_event(line: str):
    line = line.strip()
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        return json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None

def _emit(obj: dict) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps(obj) + '\n')
    sys.stdout.flush()
H3_REFERENCE_SHORT_EDGE = 2048
H3_KERNEL_SWITCH_DEFAULTS = {'STUDIO_H3_FUSED_ROPE': '1', 'STUDIO_H3_FUSED_ADALN': '0', 'STUDIO_H3_FUSED_INT8': '1', 'STUDIO_H3_FUSED_W4A8': '1', 'STUDIO_H3_INT8_ATTENTION': '0', 'STUDIO_H3_LORA_BF16': '1', 'MINIMAX_H3_VAE_AUDIO_CACHE': '1', 'MINIMAX_H3_VAE_DECODER': 'auto', 'MINIMAX_H3_VAE_ENCODER': 'auto'}
H3_PAGE_CONTROLLED_SWITCHES = frozenset({'STUDIO_H3_INT8_ATTENTION', 'MINIMAX_H3_VAE_DECODER', 'MINIMAX_H3_VAE_ENCODER'})

def kernel_switch_report(inherited, environ=None):
    environ = os.environ if environ is None else environ

    def origin(name):
        if name in H3_PAGE_CONTROLLED_SWITCHES:
            return 'page setting'
        return 'inherited' if name in inherited else 'live default'
    switches = {name: {'value': environ.get(name, default), 'origin': origin(name)} for name, default in H3_KERNEL_SWITCH_DEFAULTS.items()}
    overridden = sorted((n for n, s in switches.items() if s['origin'] == 'inherited' and s['value'] != H3_KERNEL_SWITCH_DEFAULTS[n]))
    return (switches, overridden)
_PIPELINE_EVENTS = ('sparse_attention_selected', 'lora_attached', 'dit_stream_stats', 'sigma_schedule')
H3_MAX_SOURCE_VIDEOS = 3

def _clip_source_videos(msg: dict, feedback_videos=()) -> tuple:
    wanted = [Path(str(v)) for v in msg.get('source_videos') or [] if str(v or '').strip()]
    for video in wanted:
        if not video.is_file():
            raise FileNotFoundError(f'source video not found: {video}')
    videos = tuple(wanted) + tuple(feedback_videos or ())
    if len(videos) > H3_MAX_SOURCE_VIDEOS:
        raise ValueError(f'at most {H3_MAX_SOURCE_VIDEOS} source videos per clip ({len(wanted)} supplied + {len(tuple(feedback_videos or ()))} fed back)')
    return videos

def _stdin_lines(poll_s: float=0.02):
    if os.name == 'nt':
        try:
            import ctypes
            import msvcrt
            from ctypes import wintypes
            fd = sys.stdin.fileno()
            handle = msvcrt.get_osfhandle(fd)
            peek = ctypes.windll.kernel32.PeekNamedPipe
            peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
            peek.restype = wintypes.BOOL
            avail = wintypes.DWORD(0)
            usable = bool(peek(handle, None, 0, None, ctypes.byref(avail), None))
        except Exception:
            usable = False
        if usable:
            pending = b''
            while True:
                avail = wintypes.DWORD(0)
                if not peek(handle, None, 0, None, ctypes.byref(avail), None):
                    break
                if not avail.value:
                    time.sleep(poll_s)
                    continue
                chunk = os.read(fd, int(avail.value))
                if not chunk:
                    break
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    yield line.decode('utf-8', errors='replace')
            if pending.strip():
                yield pending.decode('utf-8', errors='replace')
            return
    yield from sys.stdin

def _source_size(msg: dict, width: int, height: int):
    size = msg.get('source_size')
    if not size:
        return (int(width), int(height))
    w, h = (int(size[0]), int(size[1]))
    if w <= 0 or h <= 0 or w % 32 or h % 32:
        raise ValueError(f'source_size must be positive multiples of 32; got {size!r}')
    if w > int(width) or h > int(height):
        raise ValueError(f'source_size {w}x{h} is larger than the clip {width}x{height}')
    return (w, h)

def _resolve_source_encode_device(choice: str, main_index: int):
    choice = str(choice or 'same').strip().lower()
    if choice in ('', 'same', 'off', 'none'):
        return (None, 'same-device encode selected')
    import torch
    count = torch.cuda.device_count()
    if choice == 'auto':
        best, best_free = (None, 0)
        for index in range(count):
            if index == int(main_index):
                continue
            try:
                free, _total = torch.cuda.mem_get_info(index)
            except Exception:
                continue
            if free > best_free:
                best, best_free = (index, free)
        if best is None:
            return (None, 'no second CUDA device')
        if best_free < 3 * 1024 ** 3:
            return (None, f'cuda:{best} has only {best_free / 1000000000.0:.2f} GB free')
        return (f'cuda:{best}', None)
    if not choice.startswith('cuda:'):
        raise ValueError(f'--source-encode-device wants same|auto|cuda:N, got {choice!r}')
    index = int(choice.split(':', 1)[1])
    if index == int(main_index) or index >= count:
        return (None, f'{choice} is the main card or does not exist')
    return (choice, None)

class _SourcePrefetcher:

    def __init__(self, encoder, emit, *, keep: int=6):
        self.encoder = encoder
        self._emit = emit
        self._cond = threading.Condition()
        self._pending = None
        self._inflight = None
        self._cache: dict = {}
        self._order: list = []
        self._keep = int(keep)
        self._closed = False
        threading.Thread(target=self._run, daemon=True, name='h3-source-prefetch').start()

    @staticmethod
    def key(path, width, height, frames):
        path = Path(path)
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = 0
        return (str(path), stamp, int(width), int(height), int(frames))

    def request(self, path, width, height, frames) -> None:
        k = self.key(path, width, height, frames)
        with self._cond:
            if k in self._cache or k == self._inflight:
                return
            self._pending = (k, str(path), int(width), int(height), int(frames))
            self._cond.notify_all()

    def _store(self, k, value) -> None:
        self._cache[k] = value
        self._order.append(k)
        while len(self._order) > self._keep:
            self._cache.pop(self._order.pop(0), None)

    def _encode(self, k, path, width, height, frames, *, prefetched: bool):
        from .minimax_h3_vae_mux import encode_h3_source_video_item
        item, seconds = encode_h3_source_video_item(self.encoder, path, width=width, height=height, frames=frames)
        self._emit({'event': 'source_encoded', 'path': path, 'device': self.encoder.device, 'width': width, 'height': height, 'latent': [item.latent_t, item.latent_h, item.latent_w], 'rows': int(item.rows.shape[0]), 'seconds': round(seconds, 3), 'prefetched': prefetched})
        return (item, seconds)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and (not self._closed):
                    self._cond.wait()
                if self._closed:
                    return
                k, path, width, height, frames = self._pending
                self._pending = None
                self._inflight = k
            try:
                value = self._encode(k, path, width, height, frames, prefetched=True)
            except Exception as exc:
                value = None
                self._emit({'event': 'source_prefetch_failed', 'path': path, 'message': f'{type(exc).__name__}: {exc}'[:300]})
            with self._cond:
                if value is not None:
                    self._store(k, value)
                self._inflight = None
                self._cond.notify_all()

    def get(self, path, width, height, frames):
        k = self.key(path, width, height, frames)
        waited = time.perf_counter()
        with self._cond:
            while k == self._inflight:
                self._cond.wait(timeout=0.5)
            hit = self._cache.get(k)
            if self._pending is not None and self._pending[0] == k:
                self._pending = None
        if hit is not None:
            return (hit[0], {'source_prefetched': True, 'source_wait_s': round(time.perf_counter() - waited, 3), 'source_encode_s': round(hit[1], 3)})
        item, seconds = self._encode(k, str(path), int(width), int(height), int(frames), prefetched=False)
        with self._cond:
            self._store(k, (item, seconds))
        return (item, {'source_prefetched': False, 'source_wait_s': round(time.perf_counter() - waited, 3), 'source_encode_s': round(seconds, 3)})

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

def _reference_spec(value):
    if not isinstance(value, dict):
        return Path(value)
    path = value.get('path')
    if not path:
        raise ValueError(f'reference has no path: {value!r}')
    scale = value.get('scale')
    if scale is None or float(scale) == 1.0:
        return Path(path)
    scale = float(scale)
    if not 0 < scale <= 1.0:
        raise ValueError(f'reference scale must be in (0, 1]; got {scale}')
    from PIL import Image
    with Image.open(path) as probe:
        w, h = (probe.width, probe.height)
    short = max(1, int(round(H3_REFERENCE_SHORT_EDGE * scale)))
    if w <= h:
        width, height = (short, max(1, int(round(h * short / w))))
    else:
        width, height = (max(1, int(round(w * short / h))), short)
    return {'path': str(path), 'width': width, 'height': height}

def _pipeline_event(name: str, payload: dict) -> None:
    if name in _PIPELINE_EVENTS or (name == 'denoise_step_complete' and os.environ.get('STUDIO_H3_LIVE_STEP_EVENTS') == '1'):
        _emit({**(payload or {}), 'event': name})

def _latent_continuity_rows(latent):
    from .minimax_h3_pipeline import patchify_h3_video
    return patchify_h3_video(latent[:, :, -1:].contiguous()).clone()

def _write_handoff_png(clip: Path, tail) -> Path | None:
    if tail is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        frame = tail[:, 0] if getattr(tail, 'ndim', 0) == 4 else tail
        array = frame.detach().to('cpu').numpy() if hasattr(frame, 'detach') else np.asarray(frame)
        if array.ndim != 3:
            return None
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        out = clip.with_suffix('.last.png')
        Image.fromarray(array.squeeze()).save(out)
        return out if out.is_file() and out.stat().st_size else None
    except Exception:
        return None

def _extract_last_frame(clip: Path) -> Path | None:
    out = clip.with_suffix('.last.png')
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-sseof', '-0.1', '-i', str(clip), '-update', '1', '-frames:v', '1', str(out)], check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.is_file() and out.stat().st_size else None

def main(argv: list[str] | None=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Persistent H3 clip worker (JSON lines on stdin/stdout)')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--duration', type=float, default=15.0833)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--attention-backend', default='sageattention')
    ap.add_argument('--resident', action='store_true')
    ap.add_argument('--lora', action='append', default=[], help='path:weight, repeatable (the stack, applied in order)')
    ap.add_argument('--prepass-steps', type=int, default=0, help='run this many steps FIRST with only the adapters marked `:first`, then the remaining steps with the whole stack. 0 disables it, and a session with no prepass tags no adapter at all so it behaves exactly as before.')
    ap.add_argument('--sparse-attention', default='auto', choices=('auto', 'on', 'off'))
    ap.add_argument('--sparse-method', default='sol-attn', choices=('sol-attn', 'sla'))
    ap.add_argument('--proj-chunk', type=int, default=4096, help='STUDIO_H3_PROJ_CHUNK: token block for the INT8 projections. 4096 is the smallest size measured bit-exact, and it is what makes 384/448 fit at all.')
    ap.add_argument('--vram-fraction', type=float, default=0.934)
    ap.add_argument('--continuity-projection', choices=('float', 'uint8', 'video'), help='Experimental in-memory still/video projection of each continuation handoff')
    ap.add_argument('--continuity-projection-alpha', type=float, default=1.0)
    ap.add_argument('--seamless', action='store_true', help='Preserve a clean AV latent prefix for extensions')
    ap.add_argument('--continuity-feedback', choices=('off', 'image', 'video', 'anchor'), default='off')
    ap.add_argument('--clipback', action='store_true')
    ap.add_argument('--reference-native-res', action='store_true')
    ap.add_argument('--source-encode-device', default='same')
    ns = ap.parse_args(argv)
    if not 0 < ns.continuity_projection_alpha <= 1:
        ap.error('--continuity-projection-alpha must be in (0, 1]')
    if ns.continuity_projection_alpha != 1 and (not ns.continuity_projection):
        ap.error('Projection alpha requires --continuity-projection')
    if ns.clipback and ns.continuity_projection:
        ap.error('--clipback cannot be combined with --continuity-projection')
    if ns.seamless and ns.clipback:
        ap.error('--seamless cannot be combined with --clipback: the source tail must match the delivered video')
    if not 0 < ns.vram_fraction <= 0.934:
        ap.error('--vram-fraction must be in (0, 0.934]; never raise the WDDM guard')
    if ns.proj_chunk:
        os.environ['STUDIO_H3_PROJ_CHUNK'] = str(int(ns.proj_chunk))
    _inherited_switches = {n for n in H3_KERNEL_SWITCH_DEFAULTS if n in os.environ}
    os.environ.setdefault('STUDIO_H3_FUSED_ROPE', '1')
    os.environ.setdefault('STUDIO_H3_FUSED_ADALN', '0')
    os.environ.setdefault('STUDIO_H3_FUSED_INT8', '1')
    os.environ.setdefault('STUDIO_H3_FUSED_W4A8', '1')
    os.environ.setdefault('STUDIO_H3_INT8_ATTENTION', '0')
    from .minimax_h3_attention import resolve_h3_attention_backend
    ns.attention_backend = resolve_h3_attention_backend(ns.attention_backend)
    os.environ.setdefault('STUDIO_H3_LORA_BF16', '1')
    os.environ.setdefault('MINIMAX_H3_VAE_AUDIO_CACHE', '1')
    os.environ.setdefault('MINIMAX_H3_VAE_DECODER', H3_KERNEL_SWITCH_DEFAULTS['MINIMAX_H3_VAE_DECODER'])
    os.environ.setdefault('MINIMAX_H3_VAE_ENCODER', H3_KERNEL_SWITCH_DEFAULTS['MINIMAX_H3_VAE_ENCODER'])
    if ns.prepass_steps:
        os.environ['STUDIO_H3_PREPASS_STEPS'] = str(int(ns.prepass_steps))
    _switches, _overridden = kernel_switch_report(_inherited_switches)
    _emit({'event': 'kernel_switches', 'switches': _switches, 'overridden': _overridden})
    loras: list = []
    for spec in ns.lora:
        text = str(spec)
        first_pass = False
        head, _, tail = text.rpartition(':')
        if head and tail.strip().lower() in ('first', '1st', 'prepass'):
            text, first_pass = (head, True)
        raw, _, weight = text.rpartition(':')
        if not raw:
            raise SystemExit(f'--lora wants path:weight[:first], got {spec!r}')
        passes = ((1, 2) if first_pass else (2,)) if ns.prepass_steps else ()
        loras.append((Path(raw), float(weight), passes))
    import torch
    device = f'cuda:{ns.gpu}'
    torch.cuda.set_device(ns.gpu)
    torch.cuda.set_per_process_memory_fraction(ns.vram_fraction, ns.gpu)
    if ns.resident:
        from .minimax_h3_residency import probe_residency
        budget = probe_residency(ns.model_dir, loras=[entry[0] for entry in loras], device=ns.gpu, width=ns.width, height=ns.height, duration=ns.duration, fraction=ns.vram_fraction)
        _emit({'event': 'residency_budget', **budget})
        if not budget['resident']:
            ns.resident = False
            _emit({'event': 'residency_declined', **budget})
    from .minimax_h3_backend import H3GenerationRequest, _encode_h3_conditioning
    from .minimax_h3_pipeline import denoise_h3_t2v, h3_latent_geometry
    from .minimax_h3_text_encoder import encode_h3_text_prompt
    from .minimax_h3_vae_mux import decode_and_mux_h3
    from .minimax_h3_vae_mux import h3_motion_projection_frames
    projection_frames = h3_motion_projection_frames(ns.model_dir) if ns.continuity_projection == 'video' else 1
    from .minimax_h3_sparse import H3SparseAttentionConfig
    sparse_config = H3SparseAttentionConfig(mode=ns.sparse_attention, method=ns.sparse_method)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session: dict = {}
    prev_latent = None
    prev_audio = None
    prev_geometry = None
    decoded_handoff = None
    prev_clip = None
    anchor_frame = None
    prompt_cache: dict[str, object] = {}
    reference_cache: dict[tuple, dict] = {}
    pinned_prompts: set[str] = set()
    index = 0
    _decode_queue: 'queue.Queue' = queue.Queue(maxsize=2)
    action_cache = {}
    world_head_cache = {}
    world_initial_frame = None

    def _decode_worker() -> None:
        nonlocal decoded_handoff
        while True:
            item = _decode_queue.get()
            if item is None:
                _decode_queue.task_done()
                return
            idx, video, audio, geometry, path, phases, started, seed, validation = item
            nonlocal prev_clip, anchor_frame
            try:
                mark = time.perf_counter()

                def capture(frame, frame_index):
                    nonlocal decoded_handoff
                    decoded_handoff = (idx, frame, frame_index)
                capture_kwargs = {'decoded_frame_callback': capture, 'decoded_frame_count': projection_frames} if ns.continuity_projection else {}
                if ns.clipback:
                    capture_kwargs = {'decoded_frame_callback': capture, 'decoded_frame_count': 2, 'trim_tail_frames': 2}
                if validation is not None:

                    def validate(frames):
                        from .minimax_h3_world import pure_noise_metrics, H3PureNoiseError
                        mark_check = time.perf_counter()
                        metrics = pure_noise_metrics(frames)
                        phases['noise_check_s'] = round(time.perf_counter() - mark_check, 4)
                        _emit(dict(event='world_noise_check', index=idx, **metrics))
                        if metrics['pure_noise']:
                            raise H3PureNoiseError('World decode is pure noise; withholding clip and carry')
                        validation['accepted'] = True
                        validation['ready'].set()
                    capture_kwargs['frame_validator'] = validate
                if phases.get('extension_frames'):
                    capture_kwargs['trim_head_frames'] = phases['extension_frames']
                result = decode_and_mux_h3(Path(ns.model_dir), video, audio, geometry, path, event_callback=None, **capture_kwargs)
                if ns.seamless:
                    from .minimax_h3_extension import save_context
                    save_context(path, video, audio, geometry)
                phases['decode_mux_s'] = round(time.perf_counter() - mark, 3)
                phases['vae_decode_s'] = round(float(result.decode_seconds), 3)
                phases['mux_s'] = round(float(result.mux_seconds), 3)
                mark = time.perf_counter()
                if ns.clipback:
                    last_frame = _write_handoff_png(path, decoded_handoff[1] if decoded_handoff is not None and decoded_handoff[0] == idx else None)
                else:
                    last_frame = None if ns.continuity_projection and ns.continuity_feedback == 'off' else _extract_last_frame(path)
                phases['last_frame_s'] = round(time.perf_counter() - mark, 3)
                prev_clip = (idx, path, last_frame)
                if anchor_frame is None and last_frame:
                    anchor_frame = last_frame
                _emit({'event': 'clip_done', 'index': idx, 'path': str(path), 'last_frame': str(last_frame) if last_frame else None, 'handoff_frame_index': decoded_handoff[2] if ns.continuity_projection and decoded_handoff is not None else None, 'seconds': round(time.perf_counter() - started, 3), 'phases': phases, 'video_seconds': result.frames / 24, 'seed': seed})
            except Exception as exc:
                if validation is not None and (not validation.get('accepted')):
                    validation['error'] = exc
                else:
                    _emit({'event': 'clip_failed', 'index': idx, 'error_type': type(exc).__name__, 'message': str(exc)[:400], 'traceback': traceback.format_exc()[-1200:]})
            finally:
                if validation is not None:
                    validation['ready'].set()
                _decode_queue.task_done()
                del video, audio, item
    threading.Thread(target=_decode_worker, daemon=True, name='h3-decode').start()
    _emit({'event': 'ready', 'model_dir': ns.model_dir, 'width': ns.width, 'height': ns.height, 'duration': ns.duration, 'steps': ns.steps, 'resident': bool(ns.resident), 'attention_backend': ns.attention_backend, 'proj_chunk': ns.proj_chunk, 'loras': [[str(entry[0]), entry[1]] for entry in loras], 'sparse': {'mode': ns.sparse_attention, 'method': ns.sparse_method}})
    try:
        from .minimax_h3_vae_fast import describe_modes
        _emit({'event': 'vae_kernels', **describe_modes(Path(ns.model_dir) / 'video_vae')})
    except Exception as exc:
        _emit({'event': 'vae_kernels', 'error': f'{type(exc).__name__}: {exc}'[:300]})
    source_prefetcher = None
    _source_ready = threading.Event()
    _aux_device, _aux_reason = _resolve_source_encode_device(ns.source_encode_device, ns.gpu)
    if _aux_device is None:
        _source_ready.set()
        if str(ns.source_encode_device).strip().lower() not in ('', 'same', 'off', 'none'):
            _emit({'event': 'source_encoder_unavailable', 'requested': ns.source_encode_device, 'reason': _aux_reason})
    else:

        def _build_source_encoder() -> None:
            nonlocal source_prefetcher
            try:
                from .minimax_h3_vae_mux import H3PersistentVisualEncoder
                encoder = H3PersistentVisualEncoder(ns.model_dir, device=_aux_device)
                source_prefetcher = _SourcePrefetcher(encoder, _emit)
                _emit({'event': 'source_encoder_ready', 'device': _aux_device, 'name': torch.cuda.get_device_name(torch.device(_aux_device)), 'build_s': round(encoder.build_seconds, 3), 'resident_bytes': int(encoder.resident_bytes), 'transport': encoder.transport})
            except Exception as exc:
                _emit({'event': 'source_encoder_unavailable', 'requested': ns.source_encode_device, 'reason': f'{type(exc).__name__}: {exc}'[:300]})
            finally:
                _source_ready.set()
        threading.Thread(target=_build_source_encoder, daemon=True, name='h3-source-encoder').start()
    geometry_frames = int(h3_latent_geometry(ns.duration, ns.width, ns.height).frames)

    def _prefetch(msg: dict) -> None:
        if source_prefetcher is None:
            return
        try:
            w, h = _source_size(msg, ns.width, ns.height)
            for video in msg.get('source_videos') or []:
                source_prefetcher.request(video, w, h, geometry_frames)
        except Exception as exc:
            _emit({'event': 'source_prefetch_failed', 'message': str(exc)[:300]})
    commands: 'queue.Queue' = queue.Queue()

    def _reader() -> None:
        for raw in _stdin_lines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except ValueError as exc:
                _emit({'event': 'error', 'message': f'unparseable command: {exc}'})
                continue
            if str(parsed.get('cmd') or '') == 'source_prefetch':
                _prefetch(parsed)
                continue
            commands.put(parsed)
        commands.put(None)
    threading.Thread(target=_reader, daemon=True, name='h3-stdin').start()
    while True:
        msg = commands.get()
        if msg is None:
            break
        cmd = str(msg.get('cmd') or '')
        if cmd == 'stop':
            _decode_queue.join()
            from .minimax_h3_vae_mux import clear_video_decode_cache
            clear_video_decode_cache()
            if source_prefetcher is not None:
                source_prefetcher.close()
                source_prefetcher.encoder.close()
            _emit({'event': 'stopped', 'clips': index})
            return 0
        if cmd == 'preencode':
            wanted = [str(x) for x in msg.get('prompts') or [] if str(x).strip()]
            done, failed = (0, None)
            _t_pre = time.perf_counter()
            try:
                if any((text not in prompt_cache for text in wanted)):
                    from .minimax_h3_world import yield_world_residency
                    yield_world_residency(session, _decode_queue.join, _emit)
                for text in wanted:
                    if text in prompt_cache:
                        continue
                    prompt_cache[text] = encode_h3_text_prompt(str(Path(ns.model_dir) / 'text_encoder'), text, device=device)
                    pinned_prompts.add(text)
                    done += 1
            except Exception as exc:
                failed = f'{type(exc).__name__}: {exc}'[:200]
            _emit({'event': 'preencoded', 'encoded': done, 'requested': len(wanted), 'cached': len(prompt_cache), 'seconds': round(time.perf_counter() - _t_pre, 3), 'error': failed})
            continue
        if cmd == 'ping':
            _emit({'event': 'pong', 'clips': index})
            continue
        if cmd != 'clip':
            _emit({'event': 'error', 'message': f'unknown cmd {cmd!r}'})
            continue
        prompt = str(msg.get('prompt') or '')
        seed = int(msg.get('seed') or 1000 + index)
        started = time.perf_counter()
        _emit({'event': 'clip_started', 'index': index, 'prompt': prompt[:200], 'first_frame': msg.get('first_frame'), 'refs': list(msg.get('refs') or []), 'source_videos': [str(v) for v in msg.get('source_videos') or []], 'source_size': list(_source_size(msg, ns.width, ns.height)) if msg.get('source_videos') else None})
        try:
            _t_text = time.perf_counter()
            head_tags = None
            from .minimax_h3_world import preencode_actions, yield_world_residency
            script = msg.get('action_script')
            if script is not None:
                geometry_check = h3_latent_geometry(ns.duration, ns.width, ns.height)
                if len(script) != geometry_check.video_t:
                    raise ValueError('action_script length does not match clip latent count')
                wanted = list(dict.fromkeys(list(msg.get('action_preencode') or []) + script))
                head_key = (prompt, world_initial_frame or msg.get('first_frame'))
                needs_text = head_key not in world_head_cache or any((s not in action_cache for s in wanted))
            else:
                needs_text = prompt not in prompt_cache
            if needs_text:
                yield_world_residency(session, _decode_queue.join, _emit)
            if script is not None:
                preencode_actions(str(Path(ns.model_dir) / 'text_encoder'), wanted, action_cache, device, encode_h3_text_prompt, _emit)
            if msg.get('action_script') is not None:
                from .minimax_h3_world import encode_world_head
                world_initial_frame = world_initial_frame or msg.get('first_frame')
                if not world_initial_frame:
                    raise ValueError('H3-World requires an initial image')
                head_key = (prompt, world_initial_frame)
                if head_key not in world_head_cache:
                    world_head_cache.clear()
                    world_head_cache[head_key] = encode_world_head(ns.model_dir, prompt, world_initial_frame, ns.width, ns.height, device)
                emb, head_tags = world_head_cache[head_key]
            else:
                emb = prompt_cache.get(prompt)
            if emb is None:
                emb = encode_h3_text_prompt(str(Path(ns.model_dir) / 'text_encoder'), prompt, device=device)
                if len(prompt_cache) > 32:
                    for key in [k for k in prompt_cache if k not in pinned_prompts]:
                        del prompt_cache[key]
                prompt_cache[prompt] = emb
            action_spans = None
            if msg.get('action_script') is not None:
                from .minimax_h3_world import encode_action_script
                script = msg['action_script']
                geometry_check = h3_latent_geometry(ns.duration, ns.width, ns.height)
                if len(script) != geometry_check.video_t:
                    raise ValueError('action_script length does not match clip latent count')
                emb, action_spans = encode_action_script(str(Path(ns.model_dir) / 'text_encoder'), emb, script, action_cache, device, encode_h3_text_prompt)
                _emit({'event': 'world_conditioning', 'index': index, 'action_script': script, 'spans': action_spans, 'routing': 'directed'})
            _text_s = round(time.perf_counter() - _t_text, 3)
            path = out_dir / f'clip_{index:05d}.mp4'
            first_frame = msg.get('first_frame')
            pinned_last = msg.get('last_frame')
            refs = tuple((_reference_spec(r) for r in msg.get('refs') or [] if r))
            use_latent = bool(msg.get('continue_latent')) and prev_latent is not None and (not first_frame)
            use_extension = use_latent and ns.seamless
            use_projection = use_latent and bool(ns.continuity_projection) and (not use_extension)
            if use_projection:
                _decode_queue.join()
                if decoded_handoff is None or decoded_handoff[0] != index - 1:
                    raise RuntimeError('Previous decode did not provide an in-memory frame')
            feed_images, feed_videos = ((), ())
            continuing = index > 0
            if continuing and ns.continuity_feedback == 'anchor':
                if anchor_frame is not None:
                    feed_images = (Path(anchor_frame),)
            elif continuing and ns.continuity_feedback != 'off':
                _decode_queue.join()
                if prev_clip is not None and prev_clip[0] == index - 1:
                    if ns.continuity_feedback == 'video' and prev_clip[1] and Path(prev_clip[1]).is_file():
                        feed_videos = (Path(prev_clip[1]),)
                    elif ns.continuity_feedback == 'image' and prev_clip[2]:
                        feed_images = (Path(prev_clip[2]),)
            request = H3GenerationRequest(model_dir=Path(ns.model_dir), prompt=prompt, mode='t2v', out=path, image=Path(first_frame) if first_frame else None, last_frame=Path(pinned_last) if pinned_last else None, reference_images=refs + feed_images, source_videos=_clip_source_videos(msg, feed_videos), reference_resolution=max(ns.width, ns.height) if ns.reference_native_res and (feed_images or feed_videos) and (not refs) else None, duration=ns.duration, width=ns.width, height=ns.height, steps=ns.steps, seed=seed, gpu=ns.gpu, resident=bool(ns.resident), attention_backend=ns.attention_backend)
            source_overrides: dict = {}
            source_phases: dict = {}
            if msg.get('source_videos'):
                src_w, src_h = _source_size(msg, ns.width, ns.height)
                if (src_w, src_h) != (int(ns.width), int(ns.height)):
                    for video in msg.get('source_videos') or []:
                        source_overrides.setdefault(str(Path(str(video))), {}).update({'width': src_w, 'height': src_h})
                if _aux_device is not None:
                    _source_ready.wait()
                if source_prefetcher is not None:
                    for video in msg.get('source_videos') or []:
                        item, info = source_prefetcher.get(str(Path(str(video))), src_w, src_h, geometry_frames)
                        source_overrides.setdefault(str(Path(str(video))), {})['encoded'] = item
                        source_phases = {**info, 'source_device': source_prefetcher.encoder.device, 'source_rows': int(item.rows.shape[0]), 'source_latent': [item.latent_t, item.latent_h, item.latent_w]}
            phases: dict = {'text_encode_s': _text_s, **source_phases}
            _mark = time.perf_counter()
            geometry = h3_latent_geometry(ns.duration, ns.width, ns.height)
            if use_extension:
                from .minimax_h3_extension import extension_prefix
                conditioning = _encode_h3_conditioning(request, geometry, root=Path(ns.model_dir), device=device, event_callback=lambda name, payload: _emit({'event': name, **payload}), source_video_overrides=source_overrides)
                conditioning['extension_context'] = extension_prefix(prev_latent, prev_audio, prev_geometry)
                phases.update(continuity='masked_latent', extension_frames=39)
            elif use_projection:
                conditioning = _encode_h3_conditioning(request, geometry, root=Path(ns.model_dir), device=device, source_video_overrides=source_overrides, event_callback=lambda name, payload: _emit({'event': name, **payload}), frame_tensor=decoded_handoff[1], quantize_frame=ns.continuity_projection == 'uint8', projection_video=ns.continuity_projection == 'video', projection_alpha=ns.continuity_projection_alpha, carried_latent=prev_latent)
                phases['continuity'] = 'projection_' + ns.continuity_projection
                phases['projection_alpha'] = ns.continuity_projection_alpha
                phases['handoff_frame_count'] = projection_frames
                phases['handoff_frame_index'] = decoded_handoff[2]
                decoded_handoff = None
            elif use_latent:
                _anchor_key = (str(anchor_frame), int(ns.width), int(ns.height), bool(ns.reference_native_res)) if ns.continuity_feedback == 'anchor' and feed_images and (not msg.get('source_videos')) else None
                _cached = reference_cache.get(_anchor_key) if _anchor_key else None
                if _cached is not None:
                    request = replace(request, reference_images=refs)
                    conditioning = _encode_h3_conditioning(request, geometry, root=Path(ns.model_dir), device=device, event_callback=lambda name, payload: _emit({'event': name, **payload}), carried_latent=prev_latent, source_video_overrides=source_overrides)
                    conditioning.update(_cached)
                    phases['reference_cached'] = True
                else:
                    conditioning = _encode_h3_conditioning(request, geometry, root=Path(ns.model_dir), device=device, event_callback=lambda name, payload: _emit({'event': name, **payload}), carried_latent=prev_latent, source_video_overrides=source_overrides)
                    if _anchor_key is not None:
                        reference_cache[_anchor_key] = {k: conditioning[k] for k in ('reference_video_rows', 'reference_audio_rows', 'ref_blocks') if k in conditioning}
                phases['continuity'] = 'latent+pinned_end' if pinned_last else 'latent'
            else:
                _key = (str(anchor_frame), int(ns.width), int(ns.height), bool(ns.reference_native_res)) if ns.continuity_feedback == 'anchor' and feed_images and (not msg.get('source_videos')) else None
                _hit = reference_cache.get(_key) if _key else None
                if _hit is not None:
                    conditioning = _encode_h3_conditioning(replace(request, reference_images=refs), geometry, root=Path(ns.model_dir), device=device, event_callback=None)
                    conditioning.update(_hit)
                    phases['reference_cached'] = True
                else:
                    conditioning = _encode_h3_conditioning(request, geometry, root=Path(ns.model_dir), device=device, event_callback=lambda name, payload: _emit({'event': name, **payload}), source_video_overrides=source_overrides)
                    if _key is not None:
                        reference_cache[_key] = {k: conditioning[k] for k in ('reference_video_rows', 'reference_audio_rows', 'ref_blocks') if k in conditioning}
                phases['continuity'] = 'image' if first_frame else 'none'
            if feed_videos or feed_images:
                phases['feedback'] = ns.continuity_feedback
            phases['condition_s'] = round(time.perf_counter() - _mark, 3)

            def render(attempt):
                if action_spans is not None and ns.resident:
                    from .minimax_h3_world import check_world_workspace
                    check_world_workspace(session, budget, device)
                from .minimax_h3_int8 import packed_linear_stats
                packed_before = packed_linear_stats()
                _mark = time.perf_counter()
                video, audio, geometry = denoise_h3_t2v(ns.model_dir, emb, duration=ns.duration, width=ns.width, height=ns.height, steps=ns.steps, seed=seed, device=device, resident=bool(ns.resident), attention_backend=ns.attention_backend, session=session, loras=tuple(loras), sparse_attention=sparse_config, event_callback=_pipeline_event, action_text_spans=action_spans, world_head_tags=head_tags, **conditioning)
                phases['denoise_s'] = round(time.perf_counter() - _mark, 3)
                phases['packed_linear_calls'] = {k: v - packed_before[k] for k, v in packed_linear_stats().items()}
                torch.cuda.empty_cache()
                from cosmos_windows.weight_streaming.minimax_h3_backend import stream_stats
                _ds = stream_stats()
                phases['dit_binds'] = _ds['binds']
                phases['dit_wait_s'] = round(_ds['wait_s'], 3)
                phases['dit_gb'] = round(_ds['bytes'] / 1000000000.0, 3)
                if 'dll_uploaded_bytes' in _ds:
                    phases['dit_uploaded_gb'] = round(_ds['dll_uploaded_bytes'] / 1000000000.0, 3)
                from cosmos_windows.weight_streaming.minimax_h3_native import prologue_stats
                _pro = prologue_stats()
                phases['prologue_fused'] = _pro['fused']
                phases['prologue_chunked'] = _pro['chunked']
                from cosmos_windows.weight_streaming.minimax_h3_attention import int8_attention_stats
                phases['int8_attention_calls'] = int8_attention_stats()
                validation = {'ready': threading.Event()} if action_spans is not None else None
                phases['noise_retries'] = attempt
                _decode_queue.put((index, video, audio, geometry, path, dict(phases), started, seed, validation))
                if validation is not None:
                    if not validation['ready'].wait(timeout=120):
                        raise RuntimeError('World decode made no progress for 120 seconds')
                    if 'error' in validation:
                        raise validation['error']
                return (video.detach(), audio.detach(), geometry)
            if action_spans is not None:
                from .minimax_h3_world import retry_world_noise
                prev_latent, prev_audio, prev_geometry = retry_world_noise(render, lambda: yield_world_residency(session, _decode_queue.join, _emit), lambda ev: _emit(dict(index=index, **ev)))
            else:
                prev_latent, prev_audio, prev_geometry = render(0)
            _emit({'event': 'clip_denoised', 'index': index, 'denoise_s': phases['denoise_s'], 'seed': seed})
            index += 1
        except Exception as exc:
            _emit({'event': 'clip_failed', 'index': index, 'error_type': type(exc).__name__, 'message': str(exc)[:400], 'traceback': traceback.format_exc()[-1200:]})
            index += 1
    _decode_queue.join()
    from .minimax_h3_vae_mux import clear_audio_component_cache, clear_video_decode_cache
    clear_video_decode_cache()
    clear_audio_component_cache()
    _emit({'event': 'stopped', 'clips': index, 'reason': 'stdin closed'})
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
