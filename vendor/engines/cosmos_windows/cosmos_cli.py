# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/cosmos_cli.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import base64
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
MODELS_ROOT = os.environ.get('COSMOS_MODELS_ROOT', '')
MODELS = {'nano': dict(dir='Cosmos3-Nano', label='Cosmos3-Nano (16B omni)', modalities=['reason', 't2i', 'i2i', 't2v', 'i2v', 'v2v', 'av', 'action'], offload='model', reasoner='mac'), 'super': dict(dir='Cosmos3-Super', label='Cosmos3-Super (64B omni)', modalities=['reason', 't2i', 'i2i', 't2v', 'i2v', 'v2v', 'av', 'action'], offload='sequential', reasoner='mac', note='64B -> offload=sequential, or /stream for DirectStorage weight-streaming (fast on 32GB).'), 'super-t2i': dict(dir='Cosmos3-Super-Text2Image', label='Cosmos3-Super-Text2Image (high-fidelity T2I)', modalities=['t2i', 'i2i'], offload='sequential', reasoner=None), 'super-i2v': dict(dir='Cosmos3-Super-Image2Video', label='Cosmos3-Super-Image2Video', modalities=['i2v', 'v2v'], offload='sequential', reasoner=None), 'droid': dict(dir='Cosmos3-Nano-Policy-DROID', label='Cosmos3-Nano-Policy-DROID (robot action policy)', modalities=['policy'], offload='model', reasoner=None, note='DROID platform: language instruction + visual observation -> robot action trajectory.')}
SAMPLING = {'t2i': dict(guidance=4.0, flow_shift=3.0, steps=50), 'i2i': dict(guidance=4.0, flow_shift=3.0, steps=40), 't2v': dict(guidance=6.0, flow_shift=10.0, steps=35), 'i2v': dict(guidance=6.0, flow_shift=10.0, steps=35), 'v2v': dict(guidance=6.0, flow_shift=10.0, steps=35), 'av': dict(guidance=6.0, flow_shift=10.0, steps=35)}
_SAMPLERS = {'unipc': 'UniPCMultistepScheduler', 'dpm2m': 'DPMSolverMultistepScheduler', 'dpmpp2m': 'DPMSolverMultistepScheduler', 'dpm2m_sde': 'DPMSolverMultistepScheduler', 'dpm_single': 'DPMSolverSinglestepScheduler', 'deis': 'DEISMultistepScheduler', 'euler': 'EulerDiscreteScheduler', 'euler_a': 'EulerAncestralDiscreteScheduler', 'heun': 'HeunDiscreteScheduler', 'kdpm2': 'KDPM2DiscreteScheduler', 'kdpm2_a': 'KDPM2AncestralDiscreteScheduler', 'lms': 'LMSDiscreteScheduler'}
_SIGMAS = {'flow': 'use_flow_sigmas', 'karras': 'use_karras_sigmas', 'exponential': 'use_exponential_sigmas', 'beta': 'use_beta_sigmas', 'normal': None, 'simple': None}
try:
    from cosmos_windows.res4lyf_sampling import RES4LYF_SIGMAS, configure_diffusers_flow_scheduler, extend_unique, is_res4lyf_sampler, is_res4lyf_sigma, res4lyf_sampler_options
    for _name in res4lyf_sampler_options():
        _SAMPLERS.setdefault(_name, 'FlowMatchEulerDiscreteScheduler')
    _SIGMAS = {name: _SIGMAS.get(name) for name in extend_unique(_SIGMAS.keys(), RES4LYF_SIGMAS)}
except Exception:
    RES4LYF_SIGMAS = []
    configure_diffusers_flow_scheduler = None
    is_res4lyf_sampler = lambda _name: False
    is_res4lyf_sigma = lambda _name: False
DEFAULTS = {'reasoner_host': '<mac-ip>', 'reasoner_port': 9200, 'temperature': 0.0, 'top_p': 0.95, 'max_new_tokens': 512, 'system': '', 'model': '', 'steps': 35, 'guidance': 6.0, 'height': 720, 'width': 1280, 'num_frames': 1, 'fps': 24, 'flow_shift': 10.0, 'negative': '', 'seed': 42, 'offload': 'model', 'gpu_index': '1', 'weight_stream': True, 'weight_backend': 'directstorage', 'prefetch': True, 'sampler': 'unipc', 'sigmas': 'karras', 'v_pred': False, 'clip_skip': 0, 'cfg_batch': False, 'enable_sound': False, 'memeff_attn': True, 'vae_tile': 'auto', 'mode': 'reason', 'loop_render': 't2i', 'out': '', 'auto_prefill': 'on', 'prefill_min_tokens': 1500, 'prefill_max_tokens': 64000, 'remote_host': '<mac-ip>', 'remote_port': 9100, 'mac_user': 'private', 'mac_key': str(Path.home() / '.ssh' / 'mac_ed'), 'mac_python': '/tmp/cosmos3_mps_venv/bin/python', 'mac_model': ''}
_INT = {'reasoner_port', 'max_new_tokens', 'steps', 'height', 'width', 'num_frames', 'fps', 'seed', 'remote_port', 'prefill_min_tokens', 'prefill_max_tokens'}
_FLOAT = {'temperature', 'top_p', 'guidance', 'flow_shift'}
_BOOL = {'weight_stream', 'prefetch', 'cfg_batch', 'memeff_attn', 'enable_sound'}

def _avail_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return int(ms.ullAvailPhys)
    except Exception:
        return None

def _safe_stamp(text, fallback='out'):
    s = ''.join((c if c.isalnum() or c in '-_' else '_' for c in (text or '')[:32])).strip('_')
    return s or fallback

def _unique_path(p):
    p = Path(p)
    if not p.exists():
        return p
    i = 1
    while True:
        cand = p.with_name(f'{p.stem}_{i}{p.suffix}')
        if not cand.exists():
            return cand
        i += 1

def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'

def mux_sound_into_video(mp4_path, wav, sample_rate, outdir, stem):
    import numpy as np
    from scipy.io import wavfile
    try:
        a = np.asarray(wav, dtype=np.float32)
        if a.size == 0 or not np.isfinite(a).all() or a.ndim not in (1, 2):
            raise ValueError('Generated audio must be a finite, nonempty mono/stereo waveform')
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] in (1, 2) and a.shape[0] < a.shape[1]:
            a = a.T
        if a.ndim != 2 or a.shape[1] not in (1, 2) or int(sample_rate) <= 0:
            raise ValueError('Generated audio must have one or two channels and a positive sample rate')
        a = np.clip(a, -1.0, 1.0)
        pcm = (a * 32767.0).astype(np.int16)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        wav_path = _unique_path(Path(outdir) / f'{stem}_sound.wav')
        wavfile.write(str(wav_path), int(sample_rate), pcm)
        out = _unique_path(Path(outdir) / f'{stem}_with_audio.mp4')
        cmd = [_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(mp4_path), '-i', str(wav_path), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-map', '0:v:0', '-map', '1:a:0', str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and out.is_file():
            print(f'  [audio] muxed generated sound -> {out}', flush=True)
            return out
        raise RuntimeError(f"Generated sound mux failed (ffmpeg rc={r.returncode}): {(r.stderr or '')[-800:]}")
    except Exception as e:
        raise RuntimeError(f'Could not deliver requested Cosmos audio: {e}') from e

class Cli:

    def __init__(self, st):
        self.st = st
        self._pipe = None
        self._pipe_key = None
        self._streamer = None
        self._custom_sigmas = None
        self._active_steps = int(st.get('steps', 35) or 35)
        self._user_set = set()
        self.current = 'nano'

    @staticmethod
    def _model_status(p):
        if not p.exists():
            return 'ABSENT'
        cache = p / '.cache' / 'huggingface' / 'download'
        if cache.exists() and next(cache.rglob('*.incomplete'), None) is not None:
            has_weights = next(p.rglob('*.safetensors'), None) is not None or next(p.rglob('*.gguf'), None) is not None or (p / 'model_index.json').exists()
            if not has_weights:
                return 'downloading'
        return 'present'

    def models_list(self):
        print('models (use /model NAME):')
        for name, m in MODELS.items():
            status = self._model_status(Path(MODELS_ROOT) / m['dir'])
            mark = '*' if name == self.current else ' '
            print(f" {mark}{name:10s} {status:11s} {m['label']}")
            print(f"            use: {', '.join(m['modalities'])}")

    def select_model(self, name):
        if name not in MODELS:
            print(f'  unknown model {name!r}. /models to list.')
            return
        m = MODELS[name]
        p = Path(MODELS_ROOT) / m['dir']
        status = self._model_status(p)
        if status == 'ABSENT':
            print(f'  {name} not downloaded yet ({p}). /models shows status.')
            return
        if status == 'downloading':
            print(f'  {name} is still downloading (incomplete shards in {p}). Wait for it to finish — selecting it now would fail on missing weights. /models shows status.')
            return
        self.current = name
        self.st['model'] = str(p)
        self.st['offload'] = m['offload']
        self._pipe = None
        print(f"  model = {m['label']}")
        print(f"    path={p}  offload={m['offload']}  modalities={', '.join(m['modalities'])}  reasoner={m['reasoner']}")
        if m.get('note'):
            print(f"    note: {m['note']}")

    def _supports(self, modality):
        m = MODELS[self.current]
        if modality not in m['modalities']:
            print(f"  {self.current!r} does not do {modality!r} (it does: {', '.join(m['modalities'])}). /model NAME to switch.")
            return False
        return True

    def show(self):
        print('settings:')
        for k in sorted(self.st):
            print(f'  {k:14s} = {self.st[k]!r}')

    def set(self, key, val):
        if key in ('res', 'resolution'):
            w, h = val.lower().split('x')
            self.st['width'], self.st['height'] = (int(w), int(h))
            print(f'  width={w} height={h}')
            return
        if key not in self.st:
            print(f'  unknown setting {key!r} (see /show)')
            return
        if key in _INT:
            val = int(val)
        elif key in _FLOAT:
            val = float(val)
        elif key in _BOOL:
            val = str(val).strip().lower() in ('1', 'true', 'on', 'yes', 'y')
        if key in ('weight_stream', 'weight_backend', 'model', 'offload', 'prefetch'):
            self._pipe = None
        if key in ('guidance', 'flow_shift', 'steps'):
            self._user_set.add(key)
        self.st[key] = val
        print(f'  {key} = {val!r}')
    _PERF_PROFILES = {'fast': {'cfg_batch': True, 'prefetch': True, 'vae_tile': 'auto', 'memeff_attn': True, 'offload': 'none'}, 'balanced': {'cfg_batch': False, 'prefetch': True, 'vae_tile': 'auto', 'memeff_attn': True, 'offload': 'model'}, 'lowvram': {'cfg_batch': False, 'prefetch': False, 'vae_tile': 'on', 'memeff_attn': True, 'offload': 'sequential'}, 'longvideo': {'cfg_batch': False, 'prefetch': False, 'vae_tile': 'on', 'memeff_attn': True, 'offload': 'sequential'}}

    def perf(self, profile):
        if not profile:
            print('  usage: /perf fast|balanced|lowvram|longvideo')
            print('  current: ' + ', '.join((f'{k}={self.st[k]!r}' for k in ('cfg_batch', 'prefetch', 'vae_tile', 'memeff_attn', 'offload'))))
            return
        p = self._PERF_PROFILES.get(profile)
        if not p:
            print(f"  unknown perf profile {profile!r}. choose: {', '.join(self._PERF_PROFILES)}")
            return
        print(f"  perf profile '{profile}' - applying the speed<->VRAM bundle:")
        for k, v in p.items():
            self.set(k, str(v).lower() if isinstance(v, bool) else str(v))
        print('  (tune individual knobs with /set; quality = /set steps + /set res, separate from these)')

    def _reason_request(self, prompt, image_path=None, system=None, max_new=None):
        body = {'prompt': prompt, 'temperature': self.st['temperature'], 'top_p': self.st['top_p'], 'max_new_tokens': int(max_new or self.st['max_new_tokens'])}
        if system or self.st['system']:
            body['system'] = system or self.st['system']
        if image_path:
            body['image_b64'] = base64.b64encode(Path(image_path).read_bytes()).decode()
        url = f"{self.st['reasoner_host']}:{self.st['reasoner_port']}/reason"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def reason(self, prompt, image_path=None):
        if not prompt:
            print('  usage: /reason PROMPT [image=PATH]')
            return None
        if image_path and (not Path(image_path).exists()):
            print(f'  image not found: {image_path!r} (check the path/quoting)')
            return None
        try:
            res = self._reason_request(prompt, image_path)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            print(f'  reasoner unreachable ({type(e).__name__}: {e}). Try /sync to (re)start the Mac server.')
            return None
        except Exception as e:
            print(f'  reason failed ({type(e).__name__}: {e})')
            return None
        if 'error' in res:
            print(f"  reasoner error: {res['error']}")
            return None
        print(f"\n--- reasoner ({res['seconds']}s, {res['gen_tokens']} tok) ---\n{res['answer']}\n")
        return res['answer']

    def preason(self, prompt):
        mode = None
        if prompt and prompt.split()[0] in ('--local', '--quant'):
            tok = prompt.split()[0]
            mode = tok[2:]
            prompt = prompt[len(tok):].strip()
        if not prompt:
            print('  usage: /preason [--local|--quant] PROMPT   (no flag = RTX prefill -> Mac decode)')
            return None
        if mode:
            cmd = [sys.executable, '-m', 'cosmos_windows.reasoner_prefill_stream', '--mode', mode, '--device', 'cuda', '--model', self.st['model'], '--gpu-index', self.st['gpu_index'], '--prompt', prompt, '--max-new-tokens', str(self.st['max_new_tokens'])]
            print(f'  [preason --{mode}] prefill+decode on the RTX (single machine, no Mac)...')
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            ans = None
            for line in r.stdout.splitlines():
                if line.startswith('ANSWER_JSON'):
                    ans = json.loads(line[len('ANSWER_JSON'):].strip())
            if ans:
                print(f"\n--- reasoner (RTX {mode}, {ans['prefill_tokens']} tok, {ans['seconds']}s) ---\n{ans['answer']}\n")
                return ans['answer']
            print('  local reason failed:\n', (r.stdout or '')[-600:], (r.stderr or '')[-600:])
            return None
        out = Path(self.st['out']) / 'cli_prefill'
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-m', 'cosmos_windows.reasoner_prefill_stream', '--mode', 'export', '--device', 'cuda', '--model', self.st['model'], '--gpu-index', self.st['gpu_index'], '--prompt', prompt, '--out', str(out)]
        print('  [preason] 1/2 prefilling on the RTX (CUDA)...')
        if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            print('  RTX prefill failed.')
            return None
        meta = json.loads(Path(str(out) + '.json').read_text())
        kv_bytes = Path(str(out) + '.safetensors').read_bytes()
        url = f"{self.st['reasoner_host']}:{self.st['reasoner_port']}/decode_kv"
        print(f'  [preason] 2/2 shipping {len(kv_bytes) // 1000000} MB KV -> Mac decode...')
        try:
            req = urllib.request.Request(url, data=kv_bytes, headers={'Content-Type': 'application/octet-stream', 'X-Prompt-Len': str(meta['prompt_len']), 'X-Next-Id': str(meta['next_id']), 'X-Max-New-Tokens': str(self.st['max_new_tokens'])})
            with urllib.request.urlopen(req, timeout=900) as r:
                res = json.loads(r.read())
        except Exception as e:
            print(f'  Mac decode unreachable ({type(e).__name__}). /sync to (re)start the Mac server.')
            return None
        if 'error' in res:
            print(f"  decode error: {res['error']}")
            return None
        print(f"\n--- reasoner (RTX prefill {res['prefill_tokens']} tok -> Mac decode, {res['seconds']}s) ---\n{res['answer']}\n")
        return res['answer']

    def health(self):
        url = f"{self.st['reasoner_host']}:{self.st['reasoner_port']}/health"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                print('  Mac reasoner:', r.read().decode())
                return True
        except Exception as e:
            print(f'  Mac reasoner down ({type(e).__name__}). /sync to start it.')
            return False

    def _transformer_bytes(self):
        tdir = Path(self.st['model']) / 'transformer'
        return sum((f.stat().st_size for f in tdir.glob('*.safetensors'))) if tdir.exists() else 0

    def _memory_guard(self):
        if self.st['weight_stream']:
            return True
        need = self._transformer_bytes()
        avail = _avail_ram_bytes()
        if need and (avail is None or need * 1.25 > avail):
            a = f'{avail / 1000000000.0:.1f}GB avail' if avail else 'RAM unmeasurable'
            print(f"  [guard] host-resident load needs ~{need / 1000000000.0:.1f}GB RAM ({a}) -> forcing weight-streaming (NVMe->GPU) so this won't crash the box. /set weight_stream off to override (risky).")
            return True
        return False

    def _set_scheduler(self, pipe):
        import diffusers
        base = dict(self._sched_base_config)
        name = self.st['sampler']
        sig = self.st['sigmas']
        self._custom_sigmas = None
        if configure_diffusers_flow_scheduler is not None and (is_res4lyf_sampler(name) or is_res4lyf_sigma(sig)):
            self._custom_sigmas = configure_diffusers_flow_scheduler(pipe, sampler=name, sigmas=sig, num_inference_steps=int(getattr(self, '_active_steps', self.st['steps']) or self.st['steps']), shift=self.st['flow_shift'], log_prefix='[cosmos]')
            return
        if name not in _SAMPLERS and (not is_res4lyf_sampler(name)):
            raise ValueError(f"Unsupported sampler={name!r}; choose one of {', '.join(sorted(_SAMPLERS))}")
        if sig not in _SIGMAS and (not is_res4lyf_sigma(sig)):
            raise ValueError(f"Unsupported sigma schedule={sig!r}; choose one of {', '.join(sorted(_SIGMAS))}")
        if sig in ('exponential',):
            print(f'  [warn] sigmas={sig} bypasses the flow-sigma timesteps on this flow model (would break); using base flow spacing instead.', flush=True)
        kw = dict(flow_shift=self.st['flow_shift'], use_flow_sigmas=True, use_karras_sigmas=sig == 'karras')
        if self.st.get('v_pred'):
            kw = {'prediction_type': 'v_prediction'}
            if sig == 'karras':
                kw['use_karras_sigmas'] = True
        if name == 'dpm2m_sde':
            kw['algorithm_type'] = 'sde-dpmsolver++'
        try:
            cls = getattr(diffusers, _SAMPLERS[name])
            pipe.scheduler = cls.from_config(base, **kw)
        except Exception as e:
            raise RuntimeError(f'sampler={name} sigmas={sig} unsupported: {type(e).__name__}: {str(e)[:120]}') from e

    def free(self):
        streamer = getattr(self, '_streamer', None)
        close = getattr(streamer, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._pipe = None
        self._pipe_key = None
        self._streamer = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def upscale(self, args):
        img = args[0]
        kv = dict((a.split('=', 1) for a in args[1:] if '=' in a))
        scale = kv.get('scale', '4')
        backbone = kv.get('backbone', 'flux')
        out = kv.get('out', os.path.splitext(img)[0] + '_upscaled.png')
        self.free()
        pidpy = os.environ.get('PID_VENV_PYTHON', '')
        if not os.path.isfile(pidpy):
            print(f'  [upscale] PiD venv python not found: {pidpy} (set PID_VENV_PYTHON)')
            return
        cmd = [pidpy, '-m', 'cosmos_windows.pid_upscale', '--image', img, '--out', out, '--scale', str(scale), '--backbone', backbone, '--gpu', str(self.st.get('gpu_index', '1'))]
        print(f'  PiD upscale x{scale} ({backbone}) -> {out}', flush=True)
        subprocess.run(cmd, cwd=str(REPO))

    def _get_pipe(self, need_sound=False):
        stream = self.st['weight_stream'] or self._memory_guard()
        key = (self.st['model'], stream, self.st['weight_backend'], self.st['offload'], need_sound, self.st['prefetch'])
        if self._pipe is not None and self._pipe_key == key:
            self._set_scheduler(self._pipe)
            return self._pipe
        sound_kw = {} if need_sound else {'sound_tokenizer': None}
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', self.st['gpu_index'])
        sys.path.insert(0, str(REPO))
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self._pipe = None
        self._streamer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stream:
            backend = self.st['weight_backend']
            print(f"  [load] diffusion {self.st['model']} — weight-streamed transformer ({backend} NVMe->GPU, ~0 host RAM; 91.7% of the transformer streams per-layer)...", flush=True)
            from cosmos_windows.weight_streaming.build_streamed_transformer import build_ds_streamed_transformer, build_streamed_transformer
            pf = self.st['prefetch']
            ds_bufs = 3 if pf else 1
            pt_window = 2 if pf else 1
            if backend == 'directstorage':
                try:
                    tf, self._streamer = build_ds_streamed_transformer(self.st['model'], device='cuda', n_gpu_buffers=ds_bufs)
                except Exception as e:
                    print(f'  [warn] DirectStorage init failed ({type(e).__name__}: {str(e)[:80]}); falling back to portable mmap streaming.', flush=True)
                    tf, self._streamer = build_streamed_transformer(self.st['model'], window=pt_window, device='cuda')
            else:
                tf, self._streamer = build_streamed_transformer(self.st['model'], window=pt_window, device='cuda')
            pipe = Cosmos3OmniPipeline.from_pretrained(self.st['model'], torch_dtype=torch.bfloat16, transformer=tf, enable_safety_checker=False, **sound_kw)
            for nm, comp in pipe.components.items():
                if nm != 'transformer' and isinstance(comp, torch.nn.Module):
                    comp.to('cuda')
        else:
            print(f"  [load] diffusion {self.st['model']} (bf16, offload={self.st['offload']}) on RTX...", flush=True)
            pipe = Cosmos3OmniPipeline.from_pretrained(self.st['model'], torch_dtype=torch.bfloat16, enable_safety_checker=False, **sound_kw)
            if self.st['offload'] == 'model':
                pipe.enable_model_cpu_offload()
            elif self.st['offload'] == 'sequential':
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to('cuda')
        self._sched_base_config = dict(pipe.scheduler.config)
        self._set_scheduler(pipe)
        self._pipe, self._pipe_key = (pipe, key)
        return pipe

    def _resolve_sampling(self, modality):
        samp = SAMPLING.get(modality, {})
        pick = lambda k: self.st[k] if k in self._user_set else samp.get(k, self.st[k])
        return (pick('steps'), pick('guidance'), pick('flow_shift'))

    def _apply_vae_tiling(self, pipe, modality):
        mode = str(self.st.get('vae_tile', 'auto')).lower()
        big = min(self.st['width'], self.st['height']) >= 512
        want = mode == 'on' or (mode == 'auto' and big)
        vae = getattr(pipe, 'vae', None)
        if vae is None:
            return
        try:
            if want and hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
                print('  [vae_tile] on (tiled decode — lower VRAM at high res)', flush=True)
            elif not want and hasattr(vae, 'disable_tiling'):
                vae.disable_tiling()
        except Exception as e:
            print('  [vae_tile] err:', e, flush=True)

    def generate(self, modality, prompt, image_path=None, video_path=None, num_conditional_frames=None, on_step=None, should_cancel=None):
        if not self._supports(modality):
            return
        import torch
        from PIL import Image
        from .weight_streaming.lora_stack import preflight_model_loras
        lora_prepared = preflight_model_loras(self.st['model'], self.st.get('loras'))
        edit_source = edit_mask = None
        edit_strength = None
        if modality == 'i2i':
            if not image_path:
                raise ValueError('Cosmos image editing requires a source image')
            edit_strength = float(self.st.get('denoise') if self.st.get('denoise') is not None else 1.0)
            if not math.isfinite(edit_strength) or not 0 <= edit_strength <= 1:
                raise ValueError('Cosmos edit strength must be finite and between zero and one')
            with Image.open(image_path) as src:
                edit_source = src.convert('RGB').resize((self.st['width'], self.st['height']), Image.Resampling.LANCZOS)
            if self.st.get('mask_path'):
                with Image.open(self.st['mask_path']) as mask:
                    edit_mask = mask.convert('L').copy()
            if edit_strength == 0:
                if should_cancel is not None and should_cancel():
                    raise RuntimeError('Cosmos image edit cancelled before initialization')
                outdir = Path(self.st['out'])
                outdir.mkdir(parents=True, exist_ok=True)
                path = _unique_path(outdir / f'i2i_{_safe_stamp(prompt, modality)}.png')
                edit_source.save(path)
                return path
        steps, guidance, shift = self._resolve_sampling(modality)
        audio_reference = self.st.get('audio_mode') == 'condition'
        audio_codec_path = None
        audio_waveform = audio_rate = None
        if audio_reference:
            if modality not in ('t2v', 'i2v', 'v2v', 'av'):
                raise ValueError('Cosmos audio conditioning requires a generated video')
            if not self.st.get('audio_path') or not Path(self.st['audio_path']).is_file():
                raise ValueError('Cosmos audio conditioning requires an existing reference audio file')
            from .audio_condition import require_codec
            audio_codec_path = require_codec(self.st.get('model_support_files'))
            import soundfile
            reference_frames = 17 if self.st['num_frames'] <= 1 else self.st['num_frames']
            with soundfile.SoundFile(self.st['audio_path']) as reference:
                if reference.channels not in (1, 2) or reference.frames < 1 or reference.samplerate <= 0:
                    raise ValueError('Cosmos reference audio must be a nonempty mono or stereo recording')
                audio_rate = reference.samplerate
                audio_waveform = reference.read(frames=math.ceil(reference_frames / self.st['fps'] * audio_rate), dtype='float32', always_2d=True)
        timed_rows = list(self.st.get('keyframes') or [])
        if self.st.get('image2_path'):
            timed_rows.append({'image_path': self.st['image2_path'], 'frame': 16 if self.st['num_frames'] <= 1 else int(self.st['num_frames']) - 1})
        if timed_rows:
            if modality not in ('t2v', 'i2v', 'av'):
                raise ValueError('Timed Cosmos references require text/image-to-video or joint audio-video')
            timeline_frames = 17 if self.st['num_frames'] <= 1 else self.st['num_frames']
            if image_path and (not any((row.get('image_path') == image_path for row in timed_rows))):
                timed_rows.append({'image_path': image_path, 'frame': 0})
            from .timed_frames import timeline
            timeline(timed_rows, frames=timeline_frames, temporal_scale=1)
        self.st['flow_shift'] = shift
        self._active_steps = int(steps)
        want_sound = modality == 'av' or bool(self.st.get('enable_sound')) or audio_reference
        pipe = self._get_pipe(need_sound=want_sound)
        if self.st.get('memeff_attn', True):
            from .attention_memeff import enable_memeff_attention
            allow_attention_fallback = os.environ.get('STUDIO_ALLOW_ATTENTION_FALLBACK', '').strip().lower()
            enable_memeff_attention(allow_math_fallback=allow_attention_fallback in {'1', 'true', 'yes', 'on'})
        self._apply_vae_tiling(pipe, modality)
        from .cfg_batch import enable_cfg_batch, disable_cfg_batch
        if self.st.get('cfg_batch') and guidance != 1.0:
            enable_cfg_batch(pipe)
            print('  [cfg_batch] on — 1 merged forward/step for both CFG passes (~1.8x)', flush=True)
        else:
            disable_cfg_batch(pipe)
        nf = self.st['num_frames']
        if modality in ('t2i', 'i2i'):
            nf = 1
        if modality in ('t2v', 'i2v', 'v2v', 'av') and nf <= 1:
            nf = 17
        kw = dict(prompt=prompt, negative_prompt=self.st['negative'], height=self.st['height'], width=self.st['width'], num_frames=nf, num_inference_steps=steps, guidance_scale=guidance, add_resolution_template=False, add_duration_template=False, generator=torch.Generator(device='cuda').manual_seed(self.st['seed']))
        custom_sigmas = getattr(self, '_custom_sigmas', None)
        if custom_sigmas is not None:
            try:
                import inspect as _inspect
                if 'sigmas' in set(_inspect.signature(pipe.__call__).parameters):
                    kw['sigmas'] = custom_sigmas
            except (TypeError, ValueError):
                pass
        if modality in ('t2v', 'i2v', 'v2v', 'av'):
            kw['fps'] = self.st['fps']
        if modality == 'i2v':
            if not image_path and (not timed_rows):
                raise ValueError('Cosmos image-to-video needs an input image or timed reference')
            if image_path:
                kw['image'] = Image.open(image_path).convert('RGB')
        if modality == 'av' and image_path:
            kw['image'] = Image.open(image_path).convert('RGB')
        if modality == 'v2v':
            vp = video_path or image_path
            if not vp:
                print("  usage: /v2v VIDEO_PATH PROMPT  (conditions on the input video's leading frames)")
                return
            from diffusers.utils import load_video as _load_video
            from .v2v_patch import enable_v2v, set_v2v_conditioning, _frames_to_cond_pixels
            src_frames = _load_video(vp)
            ncf = int(num_conditional_frames) if num_conditional_frames else min(len(src_frames), max(1, nf // 4))
            ncf = max(1, min(ncf, len(src_frames), nf))
            cond_pil = [f.convert('RGB').resize((self.st['width'], self.st['height'])) for f in src_frames[:ncf]]
            dev = getattr(pipe, '_execution_device', None) or 'cuda'
            dt = getattr(pipe, 'dtype', None) or torch.bfloat16
            cond_pixels = _frames_to_cond_pixels(cond_pil, dev, dt)
            enable_v2v(pipe)
            lp = set_v2v_conditioning(pipe, cond_pixels, num_conditional_frames=ncf)
            kw['image'] = cond_pil[0]
            print(f'  [gen:v2v] conditioning on {ncf} input frames -> {lp} clean latent frames', flush=True)
        if modality == 'i2i':
            from .image_edit import validate_edit
            validate_edit(pipe, edit_strength)
            kw['image'] = edit_source
        if want_sound:
            kw['enable_sound'] = True
        print(f"  [gen:{modality}] {self.st['width']}x{self.st['height']} frames={nf} steps={steps} cfg={guidance} shift={self.st['flow_shift']}...", flush=True)
        if on_step is not None or should_cancel is not None:
            try:
                import inspect as _inspect
                _accepted = set(_inspect.signature(pipe.__call__).parameters)
            except (TypeError, ValueError):
                _accepted = set()
            if 'callback_on_step_end' in _accepted:
                _total = int(steps)
                _prev_dir = Path(self.st['out']) / '.previews'
                _prev_every = max(1, _total // 4)

                def _decode_preview(_p, _ckw, step):
                    try:
                        import numpy as _np
                        import torch as _pt
                        from PIL import Image as _Im
                        lat = _ckw.get('latents')
                        vae = getattr(_p, 'vae', None)
                        if lat is None or vae is None:
                            return None
                        with _pt.no_grad():
                            l = lat
                            mean = getattr(_p, '_vae_latents_mean', None)
                            inv_std = getattr(_p, '_vae_latents_inv_std', None)
                            if l.dim() == 5 and l.shape[2] > 1:
                                l = l[:, :, :1]
                            if mean is not None and inv_std is not None:
                                _dt = vae.dtype
                                m = mean.to(device=l.device, dtype=_dt).view(1, -1, 1, 1, 1)
                                istd = inv_std.to(device=l.device, dtype=_dt).view(1, -1, 1, 1, 1)
                                z = l.to(_dt) / istd + m
                            else:
                                sf = getattr(getattr(vae, 'config', None), 'scaling_factor', None)
                                z = l / sf if sf else l
                            dec = vae.decode(z, return_dict=False)[0]
                        img = dec[0]
                        while img.dim() > 3:
                            img = img[:, 0]
                        img = (img.float().clamp(-1, 1) + 1) / 2
                        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(_np.uint8)
                        im = _Im.fromarray(arr)
                        im.thumbnail((384, 384))
                        _prev_dir.mkdir(parents=True, exist_ok=True)
                        fp = _prev_dir / f'prev_{step}.png'
                        im.save(fp)
                        return str(fp)
                    except Exception as _pe:
                        _lat = _ckw.get('latents')
                        _shape = tuple(_lat.shape) if hasattr(_lat, 'shape') else None
                        print(f'  [gen] preview decode skipped: {type(_pe).__name__}: {str(_pe)[:140]} (latent shape={_shape})', flush=True)
                        return None

                def _step_cb(_p, _i, _t, _ckw):
                    step = int(_i) + 1
                    actual_total = int(getattr(_p, 'num_timesteps', _total) or _total)
                    preview = None
                    if step % max(1, actual_total // 4) == 0 or step == actual_total:
                        preview = _decode_preview(_p, _ckw, step)
                    if on_step is not None:
                        try:
                            on_step(step, actual_total, preview)
                        except TypeError:
                            try:
                                on_step(step, actual_total)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    if should_cancel is not None and should_cancel():
                        try:
                            _p._interrupt = True
                        except Exception:
                            pass
                    return _ckw
                kw['callback_on_step_end'] = _step_cb
                kw.setdefault('callback_on_step_end_tensor_inputs', ['latents'])
                print(f'  [gen] live progress + previews + cancellable ({_total} steps)', flush=True)
        t0 = time.time()
        conditioning_receipt = None
        audio_receipt = None
        encoded_sound = None
        if audio_reference:
            from .audio_condition import load_codec, encode_reference
            if pipe.sound_tokenizer is None:
                raise ValueError('Selected Cosmos model has no joint audio tokenizer')
            codec = load_codec(pipe, audio_codec_path)
            try:
                encoded_sound, audio_receipt = encode_reference(pipe, codec, torch.from_numpy(audio_waveform), audio_rate, frames=nf, fps=self.st['fps'], should_cancel=should_cancel)
            finally:
                codec.to('cpu')
            import hashlib
            audio_receipt.update(reference=self.st['audio_path'], reference_interval_sha256=hashlib.sha256(audio_waveform.tobytes()).hexdigest(), reference_samples_read=len(audio_waveform), reference_start_seconds=0)

        def call_pipe():
            from contextlib import ExitStack
            from .weight_streaming.lora_stack import stacked_adapters
            with ExitStack() as stack:
                if lora_prepared:
                    stack.enter_context(stacked_adapters(pipe.transformer, self.st.get('loras'), device='cuda', prepared=lora_prepared))
                if encoded_sound is not None:
                    from .audio_condition import audio_condition
                    with audio_condition(pipe, encoded_sound):
                        return pipe(**kw)
                return pipe(**kw)
        try:
            if modality == 'i2i':
                from .image_edit import image_edit
                with image_edit(pipe, edit_source, edit_mask, strength=edit_strength, steps=steps, height=self.st['height'], width=self.st['width'], generator=kw['generator']) as edit:
                    kw['callback_on_step_end'] = edit['callback'](kw.get('callback_on_step_end'))
                    kw['callback_on_step_end_tensor_inputs'] = ['latents']
                    res = call_pipe()
                    res.video = [edit['protect'](frame) for frame in res.video]
            elif timed_rows:
                from .timed_frames import timed_frames
                with timed_frames(pipe, timed_rows, frames=nf, height=self.st['height'], width=self.st['width']) as conditioning_receipt:
                    res = call_pipe()
            else:
                res = call_pipe()
        finally:
            try:
                from .v2v_patch import clear_v2v_conditioning as _clear_v2v
                _clear_v2v(pipe)
            except Exception:
                pass
        frames = list(res.video) if not isinstance(getattr(res, 'video', None), list) else res.video
        outdir = Path(self.st['out'])
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = _safe_stamp(prompt, modality)
        if len(frames) == 1:
            p = _unique_path(outdir / f'{modality}_{stamp}.png')
            frames[0].save(p)
        else:
            from diffusers.utils import export_to_video
            p = _unique_path(outdir / f'{modality}_{stamp}.mp4')
            export_to_video(frames, str(p), fps=self.st['fps'], macro_block_size=1)
        snd = getattr(res, 'sound', None)
        if snd is not None:
            import numpy as np
            wav = snd.float().cpu().numpy()
            np.save(str(_unique_path(outdir / f'{modality}_{stamp}_sound.npy')), wav)
            if len(frames) > 1:
                sr = 48000
                try:
                    sr = int(getattr(getattr(pipe, 'sound_tokenizer', None).config, 'sampling_rate', 48000))
                except Exception:
                    pass
                p = mux_sound_into_video(p, wav, sr, outdir, f'{modality}_{stamp}')
        print(f'  [done] {p}  ({time.time() - t0:.1f}s)\n')
        if conditioning_receipt is not None:
            conditioning_receipt.update(fps=self.st['fps'], seed=self.st['seed'], output=str(p))
            Path(str(p) + '.conditioning.json').write_text(json.dumps(conditioning_receipt, indent=2), encoding='utf-8')
        if audio_receipt is not None:
            audio_receipt.update(output=str(p), seed=self.st['seed'])
            Path(str(p) + '.audio-conditioning.json').write_text(json.dumps(audio_receipt, indent=2), encoding='utf-8')
        if lora_prepared:
            Path(str(p) + '.loras.json').write_text(json.dumps(lora_prepared, indent=2), encoding='utf-8')
        return p

    def continue_video(self, prompt, total_frames, chunk_frames=49, overlap=9, image_path=None, on_step=None, should_cancel=None):
        import math as _math
        from diffusers.utils import load_video, export_to_video
        if not self._supports('v2v'):
            print('  [vlong] current model has no v2v modality (needs a Cosmos3 video checkpoint)')
            return None
        overlap = max(1, int(overlap))
        chunk_frames = max(overlap + 1, int(chunk_frames))
        new_per = chunk_frames - overlap
        total_frames = max(chunk_frames, int(total_frames))
        nchunks = max(1, _math.ceil((total_frames - chunk_frames) / new_per) + 1)
        prev_nf = self.st['num_frames']
        self.st['num_frames'] = chunk_frames
        self._user_set.add('num_frames')
        tmpdir = Path(self.st['out']) / '_cont_tmp'
        tmpdir.mkdir(parents=True, exist_ok=True)
        all_frames: list = []
        print(f'  [vlong] {total_frames} frames over {nchunks} chunks (chunk={chunk_frames}, overlap={overlap}, new/chunk={new_per})', flush=True)
        try:
            for i in range(nchunks):
                if i == 0:
                    mod = 'i2v' if image_path else 't2v'
                    cp = self.generate(mod, prompt, image_path=image_path, on_step=on_step, should_cancel=should_cancel)
                else:
                    tail = all_frames[-overlap:]
                    tail_path = str(tmpdir / f'tail_{i}.mp4')
                    export_to_video(tail, tail_path, fps=self.st['fps'], macro_block_size=1)
                    cp = self.generate('v2v', prompt, video_path=tail_path, num_conditional_frames=overlap, on_step=on_step, should_cancel=should_cancel)
                if not cp or not Path(str(cp)).is_file():
                    print(f'  [vlong] chunk {i} produced no output — stopping', flush=True)
                    break
                cf = load_video(str(cp))
                all_frames.extend(cf if i == 0 else cf[overlap:])
                if should_cancel and should_cancel():
                    break
                if len(all_frames) >= total_frames:
                    break
        finally:
            self.st['num_frames'] = prev_nf
        if not all_frames:
            return None
        all_frames = all_frames[:total_frames]
        out = _unique_path(Path(self.st['out']) / f"vlong_{_safe_stamp(prompt, 'cont')}.mp4")
        export_to_video(all_frames, str(out), fps=self.st['fps'], macro_block_size=1)
        print(f'  [vlong] {len(all_frames)} frames -> {out}', flush=True)
        return out

    def _native_action(self, mode, vision, prompt, checkpoint=None):
        runner = REPO / 'cosmos_windows' / 'run_native_action.py'
        cmd = [sys.executable, str(runner), '--mode', mode, '--prompt', prompt or '', '--out', str(Path(self.st['out']) / mode), '--gpu-index', self.st['gpu_index']]
        if vision:
            cmd += ['--vision', vision]
        if checkpoint:
            cmd += ['--checkpoint', checkpoint]
        print(f'  [{mode}] native cosmos_framework (16B, eager SDPA — minutes). Default sample if no --vision.')
        subprocess.run(cmd, cwd=str(REPO))

    def action(self, image_path, prompt):
        self._native_action('forward_dynamics', image_path, prompt)

    def policy(self, image_path, instruction):
        if not self._supports('policy'):
            return
        print('  [policy] DROID: instruction + observation -> robot action trajectory')
        self._native_action('policy', image_path, instruction, checkpoint=self.st['model'])

    def stream(self, prompt):
        eng = REPO / 'cosmos_windows' / 'streaming_engine.py'
        out = Path(self.st['out']) / 'stream.mp4'
        Path(self.st['out']).mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(eng), '--stream-weights', '--weight-backend', 'directstorage', '--model', self.st['model'], '--total-frames', str(max(self.st['num_frames'], 25)), '--chunk', '9', '--height', str(self.st['height']), '--width', str(self.st['width']), '--prompt', prompt, '--out', str(out)]
        print(f"  [stream] DirectStorage weight-streamed video ({self.st['model']}) -> {out}")
        subprocess.run(cmd)

    def loop(self, goal):
        if not goal:
            print('  usage: /loop GOAL')
            return
        sys_prompt = "You are a generation planner for a video/image diffusion model. Turn the user's goal into a single vivid, concrete visual prompt (one paragraph, no preamble, describe subject, scene, lighting, motion). Output ONLY the prompt."
        print('  [loop] 1/2 reasoning the render plan on the Mac...')
        plan = self.reason(goal, None) if False else None
        try:
            res = self._reason_request(goal, system=sys_prompt, max_new=220)
            plan = res.get('answer')
        except Exception as e:
            print(f'  reasoner unreachable ({type(e).__name__}); /sync first.')
            return
        if not plan:
            print('  no plan produced.')
            return
        print(f"  [loop] plan: {plan}\n  [loop] 2/2 rendering on the RTX ({self.st['loop_render']})...")
        self.generate(self.st['loop_render'], plan)

    def _ssh(self, *cmd):
        return ['ssh', '-i', self.st['mac_key'], '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=10', f"{self.st['mac_user']}@{self.st['reasoner_host']}", *cmd]

    def sync(self):
        print('  [sync] shipping reasoner server to the Mac + (re)starting it...')
        scp = ['scp', '-i', self.st['mac_key'], '-o', 'IdentitiesOnly=yes', str(REPO / 'cosmos_windows' / 'mac_reasoner_server.py'), f"{self.st['mac_user']}@{self.st['reasoner_host']}:/tmp/mac_reasoner_server.py"]
        subprocess.run(scp)
        start = f"pkill -f mac_reasoner_server.py 2>/dev/null; sleep 1; cd /tmp; nohup {self.st['mac_python']} -u /tmp/mac_reasoner_server.py --model {self.st['mac_model']} --port {self.st['reasoner_port']} > /tmp/mac_reasoner_server.log 2>&1 & echo started $!"
        subprocess.run(self._ssh(start))
        print('  [sync] waiting for the reasoner to load...')
        for _ in range(40):
            time.sleep(3)
            if self.health():
                print('  [sync] ready.')
                return
        print('  [sync] timed out waiting for health; check /tmp/mac_reasoner_server.log on the Mac.')

    def remote(self, modality, prompt):
        sys.path.insert(0, str(REPO))
        from cosmos_windows.omni_connector_sender import send_generation_request
        nf = self.st['num_frames'] if modality != 't2v' else max(self.st['num_frames'], 17)
        print(f"  [remote] TBP1 -> {self.st['remote_host']}:{self.st['remote_port']} ({modality})")
        send_generation_request(self.st['remote_host'], self.st['remote_port'], prompt=prompt, height=self.st['height'], width=self.st['width'], num_frames=nf, num_inference_steps=self.st['steps'], seed=self.st['seed'])

    def do(self, line):
        line = line.strip()
        if not line:
            return True
        if line[0] == '/':
            line = line[1:]
        try:
            parts = shlex.split(line, posix=False)
        except ValueError:
            parts = line.split()
        parts = [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and (p[0] in '"\'') else p for p in parts]
        if not parts:
            return True
        cmd, rest = (parts[0].lower(), parts[1:])
        raw = line[len(parts[0]):].strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and (raw[0] in '"\''):
            raw = raw[1:-1]
        if cmd in ('quit', 'exit', 'q'):
            return False
        if cmd in ('help', 'h', '?'):
            print(__doc__)
        elif cmd in ('show', 'settings'):
            self.show()
        elif cmd == 'set' and len(rest) >= 2:
            self.set(rest[0], ' '.join(rest[1:]))
        elif cmd == 'perf':
            self.perf(rest[0].lower() if rest else None)
        elif cmd == 'out' and rest:
            self.st['out'] = raw
            print(f'  out = {raw}')
        elif cmd == 'mode' and rest:
            self.st['mode'] = rest[0]
            print(f'  mode = {rest[0]}')
        elif cmd == 'models':
            self.models_list()
        elif cmd == 'model' and rest:
            self.select_model(rest[0].lower())
        elif cmd == 'policy':
            self.policy(rest[0] if rest else None, ' '.join(rest[1:]))
        elif cmd == 'stream':
            self.stream(raw)
        elif cmd == 'upscale' and rest:
            self.upscale(rest)
        elif cmd == 'health':
            self.health()
        elif cmd == 'sync':
            self.sync()
        elif cmd in ('reason', 'ask', 'r'):
            img = None
            toks = []
            for t in rest:
                if t.startswith('image='):
                    img = t.split('=', 1)[1]
                else:
                    toks.append(t)
            self.reason(' '.join(toks), img)
        elif cmd in ('preason', 'pr'):
            self.preason(raw)
        elif cmd in ('t2i', 't2v', 'av', 'sound'):
            self.generate('av' if cmd == 'sound' else cmd, raw)
        elif cmd == 'i2v':
            if len(rest) < 2:
                print('  usage: /i2v IMG_PATH PROMPT')
            else:
                self.generate('i2v', ' '.join(rest[1:]), image_path=rest[0])
        elif cmd == 'v2v':
            if len(rest) < 2:
                print('  usage: /v2v VIDEO_PATH PROMPT [cond=N]   (N = #leading frames to pin)')
            else:
                ncf, toks = (None, [])
                for t in rest[1:]:
                    if t.startswith('cond='):
                        try:
                            ncf = int(t.split('=', 1)[1])
                        except ValueError:
                            pass
                    else:
                        toks.append(t)
                self.generate('v2v', ' '.join(toks), video_path=rest[0], num_conditional_frames=ncf)
        elif cmd == 'vlong':
            total, chunk, ov, img, toks = (97, 49, 9, None, [])
            for t in rest:
                if t.startswith('frames='):
                    total = int(t.split('=', 1)[1] or total)
                elif t.startswith('chunk='):
                    chunk = int(t.split('=', 1)[1] or chunk)
                elif t.startswith('overlap='):
                    ov = int(t.split('=', 1)[1] or ov)
                elif t.startswith('img='):
                    img = t.split('=', 1)[1]
                else:
                    toks.append(t)
            if not toks:
                print('  usage: /vlong PROMPT [frames=N] [chunk=C] [overlap=O] [img=PATH]')
            else:
                self.continue_video(' '.join(toks), total_frames=total, chunk_frames=chunk, overlap=ov, image_path=img)
        elif cmd == 'action':
            self.action(rest[0] if rest else None, ' '.join(rest[1:]))
        elif cmd == 'loop':
            self.loop(raw)
        elif cmd == 'remote' and len(rest) >= 2:
            self.remote(rest[0], ' '.join(rest[1:]))
        else:
            self._run_mode(line)
        return True

    @staticmethod
    def _est_tokens(text):
        return int(len(text) / 3.5) + 24

    def _run_mode(self, text):
        m = self.st['mode']
        if m in ('reason', 'ask'):
            est = self._est_tokens(text)
            if self.st['auto_prefill'] == 'on' and self.st['prefill_min_tokens'] <= est <= self.st['prefill_max_tokens']:
                print(f'  (~{est} tok -> RTX prefill path; /set auto_prefill off to force Mac-only)')
                self.preason(text)
            else:
                self.reason(text)
        elif m in ('t2i', 't2v', 'av'):
            self.generate(m, text)
        elif m == 'loop':
            self.loop(text)
        else:
            print(f'  (mode {m!r}: type a /command, or /set mode reason|t2i|t2v|av|loop)')

def main():
    ap = argparse.ArgumentParser(description='Cosmos3 unified CLI', add_help=True)
    ap.add_argument('command', nargs='*', help='run one command then exit (omit for interactive REPL)')
    args = ap.parse_args()
    st = dict(DEFAULTS)
    cli = Cli(st)
    if args.command:
        cli.do(' '.join(args.command))
        return
    print('Cosmos3 CLI — /help for commands, /quit to exit. reasoner->Mac, diffusion->RTX.')
    print(f"  reasoner @ {st['reasoner_host']}:{st['reasoner_port']}  |  mode={st['mode']}  |  out={st['out']}")
    while True:
        try:
            line = input('cosmos> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cli.do(line):
            break
if __name__ == '__main__':
    main()
