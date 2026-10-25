# Derived from Studio backend/studio/audio.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; machine-specific fallback paths cleared, otherwise unchanged.
"""Daemon-side audio manager (Phase 1: ASR + TTS + voice clone).

The heavy ML engines (faster-whisper / chatterbox) live in a DEDICATED venv (``STUDIO_AUDIO_PYTHON``,
default dev audio venv); this module — imported into the lightweight daemon — only orchestrates
out-of-process calls into that venv (mirrors the Hunyuan/GLM/PiD separate-venv subprocess pattern)
and owns the voice-profile store + the fail-soft availability probe. The idle daemon never imports
torch/CTranslate2.

Everything FAILS SOFT: if the venv/worker is missing, ``available()`` is False and the endpoints
return a structured 503 the GUI uses to hide the mic/TTS controls — the rest of the app is unaffected.
"""
from __future__ import annotations

import base64
import asyncio
import json
import os
import sys
import subprocess
from . import worker_registry
import threading
import time
import uuid
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Optional

from . import paths as _paths
from .config import settings


# --------------------------------------------------------------------------------------------------
# Paths / config (env override -> setting -> installer default -> dev discovery; resolver precedence).
# --------------------------------------------------------------------------------------------------
def _dev_audio_candidates(mac_venv: str, win: str = "",
                          include_repo_venv: bool = True) -> list[str]:
    """Dev-mode python candidates for an audio venv, per OS. Windows keeps the legacy dev venv;
    macOS looks in the app-support tree. Only the MAIN audio venv (``include_repo_venv``) also falls
    back to the repo's ``.venv312`` — so dictation/transcription work in a dev checkout where
    faster-whisper is installed into the main venv, while the HQ venv stays a DISTINCT path (the
    moss/dots engines need their own transformers>=5 venv, never the shared repo venv)."""
    if sys.platform == "darwin":
        cands = [os.path.expanduser(f"~/Library/Application Support/front-eagle-stack/{mac_venv}/bin/python")]
        if include_repo_venv:
            cands.append(str(Path(__file__).resolve().parents[2] / ".venv312" / "bin" / "python"))
        return cands
    return [win]


def audio_python() -> str:
    """Path to the dedicated audio venv's python (faster-whisper / chatterbox engines).

    Resolved via the canonical precedence: ``STUDIO_AUDIO_PYTHON`` env > installer-placeable
    ``<runtime_root>/audio-python`` (packaged; ``installer_python`` tolerates both the venv and
    python-build-standalone layouts the packager can produce) > dev audio venv (last)."""
    return str(_paths.resolve(
        env="STUDIO_AUDIO_PYTHON",
        installer_default=_paths.installer_python("audio-python"),
        dev_candidates=_dev_audio_candidates("audio-venv"),
        want="file",
    ))


def hq_audio_python() -> str:
    """Path to the HIGH-QUALITY audio venv's python (dots.tts + MOSS-TTS — transformers>=5 / torch>=2.8).

    These narration engines need a newer torch/transformers than the chatterbox/whisper venv tolerates,
    so they live in their own venv. Precedence: ``STUDIO_AUDIO_HQ_PYTHON`` env > installer-placeable
    ``<runtime_root>/hq-audio-python`` > dev moss venv (last). Falls back to the main audio
    python when no HQ venv exists, so a single-venv install still resolves (dots/moss just probe absent)."""
    try:
        return str(_paths.resolve(
            env="STUDIO_AUDIO_HQ_PYTHON",
            installer_default=_paths.installer_python("hq-audio-python"),
            dev_candidates=_dev_audio_candidates("moss-venv", win="", include_repo_venv=False),
            want="file",
        ))
    except Exception:  # noqa: BLE001  (no HQ venv placed — degrade to the main audio venv)
        return audio_python()


# Engines that must run in the HQ-audio venv (newer torch/transformers); the rest use the main venv.
_HQ_ENGINES = ("dots", "moss")
# Approx transient HOST RAM each HQ engine needs DURING load (measured ~7GB dots / ~18GB MOSS, + headroom).
# Used by the pre-flight fit check so an HQ engine that can't fit free RAM fails with a clear message
# instead of OOM-ing the whole box (user 2026-06-15: "make sure the audio model doesnt immediately try to
# eat all the ram"). Override the guard with STUDIO_AUDIO_SKIP_FIT_CHECK=1.
_HQ_HOST_RAM_GB = {"dots": 8.0, "moss": 19.0}


def _check_hq_fits(engine: Optional[str]) -> None:
    """Refuse to spawn an HQ TTS worker that won't fit in free host RAM (prevents OOMing the box). No-op
    for the light engines, when psutil is unavailable, or when STUDIO_AUDIO_SKIP_FIT_CHECK=1."""
    if engine not in _HQ_ENGINES or os.environ.get("STUDIO_AUDIO_SKIP_FIT_CHECK") == "1":
        return
    need = _HQ_HOST_RAM_GB.get(engine, 8.0)
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:  # noqa: BLE001 — can't measure -> don't block
        return
    if avail < need:
        raise ValueError(
            f"the '{engine}' TTS engine needs ~{need:.0f}GB free RAM to load but only {avail:.1f}GB is "
            f"free. Assign it a GPU with enough memory (Settings -> GPU assignment), use a lighter engine "
            f"(chatterbox/kokoro), free RAM, or set STUDIO_AUDIO_SKIP_FIT_CHECK=1 to override.")


def _python_for_engine(engine: Optional[str]) -> str:
    return hq_audio_python() if engine in _HQ_ENGINES else audio_python()


def _worker_path() -> str:
    return str(Path(__file__).resolve().parent / "audio_worker.py")


def audio_dir() -> Path:
    """App-data dir for voice profiles + temp blobs. Env override else under the output root."""
    d = os.environ.get("STUDIO_AUDIO_DIR")
    base = Path(d) if d else (settings.output_dir.parent / "audio")
    base.mkdir(parents=True, exist_ok=True)
    return base


def voices_dir() -> Path:
    d = audio_dir() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _voices_index() -> Path:
    return voices_dir() / "voices.json"


def _tmp_dir() -> Path:
    d = audio_dir() / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Built-in voices (no clone reference needed). Chatterbox default + the kokoro fast voice.
_BUILTIN_VOICES = [
    {"voice_id": "default", "name": "Default (Chatterbox)", "builtin": True, "engine": "chatterbox"},
    {"voice_id": "fast", "name": "Fast (Kokoro)", "builtin": True, "engine": "kokoro"},
]

# The selectable worker TTS engines (mirrors audio_worker._TTS_ENGINES — chatterbox is the
# low-latency default, kokoro the fastest built-in voice, dots the high-quality video-speech one).
_TTS_ENGINE_NAMES = ("chatterbox", "kokoro", "dots", "moss")


# Env vars that, if inherited from the daemon parent, point the CHILD interpreter at the WRONG
# Python install. The daemon may run under uv's cpython-3.13 (PYTHONHOME/PYTHONPATH/VIRTUAL_ENV set
# to that tree) while the audio venv is a stock 3.12 (dev audio venv). With PYTHONHOME inherited, the
# 3.12 worker imports the 3.13 stdlib `re`/`_sre` and dies at `import re` with an `_sre.MAGIC`
# mismatch ("SRE module mismatch") — the exact truncated-traceback crash GET /audio/status reported.
# Scrubbing these forces the child to use its own venv's stdlib (resolved from its pyvenv.cfg home),
# which fixes the worker regardless of how the venv was built. Validated by behaviour diff
# (contaminated parent crashes at import re; scrubbed env imports the venv's own 3.12 re).
_PY_ISOLATION_VARS = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONEXECUTABLE")


def _assigned_gpu_index() -> Optional[str]:
    """The user's persisted GPU assignment for the audio service (nvidia-smi PCI-bus index), or None.
    Best-effort: a missing/broken store never blocks audio (falls back to the worker's default 3070)."""
    try:
        from . import settings_store
        return settings_store.gpu_for("audio")
    except Exception:  # noqa: BLE001
        return None


def _worker_env(engine: Optional[str] = None) -> dict:
    env = dict(os.environ)
    for k in _PY_ISOLATION_VARS:
        env.pop(k, None)
    # The HQ engines (dots 2B / MOSS 8B) legitimately use far more host RAM during load than the
    # tiny chatterbox/whisper (measured dots ~7GB, MOSS ~18GB). The default 6GB RSS guard would
    # abort them, so raise it for the HQ venv (the box is 32GB; 24GB leaves headroom). Honor an
    # explicit user override.
    if engine in _HQ_ENGINES and "STUDIO_AUDIO_RSS_LIMIT_GB" not in os.environ:
        env["STUDIO_AUDIO_RSS_LIMIT_GB"] = os.environ.get("STUDIO_AUDIO_HQ_RSS_LIMIT_GB", "24")
    # carry the audio-specific knobs through; HF cache pin if configured
    hf_home = os.environ.get("STUDIO_AUDIO_HF_HOME")
    if hf_home:
        env["HF_HOME"] = hf_home
    # The worker runs with the Hugging Face hub offline, so a bare model size ("large-v3-turbo") can
    # only load from a cache. Use the installed local faster-whisper folder, the same one world voice
    # and the mic monitors use, unless the owner configured a model explicitly.
    if "STUDIO_AUDIO_ASR_MODEL" not in os.environ:
        from .world_voice import _asr_model_default
        local_asr = _asr_model_default()
        if local_asr:
            env["STUDIO_AUDIO_ASR_MODEL"] = local_asr
    # Per-service GPU assignment (user 2026-06-09): pin the audio worker to the user's chosen card.
    # The worker reads STUDIO_AUDIO_GPU_INDEX and applies CUDA_DEVICE_ORDER=PCI_BUS_ID +
    # CUDA_VISIBLE_DEVICES before importing torch/CTranslate2, so the index is an nvidia-smi PCI-bus
    # index. An explicit environment override wins over the saved assignment.
    if "STUDIO_AUDIO_GPU_INDEX" not in os.environ:
        idx = _assigned_gpu_index()
        if idx is not None:
            env["STUDIO_AUDIO_GPU_INDEX"] = idx
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _worker_cwd() -> str:
    """A sane, always-existing cwd for the worker subprocess (never the daemon's package dir, which
    can shadow stdlib names on sys.path[0]). Use the audio app-data dir."""
    try:
        return str(audio_dir())
    except OSError:
        return str(Path(__file__).resolve().parent)


# --------------------------------------------------------------------------------------------------
# Availability (fail-soft probe).
# --------------------------------------------------------------------------------------------------
_avail_cache: dict = {"ts": 0.0, "value": None}
_avail_lock = threading.Lock()


def available() -> bool:
    return bool(status().get("available"))


def _gemma_stt_model() -> Optional[str]:
    """Path to an audio-capable gemma model used for LLM-based ASR, from STUDIO_STT_GEMMA_MODEL.
    Returns the path only if set AND it exists on disk."""
    p = (os.environ.get("STUDIO_STT_GEMMA_MODEL") or "").strip()
    return p if p and Path(os.path.expanduser(p)).is_file() else None


def stt_engines() -> list[str]:
    """Selectable speech-to-text engines (user 2026-06-15: "audio-capable models like gemma 2/4/12b
    selectable STT options"). 'whisper' = faster-whisper (default); 'gemma-audio' is offered when an
    audio-capable gemma model is configured via STUDIO_STT_GEMMA_MODEL. Always returns at least whisper."""
    engines: list[str] = []
    try:
        sp = Path(audio_python()).resolve().parent.parent / "Lib" / "site-packages"
        if (sp / "faster_whisper").is_dir():
            engines.append("whisper")
    except Exception:  # noqa: BLE001
        pass
    from . import gemma_audio_stt
    if gemma_audio_stt.readiness()['available']:
        engines.append("gemma-audio")
    return engines or ["whisper"]


def status(force: bool = False) -> dict:
    """Probe whether the audio venv + worker are usable. Cached for 30 s so the GUI can poll cheaply.

    Returns {available, python, worker, asr, tts, detail}. ``available`` is True only when the venv
    python exists AND the worker selftest imports faster-whisper (ASR is the must-have)."""
    with _avail_lock:
        now = time.time()
        if not force and _avail_cache["value"] is not None and (now - _avail_cache["ts"]) < 30:
            return dict(_avail_cache["value"])
        py = audio_python()
        worker = _worker_path()
        from . import gemma_audio_stt
        gemma = gemma_audio_stt.readiness()
        result = {"available": False, "python": py, "worker": worker,
                  "asr": False, "tts": False, "detail": "",
                  'stt_engine_details': {'gemma-audio': gemma},
                  'stt_engines': stt_engines()}
        if not Path(py).is_file():
            if gemma['available']:
                result.update(available=True,asr=True,detail=gemma['detail'])
                _avail_cache.update(ts=now,value=result)
                return dict(result)
            result["detail"] = (f"audio venv python not found at {py}; create it per "
                                f"docs/AUDIO_FEATURES_BOOTSTRAP.md or set STUDIO_AUDIO_PYTHON")
            _avail_cache.update(ts=now, value=result)
            return dict(result)
        # FAST availability: just check the package FILES are present in the venv(s). NO slow
        # model-importing subprocess (user 2026-06-15: "a check that the files are there should be
        # fine; it can pop up an error if it fails on import"). Heavy import / model-load failures
        # surface at synthesize()/transcribe() time, so this status never times out or flakes.
        def _present(venv_py: str, import_name: str) -> bool:
            """Is ``import_name`` importable from ``venv_py``'s site-packages? Handles a normal package
            dir, a single-module ``.py``, a wheel/``.dist-info`` install, AND an EDITABLE install
            (``pip install -e``) — which leaves NO dir in site-packages, only a ``.dist-info`` /
            ``.egg-link`` / ``__editable__*.pth`` reference. The old ``is_dir()`` check reported editable
            installs (e.g. MOSS-TTS) as ABSENT even though ``import`` works, so the GUI hid them
            (integration audit 2026-06-16)."""
            try:
                sp = Path(venv_py).resolve().parent.parent / "Lib" / "site-packages"
                if not sp.is_dir():
                    return False
                if (sp / import_name).is_dir() or (sp / f"{import_name}.py").is_file():
                    return True
                norm = import_name.replace("_", "-").lower()
                for di in sp.glob("*.dist-info"):       # wheel OR editable both leave <Name>-<ver>.dist-info
                    proj = di.name[: -len(".dist-info")].rsplit("-", 1)[0]
                    if proj.replace("_", "-").lower() == norm:
                        return True
                if (sp / f"{import_name}.egg-link").is_file():
                    return True
                for pth in sp.glob("*.pth"):            # __editable__.<name>-<ver>.pth or a legacy .pth
                    stem = pth.stem.lower()
                    if stem.startswith("__editable__") and norm in stem.replace("_", "-"):
                        return True
                    try:
                        if import_name in pth.read_text(encoding="utf-8", errors="ignore"):
                            return True
                    except OSError:
                        pass
                return False
            except Exception:  # noqa: BLE001
                return False
        hq_py = hq_audio_python()
        result["asr"] = _present(py, "faster_whisper") or gemma['available']
        result["tts"] = _present(py, "chatterbox")
        result["engines"] = {
            "chatterbox": _present(py, "chatterbox"),
            "kokoro": _present(py, "kokoro"),
            # dots/moss may live in the HQ-audio venv (if provisioned) or the main venv
            "dots": _present(py, "dots_tts") or _present(hq_py, "dots_tts"),
            "moss": (_present(py, "moss_tts") or _present(hq_py, "moss_tts")
                     or _present(py, "moss") or _present(hq_py, "moss")),
        }
        result["available"] = result["asr"]
        result["stt_engines"] = stt_engines()  # selectable ASR engines (whisper [+ gemma-audio if configured])
        if not result["available"]:
            result["detail"] = (f"faster-whisper not installed in {py} - see "
                                f"docs/AUDIO_FEATURES_BOOTSTRAP.md (pip install faster-whisper).")
        else:
            ready = ",".join(k for k in _TTS_ENGINE_NAMES if result["engines"].get(k))
            result["detail"] = (f"ready (engine files present): {ready or 'ASR only'} "
                                f"- models load on first use")
        result["device"] = None  # not probed (no model load); set at first synthesize/transcribe
        _avail_cache.update(ts=now, value=result)
        return dict(result)


class AudioUnavailable(RuntimeError):
    """Raised when the audio venv/worker is absent — the daemon maps it to a 503."""


def _require_available() -> None:
    st = status()
    if not st.get("available"):
        raise AudioUnavailable(st.get("detail") or "audio features unavailable")


# --------------------------------------------------------------------------------------------------
# Subprocess helpers.
# --------------------------------------------------------------------------------------------------
def _no_window() -> dict:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


def _parse_result(stdout: str) -> Optional[dict]:
    """Last ``RESULT <json>`` line wins; an ``ERROR <json>`` raises."""
    err = None
    last = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("RESULT "):
            try:
                last = json.loads(line[len("RESULT "):])
            except ValueError:
                continue
        elif line.startswith("ERROR "):
            try:
                err = json.loads(line[len("ERROR "):])
            except ValueError:
                err = {"error": line[len("ERROR "):]}
    if err is not None and last is None:
        raise RuntimeError(err.get("error") or "audio worker error")
    return last


_RUN_LOCK = threading.Lock()  # 3070 is small; serialize one-shot worker invocations


def _spawn_tts_session(key):
    py, worker, _engine, env_items = key
    return worker_registry.popen(
        [py, worker, "tts-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
        env=dict(env_items), cwd=_worker_cwd(), **_no_window())


from .audio_tts_session import TtsSession
_TTS_SESSION = TtsSession(_spawn_tts_session)
_TTS_SLOT_LOCKS = {}
_TTS_BORROWERS = {}


def _slot_lock(gpu):
    return _TTS_SLOT_LOCKS.setdefault(str(gpu), threading.RLock())


# VC1: other resident speech workers (world voice) that also make way for a workload on their GPU.
GPU_EVICTORS: list = []


def _evict_tts_gpu(gpu):
    key = _TTS_SESSION.key
    if key is not None and str(gpu) == _tts_gpu(dict(key[3])):
        _TTS_SESSION.close()
    for evict in list(GPU_EVICTORS):
        try:
            evict(str(gpu))
        except Exception:  # noqa: BLE001 -- an evictor failing must never block the workload
            pass


def release_for_gpu(gpu):
    """Release resident speech before another service performs its fit check."""
    with _slot_lock(gpu):
        _evict_tts_gpu(gpu)


def _tts_gpu(env=None):
    env = _worker_env() if env is None else env
    visible = env.get('CUDA_VISIBLE_DEVICES')
    return str(visible if visible is not None else env.get('STUDIO_AUDIO_GPU_INDEX', '0')).split(',')[0]


def _busy_tts_gpu(gpu):
    owners = _TTS_BORROWERS.setdefault(str(gpu), {})
    for key, process in list(owners.items()):
        if process is not None and process.poll() is not None:
            owners.pop(key)
    return bool(owners)


@contextmanager
def gpu_workload(gpu):
    """Evict speech before an in-process workload takes its assigned GPU."""
    gpu = str(gpu)
    token = object()
    with _slot_lock(gpu):
        _evict_tts_gpu(gpu)
        _TTS_BORROWERS.setdefault(gpu, {})[token] = None
    try:
        yield
    finally:
        with _slot_lock(gpu):
            _TTS_BORROWERS[gpu].pop(token, None)


def spawn_gpu_worker(spawn, argv, env):
    """Managed GPU children evict speech on the same device before allocation."""
    if env.get('STUDIO_GPU_SLOT_OWNER') == 'audio-tts':
        return spawn()
    gpu = env.get('CUDA_VISIBLE_DEVICES')
    argv = list(argv) if isinstance(argv, (list, tuple)) else []
    for flag in ('--gpu-index', '--gpu_index', '--gpu', '--main-gpu'):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            gpu = str(argv[argv.index(flag) + 1])
            break
    if gpu is None:
        gpu = str(getattr(settings, 'gpu_index', '0'))
    devices = str(gpu).split(',')
    with ExitStack() as locks:
        for device in sorted(set(devices)):
            locks.enter_context(_slot_lock(device))
            _evict_tts_gpu(device)
            _busy_tts_gpu(device)  # prune completed children
        process = spawn()
        for device in devices:
            _TTS_BORROWERS.setdefault(device, {})[id(process)] = process
        return process


def _run_worker(args: list[str], timeout: float = 300, python: Optional[str] = None,
                engine: Optional[str] = None) -> dict:
    py, worker = (python or audio_python()), _worker_path()
    with _RUN_LOCK:
        # HQ engines retain their one-shot memory contract. The small speech
        # engines stay resident until another workload needs their GPU slot.
        if args and args[0] == "tts" and engine not in _HQ_ENGINES:
            env = _worker_env(engine)
            env['STUDIO_GPU_SLOT_OWNER'] = 'audio-tts'
            key = (py, worker, engine, tuple(sorted(env.items())))
            with _slot_lock(_tts_gpu(env)):
                if _busy_tts_gpu(_tts_gpu(env)):
                    raise AudioUnavailable('The speech GPU is in use by another workload. Speech will reload when its slot is free.')
                return _TTS_SESSION.call(key, args, timeout)
        _TTS_SESSION.close()
        proc = worker_registry.run(
            [py, worker, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=_worker_env(engine),
            cwd=_worker_cwd(), **_no_window(),
        )
    payload = _parse_result(proc.stdout)
    if payload is None:
        raise RuntimeError((proc.stderr or proc.stdout or f"audio worker exited {proc.returncode}").strip()[:800])
    return payload


# --------------------------------------------------------------------------------------------------
# Voice-profile store.
# --------------------------------------------------------------------------------------------------
def _load_voice_index() -> list[dict]:
    p = _voices_index()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
    return []


def _save_voice_index(items: list[dict]) -> None:
    _voices_index().write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_voices() -> list[dict]:
    """Built-in voices + saved clone profiles (newest clone first)."""
    clones = sorted(_load_voice_index(), key=lambda v: v.get("created", 0), reverse=True)
    return list(_BUILTIN_VOICES) + clones


def get_voice(voice_id: str) -> Optional[dict]:
    if voice_id in ("default", "fast"):
        return next((v for v in _BUILTIN_VOICES if v["voice_id"] == voice_id), None)
    return next((v for v in _load_voice_index() if v["voice_id"] == voice_id), None)


def save_voice(name: str, wav_bytes: bytes) -> dict:
    """Store a reference clip as a cloneable voice profile. The bytes can be any format the audio
    worker / ffmpeg reads; we keep the original extension-less and let Chatterbox decode it."""
    vid = uuid.uuid4().hex[:12]
    safe = "".join(c for c in (name or "voice") if c.isalnum() or c in "-_ ").strip()[:48] or "voice"
    ref = voices_dir() / f"{vid}.wav"
    ref.write_bytes(wav_bytes)
    entry = {"voice_id": vid, "name": safe, "builtin": False, "ref": str(ref),
             "created": time.time(), "engine": "chatterbox"}
    items = _load_voice_index()
    items.append(entry)
    _save_voice_index(items)
    return {k: entry[k] for k in ("voice_id", "name", "builtin")}


def delete_voice(voice_id: str) -> bool:
    items = _load_voice_index()
    kept = [v for v in items if v["voice_id"] != voice_id]
    if len(kept) == len(items):
        return False
    dropped = next((v for v in items if v["voice_id"] == voice_id), None)
    if dropped and dropped.get("ref"):
        try:
            Path(dropped["ref"]).unlink(missing_ok=True)
        except OSError:
            pass
    _save_voice_index(kept)
    return True


# --------------------------------------------------------------------------------------------------
# One-shot operations (called by the daemon endpoints).
# --------------------------------------------------------------------------------------------------
def transcribe_bytes(blob: bytes, suffix: str = ".webm", language: Optional[str] = None,
                     stt_engine: Optional[str] = None, cancelled=None) -> dict:
    """Persist the uploaded blob and transcribe it. faster-whisper reads compressed audio via its
    bundled decoder, so we pass the file straight through (no ffmpeg pre-pass needed). ``stt_engine``
    selects the ASR engine: 'whisper' (default) or 'gemma-audio' (an audio-capable gemma LLM, when
    STUDIO_STT_GEMMA_MODEL is configured)."""
    engine = (stt_engine or "whisper").strip().lower()
    if engine not in ("whisper", "gemma-audio"):
        raise ValueError(f"unknown STT engine '{engine}'; available: {stt_engines()}")
    if engine == 'gemma-audio':
        from . import gemma_audio_stt
        ready = gemma_audio_stt.readiness()
        if not ready['available']:
            raise ValueError(ready['detail'])
        if not blob or len(blob)>gemma_audio_stt.MAX_BYTES:
            raise ValueError('Audio must contain data and be no larger than 64 MiB')
        if suffix.lower() not in ('.wav','.webm','.mp3','.mp4','.m4a','.ogg','.flac','.aiff'):
            raise ValueError('Unsupported audio format')
    else:
        _require_available()
    tmp = _tmp_dir() / f"asr_{uuid.uuid4().hex[:10]}{suffix}"
    tmp.write_bytes(blob)
    try:
        args = ["transcribe", "--audio", str(tmp)]
        if language:
            args += ["--language", language]
        if engine == "gemma-audio":
            def spawn(argv, **options):
                environment = {**options.pop('env'), **_worker_env()}
                # Retain the native runner's offline policy and selected audio device.
                environment.update(LLAMA_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                for key in ('LLAMA_ARG_HF_REPO', 'LLAMA_ARG_MODEL_URL', 'LLAMA_ARG_MMPROJ_URL', 'MTMD_TEST_RESPONSE_MARKER'):
                    environment.pop(key, None)
                index = str(environment.get('STUDIO_AUDIO_GPU_INDEX','0')).lower()
                environment['CUDA_VISIBLE_DEVICES'] = '' if index in ('cpu','-1','none') else index
                return worker_registry.popen(argv,env=environment,**options)
            return gemma_audio_stt.transcribe_path(tmp,ready['model'],language,cancelled=cancelled,spawn=spawn)
        return _run_worker(args, timeout=300)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def synthesize(text: str, voice_id: str = "default", engine: Optional[str] = None) -> dict:
    """TTS ``text`` with ``voice_id`` (built-in or a saved clone) -> {path, sr, duration, engine, bytes_b64}.

    ``engine`` optionally picks the worker engine ("chatterbox" | "kokoro" | "dots"); None keeps the
    worker's env default (STUDIO_AUDIO_TTS_ENGINE, chatterbox). The 'fast' built-in voice implies
    kokoro (an explicit ``engine`` wins). Unknown engines raise ValueError (the daemon's 400)."""
    _require_available()
    voice = get_voice(voice_id) or {"voice_id": "default"}
    if voice.get('ref') and not voice.get('builtin') and engine in (None, '', 'kokoro'):
        # A saved clone must never silently become a generic built-in voice.
        # The profile records which engine can reproduce its reference speech.
        engine = voice.get('engine') or 'chatterbox'
    if engine is not None:
        engine = str(engine).strip().lower()
        if engine not in _TTS_ENGINE_NAMES:
            raise ValueError(f"unknown tts engine '{engine}' (expected one of {list(_TTS_ENGINE_NAMES)})")
    elif voice.get("voice_id") == "fast":
        engine = "kokoro"  # the built-in fast voice routes to kokoro (per-call flag, no env juggling)
    _check_hq_fits(engine)  # fail clearly if an HQ engine won't fit free RAM, rather than OOM the box
    out = _tmp_dir() / f"tts_{uuid.uuid4().hex[:10]}.wav"
    text_file = _tmp_dir() / f"tts_{uuid.uuid4().hex[:10]}.txt"
    text_file.write_text(text or "", encoding="utf-8")
    args = ["tts", "--text-file", str(text_file), "--out", str(out)]
    if engine:
        args += ["--engine", engine]
    if not voice.get("builtin") and voice.get("ref"):
        args += ["--voice-wav", voice["ref"]]
    try:
        # dots/moss run in the HQ-audio venv (newer torch/transformers); the rest in the main venv.
        res = _run_worker(args, timeout=600 if engine in _HQ_ENGINES else 300,
                          python=_python_for_engine(engine), engine=engine)
        path = res.get("path") or str(out)
        res["bytes_b64"] = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        res["voice_id"] = voice.get("voice_id", "default")
        return res
    finally:
        try:
            text_file.unlink(missing_ok=True)
        except OSError:
            pass


async def transcribe_gemma_async(blob, suffix, language=None, *, disconnected=None):
    """Cancellation is acknowledged only after the owned native decoder is reaped."""
    from .gemma_audio_stt import TranscriptionCancelled
    stopped = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(transcribe_bytes,blob,suffix,language,'gemma-audio',stopped.is_set))
    try:
        while not worker.done():
            if disconnected and await disconnected():
                raise asyncio.CancelledError('Transcription client disconnected')
            await asyncio.sleep(.05)
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stopped.set()
        try:
            await asyncio.shield(worker)
        except (TranscriptionCancelled,Exception):
            pass
        raise


# --------------------------------------------------------------------------------------------------
# Streaming session (the WS /audio/stream backing object — the VAD / barge-in primitive).
# --------------------------------------------------------------------------------------------------
class StreamSession:
    """Wraps a persistent ``audio_worker stream`` subprocess. The WS endpoint feeds base64 PCM
    chunks via ``send_chunk`` and drains worker events via ``read_event`` (blocking, called in a
    threadpool). VAD events (``speech_start``) are what the chat page uses for barge-in."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> dict:
        _require_available()
        py, worker = audio_python(), _worker_path()
        self.proc = worker_registry.popen(
            [py, worker, "stream"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=_worker_env(),
            cwd=_worker_cwd(), **_no_window(),
        )
        # first line is the ready/error event
        return self.read_event() or {"type": "error", "error": "stream worker did not start"}

    def send_chunk(self, pcm16_b64: str, sr: int = 16000) -> None:
        if not self.proc or self.proc.stdin is None:
            return
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps({"type": "chunk", "pcm16": pcm16_b64, "sr": sr}) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def send_control(self, ctrl: str) -> None:
        if not self.proc or self.proc.stdin is None:
            return
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps({"type": ctrl}) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def read_event(self) -> Optional[dict]:
        """Blocking read of the next worker event line. Returns None at EOF. Call in a threadpool."""
        if not self.proc or self.proc.stdout is None:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except ValueError:
                return None
        if line.startswith("ERROR "):
            try:
                return {"type": "error", **json.loads(line[len("ERROR "):])}
            except ValueError:
                return {"type": "error", "error": line[len("ERROR "):]}
        return None

    def close(self) -> None:
        if not self.proc:
            return
        try:
            self.send_control("close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
