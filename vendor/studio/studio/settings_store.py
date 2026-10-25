# Derived from Studio backend/studio/settings_store.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""Persisted engine defaults (the §4.4 differentiator knobs + §4.0 Settings screen).

A small JSON-backed store of the engine defaults the GUI exposes (sampler/sigmas + the
weight-streaming / cfg-batch / memeff / vae-tile toggles + gpu index). These seed new
generations and are editable in the Settings screen. The path roots come from the
env-driven config.Settings (shown read-only — changing them affects /file confinement and
needs a restart).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import configure_model_storage_env, settings

# Curated, editable defaults (mirrors cosmos_cli DEFAULTS for the knobs the GUI surfaces).
DEFAULT_ENGINE_SETTINGS: dict = {
    "sampler": "unipc",
    "sigmas": "karras",
    "weight_stream": True,
    "weight_backend": "directstorage",
    "prefetch": True,
    "cfg_batch": False,
    "memeff_attn": True,
    "vae_tile": "auto",
    "gpu_index": os.environ.get("STUDIO_GPU_INDEX", "0"),  # explicit PCI-bus ordinal
}

_lock = threading.Lock()


def _path() -> Path:
    return settings.state_dir / "studio_settings.json"


def _token_path() -> Path:
    return settings.state_dir / "studio_pair_token.txt"


def load_token():
    """The persisted LAN pairing token (set in Settings). Separate file so engine-settings saves
    never clobber it. Returns None when no saved preference exists; returns "" when the user
    explicitly saved blank to disable token auth."""
    p = _token_path()
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return None


def save_token(token) -> None:
    """Persist a pairing token. Refuses blank, short or the retired public default (ValueError):
    turning the token layer off is no longer possible."""
    from .config import pair_token_problem

    tok = (token or "").strip()
    problem = pair_token_problem(tok)
    if problem:
        raise ValueError(problem)
    with _lock:
        p = _token_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tok, encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass


def ensure_pair_token() -> str:
    """This install's pairing token from the shared state dir, generated on first use."""
    from .config import load_or_create_install_pair_token

    with _lock:
        return load_or_create_install_pair_token(settings.state_dir)


# --------------------------------------------------------------------------------------------------
# Runtime remote-access toggle (user 2026-08-04: "remote access should be enabled/disabled from the
# GUI at runtime — never a separate startup script / relaunch"). The daemon binds 0.0.0.0 so a phone
# on the LAN CAN reach it, but the auth middleware 403-rejects every non-local caller until this
# persisted flag is true. Default FALSE => locked to localhost, byte-for-byte the old behaviour.
# Stored in its OWN file (like the token / gpu-assignment stores) so an engine-settings save never
# clobbers it; the middleware reads it live so a toggle takes effect on the very next request.
# --------------------------------------------------------------------------------------------------
def _remote_enabled_file() -> Path:
    # anchored next to the other stores (output_dir.parent) so tests/env stay isolated. NOTE: distinct
    # from studio_remote_access.json (remote_auth's legacy PASSWORD record) — different concern.
    return settings.state_dir / "studio_remote_enabled.json"


def load_remote_access_enabled() -> bool:
    """True iff the user has explicitly enabled LAN remote access at runtime. Default False (locked)."""
    p = _remote_enabled_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return bool(d.get("remote_access_enabled", False))
        except (ValueError, OSError):
            return False
    return False


def save_remote_access_enabled(enabled: bool) -> bool:
    """Persist the runtime remote-access flag. Returns the stored boolean."""
    val = bool(enabled)
    with _lock:
        p = _remote_enabled_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"remote_access_enabled": val}, indent=2), encoding="utf-8")
        from . import settings_history  # scoped commit in the private settings history
        settings_history.after_write("remote-enabled")
    return val


# --------------------------------------------------------------------------------------------------
# H3 Prompt Enhance backend selection (user 2026-08-05: "the enhancer backend choice must be a
# persisted, GUI-settable option — no hidden env-only config"). The H3 Context-IR enhancer at the
# adapter chokepoint (adapter._maybe_enhance_h3) can run on several backends; this persisted flag
# picks WHICH:
#   auto            -> the resolution chain: configured endpoint -> live LLM preset -> local vision
#                      reasoner -> fallback GGUF (the historical default; byte-for-byte the old chain)
#   endpoint        -> ONLY the STUDIO_H3_ENHANCER_ENDPOINT OpenAI-compatible sidecar
#   live-preset     -> ONLY the LLM preset selected in Connect (llm_launch.enhance_via_server)
#   local-reasoner  -> ONLY the single-machine RTX vision reasoner (llm_launch.run_local_reason)
#   gguf            -> ONLY the fallback enhancer GGUF via LlamaReasoner
# A specific (non-auto) value fails LOUD when its backend cannot produce a labelled rewrite — never a
# silent fall-through to another backend. Stored in its OWN file (like the token / remote-access /
# gpu-assignment stores) so an engine-settings save never clobbers it; adapter._maybe_enhance_h3 reads
# it LIVE so a change takes effect on the very next generation (no daemon restart).
# --------------------------------------------------------------------------------------------------
H3_ENHANCER_BACKENDS: tuple[str, ...] = ("auto", "endpoint", "live-preset", "local-reasoner", "gguf")


def _h3_enhancer_file() -> Path:
    # anchored next to the other stores (output_dir.parent) so tests/env stay isolated
    return settings.state_dir / "studio_h3_enhancer.json"


def load_h3_enhancer_backend() -> str:
    """The persisted H3 Prompt Enhance backend selection. Default 'auto' (the historical chain).
    An unreadable / unrecognised stored value degrades to 'auto' rather than breaking a generation."""
    p = _h3_enhancer_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            val = str((d or {}).get("backend", "auto")).strip().lower()
            if val in H3_ENHANCER_BACKENDS:
                return val
        except (ValueError, OSError):
            return "auto"
    return "auto"


def save_h3_enhancer_backend(backend: str) -> str:
    """Persist the H3 Prompt Enhance backend selection. Raises ValueError on an unknown value (loud —
    a typo must not silently store a backend the enhancer will reject at generation time)."""
    val = str(backend or "auto").strip().lower()
    if val not in H3_ENHANCER_BACKENDS:
        raise ValueError(
            f"invalid H3 enhancer backend {backend!r}; allowed: {', '.join(H3_ENHANCER_BACKENDS)}")
    with _lock:
        p = _h3_enhancer_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"backend": val}, indent=2), encoding="utf-8")
        from . import settings_history  # scoped commit in the private settings history
        settings_history.after_write("h3-enhancer")
    return val


# -- chat-turn concurrency (Connect page) -------------------------------------
# How many chat turns Studio dispatches at once. 1 (default) => other chats' sends QUEUE and run
# FIFO as the slot frees; N>1 => up to N turns run in parallel (the ds4/mlx server still serialises
# internally). Persisted next to the other stores; read live by the chat_queue so a change applies
# to the next turn with no restart.
CHAT_CONCURRENCY_MIN = 1
CHAT_CONCURRENCY_MAX = 8
CHAT_CONCURRENCY_DEFAULT = 1


def _chat_concurrency_file() -> Path:
    return settings.state_dir / "studio_chat_concurrency.json"


def _clamp_chat_concurrency(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return CHAT_CONCURRENCY_DEFAULT
    return max(CHAT_CONCURRENCY_MIN, min(CHAT_CONCURRENCY_MAX, n))


def load_chat_concurrency() -> int:
    """The persisted number of concurrent chat turns Studio dispatches (default 1). An unreadable /
    out-of-range stored value degrades to the default rather than breaking a chat turn."""
    p = _chat_concurrency_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return _clamp_chat_concurrency((d or {}).get("concurrency", CHAT_CONCURRENCY_DEFAULT))
        except (ValueError, OSError):
            return CHAT_CONCURRENCY_DEFAULT
    return CHAT_CONCURRENCY_DEFAULT


def save_chat_concurrency(value) -> int:
    """Persist the concurrent-chat-turn setting (clamped to [MIN, MAX]). Returns the stored value."""
    n = _clamp_chat_concurrency(value)
    with _lock:
        p = _chat_concurrency_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"concurrency": n}, indent=2), encoding="utf-8")
        from . import settings_history  # scoped commit in the private settings history
        settings_history.after_write("chat-concurrency")
    return n


# --------------------------------------------------------------------------------------------------
# Philips Hue local-control pairing (spec §10 "Philips Hue local control"). Studio pairs with the
# user's existing Hue Bridge by asking them to press the physical link button; the Bridge then mints an
# APPLICATION KEY (and an optional entertainment CLIENT KEY). That key is the ONLY credential needed for
# all subsequent Hue API V2 calls (sent as the `hue-application-key` header over HTTPS) and MUST persist
# across daemon restarts, so it lives in its OWN file (like the pair-token / remote-access / gpu stores)
# — an engine-settings save never clobbers it. We also pin the Bridge's self-signed TLS certificate
# fingerprint on first pair (trust-on-first-use) so later connections can detect a swapped bridge, per
# the spec's "record the Bridge certificate/fingerprint" requirement. The app key is a bearer secret;
# treat this file like the credential vault (§12 "Credential vault"). Absence of a stored key == UNPAIRED,
# and every control path fails LOUD in that state (no silent no-op) — spec §10 + the studio's
# no-silent-fallbacks rule.
# --------------------------------------------------------------------------------------------------
_HUE_KEYS = ("bridge_id", "bridge_ip", "app_key", "client_key", "cert_fingerprint", "paired_at")


def _hue_file() -> Path:
    # anchored next to the other stores (output_dir.parent) so tests/env stay isolated
    return settings.state_dir / "studio_hue.json"


def load_hue_credentials() -> dict:
    """The persisted Hue Bridge pairing record, or ``{}`` when UNPAIRED. Keys: bridge_id, bridge_ip,
    app_key (the `hue-application-key`), client_key (optional entertainment key), cert_fingerprint
    (pinned self-signed leaf, TOFU), paired_at (ISO-8601). An unreadable file degrades to ``{}`` (==
    unpaired) rather than raising — callers then surface the loud "press the link button" pairing flow."""
    p = _hue_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return {k: d[k] for k in _HUE_KEYS if k in d and d[k] not in (None, "")}
        except (ValueError, OSError):
            return {}
    return {}


def save_hue_credentials(record: dict) -> dict:
    """Persist the Hue pairing record. Raises ValueError when the essential fields are missing (loud — a
    half-written pairing must not masquerade as paired). Returns the stored record (app_key preserved)."""
    rec = {k: record.get(k) for k in _HUE_KEYS if record.get(k) not in (None, "")}
    if not rec.get("app_key"):
        raise ValueError("Hue pairing record requires an app_key (the hue-application-key)")
    if not rec.get("bridge_ip"):
        raise ValueError("Hue pairing record requires the bridge_ip (local Bridge address)")
    with _lock:
        p = _hue_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def clear_hue_credentials() -> None:
    """Forget the Hue pairing (spec §13.3: users can delete the Hue pairing independently). Idempotent —
    a subsequent control call then fails loud as UNPAIRED until the user re-pairs via the link button."""
    with _lock:
        try:
            _hue_file().unlink(missing_ok=True)
        except OSError:
            pass


def load() -> dict:
    data = dict(DEFAULT_ENGINE_SETTINGS)
    p = _path()
    if p.is_file():
        try:
            stored = json.loads(p.read_text())
            for k in DEFAULT_ENGINE_SETTINGS:
                if k in stored:
                    data[k] = stored[k]
        except (ValueError, OSError):
            pass
    return data


def update(patch: dict) -> dict:
    with _lock:
        data = load()
        for k, v in patch.items():
            if k in DEFAULT_ENGINE_SETTINGS:  # ignore unknown keys
                data[k] = v
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        from . import settings_history  # scoped commit in the private settings history
        settings_history.after_write("studio-settings")
        return data


def paths() -> dict:
    model_fast_root = getattr(settings, "model_fast_root", None) or settings.models_root
    model_archive_root = (
        getattr(settings, "model_archive_root", None)
        or os.environ.get("STUDIO_MODEL_ARCHIVE_ROOT")
        or str(settings.models_root.parent / "models_archive")
    )
    return {
        "output_dir": str(settings.output_dir),
        "models_root": str(settings.models_root),
        "model_fast_root": str(model_fast_root),
        "model_archive_root": str(model_archive_root),
        "datasets_dir": str(settings.datasets_dir),
        "loras_dir": str(settings.loras_dir),
    }


# Editable data-location roots (user 2026-06-03: "data locations should all be editable/browse, not
# env vars"). Persisted to a STABLE app-data file (independent of output_dir, which itself can change).
_PATH_KEYS = ("output_dir", "models_root", "model_fast_root", "model_archive_root", "datasets_dir", "loras_dir")
_PATH_ENVS = {
    "output_dir": "STUDIO_OUTPUT_DIR",
    "models_root": "STUDIO_MODELS_ROOT",
    "model_fast_root": "STUDIO_MODEL_FAST_ROOT",
    "model_archive_root": "STUDIO_MODEL_ARCHIVE_ROOT",
    "datasets_dir": "STUDIO_DATASETS_DIR",
    "loras_dir": "STUDIO_LORAS_DIR",
}


def _paths_file() -> Path:
    # anchored next to the other stores (output_dir.parent) so tests/env stay isolated
    return settings.state_dir / "studio_paths.json"


def load_paths() -> dict:
    p = _paths_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text())
            return {k: v for k, v in d.items() if k in _PATH_KEYS and v}
        except (ValueError, OSError):
            return {}
    return {}


def save_paths(patch: dict) -> dict:
    with _lock:
        cur = load_paths()
        for k in _PATH_KEYS:
            if patch.get(k):
                cur[k] = str(patch[k])
        p = _paths_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur, indent=2))
        from . import settings_history  # scoped commit in the private settings history
        settings_history.after_write("paths")
        return cur


def apply_paths(patch: dict, *, respect_env: bool = True) -> list[str]:
    """Mutate the live Settings path roots + create the dirs (takes effect immediately)."""
    applied: list[str] = []
    for k in _PATH_KEYS:
        value = patch.get(k)
        if not value:
            continue
        # Startup env is the owner when present. This keeps a stale persisted GUI value from
        # silently moving active model loads off the intended machine-local model root.
        if respect_env and os.environ.get(_PATH_ENVS[k]):
            continue
        if k in ("models_root", "model_fast_root") and not _slow_model_root_allowed(value):
            continue
        setattr(settings, k, Path(value))
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        applied.append(k)
    configure_model_storage_env(settings.models_root)
    settings.ensure_dirs()
    return applied


def _slow_model_root_allowed(value: str) -> bool:
    if os.name != "nt":
        return True
    allow = (os.environ.get("STUDIO_ALLOW_SLOW_MODEL_DRIVES") or "").strip().lower()
    if allow in ("1", "true", "on", "yes", "y"):
        return True
    try:
        from .config import _windows_slow_model_drive
        return not _windows_slow_model_drive(value)
    except Exception:
        return True


# --------------------------------------------------------------------------------------------------
# Per-model / per-service GPU assignment (user 2026-06-09: "assign models to a gpu... so everytime
# they are started they go to the same place").
#
# THE GPU-INDEX CONVENTION (adopted app-wide): indices are stored in NVIDIA-SMI PCI-BUS ORDER, i.e.
# what `nvidia-smi --query-gpu=index` reports with CUDA_DEVICE_ORDER=PCI_BUS_ID.
# Device assignments are installation-specific. This is the same index space the audio worker
# already uses (it sets CUDA_DEVICE_ORDER=PCI_BUS_ID then CUDA_VISIBLE_DEVICES=<index>), so a single
# index means the same physical card everywhere. Subprocess backends select the card BEFORE importing
# torch by exporting CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=<index> (so the child sees it
# as its device 0). The in-process Cosmos backend instead maps the nvidia-smi index -> the matching
# torch device by GPU UUID (RealCosmosBackend._cuda_device) because in-process torch may already have
# enumerated devices in a different order. The /health GPU list is the source of indices for the UI.
# --------------------------------------------------------------------------------------------------
def _gpu_assign_file() -> Path:
    # anchored next to the other stores so tests/env stay isolated
    return settings.state_dir / "studio_gpu_assignments.json"


def _norm_gpu_index(v) -> str | None:
    """Normalize a stored GPU index to a string (nvidia-smi PCI-bus index) or None when unset/CPU."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "auto", "default", "none"):
        return None
    if s in ("cpu", "-1"):
        return "cpu"
    return s if s.isdigit() else None


def load_gpu_assignments() -> dict:
    """The persisted ``{model_or_service: gpu_index}`` map (nvidia-smi PCI-bus indices, see convention
    above). Survives daemon restart. Empty -> every backend uses its current default."""
    p = _gpu_assign_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out = {}
            for k, v in (d or {}).items():
                idx = _norm_gpu_index(v)
                if k and idx is not None:
                    out[str(k)] = idx
            return out
        except (ValueError, OSError):
            return {}
    return {}


def _save_gpu_assignments(data: dict) -> dict:
    p = _gpu_assign_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    from . import settings_history  # scoped commit in the private settings history
    settings_history.after_write("gpu-assignments")
    return data


def set_gpu_assignment(model_or_service: str, gpu_index) -> dict:
    """Assign (or, with gpu_index None/auto, clear) one model/service. Returns the full updated map."""
    key = str(model_or_service or "").strip()
    if not key:
        raise ValueError("model_or_service is required")
    with _lock:
        cur = load_gpu_assignments()
        idx = _norm_gpu_index(gpu_index)
        if idx is None:
            cur.pop(key, None)  # unassign -> fall back to default behaviour
        else:
            cur[key] = idx
        return _save_gpu_assignments(cur)


def remove_gpu_assignment(model_or_service: str) -> dict:
    with _lock:
        cur = load_gpu_assignments()
        cur.pop(str(model_or_service or "").strip(), None)
        return _save_gpu_assignments(cur)


def gpu_for(*keys: str) -> str | None:
    """Resolve the assigned GPU index for the first matching key (e.g. a specific model name, then a
    coarser service/arch bucket like 'sdxl'/'audio'). Returns the nvidia-smi index string, 'cpu', or
    None when nothing is assigned (caller keeps its current default)."""
    assignments = load_gpu_assignments()
    for k in keys:
        if not k:
            continue
        idx = assignments.get(str(k))
        if idx is not None:
            return idx
    return None
