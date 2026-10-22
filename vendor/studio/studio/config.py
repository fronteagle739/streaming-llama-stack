# Derived from Studio backend/studio/config.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import hashlib
import secrets
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from . import paths as _paths
GUEST_ID: ContextVar[Optional[str]] = ContextVar('studio_guest_id', default=None)
_GUEST_SCOPED = {'output_dir': 'outputs', 'datasets_dir': 'datasets', 'data_root': ''}

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _resolved(env: str, installer_subpath: str, *dev_candidates: str, want: str='any') -> str:
    inst = _paths.runtime_root() / installer_subpath if installer_subpath else None
    return str(_paths.resolve(env=env, installer_default=inst, dev_candidates=dev_candidates, want=want))

def _resolved_model(env: str, model_subpath: str, *dev_candidates: str) -> str:
    inst = _paths.app_data_root() / 'models' / model_subpath if model_subpath else None
    return str(_paths.resolve(env=env, installer_default=inst, dev_candidates=dev_candidates, want='dir'))

def _dev_python_candidates(*windows_candidates: str) -> list[str]:
    if _paths.is_windows():
        return list(windows_candidates)
    root = _paths.repo_root()
    candidates = [root / '.venv' / 'bin' / 'python', root / '.venv' / 'bin' / 'python3', root / '.venv312' / 'bin' / 'python', root / '.venv312' / 'bin' / 'python3', Path(sys.executable)]
    return [str(p) for p in candidates]

def _env_bool(name: str, default: bool=False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'on', 'yes', 'y')

def _env_path_opt(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None
PAIR_TOKEN_FILENAME = 'studio_pair_token.txt'
MIN_PAIR_TOKEN_LEN = 16
_RETIRED_PUBLIC_TOKEN_SHA256 = 'a9aedcb4b6759861d547193ccd957cc2c8f49ea829fcb00e8131d389adf5e425'

def is_retired_public_token(token) -> bool:
    return hashlib.sha256(str(token or '').strip().encode('utf-8')).hexdigest() == _RETIRED_PUBLIC_TOKEN_SHA256

def pair_token_problem(token) -> Optional[str]:
    tok = str(token or '').strip()
    if len(tok) < MIN_PAIR_TOKEN_LEN:
        return f'the pairing token must be at least {MIN_PAIR_TOKEN_LEN} characters'
    if is_retired_public_token(tok):
        return 'that is the old shared default token, which is public and retired'
    return None

def load_or_create_install_pair_token(state_dir: Path) -> str:
    path = Path(state_dir) / PAIR_TOKEN_FILENAME
    try:
        saved = path.read_text(encoding='utf-8').strip()
    except OSError:
        saved = ''
    if saved and pair_token_problem(saved) is None:
        return saved
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(token, encoding='utf-8')
    try:
        os.chmod(tmp, 384)
    except OSError:
        pass
    os.replace(tmp, path)
    print(f'[studio] generated a new per-install pairing token in {path}' + (' (replaced the retired shared default)' if saved else ''), file=sys.stderr, flush=True)
    return token

def env_pair_token() -> Optional[str]:
    tok = (os.environ.get('STUDIO_PAIR_TOKEN') or '').strip()
    if not tok:
        return None
    problem = pair_token_problem(tok)
    if problem:
        print(f"*** [studio] ignoring STUDIO_PAIR_TOKEN: {problem}; using this install's own token", file=sys.stderr, flush=True)
        return None
    return tok

def _default_pair_token() -> Optional[str]:
    state_dir = Path(os.environ.get('STUDIO_OUTPUT_DIR', str(_default_output_dir()))).parent
    return env_pair_token() or load_or_create_install_pair_token(state_dir)

def _default_output_dir() -> Path:
    if os.environ.get('STUDIO_DATA_ROOT'):
        return Path(os.environ['STUDIO_DATA_ROOT']).expanduser() / 'outputs'
    base = os.environ.get('LOCALAPPDATA')
    root = Path(base) / 'StreamingStudio' if base else Path.home() / '.front-eagle-stack'
    return root / 'outputs'

def _windows_slow_model_drive(path: str | Path | None) -> bool:
    if not path or not _paths.is_windows():
        return False
    try:
        drive = Path(str(path)).expanduser().drive.upper()
    except Exception:
        return False
    return drive in ('C:', 'I:')

def configure_model_storage_env(models_root: str | Path, env: Optional[dict[str, str]]=None) -> dict[str, str]:
    target = os.environ if env is None else env
    root = Path(str(models_root)).expanduser()
    cache = Path(str(target.get('STUDIO_HF_CACHE') or root / 'hf_cache')).expanduser()
    if _windows_slow_model_drive(cache) and str(target.get('STUDIO_ALLOW_SLOW_MODEL_DRIVES') or '').lower() not in ('1', 'true', 'on', 'yes', 'y'):
        cache = root / 'hf_cache'
    hub = cache / 'hub'
    allow_slow = str(target.get('STUDIO_ALLOW_SLOW_MODEL_DRIVES') or '').strip().lower() in ('1', 'true', 'on', 'yes', 'y')

    def put_default(name: str, value: Path) -> None:
        current = target.get(name)
        if not current or (_windows_slow_model_drive(current) and (not allow_slow)):
            target[name] = str(value)
    put_default('STUDIO_HF_CACHE', cache)
    put_default('HF_HOME', cache)
    put_default('HF_HUB_CACHE', hub)
    put_default('HUGGINGFACE_HUB_CACHE', hub)
    put_default('TRANSFORMERS_CACHE', hub)
    put_default('DIFFUSERS_CACHE', hub)
    put_default('KERNELS_CACHE', cache / 'kernels')
    put_default('HF_KERNELS_CACHE', cache / 'kernels')
    put_default('TORCHINDUCTOR_CACHE_DIR', cache / 'torchinductor')
    put_default('TRITON_CACHE_DIR', cache / 'triton')
    put_default('CUDA_CACHE_PATH', cache / 'cuda')
    return target

@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env('STUDIO_HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(_env('STUDIO_PORT', '8765')))
    cosmos_windows_path: Optional[Path] = field(default_factory=lambda: _env_path_opt('STUDIO_COSMOS_WINDOWS_PATH'))
    force_mock: bool = field(default_factory=lambda: _env_bool('STUDIO_FORCE_MOCK'))
    gpu_index: str = field(default_factory=lambda: _env('STUDIO_GPU_INDEX', '1'))
    pair_token: Optional[str] = field(default_factory=_default_pair_token)
    chat_base_url: Optional[str] = field(default_factory=lambda: os.environ.get('STUDIO_CHAT_BASE_URL') or None)
    chat_model: str = field(default_factory=lambda: _env('STUDIO_CHAT_MODEL', 'default'))
    chat_api_key: Optional[str] = field(default_factory=lambda: os.environ.get('STUDIO_CHAT_API_KEY') or None)
    real_llm: bool = field(default_factory=lambda: _env('STUDIO_REAL_LLM', '1') not in ('', '0', 'false', 'False'))
    real_train: bool = field(default_factory=lambda: _env('STUDIO_REAL_TRAIN', '0') not in ('', '0', 'false', 'False'))
    hunyuan_python: str = field(default_factory=lambda: _resolved('STUDIO_HUNYUAN_PYTHON', 'hunyuan-python\\Scripts\\python.exe', '', want='file'))
    hunyuan_model: str = field(default_factory=lambda: _resolved_model('STUDIO_HUNYUAN_MODEL', 'tencent\\HunyuanImage-3.0-Instruct', ''))
    cosmos_python: str = field(default_factory=lambda: _resolved('STUDIO_COSMOS_PYTHON', 'cosmos-python\\Scripts\\python.exe', *_dev_python_candidates(''), want='file'))
    reasoner_model: str = field(default_factory=lambda: _resolved_model('STUDIO_REASONER_MODEL', 'nvidia\\Cosmos3-Nano', ''))
    reasoner_max_new_tokens: int = field(default_factory=lambda: int(_env('STUDIO_REASONER_MAX_NEW_TOKENS', '512')))
    pid_python: str = field(default_factory=lambda: _resolved('STUDIO_PID_PYTHON', 'pid-python\\Scripts\\python.exe', '', want='file'))
    pid_repo: str = field(default_factory=lambda: _resolved('STUDIO_PID_REPO', 'PiD', '', want='dir'))
    pid_model_dir: str = field(default_factory=lambda: _resolved_model('STUDIO_PID_MODEL', 'nvidia\\PiD', ''))
    ideogram_python: str = field(default_factory=lambda: _resolved('STUDIO_IDEOGRAM_PYTHON', 'ideogram-python\\Scripts\\python.exe', '', want='file'))
    ideogram_cache: str = field(default_factory=lambda: _resolved('STUDIO_IDEOGRAM_CACHE', 'hf_cache', '', want='dir'))
    qwen_image21_python: str = field(default_factory=lambda: _resolved('STUDIO_QWEN_IMAGE21_PYTHON', '', 'qwen-image-21-python\\Scripts\\python.exe', want='file'))
    qwen_image21_model: str = field(default_factory=lambda: _resolved_model('STUDIO_QWEN_IMAGE21_MODEL', 'Qwen\\Qwen-Image-2.1', '', ''))
    qwen_image21_base: str = field(default_factory=lambda: _resolved_model('STUDIO_QWEN_IMAGE21_BASE', 'Qwen\\Qwen-Image-2.1', '', ''))
    joyai_python: str = field(default_factory=lambda: _resolved('STUDIO_JOYAI_PYTHON', 'JoyAI-Echo\\.venv\\Scripts\\python.exe', '', want='file'))
    joyai_repo: str = field(default_factory=lambda: _resolved('STUDIO_JOYAI_REPO', 'JoyAI-Echo', '', want='dir'))
    joyai_config: str = field(default_factory=lambda: _resolved('STUDIO_JOYAI_CONFIG', 'JoyAI-Echo\\configs\\inference.yaml', '', want='file'))
    fdanyone_python: str = field(default_factory=lambda: _resolved('STUDIO_4DANYONE_PYTHON', 'cosmos-venv\\Scripts\\python.exe', '', want='file'))
    fdanyone_repo: str = field(default_factory=lambda: _resolved('STUDIO_4DANYONE_REPO', '4DAnyone', '', want='dir'))
    fdanyone_model: str = field(default_factory=lambda: _resolved_model('STUDIO_4DANYONE_MODEL', '4danyone', ''))
    data_root: Path = field(default_factory=_paths.app_data_root)
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get('STUDIO_OUTPUT_DIR', str(_default_output_dir()))))
    models_root: Path = field(default_factory=lambda: _paths.resolve(env='STUDIO_MODELS_ROOT', installer_default=_paths.app_data_root() / 'models', dev_candidates=[''], want='dir'))
    agent_memory_embed_model: str = field(default_factory=lambda: _env('STUDIO_AGENT_MEMORY_EMBED_MODEL', 'BAAI/bge-small-en-v1.5'))
    agent_memory_embed_model_dir: Path = field(default_factory=lambda: Path(_resolved_model('STUDIO_AGENT_MEMORY_EMBED_MODEL_DIR', 'BAAI\\bge-small-en-v1.5', '')).expanduser())
    datasets_dir: Path = field(default_factory=lambda: Path(os.environ.get('STUDIO_DATASETS_DIR', str(_default_output_dir().parent / 'datasets'))))
    loras_dir: Path = field(default_factory=lambda: Path(os.environ.get('STUDIO_LORAS_DIR', str(_default_output_dir().parent / 'loras'))))

    @property
    def storyboards_dir(self) -> Path:
        return self.output_dir.parent / 'storyboards'

    def allowed_file_roots(self) -> list[Path]:
        roots = []
        for p in (self.output_dir, self.storyboards_dir, self.data_root / 'captures', self.models_root, self.datasets_dir, self.loras_dir):
            try:
                roots.append(p.resolve())
            except OSError:
                roots.append(p)
        return roots

    def __getattribute__(self, name):
        if name in _GUEST_SCOPED:
            guest = GUEST_ID.get()
            if guest:
                root = object.__getattribute__(self, 'data_root') / 'guests' / guest
                sub = _GUEST_SCOPED[name]
                return root / sub if sub else root
        return object.__getattribute__(self, name)

    @property
    def state_dir(self) -> Path:
        return object.__getattribute__(self, 'output_dir').parent

    @property
    def owner_data_root(self) -> Path:
        return object.__getattribute__(self, 'data_root')

    def __setattr__(self, name, value):
        if name == 'chat_base_url' and value is None:
            previous = getattr(self, 'chat_base_url', None)
            if previous is not None:
                import sys as _sys
                import traceback as _tb
                print('[studio] chat_base_url CLEARED (was %r) -- chat will report no live backend until a switch re-adopts. Stack:' % (previous,), file=_sys.stderr, flush=True)
                _tb.print_stack(file=_sys.stderr)
                _sys.stderr.flush()
        object.__setattr__(self, name, value)

    def ensure_dirs(self) -> None:
        configure_model_storage_env(self.models_root)
        for p in (self.data_root, self.output_dir, self.datasets_dir, self.loras_dir, self.models_root, Path(os.environ['HF_HOME']), Path(os.environ['HF_HUB_CACHE']), Path(os.environ['KERNELS_CACHE'])):
            p.mkdir(parents=True, exist_ok=True)
settings = Settings()
configure_model_storage_env(settings.models_root)
