# Derived from Studio backend/studio/chat_artifacts.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import os
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional
_MANIFEST_LOCK = threading.Lock()
from .config import settings
_SAFE_SEGMENT = re.compile('[^A-Za-z0-9_.-]+')

def _is_within(child: Path, parent: Path) -> bool:
    c = os.path.normcase(str(child))
    p = os.path.normcase(str(parent))
    return c == p or c.startswith(p + os.sep)

def safe_conversation_id(conversation_id: Optional[str]) -> str:
    raw = str(conversation_id or '').strip()
    if not raw:
        return '_shared'
    safe = _SAFE_SEGMENT.sub('_', raw)[:96].strip('._-')
    if safe:
        return safe
    return 'c_' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

def chat_output_dir(conversation_id: Optional[str], *, create: bool=True) -> Path:
    root = settings.output_dir / 'chats' / safe_conversation_id(conversation_id)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root

def manifest_path(conversation_id: Optional[str]) -> Path:
    return chat_output_dir(conversation_id, create=False) / 'artifacts.json'

def _read_manifest(conversation_id: Optional[str]) -> list[dict]:
    p = manifest_path(conversation_id)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = data.get('artifacts', [])
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []

def _write_manifest(conversation_id: Optional[str], items: list[dict]) -> None:
    p = manifest_path(conversation_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps({'artifacts': items[-1000:]}, indent=2), encoding='utf-8')
    os.replace(tmp, p)

def record_chat_artifact(conversation_id: Optional[str], path: str | Path, *, kind: str='media', tool: str='', metadata: Optional[dict]=None) -> Optional[dict]:
    try:
        from .project_context import read_tags, tag_file
        if path and (not read_tags(path)):
            tag_file(path)
    except Exception:
        pass
    if not conversation_id:
        return None
    root = chat_output_dir(conversation_id).resolve(strict=False)
    try:
        p = Path(path).expanduser().resolve(strict=True)
    except (OSError, FileNotFoundError):
        return None
    if not _is_within(p, root):
        return None
    try:
        stat = p.stat()
    except OSError:
        return None
    now = time.time()
    rel = p.relative_to(root).as_posix()
    item = {'path': str(p), 'relative_path': rel, 'name': p.name, 'kind': str(kind or 'media'), 'tool': str(tool or ''), 'size_bytes': int(stat.st_size), 'mtime': float(stat.st_mtime), 'created': now}
    if metadata:
        item['metadata'] = metadata
    with _MANIFEST_LOCK:
        items = _read_manifest(conversation_id)
        kept = [x for x in items if str(x.get('path')) != str(p)]
        kept.append(item)
        _write_manifest(conversation_id, kept)
    return item
_IMPORT_MEDIA_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff', '.avif', '.mp4', '.webm', '.mov', '.mkv', '.wav', '.mp3', '.flac', '.ogg', '.m4a'}
_IMPORT_MAX_BYTES = 1 << 30

def output_reference_path(raw: str) -> Path:
    path = Path(raw.replace('\\', '/'))
    if not path.is_absolute():
        if path.parts and path.parts[0] == 'outputs':
            return settings.output_dir.joinpath(*path.parts[1:])
        if path.parts and path.parts[0] == 'chats':
            return settings.output_dir / path
    return path

def import_outside_media(conversation_id: Optional[str], messages: list) -> int:
    from .artifact_api import artifact_roots, _is_within as _inside
    from .guest_scope import remote_peer
    if remote_peer() is not None or not conversation_id or (not isinstance(messages, list)):
        return 0
    roots = artifact_roots()
    changed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        media = list(msg.get('media') or [])
        media.extend((part.get('media') for part in msg.get('parts') or [] if isinstance(part, dict) and part.get('type') == 'media'))
        for md in media:
            raw = md.get('path') if isinstance(md, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                continue
            from urllib.parse import urlsplit, parse_qs
            from urllib.request import url2pathname
            if raw.startswith('file:'):
                url = urlsplit(raw)
                if url.netloc not in ('', 'localhost'):
                    continue
                raw = url2pathname(url.path)
            elif raw.startswith('/artifact?'):
                raw = parse_qs(urlsplit(raw).query).get('path', [''])[0]
            src = output_reference_path(raw)
            if not src.is_absolute():
                local = chat_output_dir(conversation_id, create=False) / src
                if local.is_file():
                    src = local
            try:
                if src.suffix.lower() not in _IMPORT_MEDIA_EXT or not src.is_file():
                    continue
                if any((_inside(src, root) for root in roots)):
                    served = str(src.resolve())
                    if md['path'] != served:
                        md['path'] = served
                        changed += 1
                    continue
                st = src.stat()
                if st.st_size > _IMPORT_MAX_BYTES:
                    continue
                key = hashlib.sha1(os.path.normcase(str(src.resolve())).encode('utf-8')).hexdigest()[:10]
                dest = chat_output_dir(conversation_id) / 'imported' / f'{key}-{src.name}'
                if not (dest.is_file() and dest.stat().st_size == st.st_size and (dest.stat().st_mtime >= st.st_mtime)):
                    import shutil
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + f'.tmp{os.getpid()}.{threading.get_ident()}')
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dest)
            except OSError:
                continue
            md['path'] = str(dest)
            changed += 1
    return changed

def list_chat_artifacts(conversation_id: Optional[str], *, kind: str='', limit: int=50) -> list[dict]:
    root = chat_output_dir(conversation_id, create=False).resolve(strict=False)
    items = []
    kind = str(kind or '').strip().lower()
    try:
        limit_i = max(1, min(200, int(limit or 50)))
    except (TypeError, ValueError):
        limit_i = 50
    for item in reversed(_read_manifest(conversation_id)):
        path = str(item.get('path') or '')
        if not path:
            continue
        try:
            p = Path(path).resolve(strict=False)
        except OSError:
            continue
        if not _is_within(p, root):
            continue
        if kind and str(item.get('kind') or '').lower() != kind:
            continue
        copy = dict(item)
        copy['exists'] = p.is_file()
        items.append(copy)
        if len(items) >= limit_i:
            break
    return items

def output_dir_for_params(params=None, *, create: bool=True) -> Path:
    raw = getattr(params, 'output_dir', None) if params is not None else None
    root = settings.output_dir.resolve(strict=False)
    if raw:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = settings.output_dir / p
        out = p.resolve(strict=False)
        if not _is_within(out, root):
            raise ValueError('generation output_dir must be under the configured Studio output directory')
    else:
        out = root
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out

def apply_chat_output_dir(params, conversation_id: Optional[str]=None):
    cid = conversation_id or getattr(params, 'conversation_id', None)
    if cid and (not getattr(params, 'output_dir', None)):
        params.output_dir = str(chat_output_dir(cid))
    from .guest_scope import confine_generation_paths
    try:
        confine_generation_paths(params)
    except PermissionError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return params

def chat_media_context_prompt(conversation_id: Optional[str]) -> str:
    path = chat_output_dir(conversation_id, create=False)
    return f'Chat workspace directory (default location for files this chat creates): {path}'

def resolve_chat_relative_path(path_str: str, conversation_id: Optional[str], *, must_exist: bool=True) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve(strict=must_exist)
    joined = chat_output_dir(conversation_id) / p
    if joined.exists() or not must_exist:
        return joined.resolve(strict=must_exist)
    from_registry = _artifact_path_for_reference(conversation_id, path_str)
    if from_registry is not None:
        return from_registry.resolve(strict=must_exist)
    return joined.resolve(strict=must_exist)

def _artifact_path_for_reference(conversation_id: Optional[str], ref: str) -> Optional[Path]:
    key = str(ref or '').strip()
    if not key:
        return None
    stem = Path(key).stem
    for item in _read_manifest(conversation_id):
        if not isinstance(item, dict):
            continue
        meta = item.get('metadata') or {}
        candidates = (str(meta.get('shot_id') or ''), str(item.get('name') or ''), str(item.get('relative_path') or ''))
        if key in candidates or (stem and stem in (Path(c).stem for c in candidates if c)):
            path = item.get('path')
            if path and Path(path).is_file():
                return Path(path)
    return None
