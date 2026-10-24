# Derived from Studio backend/studio/project_context.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""CODEVID1: the Studio Project each page has open, carried EXPLICITLY on every GUI request.

The frontend sends ``X-Studio-Projects: <page>=<project id>,...`` (every page that has a project open).
A middleware puts it in a context variable for the request; ``GenerationParams`` copies it into its own
``project_ids`` field when it is built (so it travels with the params through queues and jobs), and the
provenance sidecar (``<output>.gen.json``) records it. The page timeline then lists an item under project
P on page X when its sidecar says ``project_ids[X] == P``. Generators that do not write a sidecar fall back
to the project activity windows (project_store.owns_time).
"""
from __future__ import annotations

import contextvars
from typing import Optional

HEADER = "x-studio-projects"
_current: contextvars.ContextVar[dict] = contextvars.ContextVar("studio_projects", default={})


def parse(value: Optional[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(value or "").split(","):
        page, sep, pid = part.strip().partition("=")
        if sep and page.strip() and pid.strip() and len(pid.strip()) <= 64:
            out[page.strip()[:40]] = pid.strip()
    return out


def set_current(value: dict) -> contextvars.Token:
    return _current.set(dict(value or {}))


def reset(token: contextvars.Token) -> None:
    _current.reset(token)


def current() -> dict[str, str]:
    return dict(_current.get() or {})


# -- tagging outputs that have no GenerationParams sidecar (music, 4D, world jobs, code shots, voice
# lines, builder stages): a small `<file>.project.json` beside the file. Written at creation by the job
# runner (the job captured the request's projects when it was made) or by record_chat_artifact.
_MEDIA_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov", ".wav", ".mp3", ".flac",
              ".ogg", ".m4a", ".glb", ".gltf", ".ply", ".splat", ".spz", ".fbx", ".blend")


def tag_file(path, ids: Optional[dict] = None) -> bool:
    import json
    import os
    ids = dict(ids if ids is not None else current())
    p = str(path or "")
    if not ids or not p.lower().endswith(_MEDIA_EXT) or not os.path.isfile(p):
        return False
    try:
        with open(p + ".project.json", "w", encoding="utf-8") as fh:
            json.dump({"project_ids": ids}, fh)
        return True
    except OSError:
        return False


def tag_outputs(result, ids: Optional[dict]) -> int:
    """Tag every output file path found anywhere in a job result."""
    if not ids:
        return 0
    n = 0
    stack = [result]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, str) and len(cur) < 1000:
            n += tag_file(cur, ids)
    return n


def read_tags(path) -> dict:
    """The project tags of an output: its .gen.json project_ids, else its .project.json."""
    import json
    import os
    for side in (str(path) + ".gen.json", str(path) + ".project.json"):
        if os.path.isfile(side):
            try:
                tags = (json.load(open(side, encoding="utf-8")) or {}).get("project_ids") or {}
            except (ValueError, OSError):
                tags = {}
            if tags:
                return tags
    return {}
