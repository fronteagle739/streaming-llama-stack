"""Environment inventory, CLI surface and child environment share one checked table."""
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'envvars.json'
_WARNED = set()
READ = re.compile(r'''(?:\b\w*env\w*|environ\.get|environ\.setdefault)\s*\(\s*[Lru]*["']([A-Z][A-Z0-9_]+)["']|environ\s*\[\s*["']([A-Z][A-Z0-9_]+)["']''')


def inventory():
    found = {}
    for root in (ROOT / 'engines', ROOT / 'vendor'):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in {'.c', '.h', '.cpp', '.hpp', '.cu', '.cuh', '.inc', '.m', '.mm', '.py'}:
                continue
            for line_number, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                for match in READ.finditer(line):
                    name = match[1] or match[2]
                    found.setdefault(name, []).append(f'{p.relative_to(ROOT).as_posix()}:{line_number}')
    return found


def rows():
    return json.loads(TABLE.read_text(encoding='utf-8'))


def add_flags(parser, *, suppress=False):
    import argparse
    for row in rows():
        if row['flag']:
            parser.add_argument(row['flag'], dest='env_' + row['var'], default=argparse.SUPPRESS if suppress else None,
                                help=row['doc'])


def apply(env, cfg, scope):
    """Explicit values win; inherited engine knobs never silently override a preset."""
    overrides = cfg.get('env_overrides', {})
    for row in rows():
        var = row['var']
        if not var.startswith(('DS4_', 'LLAMA_', 'GGML_', 'STUDIO_', 'MEM_GUARD_', 'COSMOS_', 'SLSTACK_')):
            continue  # Host variables (PATH, HOME, locale) keep their OS semantics.
        inherited = env.pop(var, None)
        value = overrides.get(var)
        if value is None and scope in row['scopes']:
            value = cfg.get('ds4_env', {}).get(var) if scope == 'ds4-win' else None
            if value is None:
                value = row.get('defaults_by_scope', {}).get(scope)
        if value is not None:
            value = str(value).format(**cfg)
            if row.get('allowed_values') and value not in row['allowed_values']:
                raise ValueError(f'{var}: value violates required memory-residency policy')
            env[var] = value
        if inherited is not None:
            identity = (var, inherited, value)
            if identity not in _WARNED:
                _WARNED.add(identity)
                displayed = '<redacted>' if any(word in var for word in ('TOKEN', 'SECRET', 'PASSWORD', 'KEY')) else value
                print(f'Environment override: inherited {var} ignored; selected {displayed if value is not None else "engine default"}', file=sys.stderr)
    return env


def validate_table():
    table = rows()
    present = {r['var'] for r in table}
    missing = set(inventory()) - present
    if missing or len(present) != len(table) or any(not r['doc'] for r in table):
        raise ValueError(f'Environment table out of date: {sorted(missing)}')
    return len(table)
