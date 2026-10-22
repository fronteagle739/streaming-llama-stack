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
DIRECT_READ = re.compile(r'''(?:process|import\.meta)\.env(?:\.([A-Z][A-Z0-9_]+)|\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\])|getEnvVar(?:Owned|Sentinel)\s*\([^,]+,\s*["']([A-Z][A-Z0-9_]+)["']''')
WINDOWS_READ = re.compile(r'''\b(?:GetEnvironmentVariable[AW]|getenv_s|_dupenv_s)\s*\([^;\n]*?["']([A-Z][A-Z0-9_]+)["']''')
HOST_VARIABLES = {'PATH', 'HOME', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'SYSTEMROOT',
                  'WINDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'COMSPEC', 'PATHEXT',
                  'USERNAME', 'LOGNAME', 'USER', 'SHELL', 'PWD', 'OLDPWD', 'CUDA_PATH',
                  'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'NO_COLOR', 'FORCE_COLOR'}
SCOPE_ALIASES = {'ds4-mac': 'mac-ds4', 'rtx-producer': 'rtx-llama'}


def inventory():
    found = {}
    extensions = {'.c', '.h', '.cpp', '.hpp', '.cu', '.cuh', '.inc', '.m', '.mm', '.py', '.zig', '.js', '.ts', '.svelte'}
    for root in (ROOT / 'engines', ROOT / 'vendor'):
        for directory, children, files in os.walk(root):
            children[:] = [s for s in children if s not in {'.git', 'node_modules', 'build', '.venv', '__pycache__', '.svelte-kit', 'dist'}]
            for name in sorted(files):
                p = Path(directory) / name
                if p.suffix not in extensions:
                    continue
                for line_number, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                    matches = [m[1] or m[2] for m in READ.finditer(line)]
                    matches += [next(s for s in m.groups() if s) for m in DIRECT_READ.finditer(line)]
                    matches += [m[1] for m in WINDOWS_READ.finditer(line)]
                    for name in matches:
                        reference = f'{p.relative_to(ROOT).as_posix()}:{line_number}'
                        if reference not in found.setdefault(name, []):
                            found[name].append(reference)
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
    scope = SCOPE_ALIASES.get(scope, scope)
    context = {**cfg, 'host_budget_bytes': int(cfg.get('host_budget_gib', 8) * (1 << 30)),
               'host_budget_mb': int(cfg.get('host_budget_gib', 8) * 1024),
               'physical_floor_bytes': int(cfg.get('min_available_gib', 4) * (1 << 30))}
    if cfg.get('runtime_root'):
        from .config import path
        context.update(runtime_root=str(Path(cfg['runtime_root']).resolve()),
                       studio_root=str(path(cfg.get('studio_root', 'vendor/studio'))),
                       streaming_root=str(path(cfg.get('streaming_root', 'vendor/engines'))),
                       weight_fetcher=str(path(cfg.get('weight_fetcher', 'build/llama/bin/slstack-weight-fetcher.dll'))),
                       library_pythonpath=os.pathsep.join((str(ROOT), str(path(cfg.get('studio_root', 'vendor/studio'))),
                                                          str(path(cfg.get('streaming_root', 'vendor/engines'))))))
    for row in rows():
        var = row['var']
        if var.upper() in HOST_VARIABLES and not row.get('defaults_by_scope'):
            continue  # Host variables (PATH, HOME, locale) keep their OS semantics.
        inherited = env.pop(var, None)
        applicable = scope in row['scopes']
        value = overrides.get(var) if applicable else None
        if value is None and applicable:
            value = cfg.get('ds4_env', {}).get(var) if scope == 'ds4-win' else None
            if value is None:
                value = row.get('defaults_by_scope', {}).get(scope)
                if value is not None:
                    # Only table-owned placeholders expand. User values, including
                    # JSON and braces in file names, remain literal.
                    def substitute(match):
                        if match[1] not in context:
                            raise ValueError(f'Missing launcher context {match[1]} for {var}')
                        return str(context[match[1]])
                    value = re.sub(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', substitute, str(value))
        if value is not None:
            value = str(value)
            if row.get('allowed_values') and value not in row['allowed_values']:
                raise ValueError(f'{var}: value violates required memory-residency policy')
            if row.get('equals_context') and value != str(context[row['equals_context']]):
                raise ValueError(f'{var}: must match the launcher {row["equals_context"]} setting')
            for rule, predicate in (('maximum_context', lambda a, b: a <= b),
                                    ('minimum_context', lambda a, b: a >= b)):
                if row.get(rule):
                    try:
                        valid = predicate(int(value), int(context[row[rule]]))
                    except ValueError:
                        valid = False
                    if not valid:
                        raise ValueError(f'{var}: conflicts with the admitted host-memory budget/floor')
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
