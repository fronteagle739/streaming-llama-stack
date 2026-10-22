"""Local release gate. Reports rule/path/line only, never matching secret text."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private network': r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
    'credential': r'(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{15,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',
    'machine path': r'(?i)(?<![%\w])[a-z]:[\\/]',
    'coordination file': r'DEPLOY_HOLD\.md',
    'old product name': r'\bds4stack\b|ds4-stack',
}


def forbidden_literals(root):
    """Private audit inputs stay in an ignored local file, never in this script."""
    local = root / 'release-secrets.local.json'
    if not local.exists():
        return ()
    if local.stat().st_size > 8192:
        raise ValueError('Local release-check input exceeds its bound')
    data = json.loads(local.read_text(encoding='utf-8'))
    values = data.get('forbidden_literals', [])
    if not isinstance(values, list) or any(not isinstance(v, str) or not 8 <= len(v) <= 512 for v in values):
        raise ValueError('Local forbidden literals must be strings of length 8..512')
    return tuple(set(values))


def scan(root=ROOT):
    findings = []
    forbidden = forbidden_literals(root)
    compiled = {name: re.compile(pattern) for name, pattern in RULES.items()}
    tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '-z']).decode('utf-8').split('\0'))
    candidates = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    paths = sorted(set(candidates.decode('utf-8').split('\0')) - {''})
    for relative in paths:
        p = root / relative
        rel = p.relative_to(root)
        if rel.parts[:2] == ('engines', 'mlx-serve'):
            continue  # Public upstream is independently pinned/audited; its gitlink is not vendored code.
        if any(x in {'.git', 'runtime', 'build', '__pycache__', 'node_modules', '.venv'} for x in rel.parts):
            continue
        if not p.is_file():
            continue
        if '.local.' in p.name or p.name.endswith('.local.json'):
            if relative in tracked:
                findings.append((str(rel), 0, 'local configuration must be untracked'))
            continue
        attribution = p.name in {'LICENSE', 'AUTHORS', 'NOTICE', 'NOTICES.txt'} or 'licenses' in rel.parts
        if p.suffix.lower() in {'.exe', '.dll', '.pdb', '.gguf', '.safetensors', '.pt', '.bin', '.npy', '.npz'}:
            findings.append((str(rel), 0, 'binary/weight artifact'))
            continue
        try:
            if p.suffix == '.gz':
                import gzip
                text = gzip.decompress(p.read_bytes()).decode('utf-8')
            else:
                text = p.read_text(encoding='utf-8')
        except UnicodeError:
            if attribution:
                text = p.read_bytes().decode('utf-8', errors='replace')
            else:
                findings.append((str(rel), 0, 'unreviewed non-text artifact'))
                continue
        for literal in forbidden:
            offset = text.find(literal)
            if offset >= 0:
                findings.append((str(rel), text.count('\n', 0, offset) + 1, 'forbidden audit literal'))
        candidates = {name: pattern for name, pattern in compiled.items()
                      if (not attribution or name == 'credential') and pattern.search(text)}
        if not candidates:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for name, pattern in candidates.items():
                if name != 'credential' and p.resolve() == Path(__file__).resolve() and any(line.strip().startswith(repr(key) + ':') for key in RULES):
                    continue  # Only that rule's literal pattern; credentials elsewhere still fail.
                if name == 'machine path' and (rel.parts[0] == 'examples' or p.name.endswith('.example.json')):
                    continue  # Owner allows only model/Comfy paths here; review separately.
                if pattern.search(line):
                    findings.append((str(rel), number, name))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    found = scan()
    if args.summary:
        from collections import Counter
        for name, count in Counter(row[2] for row in found).items():
            print(f'{name}: {count}')
        for row in found[:50]:
            print(f'{row[0]}:{row[1]}: {row[2]}')
    else:
        for row in found:
            print(f'{row[0]}:{row[1]}: {row[2]}')
    print(f'Release check: {len(found)} findings; local forbidden literals: {len(forbidden_literals(ROOT))}')
    return int(bool(found))


if __name__ == '__main__':
    sys.exit(main())
