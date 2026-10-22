"""Local release gate. Reports rule/path/line only, never matching secret text."""
from pathlib import Path
import argparse
import hashlib
import zlib
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


def private_names(root):
    local = root / 'release-secrets.local.json'
    if not local.exists():
        return ()
    if local.stat().st_size > 8192:
        raise ValueError('Local release-check input exceeds its bound')
    names = json.loads(local.read_text(encoding='utf-8')).get('forbidden_names', [])
    if not isinstance(names, list) or any(not isinstance(n, str) or not 2 <= len(n) <= 80 for n in names):
        raise ValueError('Private audit names must be strings of length 2..80')
    return tuple(set(names))


def reviewed_images(root):
    registry = root / 'release-artifacts.json'
    if not registry.exists():
        return {}
    if registry.stat().st_size > 65536:
        raise ValueError('Reviewed artifact registry exceeds its bound')
    rows = json.loads(registry.read_text(encoding='utf-8')).get('artifacts', [])
    if not isinstance(rows, list) or len(rows) > 200:
        raise ValueError('Invalid reviewed artifact registry')
    result = {}
    for row in rows:
        name = row.get('path', '')
        rel = Path(name)
        if (rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'examples'
            or rel.suffix != '.png' or row.get('media_type') != 'image/png'
            or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', ''))
            or not isinstance(row.get('review'), str) or len(row['review']) < 20):
            raise ValueError('Artifact review requires an example PNG path, hash and visual review')
        result[rel.as_posix()] = row['sha256']
    return result


def reviewed_png(path, expected):
    if not expected or path.stat().st_size > 5 * 1024**2:
        return False
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or not raw.startswith(bytes.fromhex('89504e470d0a1a0a')):
        return False
    # Screenshots carry pixels only. Text/EXIF/custom chunks and trailing bytes
    # must be reviewed as a new artifact, rather than silently escaping text scans.
    offset, types = 8, []
    while offset + 12 <= len(raw):
        count = int.from_bytes(raw[offset:offset+4], 'big')
        kind = raw[offset+4:offset+8]
        end = offset+12+count
        if end > len(raw) or kind not in (b'IHDR', b'IDAT', b'IEND', b'sRGB', b'gAMA', b'pHYs'):
            return False
        payload = raw[offset+8:offset+8+count]
        if zlib.crc32(kind+payload) & 0xffffffff != int.from_bytes(raw[end-4:end], 'big'):
            return False
        types.append(kind); offset = end
        if kind == b'IEND':
            return count == 0 and offset == len(raw) and types[0] == b'IHDR' and b'IDAT' in types
    return False


def scan(root=ROOT):
    findings = []
    forbidden = forbidden_literals(root)
    images = reviewed_images(root)
    compiled = {name: re.compile(pattern) for name, pattern in RULES.items()}
    names = private_names(root)
    if names:
        compiled['private identity'] = re.compile(r'(?i)(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z0-9])')
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
        if 'private identity' in compiled and compiled['private identity'].search(rel.as_posix()):
            findings.append((str(rel), 0, 'private identity in path'))
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
            if p.suffix == '.png' and reviewed_png(p, images.get(rel.as_posix())):
                continue
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
