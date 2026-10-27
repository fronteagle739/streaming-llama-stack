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
    'machine path': r'''(?i:(?<![%\w])[dlm]:[\\/]|(?<![%\w])[a-z]:[\\/]{1,2}(?:Users|scratch|GitHub|models_archive)(?:[\\/]|["']))|/(?:Users|home)/[^/\s"']+/|/Volumes/[^/\s"']+/''',
    'private identity': r'(?i)\bedw+a{1,2}rd[\s._-]*h[o]lmes(?:[\s._-]*c[r]anborne)?\b|\bh[o]lmes[\s._-]*c[r]anborne\b|c[r]anborne|e[h]olmes|e[d]studio',
    'excluded project': r'(?i)\bmc[-_]?4(?:\b|_)',
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
    return reviewed_png_data(path.read_bytes(), expected)


def reviewed_png_data(raw, expected):
    if not expected or len(raw) > 5 * 1024**2:
        return False
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


def compiled_rules(root):
    compiled = {name: re.compile(pattern) for name, pattern in RULES.items()}
    names = private_names(root)
    if names:
        compiled['private identity'] = re.compile(RULES['private identity'] + r'|(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z0-9])')
    return compiled


def scan_blob(relative, raw, compiled, forbidden, images):
    rel = Path(relative)
    findings = []
    for rule in ('private identity', 'excluded project'):
        if compiled[rule].search(rel.as_posix()):
            findings.append((str(rel), 0, rule + ' in path'))
    if '.local.' in rel.name or rel.name.endswith('.local.json'):
        return findings + [(str(rel), 0, 'local configuration must be untracked')]
    attribution = rel.name in {'LICENSE', 'AUTHORS', 'NOTICE', 'NOTICES.txt'} or 'licenses' in rel.parts
    if rel.suffix.lower() in {'.exe', '.dll', '.pdb', '.gguf', '.safetensors', '.pt', '.bin', '.npy', '.npz'}:
        return findings + [(str(rel), 0, 'binary/weight artifact')]
    try:
        if rel.suffix == '.gz':
            import gzip
            text = gzip.decompress(raw).decode('utf-8')
        else:
            text = raw.decode('utf-8')
    except (UnicodeError, OSError, EOFError):
        if rel.suffix == '.png' and reviewed_png_data(raw, images.get(rel.as_posix())):
            return findings
        if attribution:
            text = raw.decode('utf-8', errors='replace')
        else:
            return findings + [(str(rel), 0, 'unreviewed non-text artifact')]
    for literal in forbidden:
        offset = text.lower().find(literal.lower())
        if offset >= 0:
            findings.append((str(rel), text.count('\n', 0, offset) + 1, 'forbidden audit literal'))
    candidates = {name: pattern for name, pattern in compiled.items()
                  if (not attribution or name in {'credential', 'private identity', 'excluded project'}) and pattern.search(text)}
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in candidates.items():
            if relative.replace('\\', '/') == 'scripts/release_check.py' and line.strip().startswith(repr(name) + ':'):
                if name != 'credential':continue
            example = (rel.parts[0] == 'examples' or '.example.' in rel.name) and rel.suffix in {'.py', '.ps1', '.sh', '.json', '.cmd', '.bat'}
            if name == 'machine path' and example:
                continue
            if pattern.search(line):
                findings.append((str(rel), number, name))
    return findings


def scan(root=ROOT):
    findings = []
    forbidden = forbidden_literals(root)
    images = reviewed_images(root)
    compiled = compiled_rules(root)
    tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '-z']).decode('utf-8').split('\0'))
    candidates = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    paths = sorted(set(candidates.decode('utf-8').split('\0')) - {''})
    for relative in paths:
        p = root / relative
        rel = p.relative_to(root)
        if rel.parts[:2] == ('engines', 'mlx-serve'):
            continue  # Public upstream is independently pinned/audited; its gitlink is not vendored code.
        if relative not in tracked and any(x in {'.git', 'runtime', 'build', '__pycache__', 'node_modules', '.venv'} for x in rel.parts):
            continue
        if not p.is_file():
            continue
        if relative not in tracked and ('.local.' in p.name or p.name.endswith('.local.json')):
            continue
        findings.extend(scan_blob(relative, p.read_bytes(), compiled, forbidden, images))
    return findings


def scan_index(root=ROOT):
    """Read the exact staged bytes, including additions absent from HEAD."""
    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args])
    compiled, forbidden, images = compiled_rules(root), forbidden_literals(root), reviewed_images(root)
    findings = []
    rows = git('ls-files', '--stage', '-z').split(b'\0')
    batch = subprocess.Popen(['git', '--no-replace-objects', '-C', str(root), 'cat-file', '--batch'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for row in rows:
            if not row:continue
            meta, name = row.split(b'\t', 1)
            mode, oid, stage = meta.split()
            relative = name.decode('utf-8')
            if stage != b'0':
                findings.append((relative, 0, 'unmerged index entry'));continue
            if mode == b'160000':
                if relative != 'engines/mlx-serve' or oid != b'6991ba3ad3d356891d43211dec941cc6f9ff6dd6':
                    findings.append((relative, 0, 'unreviewed submodule'))
                continue
            batch.stdin.write(oid + b'\n');batch.stdin.flush()
            header = batch.stdout.readline().split()
            if len(header) != 3 or header[1] != b'blob':raise ValueError('Invalid staged blob')
            raw = batch.stdout.read(int(header[2]))
            if len(raw) != int(header[2]) or batch.stdout.read(1) != b'\n':raise ValueError('Incomplete staged blob')
            findings.extend(scan_blob(relative, raw, compiled, forbidden, images))
    finally:
        batch.stdin.close();batch.stdout.close();batch.wait()
    return findings


def scan_identities(root=ROOT):
    findings = []
    expected = r'Front Eagle <fronteagle739@users\.noreply\.github\.com> \d+ [+-]\d{4}'
    for variable in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
        identity = subprocess.check_output(['git', '-C', str(root), 'var', variable]).decode().strip()
        if not re.fullmatch(expected, identity):
            findings.append((variable, 0, 'commit requires Front Eagle identity'))
    return findings


def scan_history(root=ROOT, revision='HEAD'):
    """Audit reachable commit metadata and every historical tree/blob, not other refs."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    tip = git('rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()
    commits = git('rev-list', tip).decode().splitlines()
    compiled, forbidden, images = compiled_rules(root), forbidden_literals(root), reviewed_images(root)
    findings, seen = [], set()
    for commit in commits:
        raw = git('cat-file', 'commit', commit)
        headers, _, message = raw.partition(b'\n\n')
        identities = [line for line in headers.decode().splitlines() if line.startswith(('author ', 'committer '))]
        expected = r'(?:author|committer) Front Eagle <fronteagle739@users\.noreply\.github\.com> \d+ [+-]\d{4}'
        if len(identities) != 2 or not all(re.fullmatch(expected, line) for line in identities):
            findings.append((commit[:12], 0, 'history requires Front Eagle author and committer'))
        findings.extend(scan_blob(commit[:12] + '/commit-message', message, compiled, forbidden, {}))
    if findings:
        return findings  # Reject metadata before reading a potentially large private history.
    batch = subprocess.Popen(['git', '-C', str(root), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for commit in commits:
            for row in git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
                if not row:continue
                metadata, name = row.split(b'\t', 1)
                mode, kind, oid = metadata.split()
                relative = name.decode('utf-8')
                if kind == b'commit':
                    if relative != 'engines/mlx-serve' or oid != b'6991ba3ad3d356891d43211dec941cc6f9ff6dd6':
                        findings.append((relative, 0, 'unreviewed submodule'))
                    continue
                key = (oid, relative)
                if key in seen:continue
                seen.add(key)
                batch.stdin.write(oid + b'\n');batch.stdin.flush()
                header = batch.stdout.readline().split()
                if len(header) != 3 or header[1] != b'blob':raise ValueError('Invalid historical blob')
                size = int(header[2])
                raw = batch.stdout.read(size)
                if len(raw) != size or batch.stdout.read(1) != b'\n':raise ValueError('Incomplete historical blob')
                findings.extend((commit[:12] + ':' + path, line, rule) for path, line, rule in scan_blob(relative, raw, compiled, forbidden, images))
    finally:
        batch.stdin.close();batch.stdout.close();batch.wait()
    return findings


def scan_generated_reference():
    """Keep the variable reference at its published documentation path."""
    sys.path.insert(0, str(ROOT))
    from slstack.envdocs import render, reference_table
    reference = ROOT / 'docs' / 'VARIABLES.md'
    if not reference.is_file() or reference.read_text(encoding='utf-8') != render(reference_table()):
        return [('docs/VARIABLES.md', 0, 'regenerate with python -B -m slstack.envdocs')]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--history', metavar='REF', help='also audit all source and commit metadata reachable from this release ref')
    parser.add_argument('--staged', action='store_true', help='audit exact index bytes instead of working files')
    parser.add_argument('--identities', action='store_true', help='require Front Eagle author and committer')
    parser.add_argument('--commit-message', type=Path, help='audit a pending commit message')
    args = parser.parse_args()
    found = scan_index() if args.staged else scan()
    if not args.staged:
        found.extend(scan_generated_reference())
    if args.identities:
        found.extend(scan_identities())
    if args.commit_message:
        found.extend(scan_blob('commit-message', args.commit_message.read_bytes(), compiled_rules(ROOT), forbidden_literals(ROOT), {}))
    if args.history:
        found.extend(scan_history(revision=args.history))
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
