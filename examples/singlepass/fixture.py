"""Write the frozen recall fixture used by the paired prefill measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def tokens(count):
    seed = json.loads(Path(__file__).with_name('fixture.json').read_text(encoding='utf-8'))
    head, body, tail = (seed[k] for k in ('head', 'body', 'tail'))
    need = count-len(head)-len(tail)
    if need < 0 or count > 1_000_000:
        raise ValueError('Fixture size must be between 192 and 1,000,000 tokens')
    return head+(body*(need//len(body)+1))[:need]+tail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ids = tokens(a.count)
    raw = struct.pack('<'+'I'*len(ids), *ids)
    with a.output.open('xb') as target:
        target.write(raw)
    print(json.dumps(dict(tokens=len(ids), sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == '__main__':
    main()
