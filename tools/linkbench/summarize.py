"""Turns linkbench RESULT lines (one or more files) into a markdown table:
one row per label, p50/p99 per size. Multiple files: median of the per-run p50/p99."""
import re, sys, statistics
from collections import defaultdict, OrderedDict
data = OrderedDict()
for path in sys.argv[1:]:
    for line in open(path, encoding='utf-8', errors='replace'):
        if not line.startswith('RESULT'):
            continue
        kv = dict(re.findall(r'(\w+)=(\S+)', line))
        d = data.setdefault(kv['label'], defaultdict(list))
        d[int(kv['size'])].append((float(kv['p50_us']), float(kv['p99_us']), int(kv['retx']), int(kv['lost'])))
sizes = sorted({s for d in data.values() for s in d})
print('| config | ' + ' | '.join(f'{s//1024} KB p50 / p99 (us)' for s in sizes) + ' | runs |')
print('|---|' + '---:|' * len(sizes) + '---:|')
for label, d in data.items():
    cells = []
    for s in sizes:
        v = d.get(s)
        if not v:
            cells.append('-'); continue
        p50 = statistics.median(x[0] for x in v); p99 = statistics.median(x[1] for x in v)
        retx = sum(x[2] for x in v)
        cells.append(f'{p50:.0f} / {p99:.0f}' + (f' (retx {retx})' if retx else ''))
    print(f'| {label} | ' + ' | '.join(cells) + f' | {max(len(v) for v in d.values())} |')
