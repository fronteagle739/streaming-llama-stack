"""Generate the variable reference and literal default evidence from the environment table."""
import argparse
import ast
import json
from pathlib import Path
import re
from .envvars import ROOT, TABLE, inventory, rows


def source_defaults(table):
    cache = {}
    python_calls = {}
    for row in table:
        evidence = []
        pinned = {item['source']: item for item in row.get('code_defaults', [])
                  if item['source'].startswith('engines/mlx-serve/')}
        for reference in row['reads']:
            if reference in pinned:
                evidence.append(pinned[reference])
                continue
            relative, number = reference.rsplit(':', 1)
            file = ROOT / relative
            if not file.is_file():
                continue
            if relative not in cache:
                text = file.read_text(encoding='utf-8')
                cache[relative] = text.splitlines()
                if file.suffix == '.py':
                    python_calls[relative] = [n for n in ast.walk(ast.parse(text))
                                              if isinstance(n, (ast.Call, ast.Subscript))]
            lines = cache[relative]
            line = int(number) - 1
            if line >= len(lines):
                continue
            exact = None
            default = None
            for node in python_calls.get(relative, []):
                if node.lineno != line + 1:
                    continue
                if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == row['var']:
                    exact = ast.unparse(node)
                    if ast.unparse(node.func).endswith(('getenv', 'environ.get', 'environ.setdefault')):
                        default = ast.unparse(node.args[1]) if len(node.args) > 1 else 'None'
                    break
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == row['var']:
                    exact = ast.unparse(node)
                    default = ('Assigned by the runtime; see literal consumer' if isinstance(node.ctx, ast.Store)
                               else 'KeyError unless an earlier assignment supplies the variable')
            consumer = '\n'.join(lines[max(0, line - 2):line + 5])
            if default is None:
                default = ('NULL from getenv; the literal consumer below supplies any fallback'
                           if re.search(r'\b(?:std::)?getenv\s*\(', lines[line]) else
                           'Wrapper/conditional default: see the literal call and consumer below')
            evidence.append({'source': reference, 'unset': default,
                             'read': exact or lines[line].strip(), 'consumer': consumer})
        row['code_defaults'] = evidence
        row['code_default'] = '; '.join(dict.fromkeys(e['unset'] for e in evidence)) or 'Launcher/runtime supplied; no source read in this checkout'
    return table


def render(table):
    from .native_broker import vendor
    vendor()
    from studio.quality_speed import NAMED, PHASES
    out = ['# Environment variables', '',
           'The launcher constructs each child environment. No exports are needed. '
           'Precedence is `--flag` > `env_overrides` in the config > the DSV4.1 table default > the engine default. '
           'Inherited engine settings are removed with a warning. OS identity, locale and toolchain paths keep their normal meaning.', '',
           'Each row below comes from `envvars.json`, which also supplies the CLI flags and child defaults. '
           'Use `python -m slstack.envdocs --check` to verify this reference and the source inventory.', '',
           '`slstack install --all` installs the pinned isolated media runtime and builds the native engines and chat UI. '
           'The UI builds before the server embeds its generated assets. Media workers use the isolated runtime automatically; '
           'an explicit `studio_python` setting still overrides that choice. `install --all --check` checks packages and build outputs without rebuilding.', '',
           '## Prefill modes', '',
           "`slstack rtx --mode accurate|fast|ultra|hyper` selects Studio's delivered RTX prefill arithmetic profile. "
           'The default is accurate. Image/video residency uses separate `--image-mode streaming|resident` and '
           '`--video-mode streaming|resident` flags. All four profiles use locked RAM carry with no pageable resident copy.', '',
           'Accurate preserves the original default arithmetic; Fast enables the delivered BF16 indexer and split-F32 scores. '
           'Ultra also uses the delivered reduced-precision expert/dense/attention paths. Hyper adds transient FP4 weight/activation '
           'conversion on the GPU. Hyper never writes a requantized model file; selecting it explicitly opts into those arithmetic changes. '
           "These profile definitions are separate from the bias tool and do not imply a new build has passed every mode's parity or performance test.", '',
           '## Quality and speed', '',
           'The chat model can call `set_quality_speed` and `get_quality_speed`. '
           'The router executes these tools against the native control endpoint. '
           'Modes above Balanced are faster but may produce worse answers.', '',
           '| Setting | Bias |', '|---|---|',
           *[f'| `{name}` | {value:g} |' for name, value in NAMED.items()], '',
           'Use `value` for one setting, or provide independent ' +
           ', '.join(f'`{name}`' for name in PHASES) + ' arguments. '
           'Partial phase updates preserve the other settings. Numeric values must be finite and in [0,1.36125].', '',
           'Tool-call examples:', '', '```json',
           '{"name":"set_quality_speed","arguments":{"value":"balanced"}}',
           '{"name":"set_quality_speed","arguments":{"reasoning":"fast","answer":"balanced","code":"max_quality"}}',
           '{"name":"get_quality_speed","arguments":{}}', '```', '',
           'Native controls take effect at token boundaries. The response reports requested and applied '
           'values, revision, phase counters and the latest throughput window. Its scope field describes '
           'whether the backend applies the setting server-wide. An unsupported backend must report '
           'that capability explicitly. Short throughput windows are observations, not benchmark guarantees.', '',
           '## Launcher settings', '',
           '`runtime_min_free_gib` and `--runtime-min-free-gib` default to 0.5 GiB '
           'of available physical RAM during execution; 0 disables this floor. '
           'Admission still requires the declared host budget plus its reserve. '
           'The host RSS budget and paging stop remain enabled. Shutdown requests cooperative '
           'cancellation first, waits while diagnostics or memory reclamation advance, and '
           'allows a logged force kill only after a full quiet grace period.', '',
           '| Flag | Scope and DSV4.1 default | Purpose |', '|---|---|---|']
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for row in table:
        if row['flag']:
            values = '; '.join(f'{s}: `{v}`' for s, v in row.get('defaults_by_scope', {}).items()) or 'Explicit runtime value'
            out.append(f"| `{row['flag']}` | {cell(values)} | {cell(row['doc'])} |")
    out += ['', 'Budget placeholders use the admitted child budget: `{host_budget_bytes}`, '
            '`{host_budget_mb}`, `{physical_floor_bytes}` (admission) and `{runtime_floor_bytes}` (runtime). `{host_budget_gib}` and '
            '`{min_available_gib}` use GiB. User overrides are literal strings, including JSON and braces.', '',
            'Hardware commands use the pinned shared ownership/paging guard. Model operations wait while progress advances; only stalled work expires. '
            'It enforces the physical budget/floor and a separate Windows device-backing commit allowance. '
            'Windows owned-page observations are bounded and incremental; Darwin disk-only process attribution is unavailable. '
            'Zero observed pages does not establish exhaustive zero pagefile I/O. Resource leases remain held if owned cleanup cannot be confirmed.', '',
            'Examples (edit paths in the config first):', '', '```text',
            'slstack rtx --config config.json --llama-dsv41-topk-radix 1',
            'slstack rtx --config config.json --ds4-cuda-streaming-expert-cache-reserve-gb 2',
            'slstack mac --config config.json --ds4-metal-disable-v41-layer-prefill 1',
            'slstack rtx --config config.json --mem-guard-min-free-gb 4', '```', '',
            'The core-eviction flag must remain 0 and the wired Metal flag must remain 1. '
            'Requests to disable these residency requirements fail before an engine starts.', '',
            '## Advanced source inventory', '',
            'Variables without a launcher flag retain their built-in default. The literal read and immediate '
            'consumer are recorded below, including conditional and wrapper defaults. '
            'A null/None read is distinct from the fallback used by its consumer. '
            'Runtime-provided paths and secrets are described by purpose; values from a running machine are never copied here.', '']
    for row in table:
        out += [f"### {row['var']}", '', row['doc'], '',
                f"Scope: {', '.join(row['scopes'])}. Class: {row['classification']}.", '']
        for item in row.get('code_defaults', []):
            out += [f"Source: `{item['source']}`. Unset read: {item['unset']}.", '', '```text', item['consumer'], '```', '']
    return '\n'.join(out).rstrip() + '\n'


def reference_table():
    """Resolve source evidence without changing runtime flags or defaults."""
    table = rows()
    references = inventory()
    declared = {row['var'] for row in table}
    missing = set(references) - declared
    if missing or len(declared) != len(table) or any(not row['doc'] for row in table):
        raise ValueError(f'Environment reference has missing or invalid rows: {sorted(missing)}')
    # Source locations and literal consumers are documentation metadata. Derive
    # them from this checkout without rewriting the runtime environment table.
    for row in table:
        # The separately pinned upstream gitlink may be uninitialized. Its
        # checked-in source evidence stays valid at that immutable revision.
        external = [ref for ref in row['reads'] if ref.startswith('engines/mlx-serve/')]
        row['reads'] = list(dict.fromkeys(references.get(row['var'], []) + external))
    return source_defaults(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-defaults', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    table = reference_table()
    if args.refresh_defaults:
        TABLE.write_text(json.dumps(table, indent=2) + '\n', encoding='utf-8')
    expected = render(table)
    guide = ROOT / 'docs' / 'VARIABLES.md'
    if args.check:
        if not guide.is_file() or guide.read_text(encoding='utf-8') != expected:
            raise ValueError('docs/VARIABLES.md differs from envvars.json; regenerate it')
        print(f'Environment table/reference: {len(table)} documented variables')
    else:
        guide.write_text(expected, encoding='utf-8')


if __name__ == '__main__':
    main()
