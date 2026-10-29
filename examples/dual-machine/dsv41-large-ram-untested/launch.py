"""UNTESTED example: RTX ULTRA exact-cone export to a large-RAM ds4-win box.

Only --dry-run is implemented: the CPU mode3/YOCO receiver is missing.
No model, runtime directory, network connection or subprocess is opened.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

GAP = (
    'ds4-win has native V4.1 import for GPU modes 1/2, but lacks the '
    'RAM-resident CPU receiver for mode3 DSpark state and exact cone v2 used '
    'by the existing RTX-to-Mac handoff. See this example README.'
)


def plan(machine, config):
    from slstack.envvars import redact_cfg
    from slstack.combo_producer import prefill_arguments
    rtx = config['rtx']
    decode = config['decode_plan']
    if (rtx['producer_profile'], rtx['native_tier'], rtx['yoco_keep']) != ('ultra', 'ultra', 2541):
        raise ValueError('This example requires ULTRA exact cone2541')
    if (decode['native_state_mode'], decode['yoco_state_version'], decode['yoco_keep']) != (3, 2, 2541):
        raise ValueError('This example requires mode3 and exact cone v2')
    for cfg in (rtx, decode):
        budget = float(cfg['host_budget_gib'])
        floor = float(cfg['runtime_min_free_gib' if cfg is rtx else 'min_available_gib'])
        if not math.isfinite(budget) or budget <= 0 or not math.isfinite(floor) or floor < .5:
            raise ValueError('Set a finite positive host budget and at least 0.5 GiB runtime free-RAM floor')
    # Config/argv preview only: a representative 50K request and hypothetical
    # available RAM, not a physical-memory admission or an engine launch.
    flags, _ = prefill_arguments(rtx, 49787, int((rtx['host_budget_gib'] + .5) * 2**30), export_state=True)
    return {
        'status': 'UNTESTED example',
        'launchable': False,
        'machine': machine,
        'gap': GAP,
        'config': redact_cfg(rtx if machine == 'rtx' else decode),
        # This is the existing broker command, usable only after a compatible
        # receiver exists. The effective config is deliberately never written.
        'proposed_argv': [sys.executable, '-B', str(ROOT / 'scripts/bootstrap.py'),
                          '--config', str(Path(rtx['runtime_root']) / 'rtx.local.json'), 'rtx']
                         if machine == 'rtx' else None,
        'state_export': {'builder': 'slstack.combo_producer.prefill_arguments',
                         'export_state': True, 'mode': 3,
                         'preview_prompt_tokens': 49787, 'flags': flags},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True, choices=('rtx', 'decode'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.example.json'),
                        help='Copy the example to a private *.local.json and edit both machine sections')
    parser.add_argument('--dry-run', action='store_true', help='Print the proposal; start nothing and write nothing')
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error('UNTESTED example cannot launch: ' + GAP)
    try:
        result = plan(args.machine, json.loads(args.config.read_text(encoding='utf-8-sig')))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
