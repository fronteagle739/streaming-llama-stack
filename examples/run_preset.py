"""Portable example launcher. Local paths and credentials come from --config."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def merge(base,local):
    result=dict(base)
    for key,value in local.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
    return result


def prepare(args,local):
    if args.single:
        spec=importlib.util.spec_from_file_location('single_preset',ROOT/'scripts/dsv41_single.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.overlay(args.context,local,profile=args.single)
        result['webui']=bool(args.webui)
        return result,'start'
    recipe=ROOT/'examples/dual-machine'/args.preset/'preset.json'
    data=json.loads(recipe.read_text())[args.machine]
    result=merge(data,local)
    if args.machine=='rtx':
        result['webui']=True
        for key in ('native_tier','yoco_keep'):
            if key in data:result[key]=data[key]
        return result,'rtx'
    return result,'mac' if result.get('backend')=='ds4' else 'decode'


def missing_settings(value,prefix=''):
    if isinstance(value,dict):
        return [name for key,item in value.items() for name in missing_settings(item,prefix+'.'+key if prefix else key)]
    if isinstance(value,str) and (value.startswith(('SET_','MAC_INTERFACE_')) or 'MAC_HOST' in value):return [prefix]
    return []


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single',choices=('accurate','fast','ultra','ultra-yoco'))
    group.add_argument('--preset',choices=sorted(p.parent.name for p in (ROOT/'examples/dual-machine').glob('*/preset.json')))
    parser.add_argument('--machine',choices=('mac','rtx'))
    parser.add_argument('--context',choices=('262k','1m'),default='262k')
    parser.add_argument('--webui',action='store_true')
    parser.add_argument('--config',required=True,help='Your private config; never a file containing model weights')
    parser.add_argument('--dry-run',action='store_true',help='Show the generated config; start nothing and write nothing')
    parser.add_argument('--setup',action='store_true',help='Prepare the Mac MLX or native DS4 backend')
    parser.add_argument('--wheelhouse',help='Optional offline MLX dependency wheels')
    args=parser.parse_args(argv)
    if args.preset and not args.machine:parser.error('--machine is required with --preset')
    if args.setup and (args.single or args.machine!='mac'):parser.error('--setup applies to a Mac dual-machine preset')
    local=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    cfg,command=prepare(args,local)
    if args.setup:command='mac-setup' if cfg.get('backend')=='ds4' else 'setup-mlx'
    if args.wheelhouse and command!='setup-mlx':parser.error('--wheelhouse applies to MLX --setup')
    missing=missing_settings(cfg)
    if args.dry_run:
        from slstack.envvars import redact_cfg
        print(json.dumps(dict(command=command,config=redact_cfg(cfg),missing_local_settings=missing),indent=2));return 0
    if missing:parser.error('Set these fields in your private config: '+', '.join(missing))
    runtime=Path(cfg.get('runtime_root','runtime')).expanduser()
    runtime=(runtime if runtime.is_absolute() else ROOT/runtime).resolve()
    runtime.mkdir(parents=True,exist_ok=True)
    name=(args.single or args.preset)+'-'+(args.machine or 'single')
    effective=runtime/('example-'+name+'.local.json')
    effective.write_text(json.dumps(cfg,indent=2)+'\n')
    argv=[sys.executable,'-B',str(ROOT/'scripts/bootstrap.py'),'--config',str(effective),command]
    if args.wheelhouse:argv.extend(['--wheelhouse',str(Path(args.wheelhouse).resolve())])
    return subprocess.call(argv,cwd=ROOT)


if __name__=='__main__':raise SystemExit(main())
