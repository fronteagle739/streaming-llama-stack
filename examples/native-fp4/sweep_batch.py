"""Matched 100K ultra-cone expert batching experiment on the lossy MXFP4 proxy.

Build the ABI4 bridge and the complete modified producer first. Supply a config
with explicit host/physical/device limits and the verified proxy model receipt.
Each child uses speed_proxy.py's shared GPU lease, paging guard and dispatch
witnesses. No production preset or model file is modified.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--tokens',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--continuation',type=Path)
    parser.add_argument('--profile-stages',action='store_true')
    args=parser.parse_args()
    if args.tokens.stat().st_size!=100000*4:
        parser.error('Supply exactly 100000 little-endian uint32 prompt tokens')
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('producer_profile','ultra')!='ultra' or cfg.get('context',0)<100008:
        parser.error('ULTRA and context >=100008 are required')
    cfg['yoco_keep']=2541
    cfg.setdefault('env_overrides',{}).update(LLAMA_DSV41_FLASHINFER_MXFP4='1')
    for tokens in [4096,8192]:
        candidate=copy.deepcopy(cfg)
        candidate['env_overrides']['LLAMA_DSV41_FLASHINFER_BATCH_TOKENS']=str(tokens)
        name=str(args.output_prefix)+'-'+str(tokens)
        config=Path(name+'-config.json');output=Path(name+'.json')
        if config.exists() or output.exists():
            raise FileExistsError('Preserve existing experiment artifacts: '+name)
        config.write_text(json.dumps(candidate,indent=2)+'\n',encoding='utf-8')
        command=[sys.executable,'-B',str(Path(__file__).with_name('speed_proxy.py')),
                 '--config',str(config),'--tokens',str(args.tokens),'--arm','proxy','--output',str(output)]
        if args.continuation:command+=['--continuation',str(args.continuation)]
        if args.profile_stages:command+=['--profile-stages']
        subprocess.run(command,check=True)


if __name__=='__main__':main()
