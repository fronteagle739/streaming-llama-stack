"""Run the GLM external-state ownership check with stack memory/GPU guards."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from slstack.config import load
from slstack.process import OwnedProcess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','model','library','cuda-runtime','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();cfg=load(args.config)
    if not (0<cfg['host_budget_gib']<=3 and 0<cfg['job_commit_gib']<=32):
        raise ValueError('Require explicit RSS <=3 GiB and WDDM commit <=32 GiB caps')
    if cfg['min_available_gib']<4 or cfg['runtime_min_free_gib']<4:
        raise ValueError('Require a 4 GiB available-RAM floor')
    if not cfg.get('gpu_uuid') or not cfg.get('hold_file'):
        raise ValueError('Require selected GPU and shared lease ledger')
    output=args.output.resolve()
    if not output.is_relative_to(Path(cfg['runtime_root']).resolve()):
        raise ValueError('Output must remain under runtime_root')
    if output.exists():raise ValueError('Output already exists')
    command=[sys.executable,'-B',str(Path(__file__).with_name('test_glm_external.py'))]
    for name in ('model','library','cuda_runtime','output'):
        command.extend(['--'+name.replace('_','-'),str(getattr(args,name).resolve())])
    job=OwnedProcess(cfg,command,'glm53-external-state',gpu=True,scope='ds4-win')
    receipt=job.wait();result=json.loads(output.read_text())
    if receipt['exit_code'] or result['status']!='PASS':
        raise RuntimeError('Native ownership check failed')
    result['guard']={k:receipt[k] for k in ('host_budget_gib','job_commit_gib','floor_gib',
        'peak_tree_rss','min_available','cleanup_confirmed')}
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
