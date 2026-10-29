"""Run an E19 command under shared GPU ownership and explicit host/paging limits.

Example: python -B examples/kernels2/run_guarded.py --config CONFIG --host-gib 5
--commit-gib 32 --name official-layer -- python -B examples/kernels2/official_layer.py ...
CONFIG must put runtime_root and every dependency/build/cache beneath task scratch.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from slstack.config import load
from slstack.process import OwnedProcess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--host-gib',type=float,required=True)
    p.add_argument('--commit-gib',type=float,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('A guarded command is required')
    cfg=load(overrides={**json.loads(a.config.read_text()),'host_budget_gib':a.host_gib,
        'job_commit_gib':a.commit_gib,'min_available_gib':.5,'runtime_min_free_gib':.5,
        'producer_desktop_reserve_gib':.5,'gpu_wait_seconds':600,'lease_minutes':10})
    job=OwnedProcess(cfg,command,a.name,gpu=not a.cpu,scope='rtx-producer',telemetry=True)
    try:
        receipt=job.wait()
        if receipt['exit_code'] or receipt.get('reason') or not receipt.get('cleanup_confirmed'):
            raise RuntimeError('Guarded command did not complete cleanly: '+str(job.receipt))
        print('PASS; receipt:',job.receipt)
    finally:
        if job.log.exists():print(job.log.read_text(errors='replace')[-4000:])

if __name__=='__main__':main()
