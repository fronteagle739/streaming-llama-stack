"""Run one bounded QFN task step through Studio's independent RAM/paging guard."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/studio'))
from studio.job_guard import GuardedJob
from studio.job_paging import OWNED_PAGING_LIMIT


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--commit-gib',type=float,required=True)
    p.add_argument('--device-commit-gib',type=float,default=0)
    p.add_argument('--floor-gib',type=float,required=True)
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--progress-log',type=Path,required=True)
    p.add_argument('--stream-log',action='store_true')
    p.add_argument('--blas-threads',type=int,default=1)
    p.add_argument('--cwd',type=Path,default=ROOT)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if a.blas_threads<1:p.error('--blas-threads must be positive')
    if not cmd:p.error('command required')
    if not a.label.replace('-','').replace('_','').isalnum():p.error('invalid label')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    tmp=root/'tmp';tmp.mkdir(exist_ok=True)
    def progress(doing,latest,next_step):
        line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
        with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
        print(line,flush=True)
    env=dict(os.environ)
    for name in ('TEMP','TMP','TMPDIR','CUDA_CACHE_PATH'):env[name]=str(tmp)
    env['PYTHONDONTWRITEBYTECODE']='1'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]=str(a.blas_threads)
    receipt=root/(a.label+'.guard.json')
    guard=GuardedJob(rss_limit=a.rss_gib*2**30,commit_limit=a.commit_gib*2**30,
        device_commit_allowance=a.device_commit_gib*2**30,floor=a.floor_gib*2**30,
        seconds=a.seconds,receipt=receipt,owned_paging_limit=OWNED_PAGING_LIMIT)
    progress(a.label,'bounded guarded step starting','inspect immutable receipt')
    with (root/(a.label+'.log')).open('xb') as log:
        guard.start(cmd,cwd=a.cwd.resolve(strict=True),env=env,stdout=log)
        done=threading.Event();worker=None
        if a.stream_log:
            def tail():
                with (root/(a.label+'.log')).open('r',encoding='utf8',errors='replace') as reader:
                    pending=''
                    while True:
                        piece=reader.read(65536)
                        if piece:
                            pending+=piece
                            while '\n' in pending:
                                line,pending=pending.split('\n',1);print(line.rstrip('\r'),flush=True)
                        elif done.is_set():
                            if pending:print(pending,flush=True)
                            break
                        else:done.wait(.2)
            worker=threading.Thread(target=tail);worker.start()
        try:code=guard.wait()
        finally:
            done.set()
            if worker:worker.join(timeout=5)
    row=json.loads(receipt.read_text())
    progress(a.label,json.dumps({k:row.get(k) for k in ('command_exit_code','failure','release_allowed','peak_rss','minimum_available','peak_owned_paged_bytes')}),'inspect output and next step')
    return code


if __name__=='__main__':sys.exit(main())
