"""Bounded full-vocabulary continuation comparison; explicit test instrumentation."""
import argparse,datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from slstack.model_compatibility import continuation_kl_files
from slstack.journal import atomic_json

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--vocab',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--control-skip',type=int,default=1);p.add_argument('--candidate-skip',type=int,default=1)
    p.add_argument('--reference-label',required=True);p.add_argument('--progress-log',type=Path)
    a=p.parse_args()
    def progress(done,total,mean,top1):
        if a.progress_log and (done%16==0 or done==total):
            line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing=full-vocabulary quality measurement; latest={done}/{total} computed rows,meanKL={mean:.9g},top1={top1:.6f}; next=remaining matched-prefix rows'
            with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
    result=continuation_kl_files(a.control,a.candidate,a.vocab,a.control_skip,a.candidate_skip,progress)
    result['direction']=a.reference_label
    atomic_json(a.out,result)
    print(json.dumps({k:result[k] for k in ('rows','vocab','mean_nats','max_nats','top1','direction')}),flush=True)

if __name__=='__main__':main()
