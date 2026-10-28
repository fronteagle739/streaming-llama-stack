"""Compare DirectStorage layer-major GLM graphs with the resident CPU oracle.
Generate fixtures with KDA head dimension 32 (128 convolution channels).
This is a numerical check, not a trained-model benchmark.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from slstack.config import load
from slstack.process import OwnedProcess
from run_glm53_cpu import logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'models', 'cpu-binary', 'cuda-binary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument("--unsupported-model",type=Path)
    args = parser.parse_args()
    cfg = load(args.config)
    if not (0 < cfg['host_budget_gib'] <= 3 and 0 < cfg['job_commit_gib'] <= 8):
        raise ValueError('Require explicit RSS <=3 GiB and commit <=8 GiB budgets')
    if cfg['min_available_gib'] < 4 or cfg['runtime_min_free_gib'] < 4:
        raise ValueError('Require a 4 GiB physical memory floor')
    if not cfg.get('gpu_uuid') or not cfg.get('hold_file'):
        raise ValueError('Require selected GPU and shared lease ledger')
    output=args.output.resolve()
    if not output.is_relative_to(Path(cfg['runtime_root']).resolve()):
        raise ValueError('Output must remain under runtime_root')
    output.mkdir(parents=True,exist_ok=False)
    env=dict(LLAMA_PREFILL_WF='1', LLAMA_WF_MODE='dstorage',
        LLAMA_PREFILL_LAYER_MAJOR='1', LLAMA_PREFILL_HIDDEN_MODE='cpu',
        LLAMA_PREFILL_KV_MODE='gpu', LLAMA_WF_GPU_BUFS='1', LLAMA_WF_RAM_BUFS='2',
        LLAMA_WF_PREFETCH_DEPTH='1', LLAMA_WF_LAYER_SPLITS='1',
        LLAMA_WF_DS_DIRECT='1', LLAMA_NO_HOST='1')
    rows=[]
    for model in ('ds4-full','up-full','up-shared'):
        for n,ub in (((64,64),) if model=='up-shared' else ((64,64),(65,16),(320,16))):
            ref=None
            for mode in ('cpu','stream-all','stream-last'):
                gpu=mode!='cpu';last=mode=='stream-last'
                label=f'{model}-{n}-{ub}-{mode}';target=output/(label+'.bin')
                conf={**cfg,'env_overrides':{**cfg['env_overrides'],**(env if gpu else {})}}
                command=[str((args.cuda_binary if gpu else args.cpu_binary).resolve()),
                    str(args.models.resolve()/(model+'.gguf')),str(target),str(n),str(ub),
                    '8','0','1234','0','0','-','0',str(int(gpu)),str(int(last))]
                job=OwnedProcess(conf,command,label,gpu=gpu,scope='rtx-producer' if gpu else 'studio')
                rec=job.wait()
                if rec['exit_code'] != 0 or rec.get('reason'):
                    raise RuntimeError('Guarded child failed: '+label)
                actual=logits(target,9 if last else n+8)
                if not gpu:ref=actual;continue
                expected=ref[n-1:] if last else ref
                delta=float(np.abs(actual-expected).max())
                agree=int((actual.argmax(1)==expected.argmax(1)).sum())
                row=dict(model=model,prompt=n,microbatch=ub,last_only=last,rows=len(actual),
                    max_abs=delta,top_agree=agree,peak_rss=rec['peak_tree_rss'],
                    min_available=rec['min_available'])
                rows.append(row)
                (output/'results.json').write_text(json.dumps(rows,indent=2))
                if delta>=1e-2 or agree!=len(actual):
                    raise ValueError('CPU oracle mismatch: '+json.dumps(row))
                print('PASS '+json.dumps(row),flush=True)
    refused=[]
    controls=[(args.models.resolve()/'up-shared.gguf',64,16,'shared-indexer boundaries are not supported')]
    if args.unsupported_model:
        controls.append((args.unsupported_model.resolve(),64,64,'projection divisible by 128'))
    for model,n,ub,marker in controls:
        conf={**cfg,'env_overrides':{**cfg['env_overrides'],**env}}
        command=[str(args.cuda_binary.resolve()),str(model),str(output/'refused.bin'),
            str(n),str(ub),'0','0','1234','0','0','-','0','1']
        job=OwnedProcess(conf,command,'glm53-refusal',gpu=True,scope='rtx-producer')
        try:
            rec=job.wait()
        except RuntimeError:
            if not job.receipt.is_file():raise
            rec=json.loads(job.receipt.read_text())
        if rec.get('reason') or rec['exit_code']==0 or marker not in job.log.read_text(errors='replace') or not rec['cleanup_confirmed']:
            raise ValueError('Unsupported streaming case did not refuse cleanly')
        refused.append(dict(model=model.name,reason=marker,cleanup_confirmed=True))
    (output/'refusals.json').write_text(json.dumps(refused,indent=2))
    print('PASS: 14 streamed comparisons against seven CPU references; '+str(len(refused))+' clean refusals',flush=True)


if __name__=='__main__':main()
