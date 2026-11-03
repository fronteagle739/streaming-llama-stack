"""Reconstruct the audited public-novel/synthetic-password 100K benchmark.

The reconstruction reproduces an archived fixture, not its unknown original
generator. Supply pinned corpus and tokenizer files; this script downloads
nothing, invokes no engine, and preserves the measured CRLF/LF distinction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

CORPUS_REPO='https://github.com/antirez/ds4'
CORPUS_REVISION='c97964b8aa77034a8e17afb10ca7a686c40ab511'
CORPUS_PATH='speed-bench/promessi_sposi.txt'
CORPUS_SHA256='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
TOKENIZER_SHA256='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
TOKENIZER_REPO='nvidia/Qwen3.8-Flash-Next-NVFP4'
TOKENIZER_REVISION='fc694b54fb0174e0913e6adf86691ef85a4ead47'
FIXTURE_SHA256='87fb9af861237f91724d56ca9c1a29f77c0e4aadce673ff939abe8500fa6c936'
HEAD=('<|im_start|>user\r\nRemember this password, you will be asked for it at the end: '
      'the password is AMBER-OTTER-731.\r\n\r\nHere is a long text to read:\r\n\r\n')
TAIL=('\n\nWhat is the password I gave you at the start? State the password first, '
      'then write a detailed essay of at least one thousand words about how the '
      'characters and setting develop in the long text. Explain with examples '
      'in continuous prose. Do not end your answer early.<|im_end|>\n'
      '<|im_start|>assistant\n<think>\n\n</think>\n\n')

def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('corpus','tokenizer','output','report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True,help='Existing writable scratch root for the memory guardian')
    p.add_argument('--host-budget-gib',type=float,required=True)
    p.add_argument('--runtime-floor-gib',type=float,default=4)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.host_budget_gib<=0 or a.runtime_floor_gib<4:p.error('Explicit positive host budget and at least4GiB free physical floor required')
    for key in ('corpus','tokenizer','output','report','scratch'):setattr(a,key,getattr(a,key).resolve())
    if not a.scratch.is_dir():p.error('Scratch root must already exist')
    if not a.worker:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from slstack.guard import supervise
        import psutil
        args=[sys.executable,'-B',str(Path(__file__).resolve())]
        for key in ('corpus','tokenizer','output','report','scratch'):args+=['--'+key,str(getattr(a,key))]
        args+=['--host-budget-gib',str(a.host_budget_gib),'--runtime-floor-gib',str(a.runtime_floor_gib),'--worker']
        env=dict(os.environ,TEMP=str(a.scratch),TMP=str(a.scratch),TMPDIR=str(a.scratch),
                 PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 QFN_FIXTURE_PARENT=str(os.getpid()))
        return supervise(dict(command=args,cwd=str(Path(__file__).resolve().parents[1]),env=env,
            host_budget_gib=a.host_budget_gib,job_commit_gib=a.host_budget_gib,min_available_gib=a.runtime_floor_gib,
            timeout_s=300,owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),
            runtime_root=str(a.scratch),stop=str(a.report)+'.stop',log=str(a.report)+'.log',receipt=str(a.report)+'.memory.json'))
    import psutil
    if os.environ.get('QFN_FIXTURE_PARENT')!=str(psutil.Process().ppid()):
        raise ValueError('Internal worker must be launched by its memory supervisor')
    if a.output.exists() or a.report.exists():p.error('Output/report already exists; preserve existing evidence')
    corpus=a.corpus.read_text(encoding='utf8')
    normalized=hashlib.sha256(corpus.encode()).hexdigest()
    if normalized!=CORPUS_SHA256:raise ValueError('Pinned public corpus SHA256 mismatch after LF normalization')
    tokenizer_hash=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash!=TOKENIZER_SHA256:raise ValueError('Pinned NVIDIA tokenizer SHA256 mismatch')
    import tokenizers
    tokenizer=tokenizers.Tokenizer.from_file(str(a.tokenizer))
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
    head,tail=encode(HEAD),encode(TAIL)
    # Original source preparation read text with universal newlines, selected
    # 454096 characters, then wrote that selection using Windows CRLF.
    body=encode(corpus[:454096].replace('\n','\r\n'))
    count=100000-len(head)-len(tail)
    if len(body)<count:raise ValueError('Pinned corpus prefix is too short')
    ids=head+body[:count]+tail
    raw=struct.pack('<100000i',*ids)
    identity=hashlib.sha256(raw).hexdigest()
    if identity!=FIXTURE_SHA256:raise ValueError('Exact archived fixture SHA256 mismatch; no output written')
    with a.output.open('xb') as stream:stream.write(raw)
    report=dict(verified=True,tokens=100000,bytes=len(raw),sha256=identity,
                header_tokens=len(head),body_tokens=count,tail_tokens=len(tail),
                corpus=dict(repository=CORPUS_REPO,revision=CORPUS_REVISION,path=CORPUS_PATH,
                            normalized_lf_sha256=normalized),tokenizer_sha256=tokenizer_hash,tokenizer_repository=TOKENIZER_REPO,tokenizer_revision=TOKENIZER_REVISION,
                tokenizers_version=tokenizers.__version__,
                construction='audited deterministic reconstruction of archived measured fixture',
                engine_started=False,natural_eog_policy='benchmark preserves natural EOG and stop behavior')
    with a.report.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':raise SystemExit(main())
