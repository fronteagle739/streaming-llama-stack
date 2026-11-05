"""Build one stable QFN executable under a guard and pin its actual inputs.

The build command and read-only link inputs are explicit JSON argv/path lists.
A source edit cannot be qualified by stamping an old executable. The command
must rewrite the selected output, or reuse an already valid recorded build.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from slstack.paging_limits import shared_owned_paging_limit_bytes
from slstack.config import load
from slstack.journal import atomic_json,sha256
from slstack.model_compatibility import digest
from slstack.qfn_build import source_inventory,verified_build
from slstack.process import OwnedProcess


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('engine','manifest','runtime-root','input-files-json','command-json','cwd'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('host-budget-gib','commit-gib','floor-gib','seconds'):
        p.add_argument('--'+name,type=float,required=True)
    a=p.parse_args()
    if min(a.host_budget_gib,a.commit_gib,a.floor_gib,a.seconds)<=0:raise ValueError('Explicit positive build budgets required')
    command=json.loads(a.command_json.read_text(encoding='utf8'));inputs=json.loads(a.input_files_json.read_text(encoding='utf8'))
    for values in (command,inputs):
        if not isinstance(values,list) or not values or any(not isinstance(x,str) or not x for x in values):raise ValueError('Build command and inputs must be nonempty explicit JSON string lists')
    def identity(file):
        if not file.exists():return None
        s=file.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    before=identity(a.engine);sources=source_inventory();pins={str(Path(x).resolve()):sha256(Path(x)) for x in inputs}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cfg=load(overrides=dict(runtime_root=str(a.runtime_root),host_budget_gib=a.host_budget_gib,
        job_commit_gib=a.commit_gib,min_available_gib=a.floor_gib,runtime_min_free_gib=a.floor_gib,
        owned_paging_limit_bytes=shared_owned_paging_limit_bytes(),request_timeout_s=a.seconds))
    job=OwnedProcess(cfg,command,'qfn-pinned-build',cwd=a.cwd,timeout=a.seconds,telemetry=True)
    proof=job.wait()
    certificate=json.loads(Path(proof['shared_guard_certificate']).read_text())
    if (proof.get('exit_code')!=0 or not proof.get('cleanup_confirmed') or
            proof.get('paging_telemetry_closed') is not True or
            certificate.get('command_exit_code')!=0 or not certificate.get('release_allowed') or
            not certificate.get('cleanup',{}).get('confirmed') or
            not 0 <= certificate.get('peak_owned_paged_bytes',-1) <= shared_owned_paging_limit_bytes() or
            proof.get('paging_policy',{}).get('owned_paging_limit_bytes')!=shared_owned_paging_limit_bytes()):
        raise RuntimeError('Build guard did not certify successful retirement')
    if source_inventory()!=sources:raise ValueError('QFN source changed during its build')
    if {name:sha256(Path(name)) for name in pins}!=pins:raise ValueError('A read-only linked input changed during the build')
    after=identity(a.engine)
    if after is None:raise ValueError('Build did not produce the selected stable executable')
    if after==before:
        verified_build(a.manifest,a.engine)
        print('Reused the existing valid compiled-input manifest; no new source identity was stamped',flush=True)
        return
    atomic_json(a.manifest,dict(exit_code=0,source_files=sources,source_sha256=digest(sources),
        source_revision=revision,inputs=pins,output=str(a.engine.resolve()),sha256=sha256(a.engine),
        build_commands=[command],output_identity=after,guard_receipt=str(job.receipt),guard=proof,guard_certificate=certificate,
        linked_inputs_kind='Explicit read-only library/object closure, byte pinned before and after build'))
    verified_build(a.manifest,a.engine)
    print('Actual producer executable and compiled-source/link inputs pinned: '+str(a.manifest),flush=True)


if __name__=='__main__':main()
