"""CPU-only AOT export; run under the existing host memory guard.

Requires pinned nvidia-cutlass-dsl 4.8.1 and Torch 2.11.0+cu128.
No CUDA context or device allocations are created by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--sm-count',type=int,default=170)
args=parser.parse_args()
if not 1<=args.sm_count<=256:raise ValueError('Invalid SM count')
args.output.mkdir(parents=True,exist_ok=True)
os.environ.update(CUTE_DSL_ARCH='sm_120a',CUTE_DSL_COMPILER_BACKEND='legacy',
                  CUTE_DSL_CACHE_DIR=str(args.output/'cache'))
source=Path(__file__).parent/'source'
manifest=json.loads((source/'manifest.json').read_text())
for row in manifest['files']:
    if hashlib.sha256((source/row['vendored']).read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Vendored FlashInfer source hash mismatch: '+row['vendored'])
for name in ('flashinfer','flashinfer.fused_moe','flashinfer.fused_moe.cute_dsl','flashinfer.fused_moe.cute_dsl.blackwell_sm12x'):
    package=types.ModuleType(name);package.__path__=[str(source)];sys.modules[name]=package
utils=types.ModuleType('flashinfer.utils')
utils.ceil_div=lambda a,b:(a+b-1)//b
utils.round_up=lambda a,b:utils.ceil_div(a,b)*b
sys.modules['flashinfer.utils']=utils
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import cuda.bindings.driver as cuda
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_gemm as kernel
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_epilogue import EpiMethod

@cute.jit
def launch(gA:cute.Tensor,gB:cute.Tensor,gSFA:cute.Tensor,gSFB:cute.Tensor,gD:cute.Tensor,offsets:cute.Tensor,stream:cuda.CUstream):
    PROGRAM(gA,gB,gSFA,gSFB,gD,offsets,stream)

kernel.GRANK_A=32
outputs=[]
for n,k in [(2304,5120),(5120,2304)]:
    m=cute.sym_int32(symbol='M');sf=cute.sym_int32(divisibility=128,symbol='SF')
    fake=lambda dtype,shape,stride:make_fake_tensor(dtype,shape,stride,assumed_align=16)
    descriptors=(fake(cutlass.Float8E4M3FN,(m,k),(k,1)),
        fake(cutlass.Uint8,(384,n,k//2),(n*k//2,k//2,1)),
        fake(cutlass.Int32,(k//128,sf),(sf,1)),
        fake(cutlass.Int32,(384,k//128,n),((k//128)*n,n,1)),
        fake(cutlass.BFloat16,(m,n),(n,1)),fake(cutlass.Int32,(385,),(1,)),cuda.CUstream(0))
    cfg=kernel.make_cfg((64,128,128),2,epi=EpiMethod.R2G_WG)
    PROGRAM=kernel.CuteDslSm120MoeMxfp8Mxfp4Grouped(cfg,args.sm_count)
    compiled=cute.compile(launch,*descriptors,options='--gpu-arch sm_120a --keep-cubin')
    name=f'fi_mxfp4_{n}_{k}';compiled.export_to_c(str(args.output),name)
    outputs.append(dict(N=n,K=k,header=name+'.h',object=name+'.o'))
    print('EXPORTED',name,flush=True)
(args.output/'manifest.json').write_text(json.dumps(dict(status='BUILD_ONLY',gpu_execution=False,
    flashinfer_commit=manifest['commit'],sm_count=args.sm_count,outputs=outputs),indent=2)+'\n')
