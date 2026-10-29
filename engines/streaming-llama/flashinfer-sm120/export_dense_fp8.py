"""CPU-only AOT export of pinned SM120 MXFP8 dense GEMMs (no CUDA context)."""
import argparse,ast,hashlib,json,os,sys,types
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
os.environ.update(CUTE_DSL_ARCH='sm_120a',CUTE_DSL_COMPILER_BACKEND='legacy',CUTE_DSL_CACHE_DIR=str(args.output/'cache'))
source=Path(__file__).parent/'dense-source'
manifest=json.loads((source/'manifest.json').read_text())
for row in manifest['files']:
    if hashlib.sha256((source/row['vendored']).read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Pinned dense source changed: '+row['vendored'])
for name in ['flashinfer','flashinfer.cute_dsl','flashinfer.gemm','flashinfer.gemm.kernels']:
    package=types.ModuleType(name);package.__path__=[str(source)];sys.modules[name]=package
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import dsl_user_op
# Import only the three unmodified pure CuTe helpers needed by this kernel.
# The complete pinned utility file imports Torch for unrelated tensor helpers.
utils=types.ModuleType('flashinfer.cute_dsl.utils')
utils.__file__=str(source/'utils.py.source')
utils.__dict__.update(cute=cute,dsl_user_op=dsl_user_op)
tree=ast.parse((source/'utils.py.source').read_text())
helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
    ('ceil_div','sm120_make_smem_layout_sfa','sm120_make_smem_layout_sfb')]
if len(helpers)!=3:raise ValueError('Pinned dense helper closure changed')
sys.modules[utils.__name__]=utils
exec(compile(ast.Module(body=helpers,type_ignores=[]),utils.__file__,'exec'),utils.__dict__)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel
@cute.jit
def launch(a:cute.Tensor,b:cute.Tensor,sa:cute.Tensor,sb:cute.Tensor,out:cute.Tensor,alpha:cute.Tensor,stream:cuda.CUstream):
    PROGRAM(a,b,sa,sb,out,alpha,170,stream)
rows=[]
for n,k in [(1280,5120),(32768,1280),(512,5120),(128,512),(5120,8192),(2304,5120),(5120,2304)]:
    m=cute.sym_int32(divisibility=16,symbol='M')
    fake=lambda dtype,shape,stride:make_fake_tensor(dtype,shape,stride,assumed_align=16)
    descriptors=(fake(cutlass.Float8E4M3FN,(m,k,1),(k,1,m*k)),
        fake(cutlass.Float8E4M3FN,(n,k,1),(k,1,n*k)),
        fake(cutlass.Float8E8M0FNU,(4096*k//32,),(1,)),
        fake(cutlass.Float8E8M0FNU,(n*k//32,),(1,)),
        fake(cutlass.BFloat16,(m,n,1),(n,1,m*n)),
        fake(cutlass.Float32,(1,),(1,)),cuda.CUstream(0))
    PROGRAM=DenseGemmKernel(sf_vec_size=32,mma_tiler_mn=(128,128),cluster_shape_mn=(1,1),mma_k=32,tile_k=128,enable_pdl=False)
    compiled=cute.compile(launch,*descriptors,options='--gpu-arch sm_120a --keep-cubin')
    name=f'fi_mxfp8_{n}_{k}';compiled.export_to_c(str(args.output),name)
    rows.append(dict(N=n,K=k,name=name));print('EXPORTED',name,flush=True)
(args.output/'manifest.json').write_text(json.dumps(dict(status='BUILD_ONLY',gpu_execution=False,flashinfer_commit=manifest['commit'],outputs=rows),indent=2))
