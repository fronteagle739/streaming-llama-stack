"""Ahead-of-time SM120 image for the optional packed TileLang attention adapter.

Run under the shared GPU lease and explicit host/paging guard. Put every cache,
temporary and output directory inside the one task scratch root.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scratch-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();root=args.scratch_root.resolve();output=args.output.resolve()
if not output.is_relative_to(root):raise ValueError('Image must stay inside task scratch')
sys.path[:0]=[str(root/'tilelang-deps'),str(root/'python-packages')]
os.environ.update(TILELANG_CACHE_DIR=str(root/'tilelang-cache'),TVM_FFI_CACHE_DIR=str(root/'tvm-cache'),MAX_JOBS='2')
import torch
from sparse_native import native_sparse_attn_kernel
from tilelang.contrib.nvcc import compile_cuda,default_compile_options
assert torch.cuda.get_device_capability()==(12,0)
program=native_sparse_attn_kernel()
code=program.get_kernel_source()
binary=compile_cuda(code,target_format='cubin',arch='sm_120',
    options=default_compile_options(['-O3','-Xcompiler /Zc:preprocessor'] if os.name=='nt' else ['-O3']))
output.write_bytes(binary)
output.with_suffix('.cu').write_text(code)
output.with_suffix('.json').write_text(json.dumps(dict(cubin_sha256=hashlib.sha256(binary).hexdigest(),
    cubin_bytes=len(binary),generated_cuda_sha256=hashlib.sha256(code.encode()).hexdigest()),indent=2)+'\n')
print('Built',len(binary),'bytes for sm_120; validate this image before promotion')
