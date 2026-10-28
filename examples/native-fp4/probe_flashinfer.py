import ast
import ctypes
import json
import os
from pathlib import Path
import site
import sys
import types
ROOT=Path(os.environ['NATIVEFP4_BENCH_ROOT'])
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('NATIVEFP4_GPU','0'),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUTE_DSL_ARCH='sm_120a',CUTE_DSL_CACHE_DIR=str(ROOT/'baseline-runtime/cache/cute_dsl'),
                  TEMP=str(ROOT),TMP=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
if os.name=='nt':ctypes.WinDLL('kernel32').SetErrorMode(3)
# Run inside the caller's isolated, pinned Torch/CuTe environment.
source=Path(os.environ['NATIVEFP4_FLASHINFER_SOURCE'])/'flashinfer'
# Isolate the unmodified kernel from unrelated serving/EP initialization.
for name in ('flashinfer','flashinfer.fused_moe','flashinfer.fused_moe.cute_dsl','flashinfer.fused_moe.cute_dsl.blackwell_sm12x'):
    package=types.ModuleType(name)
    package.__path__=[str(source.joinpath(*name.split('.')[1:]))]
    sys.modules[name]=package
tree=ast.parse((source/'utils.py').read_text())
utils=types.ModuleType('flashinfer.utils')
for name in ('ceil_div','round_up'):
    function=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source/'utils.py'),'exec'),utils.__dict__)
sys.modules['flashinfer.utils']=utils
import torch
import cutlass
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_gemm as kernel
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_epilogue import EpiMethod
props=torch.cuda.get_device_properties(0)
if torch.cuda.get_device_capability() not in ((12,0),(12,1)):raise ValueError('SM120/121 required')
kernel.GRANK_A=32
cfg=kernel.make_cfg((64,128,128),2,epi=EpiMethod.R2G_WG)
record={'status':'IMPORT_AND_CONFIG_PASS','torch':torch.__version__,'device':props.name,'capability':list(torch.cuda.get_device_capability()),
        'source_commit':'1d622fdb1b053fc532069914bc9a6b498bd5f722','activation_scale_granularity':32,
        'weight_scale_granularity':32,'note':'Direct standalone kernel import; declared full b12x wrapper requires Torch >=2.12; no performance or correctness claim yet'}
(ROOT/'flashinfer-probe.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
