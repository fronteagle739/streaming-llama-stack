"""CPU-only AOT export of the native-preserving QFN grouped NVFP4 GEMM.

Run under the existing host-memory guard, with python -B. Requires the pinned
CUTLASS DSL4.8.1 installation; all compiler output/cache/temp stays below the
explicitly requested workspace directory. No CUDA context is created.

Exports fi_nvfp4_640_2560 and fi_nvfp4_2560_640 .h/.o files. Each wrapper takes
gA,gB,gSFA,gSFB,gD,offsets,stream, in that order. Descriptors:
  gA: Uint8 [M,K/2], contiguous packed adjacent E2M1 codes.
  gB: Uint8 [512,N,K/2], strides [2764816,K/2,1] BYTES.
      Point at the role's start in expert0; expert blobs retain all three roles.
  gSFA: Int32 [K/64,SF], strides [SF,1], byte view [K/64,SF,4].
  gSFB: Int32 [512,K/64,N], strides [691204,N,1].
      Point at role_start+N*K/2; byte view [512,K/64,N,4].
  gD: Float32 [M,N], contiguous. No BF16 rounding and no global scaling.
  offsets: Int32 [513], stable expert-sorted cumulative routed-row bounds.
M/SF are dynamic wrapper descriptor dimensions. SF is divisible by128 and
must cover every padded expert scale row; scale row of route r in expert e is
floor((bounds[e]+3*e)/4)*4 + r-bounds[e]. Pad unused scale bytes with zeros.
Each UE4M3 scale covers16 unchanged E2M1 values. Each Int32 packs four scales
in increasing K order. Blob offsets: gate0, up921600, down1843200,
FP32 scalar tail2764800. The caller applies activation and weight globals
after each product; independent gate/up/down calibration remains mandatory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import site
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sm-count', type=int, default=170)
    parser.add_argument('--dsl-site', type=Path,
                        required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Every AOT output/cache/temp must stay under the explicit workspace')
    if not 1 <= args.sm_count <= 256:
        raise ValueError('Invalid SM count')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'temp'
    temporary.mkdir(exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUTE_DSL_ARCH='sm_120a',
                      CUTE_DSL_COMPILER_BACKEND='legacy',
                      CUTE_DSL_CACHE_DIR=str(output / 'cache'),
                      TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    site.addsitedir(str(args.dsl_site))
    here = Path(__file__).parent
    source = here / 'source'
    manifest = json.loads((source / 'manifest.json').read_text())
    for row in manifest['files']:
        if hashlib.sha256((source / row['vendored']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Pinned source hash mismatch: ' + row['vendored'])

    # An isolated package namespace keeps DSV imports and its kernel untouched.
    package = 'qfn_flashinfer'
    for suffix in ('', '.fused_moe', '.fused_moe.cute_dsl',
                   '.fused_moe.cute_dsl.blackwell_sm12x'):
        name = package + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(source)]
        sys.modules[name] = module
    utils = types.ModuleType(package + '.utils')
    utils.ceil_div = lambda a, b: (a + b - 1) // b
    utils.round_up = lambda a, b: utils.ceil_div(a, b) * b
    sys.modules[utils.__name__] = utils
    name = package + '.fused_moe.cute_dsl.blackwell_sm12x.kernel_qfn_nvfp4'
    spec = importlib.util.spec_from_file_location(name, here / 'qfn-native-nvfp4-kernel.py')
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[name] = kernel
    spec.loader.exec_module(kernel)

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    import cuda.bindings.driver as cuda

    # CuTe captures this closure at each compile, as in the existing DSV exporter.
    @cute.jit
    def launch(gA: cute.Tensor, gB: cute.Tensor, gSFA: cute.Tensor,
               gSFB: cute.Tensor, gD: cute.Tensor, offsets: cute.Tensor,
               stream: cuda.CUstream):
        program(gA, gB, gSFA, gSFB, gD, offsets, stream)

    outputs = []
    for n, k in ((640, 2560), (2560, 640)):
        m = cute.sym_int32(symbol='M')
        sf = cute.sym_int32(divisibility=128, symbol='SF')
        fake = lambda dtype, shape, stride: make_fake_tensor(dtype, shape, stride, assumed_align=16)
        descriptors = (
            fake(cutlass.Uint8, (m, k // 2), (k // 2, 1)),
            fake(cutlass.Uint8, (512, n, k // 2), (2764816, k // 2, 1)),
            fake(cutlass.Int32, (k // 64, sf), (sf, 1)),
            fake(cutlass.Int32, (512, k // 64, n), (691204, n, 1)),
            fake(cutlass.Float32, (m, n), (n, 1)),
            fake(cutlass.Int32, (513,), (1,)), cuda.CUstream(0),
        )
        program = kernel.QfnNativeNvfp4Grouped(kernel.make_cfg(), args.sm_count)
        compiled = cute.compile(launch, *descriptors,
                                options='--gpu-arch sm_120a --keep-cubin')
        stem = f'fi_nvfp4_{n}_{k}'
        compiled.export_to_c(str(output), stem)
        outputs.append(dict(N=n, K=k, header=stem + '.h', object=stem + '.o'))
        print('EXPORTED', stem, flush=True)
    (output / 'manifest.json').write_text(json.dumps(dict(
        status='BUILD_ONLY', gpu_execution=False, flashinfer_commit=manifest['commit'],
        derivative_sha256=hashlib.sha256((here / 'qfn-native-nvfp4-kernel.py').read_bytes()).hexdigest(),
        sm_count=args.sm_count, expert_blob_bytes=2764816, output_dtype='float32',
        weight_scale='ue4m3-per16', activation_scale='ue4m3-per16', outputs=outputs,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
