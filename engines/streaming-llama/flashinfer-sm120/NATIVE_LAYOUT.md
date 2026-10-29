# Experimental direct-packed FP4 experts

This format is an opt-in speed proxy when requanted from Q4_K. It is not a
production model. Native safetensors E2M1/E8M0 can use the identical byte layout
losslessly. KERNELS2 selected the existing SM120 FlashInfer W4A8 family; its
reference b12x fork is not published. No unsupported b12x layout is invented.

The GGUF tensor type remains MXFP4 (39), **with a mandatory layout flag**:
`deepseek41.expert_layout=flashinfer-sm120-mxfp8-mxfp4-v1`. Every routed tensor
contains all weight bytes `[384,N,K/2]`, followed by E8M0 bytes
`[384,K/128,N,4]`. Adjacent logical E2M1 codes occupy the low/high nibbles of
each byte. Scales cover32 K values. The four innermost scales are packed in
K order; N is the next stride. There is no additional weight permutation or
scale swizzle. Gate/up have N2304,K5120; down has N5120,K2304. File and tensor
alignment must be at least128 bytes; converter output keeps16KiB alignment.
The byte count and logical tensor dimensions remain unchanged.

Canonical loaders refuse the flag. The dedicated DirectStorage provider binds
the validated tensor bytes to the fused native FFN and refuses unknown layout,
type, shape and alignment combinations. It also refuses the canonical routed
decode-cache layout. The shared model-receipt gate admits this format only in
explicit prefill-only mode with the native bridge, full-stage reads and ds4
decode disabled. Existing decode, production presets and model paths are not
qualified for this format.

`convert_dsv41_gguf.py --requant-gguf ... --expert-layout
flashinfer-sm120-mxfp8-mxfp4-v1` performs the explicitly lossy Q4 speed proxy.
It reads one expert at a time, streams the values, and holds only that tensor's
E8M0 scales (about135MiB) before writing them. Dense/Engram bytes are unchanged.
The dedicated durable journal is `requant-native-progress.json`; resume binds
the layout, source identity and quantizer. Completed tensors and full output
are hash-checked on readback. Existing completed outputs are never overwritten.
The later source conversion uses `--expert-format native-fp4 --expert-layout
flashinfer-sm120-mxfp8-mxfp4-v1` and preserves every source code and scale.

Build `build.ps1 -NativeFfn` against the same pinned AOT objects. Set
`LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL` to the absolute resulting DLL path,
alongside the existing stored-FP4 and FlashInfer toggles. The producer and CUDA
backend must be rebuilt together. The experiment harness takes
`--prefill-only`; an optional one-token forced probe writes the untouched
frontier without evaluating a decode step. It rejects longer continuations,
warm decode, calibration and nonzero quality bias in that mode.

The fused operator sorts routes once, gathers/MXFP8-quantizes the original
input once, computes gate/up into BF16, fuses weighted SwiGLU/F16-rounding with
down-input MXFP8 quantization, then reuses gate/up storage for BF16 down output.
It combines the six routes in the original slot order into F32 token outputs.
There is no F32 routed intermediate and no384-expert weight-repack allocation.
The bridge allocates no CUDA storage. The backend reuses a fixed caller arena
for the selected4K/8K/16K chunk size, including legacy-pool lookahead, routing
maps, alignment and module headroom in admission. Tiny calls use the same
format; they never fall through to canonical MXFP4. Whole-model64K graph
lifetimes are not implemented by the component's64K call support.

For an explicit bounded-context prefill-only experiment, admission returns
only unreachable default cache rows to the graph budget, retaining the exact
VMM-rounded charge of every owner at the configured context limit. Neither
the total device ceiling nor the host-memory ceiling grows. This qualifies
16K chunks at context131072 for the100K two-sweep experiment. At the original
million-position limit,16K still refuses its graph ceiling; the100K one-sweep
attempt separately refuses current physical host-memory admission.

Accepted completion witnesses name all three products, one route map and
input gather, zero runtime weight-repack bytes and zero F32 routed bytes.
`native_ffn_check.cu` compares all4K/16K token outputs with the independently
built qualified ABI4 adapter on synthetic data. The16K comparison passes
83,886,080 bit patterns and compute-sanitizer with zero errors. The64K check
establishes bounded allocation and full-output finiteness only. These are
component checks, not native-source or broad full-model quality claims.

The bridge is MIT. It reuses the unchanged Apache-2.0 FlashInfer source and
the pinned CUTLASS DSL runtime documented in `source/manifest.json` and
[`../PROVENANCE.md`](../PROVENANCE.md). No new third-party expert kernel is imported.
