# Native FP4 conversion candidate

This is a CPU-tested converter candidate, a pinned source audit and a Q4
baseline campaign. The resumable full checkpoint download is running.
**Full conversion, native-model GPU qualification and native-vs-Q4 comparisons
have not run.** Production model selection is unchanged. No fastest-kernel or
native-model performance improvement is claimed.

The [pinned source manifest](NATIVEFP4_SOURCE_MANIFEST.json) records all 48
expected LFS SHA-256 values, sizes, header hashes and metadata identities for
[DeepSeek's release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
It explicitly marks every full shard payload as **unverified**. Downloading
headers does not verify a shard against its LFS hash.

## Audited representation

The config, index and all shard headers agree on 96,085 tensors across
510,296,708,312 bytes of safetensors (475.25 GiB). Metadata files were checked
against their Git blob identities; bounded shard ranges were checked for exact
HTTP offsets, header structure, tensor extents, index membership and scale shapes.

| Family | Released representation | Converter output |
| --- | --- | --- |
| Main routed experts | E2M1 packed in I8; E8M0 scale per 32 values | Lossless MXFP4/type 39 |
| DSpark routed experts | Same native FP4 representation | Existing Q4 sidecar recipe |
| Dense quantized matrices | E4M3; E8M0 scale per 32x32 block | Existing Q8_0 recipe |
| Engram | E4M3; E8M0 scale per 32 values | Existing packed I8 row264 |
| Other tensors | BF16/F32 | Existing per-tensor recipe |

The combined source contains 47,232 packed expert weight tensors (46,080 main,
1,152 DSpark), 355 scaled dense FP8 matrices and two scaled FP8 Engram tables.
The remaining dtype inventory is in the manifest. The released inference source
uses FP8 activations for FP4 expert GEMM; the existing HYPER W4A4 path uses a
different activation representation, requiring independent quality evaluation.

Both source FP4 and GGML MXFP4 use the same E2M1 codebook and E8M0/block32
scales. Source byte j packs adjacent values 2j/2j+1. GGML byte j packs values
j/j+16. The adapter rearranges the nibbles and interleaves the original scale;
it never converts expert weights to floating point or requantizes them.
Scale 255 is refused. Signed-zero code bits are retained; GGML's scalar
dequantizer maps either zero sign to numeric zero.

No supported native FP8 dense GGUF type exists in this runtime's loader/type
intersection. A follow-up needs an explicit FP8 tensor format and scale storage,
CPU reference decoding, GPU admission/streaming and SM120 dense kernels before
the Q8_0 fallback can be replaced. This candidate does not add those kernels.

## Commands and validation

Audit without downloading weight payloads:

```text
python -B scripts/audit_dsv41_source.py --out <metadata-directory>
```

The existing converter remains pinned to antirez/ds4
`ee7e8405232c8971000cc7ba2b945b00bf6d20d3`. No upstream source hash was changed.
An AST adapter adds one direct-copy branch to its expert callback, retaining the
original dense quantizer, Engram copier, streaming writer, disk reserve and
durable resume journal. The existing Q4 default is unchanged.

```text
python -B scripts/convert_dsv41_gguf.py --source <metadata-directory> --ds4-tools <pinned-tools> --expert-format native-fp4 --dry-run --plan <new-plan.json>
python -B scripts/convert_dsv41_gguf.py --source <verified-full-checkpoint> --ds4-tools <pinned-tools> --expert-format native-fp4 --execute --out <new-native.gguf> --quants-library <local-library> --host-budget-gib 8 --threads 1
```

The second command is a reproduction recipe, **not a completed conversion**.
Use `--runtime-root <scratch-directory>` to put guard logs, memory receipts and
temporary files on a different volume from the model output. Without that
option they remain beside the output. The model's partial file and durable
resume journal remain with the model.
It requires the fully downloaded/hash-verified checkpoint, tokenizer, optional
converter dependencies, library and adequate disk/RAM admission. Keep the
current DSpark and Vision sidecars. Never compare a native plan with a Q4
reference header expecting equality of the intentionally changed expert types.

The audited native main plan has 1,046 tensors: 120 MXFP4, 321 Q8_0, 110 F16,
493 F32 and two I8. Its aligned tensor payload is 501,603,303,424 bytes
(467.15448 GiB), versus 518,590,234,624 for Q4; GGUF header bytes are additional.
Only the 120 routed expert tensor types change. Header/layout validation on a
complete native file remains pending.

CPU checks cover every packed byte at every block position for all 255 finite
E8M0 exponents, reversible code/scale preservation, independent source/GGML
numerical oracles, multiple rows/blocks, invalid scales, the native plan and the
real pinned streaming writer. The writer fixture proves that experts bypass
dequantization/requantization while dense matrices still use their original
quantizer callback. HTTP tests reject ignored, truncated and incorrect ranges
and mismatched metadata identities, and check retry after an incomplete read.
The original GGUF/conversion regressions also pass.

Tests/audit used explicit 1-2 GiB aggregate Job commitment/RSS limits and a
4 GiB available-physical-RAM floor. Native-model GPU qualification remains pending.

## Performance evidence and gaps

The existing [ANYGGUF smoke](ANYGGUF_GPU_SMOKE.md) matches the single-engine
ULTRA+YOCO140 profile: 49,555-token prefill 1,592.1285 tok/s; warm quality
6.81180 tok/s and speed 18.15505 tok/s over 899 steps after the first 100.
These are single observations, not statistical estimates.

Fresh controls on 2026-10-28 use that same producer binary/profile, context
1,000,001, BF16 growable cache, eight split reads, and a repeated frozen recall
token body. Their input is synthetic; these runs do not establish recall
quality. Rates include native startup and initial prefill:

| Prompt tokens | Startup + prefill (s) | Q4 tok/s | Weight sweeps |
| --- | ---: | ---: | --- |
| 49,555 | 29.265 | 1,693.32 | 49,555 |
| 200,000 | 96.984 | 2,062.20 | 3 x 49,152 + 52,544 |
| 1,000,000 | 956.734 | 1,045.22 | 20 x 49,152 + 16,960 |

These are prompt-length controls, not 200K/1M tokens in one weight pass.
The historical 552.989462 tok/s million-token result predates YOCO, and the
paired 198,287-token result includes a Mac handoff; neither substitutes for
these controls. Native-vs-Q4 logits/KL, dial KL/top-1 and native gains remain
unmeasured.

| Prompt | Fetched bytes (decimal GB, whole run) | Exposed layer fetch wait (s) | Executor wall (s) |
| --- | ---: | ---: | ---: |
| 49,555 | 315.826 | 8.179 | 17.746 |
| 200,000 | 806.444 | 7.013 | 82.498 |
| 1,000,000 | 3,643.477 | 17.887 | 913.731 |

Engram tables are read on demand, not streamed wholesale. The Q4 tensor
inventory contains 305.765 GB of routed experts, 10.061 GB of other tensors
and 202.758 GB of Engram. The earlier estimate of 480 GB streamed per pass
was incorrect. Backend transfer-duration sums overlap computation; exposed
acquisition wait is the relevant ledger field. Executor wall combines expert,
attention, indexer and other operations; finer attribution remains required.

[Control receipts](../examples/native-fp4/q4-baselines.json) record input/log
hashes, sweep plans and resource observations. The 1M harness streams complete
diagnostics to disk with a 1 MiB RAM tail, overcoming the original wrapper's
8 MiB cumulative-log cap without changing the model executable. Its peak
aggregate RSS was 6.25 GB, minimum available RAM 5.91 GB, and shutdown was
graceful with no budget breach. Observed global pagefile growth was 4.23 MB,
below the enabled stop threshold; this counter is not engine paging-I/O
attribution. The original failed attempt is retained separately.

## Kernel eligibility audit and synthetic microbenchmarks

[Pinned candidate identities and source hashes](NATIVEFP4_KERNEL_AUDIT.json)
record the inspected source. No external kernel was integrated into the engine.
Standalone pinned FlashInfer CuTe DSL SM120 W4A8 and our HYPER W4A4 MMQ
passed sampled double-reference checks on identical representable operands.
These results exclude activation quantization, layout preparation, transfers,
and fused MoE epilogues. They use synthetic balanced groups, not observed
routing. [FlashInfer receipts](../examples/native-fp4/flashinfer-microbench.json)
and [HYPER receipts](../examples/native-fp4/hyper-microbench.json) retain shapes,
TFLOPS, logical effective bandwidth and reference-check counts. Logical
effective bandwidth is a byte-count estimate, not measured memory traffic.

| Candidate | Inspected source evidence | TFLOPS / effective GB/s |
| --- | --- | --- |
| Current HYPER | Source-built native MXFP4 MMQ, same operands as FlashInfer. Existing generic MXFP4 dispatch selects FP4 MMA on SM120; the separate runtime-Q4 packing experiment remains Q4-only. | 321.5-344.6 TFLOPS at M768 |
| CUTLASS | Pinned standalone W4A8/W4A4/W8A8 builds completed. Current executions fail admission/reference qualification; no accepted result. NVFP4/block16 is a different format. | Rejected/unqualified |
| FlashInfer | Pinned direct SM120 W4A8 GEMM, E8M0/block32. Apache-2.0. Full serving wrapper not imported. | 477.0-488.6 TFLOPS at M768 |
| vLLM | Inspected MXFP4 grouped kernel explicitly uses Sm100: skip that kernel on SM120. Separate NVFP4 SM120 path exists; Marlin and other exact-format paths remain to qualify. | Not measured |
| SGLang | Separate SM120 runner exists. Inspected tiny-M NVFP4 specialization targets H1024/I2688/top22, not our dimensions. General exact-format kernels remain to qualify. | Not measured |
| Upstream DeepGEMM | Pinned requirements specify SM90/SM100: skip upstream kernels on SM120. A different project's SM120 port is a separate candidate. | Not measured |
| TensorRT-LLM | Revision/license pinned; exact-format SM120 MoE eligibility remains unverified. | Not measured |

Main expert gate/up matrices require N2304/K5120, down N5120/K2304, 384
experts and top6. The balanced-routing average at 49,152 tokens is 768 tokens
per expert; this is arithmetic, **not an observed routing histogram**. Actual
HYPER graphs are limited to 4,096 tokens, so graph-level workload distribution
must also be captured. Dense examples include Q-A N1280/K5120, Q-B
N32768/K1280, KV N512/K5120 and output-B N5120/K8192.

| Experts | M/expert | N/K | FlashInfer graph ms | HYPER eager ms |
| ---: | ---: | --- | ---: | ---: |
| 1 | 128 | 2304/5120 | 0.021920 | 0.031412 |
| 6 | 1 | 2304/5120 | 0.038120 | 0.037981 |
| 6 | 1 | 5120/2304 | 0.039726 | 0.041764 |
| 384 | 64 | 2304/5120 | 1.613642 | 2.236701 |
| 384 | 64 | 5120/2304 | 1.703356 | 2.290759 |
| 384 | 768 | 2304/5120 | 14.240569 | 20.190603 |
| 384 | 768 | 5120/2304 | 14.587173 | 21.639427 |

FlashInfer eager M768 times were 13.572 and 14.095 ms. Its large-group
advantage therefore persists without CUDA graphs. FlashInfer returns BF16,
HYPER F32; comparisons use the matching final rounding. Tiny cases rotate
more than 128 MiB of weights to exceed L2. Sampled exactness does not establish
whole-tensor correctness or end-to-end quality. Nonconstant per-row/per-block
scales and independently generated scalar hash fixtures test the packing.

NVIDIA's [Blackwell whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
lists 1,676 dense FP4 TFLOPS for the 5090, versus 3,352 with structured
sparsity. The measured 488.6 is 29.2% of the nominal dense FP4 roof; this is
only a reference ratio, not a precision-matched W4A8 utilization measurement.
Neither these kernels nor the model use the advertised sparsity multiplier.

The [reference implementation notes](https://github.com/SL-x-TnT/DSV4.1-Flash-RTX-Pro-6000-Overview)
were read in full. Stock fused b12x epilogues change the route-weight point
and gate/up rounding. Any integration must preserve the required BF16 rounding,
SiLU, route weighting, BF16 output and activation-quantization order, and pass
an uncached teacher-forced gate. A GEMM win alone does not qualify that port.

Required next measurements retain identical inputs/reference checks and include
activation quantization, routing/layout preparation, kernels and end-to-end
costs separately. Capture actual expert M distributions, build eligible pinned
kernels in isolation, check correctness, then choose a kernel and record its
license in PROVENANCE. Prompt length and per-weight-pass sweep length must be
reported separately. Admission must determine the largest feasible 200K/1M
sweep before launch; no capacity or 10K tok/s claim follows from synthetic peak
FP4 arithmetic. Stream wait, expert GEMM, attention, indexer and other stage
times, measured effective bandwidth and achieved/peak SM120 FP4 rates remain
required work.

## Explicitly lossy speed proxy from the existing Q4 file

The owner authorized this exception for speed experiments. It is never a
native-precision conversion or a model to ship. The output metadata explicitly
labels it **Q4→MXFP4 requant (lossy), speed proxy for native FP4**.

```text
python -B scripts/convert_dsv41_gguf.py --requant-gguf <current-Q4.gguf> --out <new-speed-proxy.gguf> --quants-library <requant-adapter-library> --runtime-root <scratch> --hold-file <shared-GPU-ledger> --host-budget-gib 1 --runtime-floor-gib 1 --threads 16
```

Build `scripts/dsv41/requant_q4_mxfp4.cpp` as a shared library with precise
floating point and OpenMP (MSVC: `/O2 /fp:strict /openmp /LD /std:c++17`).
An AVX2 host can additionally use `/arch:AVX2`; the vector midpoint mapper
preserves the scalar quantizer's code order and exact ties.
The adapter follows GGML's scalar Q4_K decoder and MXFP4 quantizer, including
its first-code tie rule. This is an offline CPU route; it requires no GPU.
Dense and Engram tensor bytes are copied unchanged. All names and shapes
match the future native file; only the 120 routed-expert types become MXFP4.
Expected file size for the audited input is 501,609,136,128 bytes.

Source is read-only. Distinct output and a single-writer lock are mandatory.
Each bounded read checks the shared GPU holds and pauses cooperatively if a
timed GPU run starts. Memory admission includes a hard host budget and a
1 GiB available-physical-RAM floor (configurable, minimum 0.5 GiB). Completed tensors are fsynced and hashed;
`--resume` verifies their hashes and all source/header/library identities,
then rewrites only the interrupted tensor in the owned partial. The final
file is renamed into place after layout and whole-file SHA-256 verification.
For slow output disks, `--defer-readback` computes the full hash while writing
(and reading the verified resume prefix), without rereading the entire output.
Its receipt explicitly sets `readback_verified=false`; verify the stored source
bytes and destination against that hash during the later copy before timing.
The partial and journal remain available on failure. Existing output is refused.

Independent float32 decoding/nearest-code tests cover 4,096 Q4_K blocks,
zero blocks and threaded equality. A synthetic interrupted GGUF conversion
tests durable resume, the marker, and unchanged dense/source bytes. Set
`NATIVEFP4_REQUANT_LIBRARY` to run the CPU adapter tests.
The optimized midpoint mapper also matched the original GGML-style search
byte-for-byte on 253,952 random packed blocks covering every finite half bit
pattern for both Q4_K scales, including negative values and subnormals.

The complete speed proxy passed full stored-byte readback verification:
501,609,136,128 bytes, SHA-256
`cf05bdccb86b823ce794eab2ca033a649b6a2f5c03cee8a4b177d86de017ae44`.
[The conversion receipt](../examples/native-fp4/proxy-verification.json) records
all 926 unchanged tensor hashes, the 120 converted experts and memory limits.
The original Q4 file is retained. Native-source conversion and full
end-to-end speed/quality qualification remain separate work.
Timed model runs require the approved fast model volume and
the same frozen prompts/profile as the Q4 controls. Report recall and
KL/top-1 against Q4 bias zero; do not infer native quality from this proxy.
The owner's latest storage instruction is to write converted GGUFs directly
to the approved experiment folder on D:, preserving the original Q4 folder
and at least 20 GiB free. The converter checks the reserve before each chunk.
Ordinarily source reads, resume verification and final output readback pause
during shared GPU holds. A priority conversion can instead hold the shared
benchmark reservation continuously: pass its live ancestor lease file with
`--conversion-lease`. Only that exact reservation is excluded from the pause
check; a foreign hold, unreadable ledger or missing own hold fails closed.
Keep the reservation through final stored-byte verification and release it
immediately afterwards. The abandoned partial is retained for orchestrator cleanup;
only the watchdog safetensors download may continue writing to I:.

For the stored-weight experiment, set `LLAMA_DSV41_STORED_MXFP4=1` and
`GGML_CUDA_DISABLE_GRAPHS=1` with MXFP4 gate/up/down weights and the ULTRA
profile. This opt-in tags the three existing native FP4 MMA graph nodes and
requires their completed CUDA dispatch counters. It performs no runtime weight
conversion. Activations use the existing FP4 quantizer and accumulation is
F32; this W4A4 arithmetic differs from the reference engine's W4A8. Batches of
four rows or fewer retain the generic MXFP4 path and have no tensor-core claim.
Alternate expert arithmetic profiles are refused. The toggle is off by default.
Rebuild both `ds4.c` and `ds4_cuda.cu` with `DS4_EXTERNAL_DECODE=1` for the
external decode library; an older CUDA object can handle Q4 while refusing
MXFP4 external expert pointers. The rebuilt library's Q4 regression matched
64 generated IDs, the final full-vocabulary logits hash and all owner-state
hashes; [the receipt](../examples/native-fp4/decode-refresh-regression.json)
records the old and new binary identities.

Calibration files bind the non-Engram mapping size and header identity. Generate
one for each model by setting `DS4_CACHE_AFFINITY_CALIBRATE` to a new output,
selecting bias zero and decoding at least 64 tokens through the external
decode adapter. Close the session to write the 40 layer spreads. The adapter
observes routing without applying a bias during calibration. Reuse the resulting
file only for its matching model. Engram is deliberately unmapped before the
identity is calculated: compare the saved size to the first Engram offset,
not the complete file size. A different model identity is not an admissible control.
