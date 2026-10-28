# E6: sparse attention and FP4 index scoring

Original 518,596,067,328-byte Q4 GGUF, RTX 5090, 100,000 frozen tokens,
same-day fresh engines, exclusive leases, zero quality bias, YOCO disabled.
The primary sweep plan is 49,152 + 50,848 tokens. F32 growable context,
existing arithmetic profile and stored cache type are unchanged. Each pair
changes one explicit environment switch. This measures this full-prefill
configuration; it is not an exact-cone or MXFP4-weight comparison.

All throughput below uses prompt execution only: seed embeddings through frontier
logits, including every sweep and inter-sweep admission, excluding process/model
open and bulk hidden/payload export. End-to-end TTFT includes native startup
and the first returned token, but excludes lease queue and harness preparation.
The original raw `prefill_seconds` field includes producer startup; it is
retained for audit and must not be used as prompt-only time.

## TileLang attention: reject

`LLAMA_DSV41_TILELANG_ATTENTION=1` adapts native sparse membership records to
DeepSeek's TileLang kernel: BF16 operands/probabilities, 16-head groups,
128-query staging, one pipeline stage for SM120 shared-memory admission.
The existing native attention kernel is the control. Input/output conversion,
index preparation and image dispatch are included in the candidate timing.

| 100K | Prompt s | prompt tokens/s | TTFT s | Retrieval |
|---|---:|---:|---:|---|
| Native | 70.519 | 1,418.063 | 76.031 | pass |
| TileLang | 84.054 | 1,189.711 | 88.360 | pass |

The candidate takes 19.2% longer for prompt execution. Untouched-frontier KL is 0.00098803258,
top-1 agrees, and subsequent natural continuations differ. Forced rows after
those differing prefixes are not a matched quality comparison. The native
128-query/100K-key component also regresses: 0.627872 to 1.111712 ms.
The 4,242-case causal-index oracle passes; this is not an exact arithmetic claim.
Keep this experiment off. `LLAMA_DSV41_TILELANG_CUBIN` selects the locally
built image; the comparison harness verifies the pinned SHA256 before launch.
No Python/Torch/TileLang runtime is loaded into the native engine.

## Certified FP4 index scoring: adopt

`LLAMA_DSV41_INDEX_FP4=1` is the default and replaces only the existing BF16 index scorer on
admitted SM120 shapes. Queries and keys have already undergone the model's
FP4/E8M0 QAT round trip. The packer must restore every input exactly. A
conservative range certificate requires the maximum per-row binary spans
to sum to at most16 bits, with normal, non-overflowing products; the 128-term
dot then fits the FP32 significand in every accumulation order. If any input
or the certificate fails, the entire operation uses the original BF16 cuBLAS
path. This is a conservative implementation inspired by the reference's
per-tile certificate, not a copy of its private optimized kernel.

Two scaled `m16n8k64` MMAs form each dot; the original ordered F32 head
reduction, visibility and block masks remain. Packed workspace is bounded
inside the existing64MiB reserve. No per-head full-context score tensor,
host activation staging, new quantization policy or stored-KV format is added.

The component checks128 queries against100,001 keys, including11,984,327
finite scores, causal/block masks and explicit invalid-input fallback.
All scores are bit-identical to BF16. CUDA memcheck reports zero errors.
Clean median of three repetitions after two warmups: BF16 2.152544 ms,
FP4 0.973600 ms. The deliberate fallback costs2.402720 versus2.120992 ms.
These are component timings, not full-model throughput.

| Clean100K confirmation | Prompt s | prompt tokens/s | TTFT s | Retrieval |
|---|---:|---:|---:|---|
| BF16 control | 70.459 | 1,419.262 | 74.688 | pass |
| Certified FP4 | 67.746 | 1,476.101 | 71.891 | pass |

Prompt throughput improves4.00%. All17 full-vocabulary rows are bit-identical, KL0,
top-1 agrees17/17, natural answers and owner checkpoints match. The default
is enabled in every DSV4.1 example and the shared single-engine profile
overlay. Explicit `LLAMA_DSV41_INDEX_FP4=0` retains BF16. Profiles and shapes
that do not use the existing BF16 index scorer retain their existing path.

The first clean100K attempt measured74.563 to71.172 startup-inclusive seconds, but its post-run logits
hash read raced the native exclusive writer handle. The failure receipt is
retained and excluded from the accepted clean pair above. Native
publication now closes that writer before replying; the harness also bounds
its artifact-read retry to one second and never retries a decode command.
The native range-policy test passes3,808 integer-oracle cases, each in three
accumulation orders, plus boundary refusals and default/off behavior.
Launcher/environment validation passes23 tests, including explicit0
preservation across both presets and all available profiles.

The final rebuilt producer also passes the50K secondary pair: prompt execution
34.311 to33.685s (1,457.244 to1,484.357 prompt tokens/s, +1.86%), TTFT39.641
to37.266s,17 exact logit rows, KL0 and identical retrieval. Its larger
startup-inclusive improvement includes variable startup/Engram I/O.
The200K secondary also passes: prompt execution169.666 to158.399s
(1,178.787 to1,262.633 prompt tokens/s, +7.11%), TTFT175.156 to164.047s,
17 exact logit rows and identical retrieval. All ready/frontier/forced
checkpoint fields match at every context.
Real candidate FP4 dispatches are104/200/2972 at50K/100K/200K, with no
certificate fallback. See `examples/perfexp/e6/results.json` for pins and guards.

## Separate 100K synchronized stage diagnostics

| Arm | Startup-inclusive wall s | Exposed acquire s | Expert GEMM s | Sparse attention s | Index score s | Other GPU s |
|---|---:|---:|---:|---:|---:|---:|
| Native attention | 79.859 | 0.872 | 18.826 | 8.932 | 4.883 | 20.481 |
| TileLang attention | 93.250 | 0.807 | 18.680 | 22.634 | 4.847 | 20.387 |
| BF16 index control | 79.578 | 0.912 | 18.866 | 8.795 | 4.890 | 20.522 |
| Certified FP4 index | 76.938 | 0.877 | 18.876 | 8.798 | 2.056 | 20.535 |

Each arm contains80 layer records and6,685 semantic GPU-node keys. The
index pair has17 bit-identical full-vocabulary rows, KL0, identical natural
recall and top-1 agreement17/17. Attention adds13.702 seconds; the FP4 index
saves2.834 seconds. Other measured stages remain similar. Instrumentation
synchronizes operations: use the separate clean pairs for headline speed.
Other wall contains startup/carry/host scheduling and other GPU operations;
it is not an independently measured compressor timer. Exposed acquire is
already included in wall time.

## Attributing the remaining GPU and wall time

A separate matched100K diagnostic validates optional semantic names against
names disabled:17 exact logit rows, KL0, identical recall. It fixes the index
to BF16 to explain the original control. GPU events total53.125s, of which
18.896s is routed expert GEMM,8.806s sparse attention and4.883s index scoring.
The remaining20.540s has these disjoint constructor/weight-identity groups:

| Group | GPU s |
|---|---:|
| Attention output, including split projections and epilogue | 4.732 |
| Attention input projections | 2.298 |
| HC post/expand | 2.001 |
| Copies/layout with no retained owner label | 1.822 |
| Index selection | 1.806 |
| Shared expert, excluding nested activation group | 1.180 |
| Routed expert non-GEMM work | 1.054 |
| FFN activations | 1.039 |
| HC projection/mixes | 0.884 |
| HC flat normalization | 0.881 |
| Other rotary operations | 0.775 |
| Router | 0.500 |
| Engram compute/projection | 0.431 |
| HC pre/collapse | 0.414 |
| Index projections | 0.155 |
| FFN combination | 0.140 |
| HC block rounding | 0.136 |
| Other normalization | 0.116 |
| Compressor, excluding separately grouped norms/rotary | 0.040 |
| HC combination | 0.035 |
| Cache publication, initialization, head and seed | 0.053 |
| Semantically unassigned | 0.048 |

Dense matrix operations total6.592s across these groups; copy/concat/contiguous
operations total3.695s and QAT/round trips2.248s. These are another view of
the same events, not extra time. Split F16 matrices retain weight identities
in node names even when their physical operands are anonymous leaves; the
earlier0.191s named-operand attention-output-A subset was not its full cost.
The complete semantic-by-operation matrix is in `stage-attribution.json`.

CPU analysis of the original79.578s startup-inclusive profile partitions its26.505s wall less
summed node time exactly: prepare7.674, upload4.459, readback0.749,
compute-wall less recorded GPU7.641, outside layers4.699, exposed acquire0.912,
executor phase gap0.173, layer scaffold/carry0.186, cache/cleanup0.012.
The7.641s difference includes profiler synchronization, dispatch, copy/queue
waits and unrecorded GPU work. It is not a production launch-only measurement.

The native append-owner admission and chunk-token-preparation subclocks are
nested inside prompt execution; they are not added again to the phase sum.
Correcting this nesting leaves0.270494s harness-versus-native residual in
the original BF16 profile.

New nested clocks split the named diagnostic's7.699s prepare interval into
binding2.932s (registration2.925s inside it), graph construction0.487s,
allocator reservation3.763s, and0.517s other preparation. Outside-layer
receipts separately measure inspection/admission/allocation, identity
preflight, final hidden export, payload export and embedding/head/coordinator
work. All partitions and residuals are retained in the JSON, rather than
assigning unknown time to a guessed kernel. These are host intervals, not
CPU-cycle samples; allocator/transfer calls can wait on device work.

## Remaining experiment order

E14 has the largest conditional repeated-prefix benefit, but no cold-prefill
benefit; audit its missing native boundary capture first. Then measure E10's
actual decode launch/HC graph share and E7's short-chunk CPU/GPU component.
Those experiments target decode/short prompts, not the measured100K attention
output/projection bottleneck. E15 has no supported power-limit headroom.
E17 stays last as instructed: its memory benefit needs a real storage path.
NATIVEFP4 retains priority1, including its expert and dense work; this branch
takes only free exclusive leases. Host registration/reservation/upload work
also informs SINGLEPASS rather than duplicating that owner's carry changes.

## Reproduction and limits

Native entry points build through their existing CUDA translation units.
Component drivers, the adapted TileLang source, upstream MIT notice and
image ABI/hash manifest are in `examples/perfexp/e6/`. The image was generated
with TileLang0.1.15, Torch2.11.0+cu128, Python3.13 and CUDA13.2. Compile
`sparse_attn_kernel(16,512,512**-.5)` with the cache and all temporary paths
inside the one experiment scratch directory; its CUDA binary cache contains
the image described by `tilelang-image.json`. A different image is a new
experiment and must be separately pinned and validated.

Run every component and native comparison under `slstack.lease.gpu_lease`
and `slstack.process.OwnedProcess`, with an explicit host cap, physical floor,
paging stops and confirmed owned cleanup. The recorded component cap is3GiB;
engine caps are8.5GiB, tightened to7.5GiB for confirmation, with the original
native host plan,0.5GiB floor and512MiB desktop reserve retained. The Windows
commit allowance separately accounts for WDDM device backing. No test payload
is staged on M: and no model file is copied or changed.

Failed prototype geometry and operand-layout attempts are preserved, not
accepted. One stalled first component required the guardian's force fallback
after60 seconds without shutdown progress; later memcheck and clean components
closed successfully. A canceled confirmation controller raced admission;
its missing final guardian receipt is disclosed and an independent process,
idle-GPU and health audit preceded recovery of only its own stale lease.
All such attempts are excluded from accepted speed/quality evidence.
