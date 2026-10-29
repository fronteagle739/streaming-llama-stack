# OTHER22 round 3: measured opt-in CUDA experiments

## Ranked clean 100K timeline

| Rank | Kernel family | Baseline s | Queued + fused + table s |
|---:|---|---:|---:|
| 1 | FP16 / Q8 / F32 GEMM incl fixup | 3.209965 | 3.200685 |
| 2 | Attention | 3.184703 | 3.177599 |
| 3 | Expert incl packing | 2.937308 | 2.916746 |
| 4 | QAT / quantization | 1.272202 | 1.219496 |
| 5 | Cast / copy | 1.240505 | 1.222043 |
| 6 | FP4 index score / certificate | 1.004708 | 1.005296 |
| 7 | Top K | 0.807876 | 0.805053 |
| 8 | RoPE | 0.653350 | 0.648451 |
| 9 | RMS norm | 0.620154 | 0.506921 |
| 10 | Remaining kernels | 0.604114 | 0.605731 |
| 11 | HC expand | 0.522456 | 0.517312 |
| 12 | Concat | 0.331886 | 0.328398 |
| 13 | HC collapse | 0.272341 | 0.000000 |
| added | Fused HC collapse / RMS / weight / BF16 | 0 | 0.276350 |

GPU idle (first host tile preparation through final completion): **13.902916 -> 22.522715s**. Kernel/copy union 18.311272 -> 17.918499s; kernel busy 16.661568 -> 16.430082s; physical kernels 87,420 -> 86,523. Idle is uncovered envelope time, not CPU utilization. Kernel families, API sums and nested host phases overlap; do not add them. The legacy JSON envelope field retains its name, and the explicit envelope_scope defines this common measurement.

Every selected capture covers 563 tiles, 61 layer passes and 456,303,081,592 fetched bytes. Nsight node tracing and native QPC markers are used without per-node synchronized profiling. All submitted async tiles have matching completed CUDA events. Driver13.3/tracing13.2 and absent-NVTX warnings are retained; no severity3 or lost/dropped-event warning occurs in these clean captures.

## Prompt, TTFT and before/after changes

**The greater-than-20-second saving and under17-second prompt target remain unmet.** No reliable whole-prompt gain is demonstrated. The new paths remain opt-in, with production defaults unchanged. This is a partial implementation, not completion of whole-layer fusion or whole-layer/sweep pointer-independent replay.

| Arm | Prompt s | TTFT s | GPU idle s | Stream sync calls |
|---|---:|---:|---:|---:|
| Base450c6df eager | 32.788461 | 36.313000 | 13.902916 | 6118 |
| Same-build eager control | 33.829253 | 37.422000 | 14.606822 | 6118 |
| 1. Queued layers / GPU decisions | 37.993648 | 41.328000 | 19.288012 | 539 |
| 2. Queued + fused norm chain | 37.749976 | 41.110000 | 19.158193 | 538 |
| 3. Queued + fused pointer-table norm graphs | 41.114049 | 44.625000 | 22.522715 | 538 |

Single measurements follow the owner override; no medians are claimed. Same-build control -> queued is 4.164395s slower; queued -> fused saves 0.243672s; fused -> corrected table plan is 3.364072s slower. These samples do not establish gains above the stated2-3s noise. Final combined prompt is7.284796s slower than same-build eager and8.325588s slower than the base capture.

All selected arms use identical100,000 token bytes, cone2541, bias0, context131072 and sweeps49152/50848. All129,280 F32 frontier values match bitwise (SHA256 afee483112599c260dc1c05a39ad8bbd6dfb825e01a43260083e5b5114f9574a). This is the unchanged lossy packed-FP4 proxy. Prompt-only excludes launch/open/final certification; TTFT includes them. The harness legacy prefill_seconds includes startup. External Nsight wrapping is recorded by these trace files, independently of the harness profiled flag.

## 1. Remove per-tile CPU decisions and waits

LLAMA_DSV41_ASYNC_LAYER=1 retains uploads, graph metadata, pin registrations and copy ownership through bounded CUDA completion events. Original full carry validators still run; a dependent GPU failure gate executes before consumers. The original index FP4 certificate now drives a conditional graph in queued eager execution, retaining both exact-certified FP4 and original BF16 branches. No certificate, ID range or finite-value check was omitted. Odd pool tails use fully validated GPU copies.

The existing64MiB carry staging slot now uses stream-ordered CPU pack/scatter callbacks and asynchronous transfers. Metadata/upload ceilings are128MiB/1536MiB, charged as1664MiB additional host admission. Flights are bounded at16 and are never overwritten in flight. A provisional carry completion cannot commit until finish_layer drains both streams and certifies every tile. Weights are released after that certificate. Failed work keeps the carry poisoned. Error cleanup drains before releasing retained storage. finish_layer also takes the executor mutex.

Measured stream sync calls fall6,118 ->538, and stream-sync API sums17.890836 ->13.374683s over the common envelope. This does **not** mean every synchronization was removed. Layer transaction/certification and acquisition/cleanup boundaries remain. Requested hash/attribution/boundary/routing observers retain real host barriers, as do cast audits and DSpark publication. Remaining D2H copies preserve spill, IDs and required RAM mirrors:9,431 ->8,184 calls,15,345,458,048 ->15,345,450,468 bytes. Scalar CPU decisions were moved; required data exports were made asynchronous, not discarded.

Idle regresses13.902916 ->19.288012s after queued execution alone. On the final combined path it reaches22.522715s. The local indexer still constructs108 conditional graphs per100K pass. Final combined trace has109 graph instantiations (including the norm template),1.930618s instantiate API time and2.795913s executable-destruction API time. Deferred retirement avoids destroying in-flight graphs, but Windows driver handling remains expensive. CPU preparation/acquisition and layer drains remain; removing every layer-loop host gap is unfinished.

## 2. Exact fused small-op chain

LLAMA_DSV41_LAYER_FUSION=1 matches the private HC collapse -> BF16 -> RMS -> vector weight -> BF16 chain. It retains the original HC evaluation order and both BF16 rounding boundaries, and reproduces the existing CUDA RMS reduction and fast-math FTZ/divide/rsqrt policy explicitly. Shape, type, strides, output visibility and external consumer checks prevent fusion of incompatible graphs; their original operations execute.

The production100K path executes1,158 fusions. Earlier round HC expansion, CONCAT, HC collapse and QAT batching remain. The new fused family takes0.276350s in the final graph trace. This covers the two related norms per tile plus scoped heads; it is **not** a fused entire layer. RoPE, general quantization, router/index TOP_K and remaining concat/copies still execute in their existing kernels. Global reductions, GEMMs and changed rounding boundaries were not silently bypassed. No additional dense-GEMM experiment beyond the round2 FP8/control comparison is claimed in round3.

## 3. Binding-independent norm graphs

LLAMA_DSV41_LAYER_FUSION=2 updates a device-side descriptor containing residual/pre/weight/output pointers, width, rows and epsilon. One kernel graph per block-size/stream runs a fixed4096-block grid with dynamic row iteration;64 pinned descriptor/event slots retain CPU upload lifetimes, with loud refusal before overwriting an in-flight slot. Changed pointers and row counts reuse the executable.

Full100K: **1,126 actual norm graph launches, one construction**, plus32 direct fused head calls inside the existing explicit graph-disable scope. Those scoped calls remain eager as requested by that API. The witness still demands both layer norm graphs per FFN tile, reports scoped calls separately, and rejects embedding-only or insufficient reuse. Graph count/clear APIs include these resources and release them correctly.

Mode2 requires explicit GGML_CUDA_DISABLE_GRAPHS=0; the harness visibly overrides its inherited eager preset for this request. It selects outer eager node dispatch plus the norm graph, and explicitly rejects simultaneous outer streamed-template requests. Scoped graph disable still works. Other graph modes remain available; no explicit graph request passes as an eager no-op.

This table covers the norm chain only. The separate conditional indexer remains bound to each operation and is reconstructed; full-layer/sweep mutable tables, cross-tile execution without CPU dispatch and whole-layer PDL remain unfinished. One norm template is not labeled whole-layer replay.

## Validation and rejected attempts

CUDA memcheck reports zero errors for30 full-output fused/changed-binding comparisons (widths31/1024/5120, rows1/17/4096/8192/16384), graph resource count/clear,18 graph/counter regressions,8 conditional FP4/BF16 cases and4 threaded CPU comparisons.100 asynchronous validator/consumer checks pass. Six invalid inputs (residual, pre, mean, Engram, selected IDs, candidate IDs) each fail on GPU before the dependent consumer. The carry fixture rejects uncertified commit and accepts certified commit.

Python:21 environment, witness and inherited kernel-contract tests pass. Full release_check with identities reports zero findings; staged checks/hooks are required for publication. All changed CUDA and producer units were rebuilt with consistent graph flags, reusing qualified unchanged libraries and the decode DLL. This is not a fresh full-project build or a broad release qualification for decode, other models, web UI or two-machine use.

Rejected trials remain in private receipts: a missing GPU fence before spill pre-state D2H produced different full logits and was corrected; a rare BF16 tie exposed mismatched RMS fast-math arithmetic and was corrected; an inconsistent incremental graph macro caused a CUDA context ABI mismatch and was corrected with consistent compilation plus a runtime ABI check. No failed output is included as passing evidence.

One corrected100K table run passed full numerical checks but was rejected by a witness that incorrectly required graphs inside scoped head disable. The witness was fixed to require all1,126 layer graph calls and explicitly count32 eager head fusions; the subsequent clean passing capture above supplies that evidence. The older table sample that ignored global/scoped policy is superseded and excluded. Physical-memory and hard-working-set admission failures remain rejected; no reserve or paging guard was reduced.

## Capability and memory accounting

**Source capability reductions: NONE.** No existing feature, field, sampler, option, backend or numerical branch was stripped, hidden or made a no-op. Original eager implementations remain selectable. New experiment limits are explicit: queued execution requires the persistent resident BF16/IDs/metadata plan with retained async inputs/post-head export, no prefetch and sufficient host admission; it refuses a live diagnostic RAM view. Requested host observers retain barriers. The norm fusion falls back for nonmatching graphs; requesting it on HIP/MUSA fails explicitly while original backend operations remain. Mode2 and outer streamed graph templates are separate requested plans. Conditional queued indexing requires CUDA12.8+. These limits are not claims of support for those new combinations.

Benchmark capability restrictions, explicitly retained: inherited lossy Q4-to-FP4 speed proxy; prompt-only with LLAMADECODE_DS4_DECODE=0 and LLAMADECODE_PREFILL_ROUTED=0; no sampled generation, evaluated decode or recall; one fixed frontier; cone2541 and bias0. No new lossy default or production/model/store edit was made.

GPU work uses the exclusive shared lease, explicit aggregate RSS budgets and paging stops. Final outer/inner caps8.5/8.4GiB; native base7.69GiB, maximum8.19GiB, with the full100K plan8,435,466,240 bytes including carry/Engram delta. The additional1664MiB queue admission stays intact. Physical/runtime floor0.75GiB, the native separate1GiB admission floor and512MiB desktop reserve stay enforced. A separate48GiB private-commit limit accounts for WDDM device backing; it does not permit host paging. CPU checks/builds use2GiB RSS/3GiB private commit and serial native compilation. All timed state stays on GPU/RAM; read-only model storage is on the fast approved volume.

Resource audit at generation: 93 guardian receipts, all owned trees cleaned; peak aggregate RSS 7,390,420,992 bytes, minimum available physical RAM 3,311,620,096 bytes. System-wide pagefile-growth warnings are retained and not attributed to the job without evidence. Compiler-helper-only forced retirements, where present, followed cooperative requests and60s without progress; no engine or foreign process was force-killed.

Final-source and odd-length qualification records are in round3-evidence.json. The profiled table predates the final executor certification mutex and radix witness correction; subsequent full100K and odd-length checks below qualify those final changes. Publication and recoverable checkout/scratch retirement are recorded in the private task result.

| Additional qualification | Tokens | Status | Prompt s | TTFT s |
|---|---:|---|---:|---:|
| eager-async-control-odd-qualified | 8193 | PASS | 18.529901 | 20.85899999999674 |
| eager-async-table-odd-hist-qualified | 8193 | PASS | 17.6270636 | 19.68800000002375 |
| eager-async-control-clean-final-100k | 100000 | PASS | 32.911997500000005 | 35.75 |
| eager-async-control-fused-only-final-100k | 100000 | PASS | 33.6275401 | 36.53200000000652 |
| eager-async-table-source-final-100k | 100000 | PASS | 37.899513899999995 | 40.84400000004098 |



Final source (including certification mutex and corrected one-row radix witness) qualifies all five additional runs. Eager-only norm fusion changes prompt 32.911998 -> 33.627540s (+0.715543s); no gain is established. Final queued/table prompt 37.899514s, TTFT 40.844000s, full100K frontier bitwise identical. The8193-token eager and queued/table+histogram frontiers also match all129,280 values (SHA256 86985e3b559e55b31f18c1c636860e1c6fda559271fb853ecbdf0caa2ca0a524); 101 actual histograms and 101 explicit observer barriers are recorded. These short-prompt times are diagnostic, not the100K speed comparison.

The odd-tail probe initially exposed a stale native radix dispatch witness: the backend correctly uses its deterministic tiled network for one row, while the witness excluded every one-row call. The corrected witness counts the real stages and still honors explicit CUB/forced-bitonic choices. It was not removed or relaxed to ignore unexpected work. A private fractional host-budget request was also rounded upward to64KiB to match Windows reservation granularity; exact hard-minimum admission stays unchanged.


## Remaining synchronization inventory

| Site | Queued path / remaining requirement |
|---|---|
| Input/output validator scalar flags | Full GPU scan plus dependent failure gate; per-tile4-byte CPU decision removed. |
| Index certificate | GPU IF/ELSE retains both arithmetic branches;28-byte proof D2H decision removed in queued execution. |
| Tile pin rebinding | Retain registrations to completed events; original per-tile copy-stream wait removed. Driver registration itself may still block. |
|64MiB staged pack/scatter | Stream-ordered CPU callbacks retain copies; chunk-level host waits removed. |
| Graph allocator release, main compute, cast-cache completion, resident hidden D2D | Deferred to stream/event lifetimes; audits still synchronize when requested. |
| Spill/compact-ID/RAM export | Real asynchronous D2H data remains; GPU scratch-reuse dependency protects transfer lifetime. |
| Odd pooled tail | Full validation plus D2D; no CPU download/re-upload decision. |
| Layer completion and weight ownership | Compute and copy drains remain at transaction boundary before carry commit/provider release. |
| Weight acquisition and driver graph APIs | Fetch completion and graph construct/retirement costs remain measured; CPU-free sweep is unfinished. |
| Routing/hash/attribution/boundary/DSpark/audit observers | Real requested host observation/publication barriers remain; no diagnostic no-op. |
| Final head and hidden export | Existing scoped graph-disable and required final readbacks remain outside the layer tile compute loop. |

Compiler discovery warned that vswhere was unavailable before the explicit installed toolchain succeeded; the linker retained its existing LIBCMT/defaultlib conflict warning. No warning-suppression or guard bypass was used.

STATUS: PARTIAL - opt-in queued lifetimes, exact norm fusion and mutable-pointer norm graphs are implemented and measured; >20s saving, removal of all layer-loop waits, whole-layer fusion and whole-layer/sweep graph reuse remain unfinished.
