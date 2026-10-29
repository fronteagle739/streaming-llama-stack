# OTHER22 round 2

## Ranked clean 100K timeline

| Rank | CUDA kernel family | Control s | Fused s |
|---:|---|---:|---:|
| 1 | FP16 / Q8 / F32 GEMM incl fixup | 3.204454 | 3.205497 |
| 2 | Attention | 3.184424 | 3.183454 |
| 3 | Expert incl packing | 2.934017 | 2.934155 |
| 4 | BF16 GEMM / score epilogue | 1.836543 | 1.836547 |
| 5 | QAT / quantization | 1.409504 | 1.277702 |
| 6 | Remaining kernels | 1.264926 | 0.630433 |
| 7 | Cast / copy | 1.256458 | 1.260870 |
| 8 | Top K | 0.805164 | 0.805333 |
| 9 | RoPE | 0.666450 | 0.667488 |
| 10 | RMS norm | 0.570622 | 0.623012 |
| 11 | HC expand | 0.522692 | 0.525019 |
| 12 | Concat | 0.332622 | 0.332495 |
| added fused family | HC collapse | 0 | 0.272944 |

Both clean captures cover all563 tile windows, 61 layer passes and 456,303,081,592 fetched bytes. Kernels: 186,038 -> 181,534; kernel busy 17.987877 -> 17.554949s (0.432928s saved). The control already retains round1 HC expansion/CONCAT improvements.

Nsight process-tree CUDA timeline, native QPC alignment, no per-node synchronization. Kernel families are physical kernel-name groups, not complete semantic stages. Driver13.3 versus tracing-library13.2 and absent NVTX warnings are retained. The two ranked captures have no severity3 diagnostic.

| Host observation (nested/overlapping) | Control s | Fused s |
|---|---:|---:|
| host_prepare_seconds | 2.067656 | 2.061727 |
| host_bind_seconds | 0.819196 | 0.789612 |
| host_register_seconds | 0.816224 | 0.786733 |
| host_graph_seconds | 0.187503 | 0.180799 |
| host_reserve_seconds | 0.500795 | 0.445791 |
| host_upload_seconds | 3.847023 | 3.872058 |
| host_launch_seconds | 3.798077 | 6.321032 |
| host_sync_seconds | 17.117843 | 16.646306 |
| host_readback_seconds | 1.680700 | 1.671368 |
| layer acquisition wait | 8.034175 | 8.026777 |

Control first-to-last-compute envelope 37.995795s; kernel/copy union 19.612374s. These host timers overlap/nest and must not be added to kernel time or to each other. Uncovered envelope time is not CPU utilization.

## Outcome against the owner target

**The >20s saving and under17s prompt objective remain unmet.** Exact small-op improvements and actual streamed CUDA graph support are implemented. No reliable whole-prompt speedup is demonstrated. Full-layer/sweep reuse with the CPU out of the tile loop is unfinished; it is not relabeled as completed replay. No owner decision is needed merely because the target was missed.

Latest owner override used: one100K measurement per arm; no medians claimed. Same100,000 token bytes, cone2541, bias0, context131072, sweeps49152/50848, stored FP4 4K. FP4 remains the inherited lossy Q4-to-FP4 speed proxy. Prompt-only and TTFT are reported separately; the harness legacy prefill_seconds field includes startup.

| Arm | Prompt s | Prompt tok/s | TTFT s |
|---|---:|---:|---:|
| final-r2-control-nsys | 38.710063 | 2583.31 | 49.203000 |
| r2-fusion-nsys | 40.640998 | 2460.57 | 49.719000 |
| graph-complete-100k | 34.987688 | 2858.15 | 38.516000 |
| eager-rebuilt-100k | 34.506991 | 2897.96 | 37.687000 |
| eager-fp8-100k | 33.783610 | 2960.02 | 37.063000 |
| eager-dense-control-100k | 32.615655 | 3066.01 | 35.797000 |
| graph-final-nsys | 38.466298 | 2599.68 | 42.657000 |
| graph-nodes-nsys | 38.207179 | 2617.31 | 41.719000 |
| graph-topology-nsys | 38.857479 | 2573.51 | 42.453000 |
| graph-final-source-100k | 32.373785 | 3088.92 | 35.406000 |

## Adopted exact fusion

HC collapse combines four products, three left-associated sums and the existing BF16 boundary into one kernel. Explicit RN multiplies/adds preserve the original evaluation order. CPU support uses the same portable arithmetic. Original decomposition remains selectable with LLAMA_DSV41_HC_COLLAPSE_FUSION=0; noncontiguous inputs retain it. The new kernel is0.272944s in the matched fused trace, replacing the decomposed chain;4,504 physical launches disappear across1,126 collapse calls.

QAT batches independent16/32-value format groups into256-thread CTAs. The shuffle reduction preserves the reference pairings and the original scale/quantization helpers, BF16 boundary, in-place support, all formats and V4 modes. LLAMA_DSV41_QAT_TILED=0 selects the old per-group launch. Total QAT family1.409504 ->1.277702s; the raw non-BF16 per-group kernels themselves were only0.060450s before. This is not removal of activation quantization.

Combined kernel busy falls0.432928s, while the clean profiled prompt goes38.710063 ->40.640998s. There is no end-to-end speedup claim. Round1 HC expansion/CONCAT changes remain intact.

## Actual graph support and remaining structural work

The explicit speed_proxy.py --cuda-graphs experiment now forwards its registered options and requires actual graph launch witnesses covering at least the native FFN graph count. GGML_CUDA_DISABLE_GRAPHS=0 now explicitly allows capture; the production preset stays1. Other present values keep legacy disable semantics. No explicit graph request can pass as eager execution.

Implemented support: one bounded CUDA executable; full pointer/type/stride/shape/op-parameter snapshots before reuse; continued updates beyond the former four-update cutoff; cuBLAS initialization and admitted VMM temporary mapping before capture; loud refusal of pool growth inside capture; resident BF16 conversion, pinned asynchronous uploads and existing validation/lifetime drains; actual dispatcher counter deltas replayed only after a successful CUDA graph launch. Changed node counts re-instantiate directly instead of issuing an invalid graph-update API call.

The indexer preserves both its exact FP4 integer-sum certificate/MMA path and its original BF16 fallback. A GPU conditional IF/ELSE chooses on each execution, including replay with changed input values. The mutually exclusive branches alias their temporary storage, retaining the original index workspace ceiling. The pack, certificate, masking, accumulation and both branches remain active. CUDA12.8+ conditional APIs are guarded so older-toolkit eager support is retained.

Full100K qualification:1,126 actual launches,1,104 capture/update submissions and22 unchanged replays. The22 replays are embedding/seed reuse, not layer compute reuse. Layer/tile bindings change; the existing CPU tile loop, validation/readback boundaries and recapture work remain. Ordinary graph34.987688s versus same-build eager34.506991s is0.480697s slower. Final source after branch-workspace aliasing takes32.373785s,2.133206s below that eager run, still inside the stated2-3s noise range. Neither comparison demonstrates a reliable speedup. Graph mode stays experimental and opt-in.

Next structural requirement: retain per-layer sweep templates plus their streamed-weight, temporary, metadata-upload and carry lifetimes; patch changing bindings without redispatching every node; include all required copies and GPU checks, then certify/read back at the existing transaction boundary. This task does not claim that architecture is complete.

## Final graph diagnostic (not a clean ranked claim)

Last node-granularity trace (before the final mutually-exclusive workspace alias): prompt38.857479s, 87,634 observed kernels, observed kernel busy16.628890s. Every tile has kernel coverage. CUPTI reports invalid original graph/node IDs with this driver/toolkit combination; trace_quality is DIAGNOSTIC_ONLY_METADATA_ERRORS. Those errors are retained and exclude this capture from the clean table above. A previous graph-granularity capture was rejected for missing per-tile kernel coverage; node tracing was then explicitly enabled. No coverage check was removed. The final alias was subsequently checked under memcheck and a fresh full100K run, not another profiled family ranking.

| Observed diagnostic family | Seconds |
|---|---:|
| FP16 / Q8 / F32 GEMM incl fixup | 3.197471 |
| Attention | 3.179440 |
| Expert incl packing | 2.922423 |
| QAT / quantization | 1.269786 |
| Cast / copy | 1.243717 |
| FP4 index score / certificate | 1.005187 |
| Top K | 0.804750 |
| RoPE | 0.650879 |
| RMS norm | 0.619851 |
| Remaining kernels | 0.600971 |
| HC expand | 0.520058 |
| Concat | 0.333337 |
| HC collapse | 0.281020 |

A wider backend rebuild also activates the already-present index FP4 implementation in source. Its eager frontier differs from the reused-library round1 binary (KL5.19e-9, top1 same); that is not attributed to graph replay. The final graph and same-build eager vectors match all129,280 F32 bit patterns (KL0). Source changes to exact fusion were separately qualified against the retained control with all bits equal.

## Remaining priorities evaluated

| Item | Evidence and decision |
|---|---|
| Whole-layer small-op chain | HC collapse/BF16 fused; QAT groups batched. Whole-layer fusion across GEMMs/global reductions is not implemented. |
| Router TOP_K | Clean trace TOP_K0.805164s is dominated by index selection: radix stages0.669758s. Router fusion cannot remove those indexer stages. No top-k capability removed; additional router fusion deferred under the owner instruction to skip gains below2-3s noise. |
| CONCAT/copy | Round1 exact changes retained. Remaining concat0.332622s and cast/copy1.256458s are below the noise ceiling even if eliminated. Views must retain cache ownership/rounding; no unsafe alias substitution adopted. |
| RoPE, add/norm | RoPE0.666450s and RMS norm0.570622s; HC collapse addresses its related add chain. Projection/RoPE and general norm/add fusion are not implemented; no rounding boundary removed. Further isolated polishing deferred under the owner override. |
| Layer/sweep CUDA graphs | Actual capture/update/replay and missing pool, counters, pinned/BF16 and index conditional support implemented/tested. Full-sweep replay and CPU-loop removal remain unfinished. |
| Programmatic dependent launch | Actual PDL graph-chain component: ordinary0.135122ms, PDL0.252779ms;4,194,304 values match bitwise,16 kernels,20 timed graph repeats.32MiB working set fits cache. This is not a100K model measurement or a proof against every PDL schedule. No whole-layer PDL path adopted. |
| Dense GEMMs | Same-build original-Q8 BF16-store control32.615655s versus FP8 proxy33.783610s: FP8 is1.167955s slower, below2-3s noise. FP8 remains available. No new lossy default, projection batching or q/kv/o GEMM fusion adopted. |
| CPU preparation/overlap | Clean host graph-build sum0.187503s; launch3.798077s; acquisition8.034175s; upload3.847023s. They overlap. No claim that removing graph construction alone saves20s; no newly defaulted staging/pinning option. |

## Correctness and validation

64 full-output CUDA fusion cases include production5120x4096 HC and QAT shapes, BF16 ties/signed zero/subnormals, odd tails, three V4.1 quantized formats, in-place QAT and an independent HC CPU oracle. Final CUDA memcheck:0 errors.

Graph suite:18 full-output/counter checks exercise repeated replay, changed tensor contents, more than four updates and replacement tensor lifetimes; four additional index cases alternate certified FP4 and original BF16 branches on the same replayed graph. Four threaded CPU backend comparisons (one/seven threads, widths31/5120) also compare the new custom callback to the original decomposed graph. Final CUDA memcheck:0 errors with API error reporting enabled. A preliminary run exposed a recoverable invalid topology-update call; the node-count precheck fixed that case rather than suppressing sanitizer errors. Other incompatible same-count topology updates retain the existing explicit re-instantiation path;21 recoverable update failures remain visible in the full model diagnostic trace.

Python:10 environment-contract tests,3 graph-witness tests and5 inherited kernel-contract tests pass. New options reach the actual native child; the default eager preset remains1; insufficient/embedding-only launch witnesses fail. Full and staged release checks plus commit hooks are required before publication. Front Eagle identity and hooks remain enabled.

Both exact fusion and graph/eager matched-build comparisons have all129,280 frontier values identical. FP8 probe has KL4.28e-9, same top1, maximum logit difference2.593206. One peaked frontier is not broad recall or decode qualification. No recall/decode benchmark or new lossy default is claimed. Producer rebuilt all changed CUDA/producer units and required counter-bearing units, reusing other qualified libraries and the decode DLL; this is not a fresh whole-project build.

## Capabilities, failed attempts and resource limits

**Source capability reductions: NONE.** No field, sampler, precision, bias/vision routing, option, operation or backend was removed/hidden/no-op to load or pass. Original CPU paths and alternative kernels remain. Existing graph-mode incompatibilities such as legacy synchronized expert/routing-map experiments still fail explicitly; they remain available with their original eager policy. The new graph experiment requires VMM temporary reservation and CUDA12.8+ for conditional indexing; production defaults are unchanged.

Benchmark capability reductions/restrictions explicitly retained: inherited lossy FP4 proxy; prompt-only with LLAMADECODE_DS4_DECODE=0 and LLAMADECODE_PREFILL_ROUTED=0; no sampled generation, recall or evaluated decode; one untouched frontier probe; requested cone2541 and bias0. The earlier harness forced eager execution; the new explicit graph arm preserves the same requested features. These are benchmark scope choices, not production feature removals. FP8 is an optional lossy dense proxy. No model bytes, production store/preset, Altium repository or foreign branch was modified.

Exclusive shared GPU ledger,512MiB desktop reserve,0.75GiB physical/runtime floor, paging stop, aggregate9GiB RSS (nested8.5GiB; native producer7.25GiB). CPU builds/checks2GiB RSS/3GiB private commit, serial compilation. All new work, builds/caches/temp/outputs stayed in the sole task scratch root. Only authorized result/hold records were written outside it.

A graph attempt failed loudly at6.4GB RSS/35GB Windows private commitment with about5.4GB physical RAM still available. The separate WDDM device-backing allowance was raised32 ->48GiB while RSS, physical floor and paging stops stayed enforced. This is not permission to page host data. Another profile was rejected because an owned paging handle was unavailable. All failed receipts remain in recoverable local evidence, excluded from performance tables. System-wide small pagefile-growth warnings are retained and not attributed to the task without evidence.

Other rejected attempts: unregistered graph option/preset made the first request eager (not graph evidence); resident BF16/async/stored-FP4 capture guards, VMM pool synchronization and CPU index proof initially blocked capture and were then implemented; one conditional capture-info call had an invalid null status parameter and was corrected. Build retries added required MSVC preprocessor/extended-lambda flags. No guard, operation, certificate or tracing-coverage requirement was disabled to make these pass.

Two owned Visual Studio vctip helpers required the guard's last-resort force retirement after cooperative shutdown and60.03s without log/RSS/member progress. Their receipts confirm cleanup. No engine or foreign process was force-killed.

At report generation,87 guard receipts audited; all owned process trees have confirmed cleanup. Publication and scratch retirement receipts follow after push verification. No main or foreign-owned branch is a publication target.

STATUS: PARTIAL - exact fusion and streamed graph support qualified; >20s saving, under17s prompt and full-sweep replay remain unmet.
