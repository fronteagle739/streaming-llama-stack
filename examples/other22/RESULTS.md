# OTHER22 measured partial result

**BLOCKED-OWNER: the 10,000 tok/s objective and whole-sweep graph replay are unfinished.** Two exact CUDA kernel improvements are implemented and qualified. No reliable full-prompt speedup is established. Source base `1fc2d08201355ab57b98c9bf7fe9939262f865f2`; publication branch `agent/other22`, for orchestrator merge only.

## Matched setup and measurement

100,000 identical tokens, cone2541, bias0, ULTRA, context131072, sweeps49152/50848, packed FP4 4K. Existing Q4-to-FP4 requant remains a **lossy speed proxy**, unchanged by this work. Prompt-only seconds exclude launch/model-open/frontier certification; TTFT includes them. All model bytes and presets were read-only.

Nsight Systems2025.6.3 process-tree CUDA capture; per-node synchronized profiling OFF. Native QPC compute windows align to SQLite startTime; every563 tile has CUDA coverage. Each accepted capture has186,038 kernels,61 layer passes and456,303,081,592 fetched bytes. Kernel category sums are a second view of the same trace; do not add them to kernel busy or host phases. Categories group kernel names, not complete semantic layer stages.

The profiler warns that driver13.3 is newer than its13.2 tracing libraries; retained in evidence. It also reports no NVTX events; this producer emits native QPC markers instead. No lost/dropped-event diagnostic was observed for the selected producer. The analyzer rejects failed guardian receipts, missing tile coverage and reported dropped events.

## Ranked CUDA seconds, baseline to final

| Rank | Kernel family | Before s | Final s |
|---:|---|---:|---:|
| 1 | FP16 / Q8 / F32 GEMM incl fixup | 3.223889 | 3.251013 |
| 2 | Attention | 3.188808 | 3.202367 |
| 3 | Expert incl packing | 2.963750 | 3.004016 |
| 4 | BF16 GEMM / score epilogue | 1.840357 | 1.844372 |
| 5 | QAT / quantization | 1.401303 | 1.403688 |
| 6 | Remaining kernels | 1.254108 | 1.261875 |
| 7 | Cast / copy | 1.249664 | 1.254040 |
| 8 | Top K | 0.807989 | 0.810746 |
| 9 | Concat | 0.796129 | 0.331853 |
| 10 | HC expand | 0.693833 | 0.520214 |
| 11 | RoPE | 0.665613 | 0.666340 |
| 12 | RMS norm | 0.568234 | 0.567525 |

Baseline kernel busy18.653677s; final18.118049s. Baseline kernel-or-copy union20.292585s within35.741957s first-to-last-compute envelope. The remaining envelope includes host and I/O waiting; it is not CPU utilization. Initial/final phases outside this envelope remain in prompt/TTFT measurements.


## Host and I/O clocks

These observed wall intervals are nested or overlap GPU work; do not add them to the kernel table or to each other.

| Baseline observation | Seconds |
|---|---:|
| Exposed layer acquisition wait | 7.577558 |
| Host upload wall | 3.901357 |
| Host prepare wall | 1.973314 |
| Host readback wall | 1.594608 |
| Host bind wall (includes register) | 0.742346 |
| Host register wall (subset of bind) | 0.739337 |
| Host allocator reserve wall | 0.496019 |
| Host graph build wall | 0.189871 |

## Before/after by adopted change

1. HC BF16 expansion: one lane loads each of four residual channels once and computes all four outputs. Explicit RN multiply/add retains source accumulation order and the identical BF16 boundary. No extra workspace. HC-only capture:0.693833 to0.517287s; CONCAT stays0.794091s. Final HC0.520214s. Original F32 path remains available.
2. Contiguous F32 CONCAT: flatten dimensions0/1/2 into repeated input spans; aligned spans use float4, arbitrary/misaligned spans use scalar copies. Dimensions3, non-F32 and noncontiguous implementations are preserved. HC-only to combined capture:0.794091 to0.331853s. Final two changed families save0.637895s, with identical kernel counts. These copy/kernel implementations are the new exact defaults; they do not remove CONCAT or the HC operation.

The host graph-build sum is only0.189871s in the baseline. Full graph replay, PDL, projection fusion, packed activation handoff, router/top-k fusion and HC norm/add fusion are **not implemented or claimed**. Index-selection top-k is separate from router top-k and cannot be removed by naming it a router fusion.

## All accepted100K runs

Names containing nsys are diagnostic captures, separate from ordinary throughput runs. Cold startup and run variance are retained.

| Arm | Prompt s | Prompt tok/s | TTFT s |
|---|---:|---:|---:|
| baseline-clean | 39.941389 | 2503.67 | 61.875 |
| baseline-nsys | 36.457880 | 2742.89 | 41.031 |
| dense-control-1 | 35.489252 | 2817.75 | 38.687 |
| fp8-1 | 36.555445 | 2735.57 | 39.704 |
| fp8-2 | 36.143162 | 2766.78 | 39.218 |
| dense-control-2 | 33.979234 | 2942.97 | 37.047 |
| pin-1 | 34.260683 | 2918.80 | 37.297 |
| hc-1 | 33.892430 | 2950.51 | 42.062 |
| baseline-repeat | 35.801224 | 2793.20 | 38.875 |
| hc-2 | 37.872280 | 2640.45 | 40.985 |
| hc-nsys-verified | 43.415458 | 2303.33 | 47.375 |
| concat-1 | 34.901002 | 2865.25 | 42.922 |
| concat-nsys | 35.089290 | 2849.87 | 38.547 |
| concat-2 | 35.306253 | 2832.36 | 38.422 |
| final-1 | 36.705571 | 2724.38 | 42.922 |
| baseline-final-control | 36.066110 | 2772.69 | 39.703 |
| final-2 | 40.008358 | 2499.48 | 43.172 |
| final-nsys | 37.265744 | 2683.43 | 40.937 |
| pin-2 | 41.690682 | 2398.62 | 45.031 |
| stage-1 | 33.161783 | 3015.52 | 36.188 |
| stage-2 | 35.491623 | 2817.57 | 38.641 |
| baseline-last | 34.297027 | 2915.70 | 37.312 |
| stage-3 | 33.376121 | 2996.15 | 36.500 |

FP8 dense repeats36.555445/36.143162s versus matched Q8 BF16-store controls35.489252/33.979234s: no gain, no default change. FP8 remains fully available. Pin-window repeats34.260683/41.690682s are inconsistent; retain its existing option/default. Forced persistent staging trials are retained as experiments, with no default change or guarantee.

The final kernel savings are smaller than I/O variance. Exposed layer acquisition is7.577558s baseline,14.536744s HC-only and8.864378s final. The slower HC-only prompt is not a kernel regression or a throughput success. At a hypothetical25GB/s, unchanged456.303GB fetches alone need18.252s;10s requires fewer passes/bytes or substantially more bandwidth. This is a conditional bound, not a new disk specification.

## Correctness, build and remaining support

HC:24 component cases compare every F32 bit to a separate scalar GPU reference, with sampled CPU oracle, BF16 ties, signed zero, subnormals and partial blocks. Maximum case83,886,080 values. CONCAT:54 full-output CPU byte oracles (including unequal spans) cover dimensions0/1/2, odd spans, vector/scalar, aligned/misaligned pointers, NaN payloads, infinity, signed zero and subnormals. CUDA memcheck:zero errors for both suites. Final checked-in HC test including CPU oracle was rebuilt and rerun. Final producer rebuilt changed CUDA units and allocator, linking qualified existing dispatcher/backend objects and libraries; **not a clean full-backend rebuild**.

All exact candidate frontier vectors match all129,280 F32 bit patterns against control:KL0, top1 identical. The FP8 probe has KL2.02195e-9, matching top1 and max logit difference2.01412; this peaked single frontier does not establish broad recall/decode quality. No new lossy default is adopted. No recall benchmark was run.

The inherited proxy harness explicitly forces `GGML_CUDA_DISABLE_GRAPHS=1` in `examples/native-fp4/speed_proxy.py`; `ggml-cuda.cu` treats its presence as disabling graphs. The executor still builds/binds tile graphs and synchronizes at existing tile boundaries. This work does not implement a captured layer sweep with correct streamed-weight, carry and workspace lifetimes. It does not relabel the existing graph cache or eager execution as replay. That missing support and the unmet throughput target are reported BLOCKED-OWNER, rather than silently dropping operations or claiming completion.

## Capabilities and guards

Source capability reductions:**none**. No field, option, sampler, type or backend was stripped, hidden or made a no-op. Benchmark restrictions, explicitly inherited/requested:FP4 proxy opt-in; CUDA graphs disabled in that proxy harness; prompt-only/no evaluated decode; bias0. These are not production default changes and are not evidence of sweep-graph/decode qualification.

Exclusive shared GPU holds;512MiB desktop reserve; runtime/physical floor0.75GiB; aggregate model RSS cap9GiB and nested8.5GiB, monitored physical RAM and paging stops. Windows Job private-commit allowance32GiB includes WDDM accounting and is not an RSS allowance. Build/check Jobs2GiB RSS/3GiB commitment. Receipts retain paging-observer availability and counters; no claim of zero system-wide paging is inferred from private commit.

One HC Nsight finalization was rejected on unavailable shared-ledger access; excluded despite its model worker producing a PASS record. Its owned tree retired cooperatively. Initial configuration/path/BOM and sanitizer-path helper failures ran no accepted engine timing. All failed receipts remain in local evidence. The final audit corrected a test-build path mismatch; final24/54-case claims use explicitly rebuilt checked-in suites, not the earlier scratch copies. One report encoding gate failure was corrected by writing UTF-8. One owned Visual Studio vctip helper required the last-resort force fallback after cooperative retirement and60.1seconds without progress; cleanup confirmed. No engine or foreign process was forced. No production service was restarted, no model file/Altium/foreign branch changed, and no deletion was used for space clearing.

The sole existing checkout was registered as one linked worktree, with its Git common store inside the same scratch root, so it can be removed after push verification. Public evidence contains sanitized timings, kernel families, quality probes and guard summaries; private raw traces and receipts remain recoverable through the task scratch retirement described in the owner result.

STATUS: BLOCKED-OWNER — exact partial changes qualified;10K objective and whole-sweep replay unfinished.
