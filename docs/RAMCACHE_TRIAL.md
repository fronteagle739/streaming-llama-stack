# Host-RAM cache admission trial

2026-10-27, RTX 5090, 31.362 GiB physical RAM.
**The requested larger-cache benchmarks were blocked by physical RAM admission.**
No engine was launched and no new decode throughput, cache hit rates, KL or top-1
measurement is claimed. The baseline below is the earlier
[Any-GGUF GPU smoke](ANYGGUF_GPU_SMOKE.md).

## Which cache the controls actually size

- `LLAMADECODE_ENGRAM_CACHE_MB` sizes a host-RAM LRU of immutable 264-byte Engram
  rows used by the producer's early/prefill row loader. The shipping presets set
  it to `0`. An unset variable can auto-size to at most 4 GiB of admitted
  headroom; an explicit size is exact or refused. See
  [preset](../scripts/dsv41_single.py),
  [row cache](../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-engram-row-cache.h),
  [admission](../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-sweep.cpp),
  and [row-loader counters](../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-engram-prefetch.cpp).
- `LLAMADECODE_L2` enables/disables the **VRAM** victim tier after the decode
  graph frees memory. It is not a host-RAM cache size. The provider's L1 and L2
  expert slots are both device storage; a miss reads the GGUF through
  DirectStorage. See [provider](../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-weights.cpp)
  and [L2 sizing/token counters](../engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill/live-session.inc).
- ds4-win's standalone host expert cache does not start in this external-decode
  path: the producer's expert hook owns the weights/cache. The
  [adapter](../engines/ds4-win/ds4_decode_adapter.inc) explicitly disables
  standalone streaming/host tiers. Increasing the Engram row cache therefore
  does not add host expert slots to the quality/speed dial's residency view.

## Admission and baseline

Studio was stopped gracefully while the shared hold ledger was clear and the
RTX was idle. Ten paused samples showed 13.939
to 14.120 GiB available. Chrome,
the desktop and other agents' processes were preserved; this is the actual
shared desktop, not an assertion that every nonessential application was closed.
The physical floor is 4 GiB plus the 0.5 GiB desktop reserve, enforced together
as 4.5 GiB. Page-file capacity is excluded from admission.

| Host row cache | Available RAM | Required native plan + 4.5 GiB | Warm expert hits: VRAM / RAM / NVMe | Warm quality / speed tok/s | Outcome |
|---|---:|---:|---|---|---|
| 0, earlier smoke | 7.602 GiB minimum during smoke | Earlier run, separate budget | 60.490% / 0% / 39.510% quality; 97.312% / 0% / 2.688% speed | 6.81180 / 18.15505 | Historical GPU measurement |
| +3.5 GiB (3584 MiB) | 13.631 GiB at paused admission | 14.291 GiB | Not measured | Not measured | Refused before launch |
| +7 GiB (7168 MiB) | At most 14.120 GiB in saved paused samples | 17.791 GiB | Not measured | Not measured | Cannot fit even the highest paused reading |

Both proposed points reuse the exact recorded smoke requests: the 49,553-token
archive prompt, recall/follow-up, and 1,000-token quality/speed requests at bias
0/1.36125. Outer ceilings were raised for the trial only, from 8.5 GiB to
12/15.5 GiB; producer bases from 6.25 GiB to 9.75/13.25 GiB and maxima from
7.75 GiB to 11.25/14.75 GiB. The fold planner produced 9.79059/13.29059 GiB
native plans for the initial prompt. The admission requires those plans plus
4.5 GiB available, before any engine launches. Controller/monitor overhead
would need further room; the required figures are lower bounds, not a full
desktop recommendation. No +12 GiB point was attempted.

The +7 GiB live preflight check ran during Studio restoration and saw
12.683 GiB; its refusal is independently supported
by every saved paused sample. A config-name formatting error initially stopped
that invocation before admission; it was corrected. Neither invocation launched
an engine. No performance retry or tuning campaign occurred.

The baseline expert percentages were recomputed from existing native logs:
899 generated decode steps after the first 100 steps, **excluding append
prefill**. Match each `LLAMADECODE_STEP position=P` to the
`LLAMADECODE_DS4X_TOKEN position=P-1` row. Every step makes 240 expert
lookups. Quality totals: L1 77,341, L2 53,173,
NVMe 85,246. Speed: L1 116,572,
L2 93,388, NVMe 5,800.
RAM expert hits are structurally zero in this path. These percentages describe
experts only; Engram row hits are separate counters. No larger-row-cache
hit rate is available because admission refused the allocations.

## Recommendation

Keep the shipping cache setting unchanged. There is no measured >5% gain and
neither requested point fits the current shared desktop with the required
physical reserve. Before any retry, free RAM through owner-coordinated app/work
closure, then remeasure after pausing Studio. Even the native-only lower bounds
are 14.291 GiB available for +3.5 and 17.791 GiB for +7, plus controller overhead.
Do not infer faster expert decode from a larger Engram row-cache allocation.
A host **expert** victim tier would require implementation and separate
correctness/performance qualification; no such feature is delivered here.
KL/top-1 is skipped because no larger-cache decode ran.

Studio was restored by its canonical launcher: /health 200, Remote Link running,
serving and end-to-end OK. Restoration had an 8 GiB charged-memory ceiling and
4 GiB physical floor; observed minimum available RAM was
11.886 GiB. No task GPU hold remains.
No model weights, defaults, production data or presets were changed. All new
scratch/build/cache files stayed in the single task scratch root;
no Mac or M: writes occurred. Native binaries were copied and hashed from the
previous qualified smoke; none was executed. Only evidence/documentation is
published. See [receipts](../examples/ramcache/summary.json).

STATUS: BLOCKED - larger-cache performance runs refused physical RAM admission;
baseline/source evidence delivered, Studio and Remote Link restored.
