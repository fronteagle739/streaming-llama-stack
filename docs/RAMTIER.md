# Single-engine host expert tier

The producer can now use the host-cache implementation from standalone
ds4 between its VRAM L2 and DirectStorage. The versioned `ds4x_host_cache_v1`
bridge reuses ds4's allocation/backoff, LRU/admission and pinned upload code.
L2 victims populate RAM; with L2 disabled, L1 victims populate RAM. RAM hits
upload the original expert bytes and avoid an NVMe read. Copies complete before
publishing a key or reusing a source. The payload uses ds4's host allocator and
is fully locked with `VirtualLock`; one expert's pinned ds4 buffers stage both
directions. CUDA never receives a pageable pointer. This avoids WDDM's observed
whole-tier CUDA pinning limit. Mixed quantization transfers only the original
layer's extent. The quality/speed residency hook includes host experts.

History searches in the clean stack and the earlier stack found no host expert
tier in the single-engine provider. The earlier LLAMADECODE plan proposed one,
and the generic weight fetcher had a different host cache, but neither was wired
into this provider. Studio's standalone ds4 runtime does configure a host expert
cache. The Engram row cache is a separate feature.

Set `LLAMADECODE_HOST_CACHE_MB` through `env_overrides` (or the corresponding
launcher flag) to `0`, a MiB upper bound such as `3584`, or `auto`. The initial
qualification default is off. `DS4_CUDA_STREAMING_HOST_CACHE_N` optionally adds
the standalone setting's expert-count cap. Both process and physical headroom
limit the allocation; a requested size is an upper bound, never permission to
page. The matching producer and decode DLL must be built together.

`producer_decode_host_budget_gib` optionally sets a larger process ceiling than
the prefill workspace plan. It must fit the aggregate `host_budget_gib`, leaving
512 MiB for the controller, and passes physical admission before launch. It
does not enlarge the prefill fold planner's workspace. The runtime floor defaults
to 0.5 GiB on RTX and Mac; the desktop admission reserve remains 512 MiB and all
paging stops remain enabled. Cache admission keeps the runtime floor **plus**
the desktop reserve and an additional 256 MiB allocation margin.

At the transition to external decode, completed Engram prompt/row buffers and
prefill carry pages are released before measuring cache headroom. The carry
keeps its completion certificate; native exports use separate tokens, logits
and GPU owner state, and never read the retired hidden rows. Before an append
restores the full prefill plan, the expert tier is dropped first, including
before speculative Engram allocation. The carry bank is recommitted and locked
at the same address, then a fresh store seeds the new rows. The expert tier is
re-admitted after VRAM L2 growth at the next decode edge; no old/new generations
overlap. Whole-bank CUDA-registered carry is retained rather than decommitted.
The log records both exact carry bytes and available RAM before/after release.

Per-token `LLAMADECODE_DS4X_TOKEN` counters distinguish `l1_hits`, `l2_hits`,
`host_hits`, and NVMe `misses`. `host_slots` and `host_bytes` report the actual
allocation. Engram counters are not expert hits. Warm rates exclude the first
100 generated tokens and exclude append prefill.

## Measurements

One RTX 5090, 32 GB host RAM, the original Q4_K/Q8_0 checkpoint, actual stack
WebUI in headless Chromium. Each setting generated 1,000 tokens; each warm
interval contains 899 evaluated steps after the first 100. Bias was 0/1.36125
with the qualified sampler settings. Recall and prefix reuse passed. The
1,000-token Max quality output matched cache-off exactly at both tier sizes.

| Actual host tier | Mode | L1 / L2 / RAM / SSD lookups (%) | Warm tok/s |
|---|---|---|---:|
| 0 | Max quality | 35.85 / 24.64 / 0 / 39.51 | 7.26 |
| 0 | Max speed | 54.03 / 43.28 / 0 / 2.69 | 18.89 |
| 3.49 GiB, 188 slots | Max quality | 35.85 / 24.41 / 3.47 / 36.27 | 3.50 |
| 3.49 GiB, 188 slots | Max speed | 52.04 / 42.88 / 2.60 / 2.48 | 12.43 |
| 10.99 GiB, 593 slots | Max quality | 35.85 / 24.41 / 8.57 / 31.17 | 3.36 |
| 10.99 GiB, 593 slots | Max speed | 50.43 / 42.19 / 5.39 / 2.00 | 11.32 |

The historical baseline is 6.8118/18.1550 tok/s. The new cache-off control
reproduced its exact expert lookup counts. VRAM L2 varied from 668 slots in
the control to 658 in the cache runs. The implementation evolved while fixing
admission: the final maximum point includes carry release, while the middle
point retains carry pages. Per-point producer hashes and final binary hashes
are in the [summary](../examples/ramtier/summary.json). These observations are
not an identical-binary statistical comparison.

RAM hits avoid SSD reads, but a victim admission requires a device-to-host
transfer and staging copies. The measured configurations are slower, so
`LLAMADECODE_HOST_CACHE_MB=0` remains the default. No speedup is claimed.

The maximum trial used a 13.75 GiB aggregate task budget and 12.5 GiB producer
ceiling. It released exactly 4,574,535,680 bytes (4.26 GiB) of carry pages;
the observed headroom increase was 4,599,169,024 bytes (4.28 GiB), including
rows and concurrent system variation. Auto admitted 11,804,590,080 payload
bytes plus one expert's pinned staging, inside the ceiling and physical guard.
Minimum observed available RAM was 1.79 GiB. A 419-token append reused 51,733
tokens, released the tier, recommitted the carry bank, and regrew the tier;
it returned the original retrieval key. With the former 4 GiB runtime floor,
the same admission policy would reserve another 3.5 GiB, leaving approximately
7.5 GiB for this tier; that is a capacity estimate, not a separate measured run.

All three accepted producer receipts show zero budget breaches and confirmed
graceful exit. Global pagefile growth was 9.5/202.1/64.4 MB for off/middle/auto,
below the existing 256 MB stop. This is not attribution to the producer and is
not a zero-paging claim. The payload and carry allocations are explicitly locked.
The GPU byte oracle passed 69 comparisons, including mixed extents, LRU wraps,
padding and teardown. The CPU carry test passed 40 unchanged layers followed
by certificate retention, physical release, recommit/reseed and suspended-bank
destruction. Mac floor changes are source-tested here, not hardware-qualified.

Failed attempts remain disclosed: allocating the whole pinned tier before L2
caused a CUDA allocation refusal; allocating it after L2 backed 3.5 GiB down
to 0.93 GiB, motivating locked payload plus small pinned staging. A failed first
maximum trial exceeded its aggregate RSS ceiling by 16 KiB. Its shutdown hit
the stack's forced owned-job stop fallback, violating the experiment's
no-force-kill rule. That trial is excluded. Subsequent task cleanup used an
independent early cooperative-cancellation watcher; the accepted maximum run
needed no cancellation and has its own graceful native cleanup certificate.
