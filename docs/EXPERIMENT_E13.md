# E13: selected Engram row I/O

Opt-in 512 B reads reduced requested Engram bytes by about 82% with exact
results. A 64 MiB row cache eliminated nearly all reads on repeated turns.
Neither change establishes a general TTFT or prefill speedup on this machine.
Verdict: **adopt the opt-in reader and bounded row-reuse capability; retain the
existing defaults and reject a generalized speedup claim**.

The weight ledger already excludes the 202,758,032,400 B Engram tables from the
315,825,603,008 B full weight sweep. The provider hashes actual tokens, sorts and
deduplicates row IDs, and loads selected rows through bounded IoRings. Whole-table
streaming was already absent; early submission and the row LRU already existed.

`LLAMADECODE_ENGRAM_READ_ALIGN=512` selects 512 B aligned reads; 4096 remains the
default. The model file's logical sector size must divide the selected alignment.
Incompatible geometry fails before submission. A 264 B row crossing a sector
boundary reads both sectors. This changes I/O geometry without changing row bytes
or arithmetic. Logical sectors were 512 B and physical sectors 4096 B in these runs.
Counters below measure API request bytes, not physical NVMe traffic.

## Matched controls, 2026-10-28

Every pair used one rebuilt producer and the same current Q4 model, input IDs,
ULTRA arithmetic and F32 context cache. Each fresh engine acquired the shared GPU
lease, with an 8.5 GiB aggregate host RSS cap, paging stop, a 512 MiB desktop
reserve and physical/runtime floors of 0.5 GiB. Only the named setting varied.

Producer SHA256:
`51935cd6df20b975d2093af6ac9d34b2206ea8fba24b3b72c57903e11d40da73`.
Its build also contained default-off E8 and KV diagnostic candidates; those arms
were disabled throughout E13. The model was never modified or copied.

Thirteen pairs completed: four sparse-sector pairs, three diverse-sector pairs,
three cache pairs and three early-submission pairs. Every initial/turn recall
passed. All 230 compared full-vocabulary rows were bit-exact: KL 0, top-1 100%.
Each ordinary pair compared an untouched frontier and 16 forced rows; cache pairs
also compared three turn frontiers. Forced rows were compared only after checking
that the preceding generated token sequences agreed.

Across 26 successful arms, sampled peak tree RSS was 6,173,294,592 B and minimum
available RAM 8,320,475,136 B. All closed gracefully with owned cleanup confirmed.
Bounded owned-page observations found zero paged bytes; maximum global pagefile
growth was 3,145,728 B. These counters do not establish exhaustive zero pagefile I/O
or attribute the system-wide growth to the engine.

## Diverse document: 4096 versus 512 B reads

The frozen 32K fixture contains synthetic programme records between the original
retrieval key and final question. Smaller fixtures retain both ends. Fixture SHA256:
`1619fb42193f8c72f165bb4cc5b57d159297b92469d3417bbd554ed5b7d3a33f`.
The 512 B arm ran first. First output was timed before correctness readback.

| Tokens | 4KiB TTFT s | 512B TTFT s | 4KiB prefill tok/s | 512B prefill tok/s | 4KiB requested B | 512B requested B |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 20.063 | 19.265 | 105.960 | 110.333 | 87,093,248 | 15,371,776 |
| 8192 | 19.891 | 20.500 | 427.312 | 414.449 | 206,983,168 | 36,540,928 |
| 32768 | 24.375 | 23.094 | 1388.828 | 1464.492 | 522,797,056 | 92,324,864 |

Prefill seconds were 19.328/18.562, 19.171/19.766 and 23.594/22.375 respectively.
At 32K, Engram wall time increased from 1915.988 to 2179.324 ms despite fewer
requested bytes. IOPS/setup and the fixed weight stream remain relevant; the
logical byte reduction must not be presented as measured physical bandwidth savings.

## Sparse repeated-background fixture

| Tokens | 4KiB prefill s | 512B prefill s | 4KiB tok/s | 512B tok/s | 4KiB requested B | 512B requested B |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 25.109 | 20.937 | 81.564 | 97.817 | 13,684,736 | 2,419,200 |
| 8192 | 25.094 | 19.875 | 326.453 | 412.176 | 13,688,832 | 2,419,712 |
| 32768 | 24.641 | 22.843 | 1329.816 | 1434.488 | 13,684,736 | 2,419,200 |
| 32768, reversed order | 21.578 | 21.703 | 1518.584 | 1509.837 | 13,684,736 | 2,419,200 |

This fixture hashes to only 3152 unique rows at every length. Apparent first-pass
speed gains did not survive reversal: Engram wall time was 1127.484/1158.502 ms
in the reversed pair. These early runs timed output after a frontier probe;
their TTFT method differs from the diverse-document runs. An early reporting
helper also duplicated the first word because the engine returns its last sampled
token without evaluating it. This did not alter native state or the logit gates.

## RAM row reuse: 0 versus 64 MiB

The existing `LLAMADECODE_ENGRAM_CACHE_MB` LRU is charged inside the provider's
explicit host admission. The 64 MiB arm held up to 226,705 rows and had no eviction
in these small A/B/A turn fixtures. Immediately preceding prompt reuse was not
assumed to establish reuse across intervening turns.

| Initial tokens | Cold prefill tok/s, 0/64MiB | Repeated A requested B, 0/64MiB | Engram ms, 0/64MiB | Full turn s, 0/64MiB |
|---|---:|---:|---:|---:|
| 2048 | 86.007 / 94.365 | 33,972,224 / 421,888 | 75.094 / 7.862 | 18.062 / 17.594 |
| 8192 | 436.907 / 437.654 | 33,959,936 / 409,600 | 64.638 / 10.032 | 21.938 / 17.531 |
| 32768 | 1480.036 / 1437.382 | 33,968,128 / 0 | 61.839 / 5.963 | 17.406 / 17.484 |

Cold prefill seconds were 23.812/21.703, 18.750/18.718 and 22.140/22.797.
The cache has no initial hits, so those cold timing differences do not measure
row reuse. Cold TTFT was 24.578/22.437 s with a preceding probe at 2K and
22.891/23.500 s before the probe at 32K. The 8K TTFT pair changed probe order
between arms and is not a valid timing comparison. No cache TTFT gain is adopted.
The strongest repeated-turn I/O reduction still left full turn time unchanged
at 32K because weight streaming dominates.

## Early hash submission: auto versus deferred

`LLAMADECODE_ENGRAM_EARLY=auto` was the control, versus `0` (deferred submission).
Both used the same deduplicated rows, 4KiB geometry and disabled row LRU.

| Tokens | Auto TTFT s | Deferred TTFT s | Auto prefill tok/s | Deferred prefill tok/s |
|---|---:|---:|---:|---:|
| 2048 | 19.047 | 19.454 | 111.931 | 109.320 |
| 8192 | 19.266 | 19.688 | 441.689 | 432.227 |
| 32768 | 22.328 | 22.157 | 1517.529 | 1529.642 |

Prefill seconds were 18.297/18.734, 18.547/18.953 and 21.593/21.422.
Byte counts were unchanged. The auto loads finished before the weight-stream
marker in these small fixtures. The deferred arm intentionally disables that
marker in the existing code, so its zero overlap field is uninstrumented and
cannot prove zero actual overlap. Lower deferred Engram wall time includes
different setup timing; it is not an end-to-end speedup. Retain auto.

## Verification and reproducibility

`scripts/perfexp_pair.py` records model and input identity, first-output timing,
full logits, recall, GPU samples, cleanup and post-run Studio/Remote Link checks.
It refuses evidence overwrite, bounds counts and host memory, respects shared
holds, and retains ownership if retirement cannot be certified. Example:

```text
python -B scripts/perfexp_pair.py --root <scratch> --experiment E13-sector --candidate LLAMADECODE_ENGRAM_READ_ALIGN=512 --count 8192
```

Native tests recover exact bytes for sparse/random rows, hits, duplicates and
sector crossings, reject invalid geometry, and preserve existing read-plan
regressions. The 400K-operation row-LRU stress and early-decision tests pass.
Two new lock tests and 20 existing lease tests pass. Transient ledger byte-range
locks are retried for at most one second using fresh reads; persistent lock
failures still fail closed. All builds/tests used explicit bounded host budgets.
