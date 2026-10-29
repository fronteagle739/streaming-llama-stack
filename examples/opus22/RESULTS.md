# OPUS22: where the 100K prefill time goes

Setup for every run: 100,000 identical tokens, cone 2541, bias 0, ULTRA, context
131072, two row sweeps 49152 + 50848, packed FP4 4K speed proxy (the inherited
lossy Q4-to-FP4 proxy), prompt-only (`--prefill-only`, one untouched frontier
probe), `LLAMADECODE_TILE_TIMING=1`, no Nsight, no per-node synchronization.
All 14 runs below return the same 129,280-float frontier
(SHA `afee4831...f9574a`), identical to the round-3 OTHER22 baseline.

Reproduce the tables with `python scripts/opus22_decompose.py TRACE PROMPT_SECONDS`.

## 1. The wall time is the weight stream

The 100K prompt streams 456.3 GB of weights in 64 DirectStorage stage fetches:
layers 0-20 twice (once per row sweep, 156 GB each) and layers 21-39 once for
the 5,792 padded cone rows (141 GB), plus the seed and head stages. One stage is
about 7.4 GB and the producer fetches one stage at a time.

Decomposition of the clean round-3 baseline (prompt 32.79 s); the rows add up
to the prompt:

| Pass kind | Passes | Wall s | Covered by own stage fetch s | Above fetch s | Host-serial + scaffold s | GPU tile busy s |
|---|---:|---:|---:|---:|---:|---:|
| Sweep 1, layers 0/1/2/8/14/20 | 6 | 5.57 | 2.82 | 2.75 | 2.50 | 2.84 |
| Sweep 1, other layers | 15 | 6.10 | 6.07 | 0.02 | 0.49 | 4.77 |
| Sweep 2, layers 0/1/2/8/14/20 | 6 | 5.23 | 2.99 | 2.24 | 1.76 | 3.40 |
| Sweep 2, other layers | 15 | 6.82 | 6.72 | 0.10 | 1.66 | 5.02 |
| Tail, layers 21-39 (cone rows) | 19 | 7.71 | 7.53 | 0.18 | 1.20 | 0.94 |
| Outside layer passes (seed, head, admission) | | 1.36 | | | | |
| **Prompt** | | **32.79** | **26.13** | **5.29** | | |

- 26.1 s of the 31.4 s layer wall is covered by the pass's own weight fetch.
  On 49 of the 61 passes the wall equals the fetch time to within 0.2 s.
  The total fetch time is 27.4 s (16.7 GB/s).
- Only 5.3 s is above the weight stream, almost all of it (5.0 s) on the 12
  passes over the first, Engram and KV-source layers (0, 1, 2, 8, 14, 20). There,
  serial host work runs while the GPU waits: per-tile `cudaHostRegister` of
  378 MB carry windows, seed-row and RAM-carry uploads (15-58 ms per tile), and
  Engram CPU look-ahead (about 0.1 s per Engram pass).
- The tail is pure weight streaming: 19 passes, 7.7 s wall, 0.94 s of GPU work,
  6.4-7.6 s of exposed weight wait.
- Kernel time (18.3 s kernel/copy union) is mostly hidden under the stream.
  Removing kernel time only helps on the 12 special passes.

Across 14 runs, prompt = total stage fetch time + 3.9 to 5.4 s (rejected
pipelined arm: +3.0 s). Run-to-run noise in the prompt (32.9-38.1 s for the same
binary and settings) is DirectStorage throughput noise (13.3-16.7 GB/s).

## 2. Structural causes, quantified

| Cause | Seconds of the 32.8 s | What removes it |
|---|---:|---|
| Weight-stream volume: 456 GB at 14-17 GB/s | 27.4 (fetch time), 26.1 on the critical path | Fewer bytes or a faster stream |
| of which: layers 0-20 fetched twice (two row sweeps) | 9.2-9.8 | One sweep. 100K needs a 12.23 GiB producer host budget (13.1 GB); the machine has about 11.8 GB free |
| of which: tail layers 21-39 for 2,541 cone rows | 7.5-8.3 | Faster storage. The packed FP4 path fetches full stages; whether the cone rows leave any of the 384 experts per layer unused was not measured |
| Serial host work on the 12 special passes | 5.0 | Overlap tile N+1 host prep/upload with tile N GPU work (OTHER22's async-layer line) |
| Outside layer passes | 1.2-1.4 | Seed and head |

The candidate causes listed in the brief, measured: the 4K tiling and per-tile
ggml graph rebuild are not on the critical path for 49 of 61 passes, because
those passes wait on the weight stream anyway (host graph build totals 0.18 s).
Routed-expert demand waits do not occur: the packed FP4 path fetches full
stages. Carry/host staging and pin registration cost about 4 s, all of it on
the 12 special passes. The per-tile DirectStorage bank handoff is about 5 ms per
stage.

**Bounds.** At the measured 16.7 GB/s, the two-sweep stream alone is 27.4 s.
A single sweep (about 300 GB) is 18 s at 16.7 GB/s and 12 s at the 25 GB/s
that earlier raw-array measurements reached. So 10,000 tok/s at 100K (10.0 s) cannot be reached on
this machine with this model's bytes, even with zero compute. That needs fewer
bytes per pass, more storage bandwidth (for example several volumes read in
parallel), or weights resident in memory.

## 3. DirectStorage experiments (exact, opt-in, defaults unchanged)

All of these change only how the same requests are submitted: same bytes, same
destinations. Every arm reproduces the frontier bit for bit. Each run is one
100K measurement; controls are interleaved with the candidates on the same
binary.

| Arm | Setting | Prompt s | Fetch GB/s | Control prompt s (fetch GB/s) | Verdict |
|---|---|---:|---:|---:|---|
| Pipelined waves | stage's waves submitted without host fence waits (reverted; `rejected-pipelined-waves.patch`) | 43.33 | 11.32 | 35.34 (14.86) | **Rejected, about 8 s slower.** DS sweep rate fell to 10.1 GB/s |
| Submit threads 4 | `LLAMA_WF_DS_SUBMIT_THREADS=4` | 33.07 | 16.14 | 34.09 (15.73) | Within noise |
| Submit threads 2 | `LLAMA_WF_DS_SUBMIT_THREADS=2` | 32.91 | 16.34 | 34.09 (15.73) | Within noise |
| Factory staging 512 MiB | `LLAMA_WF_DS_FACTORY_STAGING_MIB=512` | 33.44 | 16.17 | 33.26 (16.22) | No change |
| Staging 512 + 2 threads | both | 33.34 | 16.10 | 33.26 (16.22) | No change |
| 32 MiB requests | `LLAMA_WF_DS_REQUEST_MIB=32` | 35.34 | 14.62 | 33.13 (16.14) | Slower |
| 2 MiB requests | `LLAMA_WF_DS_REQUEST_MIB=2` | 34.95 | 15.05 | 33.13 (16.14) | Slower |

Earlier controls in the same session: 35.34 / 35.32 / 38.12 / 36.35 s at
14.86 / 14.89 / 13.32 / 14.31 GB/s. Two of these were intended as submit-thread
arms, but the variable was not yet registered in `envvars.json`, so the launcher
dropped it. No config line was printed, so they count as controls. The default (about 7.7 MB per request,
1,000 requests per stage, one fenced Submit per bank quarter) is the best of the
tested submission shapes. None of the existing software knobs on the
compute-direct path gets close to the array's raw rate. Submitting more requests
at once, across quarters, made the stream much slower. This matches the earlier
finding that OTHER22's sync-free async version lost time to weight-acquisition
waits.

The three typed settings are kept for reproducibility, documented in
`docs/VARIABLES.md`, and off by default. `LLAMA_WF_DS_FACTORY_STAGING_MIB`
accepts only 256 or 512, which stays inside the producer's existing 512 MiB
per-tier factory allowance.

## 4. What would actually move the number

1. Storage bandwidth or bytes, the only lever larger than about 5 s. One sweep
   saves 156 GB (about 9 s of stream at today's rate). It needs about 1.3 GB more
   host RAM than is free, or a smaller host carry record. The RAM record is
   about 92 KB per row; a BF16 residual would make it about 51 KB, and 100K would
   fit near today's budget. One sweep also needs the spill rows beyond the
   45,056 VRAM-resident rows copied asynchronously; CONEALL measured
   synchronous one-pass as no faster. A second volume read in parallel, or a
   whole-pass DirectStorage ring with CUDA-signalled fences, are the stream-rate
   options. Pipelining the existing per-quarter submissions is not one of them
   (section 3).
2. The 5.0 s of serial host work on the 12 special passes. Overlap it, and do not
   add DirectStorage contention while doing so.
3. Kernel work outside the special passes does not shorten the prompt until
   items 1 and 2 are done.
