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

## 5. Why the stream runs at 14-17 GB/s (standalone DirectStorage bench)

`examples/opus22/dsbench.cpp` reads 7.4 GB (one stage) from a file offset into a GPU buffer with one
DirectStorage queue (capacity 8192, priority HIGH, 256 MiB factory staging, fixed request size, one
fence), which is the producer's stage pattern without the producer. GPU leased, nothing else running.

| File | Offset GB | 1 queue, 8 MiB, GB/s (repeat 0 / 1, two sessions) | 2 queues | 4 queues |
|---|---:|---|---:|---:|
| Q4 GGUF (2,207 NTFS extents) | 50 | 15.2/15.5, 15.2/15.5 | | |
| Q4 GGUF | 120 | 21.6/22.4 | | |
| Q4 GGUF | 150 / 250 / 350 | 10.3-12.7 / 14.6-15.4 | | |
| Q4 GGUF | 420 | 24.1/23.6, 24.2/24.5 | | |
| Q4 GGUF | 200 | | 8.7-12.3 | 11.5-11.7 |
| FP4 proxy (93,776 NTFS extents) | 50-420 | 12.9-16.2 (second repeat 14.9-16.2) | 11.1-11.4 | 7.8-10.8 |

- The producer's settings match the qualified path. A plain sequential DirectStorage read with the same pattern gets
  the same 14-17 GB/s, so neither the request pattern nor the provider is the limiter. 2 MiB and 32 MiB
  requests are slower than 8 MiB, and more queues are slower.
- The rate depends on where the data is. The same file, request size and queue give 24 GB/s at one offset
  and 15 GB/s at another, and each region repeats its own rate across sessions. NTFS fragmentation
  does not explain it: the 420 GB region (24 GB/s) and the 250/350 GB regions (15 GB/s) each lie in a
  single 16-37 GB extent. The FP4 file's 93,776 extents cost little: it reads like the slow Q4 regions.
  The likely cause is the drives' internal placement from how the data was written. A fresh
  sequential rewrite might restore the fast rate, but there is no room on the volume to test that.
- The first 7.4 GB burst after an idle gap is 15-25% slower than an immediate repeat. That points at a
  drive or link power-state wake-up between stage reads, which the stage pipeline incurs after every
  wait for a free bank.
- With no prefill job running, D: showed 0 MB/s read and 0.1 MB/s write from other processes
  (5 two-second samples). Region rates repeat across sessions, so other-process contention does not
  explain them. A per-process log during the bench runs was attempted but recorded nothing, so
  contention during the runs themselves is not ruled out.

## 6. Placement and power-management tests

**Sequential copy.** Bytes 130-180 GB of the Q4 GGUF were copied into a new file on the same
array with one unbuffered sequential write (9 extents), then benched in the same session.
Each cell is the second read of a block; the first read in each pair is the process cold start.

| Block | Original GB/s | Copy GB/s |
|---|---:|---:|
| +0 GB | 17.5-18.2 | 16.6 |
| +7.4 GB | 14.9 | 20.2 |
| +14.8 GB | 14.7 | 23.7-24.0 |
| +22.2 GB | 23.1-24.2 | 20.7-20.8 |
| +29.6 GB | 13.9-14.2 | 14.7-14.8 |
| +37.0 GB | 12.0 | 14.5-14.6 |

The copy is not uniformly fast. The rate follows the physical region the data lands in, whether
it is original or rewritten. A sequential rewrite on its own does not give 24 GB/s.

**Pause sweep and keep-alive.** The same 7.4 GB block was read four times with 0, 10, 50, 200 or
1000 ms idle pauses between reads, then again with a 4 KB DirectStorage keep-alive read every
3 ms during 200 and 1000 ms pauses:

- Fast block (offset 152.2 GB): 23.0-24.5 GB/s after every pause length, with or without keep-alive.
- Slow block (offset 137.4 GB): 14.4-15.1 GB/s after every pause length, with or without keep-alive.
- The only slow read is the first read of each process (8.8-11 GB/s at pause 0 and 10). That is
  DirectStorage or D3D12 cold start, not an idle penalty.

Power management therefore does not cause the gap. Read-only `powercfg` for the active
"High performance" plan (AC values): PCI Express Link State Power Management 0 (off), primary NVMe
idle timeout 200 ms, secondary NVMe idle timeout 2000 ms, both NVMe latency tolerances 0, NOPPME 1,
AHCI link power management 0, turn off hard disk after 1200 s.

**Conclusion.** D: is one AMD RAID volume ("AMD-RAID Array 1", 4 TB); its member drives are not
visible to Windows. Read rate depends on the block address: about 24 GB/s in some regions and about
15 GB/s in others, stable across sessions, pauses and file copies. The array's own tools (member
drive health, temperature, firmware, stripe settings, per-drive benchmark) are the next place to
look. A uniform 24 GB/s would bring today's two-sweep 100K stream from about 28 s to about 19 s.

**After the owner's power change** (primary NVMe idle timeout 0, NOPPME off, ASPM already off),
the same pause sweep gives the same results: the fast block reads 23.4-24.6 GB/s and the slow block
14.5-15.1 GB/s for every pause and keep-alive setting. Occasional first reads of 9.7-10.2 GB/s
remain process cold start. The producer at 100K gives 35.44 s prompt (TTFT 41.05 s), streaming at
15.55 GB/s, with a bit-identical frontier. That is inside the earlier control range (32.9-38.1 s),
so the power change has no measurable effect.
