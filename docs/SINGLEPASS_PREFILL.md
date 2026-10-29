# Prefill carry in VRAM

The 1M exact-cone preset now keeps more prefill carry in VRAM. It borrows the
growable cache's unused physical reservation during the initial prefill, then
returns that space before decode or a later append grows the cache. The total
device admission, 1 GiB prefill reserve, 512 MiB decode reserve, physical RAM
floor and paging stops remain enforced.

N=1 remains the default. Grouping two legacy chunks makes 100K a single weight
pass and halves the encoder weight reads, but the optimized two-pass control
is faster on this machine. The grouped path remains available explicitly.

## Results

RTX 5090, original Q4_K/Q8_0 GGUF, exact cone, routed tail reads, growable BF16
cache, 1,000,001-position context. Times include startup and the initial
checkpoint publication. The frozen fixture has one archive key, repeated
weather-station background, and the final retrieval question. These results
describe that fixture; prompt diversity can change routed weight reads.

| Tokens | N=1 control | N=2 control | N=2 with VRAM carry | N=1 with VRAM carry |
|---:|---:|---:|---:|---:|
| 50,000 | 25.344 s | same single-pass geometry | 24.203 s | 24.203 s |
| 100,000 | 45.140 s | 52.235 s | 44.632 s | 43.063 s |
| 200,000 | 93.141 s | 109.000 s | 95.672 s | 92.407 s |

100K N=1 controls: 45.719, 44.562 s. N=1 with VRAM carry: 42.954, 43.172 s. Grouped VRAM runs: 44.937, 44.328 s. The 100K table uses means; secondary lengths are individual runs.


All accepted controls and candidates preserve the complete frontier logits
hash and every owner/cache hash. Retrieval returns `AMBER OTTER 731`.
The timing runs also compare full-vocabulary, teacher-forced F32 logits after
the same natural continuation. Their bytes match, so their KL divergence is
zero. The instrumented profiles independently preserve the initial frontier.

## Where the grouped-pass overhead went

The old grouped plan kept the same 49,152-row GPU bank. Its last 4K tile served
as a spill slot, leaving only 45,056 rows resident. At 100K, grouping therefore
spilled 54,944 encoder rows; the N=1 control only spilled the 5,792-row end of
its final sweep. Those extra transfers and CUDA host registrations consumed
the expected weight-stream saving. Exposed layer acquisition was already
almost completely hidden by compute.

### Uninstrumented layer clocks

| Case | Acquire s | Executor s | Prepare s | Upload s | Compute/sync s | Readback s | Weight GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched-n1-50k | 0.004 | 22.127 | 2.823 | 1.737 | 17.168 | 0.338 | 228.018 |
| default-vram-n1-50k | 0.009 | 21.036 | 2.380 | 1.176 | 17.249 | 0.171 | 228.018 |
| matched-n1-100k | 0.016 | 40.031 | 4.886 | 2.927 | 31.592 | 0.521 | 393.723 |
| matched-n2-100k | 0.014 | 46.908 | 6.848 | 5.847 | 31.753 | 1.374 | 227.341 |
| matched-vram-n2-100k | 0.009 | 39.049 | 4.454 | 2.246 | 31.917 | 0.326 | 227.341 |
| matched-vram-n1-100k | 0.011 | 38.727 | 4.371 | 2.284 | 31.641 | 0.327 | 393.723 |
| matched-n1-200k | 0.025 | 86.700 | 8.779 | 5.050 | 71.859 | 0.813 | 725.115 |
| matched-n2-200k | 0.011 | 100.749 | 13.212 | 11.090 | 73.646 | 2.599 | 392.350 |
| matched-vram-n2-200k | 0.007 | 88.207 | 8.732 | 4.774 | 73.703 | 0.807 | 392.350 |
| default-vram-n1-200k | 0.011 | 85.853 | 8.566 | 4.572 | 71.892 | 0.628 | 725.115 |

### CUDA diagnostics

| 100K stage (seconds) | N=1 control | N=2 control | N=2 VRAM | N=1 VRAM |
|---|---:|---:|---:|---:|
| Layer wall | 43.458 | 49.521 | 42.481 | 42.707 |
| Executor wall | 43.218 | 49.298 | 42.151 | 42.474 |
| Exposed layer acquire | 0.063 | 0.022 | 0.023 | 0.046 |
| Host prepare (includes registration) | 4.924 | 6.918 | 4.439 | 4.446 |
| CUDA host registration | 1.893 | 3.835 | 1.383 | 1.430 |
| Graph build | 0.258 | 0.268 | 0.258 | 0.255 |
| Graph reservation | 2.233 | 2.276 | 2.244 | 2.222 |
| Upload/validation wall | 2.867 | 5.911 | 2.289 | 2.278 |
| Expert GEMM GPU | 9.584 | 9.594 | 9.607 | 9.607 |
| Attention/indexer GPU | 10.863 | 11.099 | 11.098 | 10.872 |
| Named cache publication GPU | 0.089 | 0.112 | 0.112 | 0.089 |
| Other GPU | 7.311 | 7.335 | 7.332 | 7.332 |
| Readback/validation wall | 0.502 | 1.352 | 0.306 | 0.309 |
| Final tile synchronization wall | 0.019 | 0.019 | 0.019 | 0.019 |

| Prompt / profile | Register s | Upload s | Readback s | Expert GPU s | Attention/indexer GPU s |
|---|---:|---:|---:|---:|---:|
| diag-n1-100k | 1.893 | 2.867 | 0.502 | 9.584 | 10.863 |
| diag-n2-100k | 3.835 | 5.911 | 1.352 | 9.594 | 11.099 |
| diag-vram-n2-100k | 1.383 | 2.289 | 0.306 | 9.607 | 11.098 |
| diag-vram-n1-100k | 1.430 | 2.278 | 0.309 | 9.607 | 10.872 |
| diag-n1-200k | 3.113 | 5.051 | 0.771 | 18.373 | 25.832 |
| diag-n2-200k | 7.218 | 11.235 | 2.543 | 18.311 | 26.273 |
| diag-vram-n2-200k | 2.994 | 4.810 | 0.769 | 18.347 | 26.322 |
| diag-vram-n1-200k | 2.752 | 4.392 | 0.587 | 18.363 | 25.836 |
| diag-n1-50k | 1.128 | 1.681 | 0.328 | 5.056 | 5.121 |
| diag-vram-n2-50k | 0.742 | 1.202 | 0.163 | 5.055 | 5.129 |
| diag-vram-n1-50k | 0.720 | 1.157 | 0.162 | 5.063 | 5.121 |


The host and GPU clocks overlap. Registration is inside preparation; upload
and readback include their copy/validation waits. Final tile synchronization
is inside compute. CUDA node profiling synchronizes individual operations, so
its GPU-stage table is diagnostic evidence, separate from throughput timing.
The first snapshot of each pass also includes its seed operations. Named
cache writes are counted as publication; unnamed writes remain in other GPU
work. The per-layer data retain all of these clocks rather than adding them.

The compute tiles stay at 4K, with the existing attention and expert kernels.
Next-layer DirectStorage prefetch continues across the entire tile loop.
Graph construction and reservation did not explain the regression; the fix
removes spill-related registration, transfers and synchronization where the
carry fits.

## Bounded allocation lifetime

The original plan charges the full-context cache plus the original hidden
bank. During prefill, only the prompt's step-rounded cache prefix needs
physical backing. The temporary hidden budget is capped by the difference
between those two cache footprints. The cache checks the actual CUDA mapping
granularity and physical sum before accepting a temporary growth ceiling.
Any attempt to cross that ceiling fails before mapping more memory.

At the first decode, compaction frees the enlarged hidden bank before clearing
the cache ceiling. A direct append follows the same retirement order and
restores the original hidden-bank size and graph reservation. Initial
encoder-only continuations retain the loan until their logical prefill ends.
The cache's virtual addresses, precision, contents and public checkpoint
format are unchanged.

A paired 2,048-token direct append crossed from 100,352 to 102,400 mapped
positions after retiring the loan. Initial and appended checkpoint hashes and
the subsequent forced logits match the control. This covers the path that
appends before any decode has compacted the prefill reservation.

## Settings and limits

The 1M text exact-cone launchers enable `LLAMADECODE_SINGLEPASS_VRAM=1` when
direct open and decode compaction are available. An explicit `0` is the
control. Other launcher profiles retain their existing policy. No larger
host-memory allowance is needed for the default N=1 path.

Explicit `producer_chunks_per_pass=2` uses one pass at 100K and two at 200K.
Its larger locked RAM bank needs a correspondingly larger explicit host
budget. The original pinned RAM spill path remains the fallback when the
temporary VRAM bank cannot hold every row; a new overlapped host-staging
pipeline is not part of this change.

An entire-bank CUDA registration attempt was refused with out-of-memory
before weight execution. A bounded 16K registration window passed recall but
was slower. Matching the legacy index frontier was exact but did not improve
time. Those diagnostic options remain off. None changes the default arithmetic.

One VRAM qualification attempt stopped on system-wide pagefile growth above
the enabled 256 MiB threshold and is excluded from timing/quality claims.
Its owned cleanup was cooperative. The successful retry retained the same
guard. Bounded owned-page observations and global counters are reported in
the receipts; they are not an exhaustive zero-pagefile-I/O proof.

[Reproduction](../examples/singlepass/README.md) |
[Qualification](SINGLEPASS_QUALIFICATION.json) |
[Per-layer clocks](SINGLEPASS_PER_LAYER.csv).
