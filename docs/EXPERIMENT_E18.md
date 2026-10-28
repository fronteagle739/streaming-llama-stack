# E18: reuse a bounded host transfer registration

Adopt `LLAMADECODE_HOST_COPY_STAGING=auto`, preserving explicit0 for the
original direct tile copies and1 for forced staging. Every example and shared
profile uses auto. Native auto selects the64MiB slot only when a supported
fixed plan's logical rows exceed its resident hidden-bank rows. Short and
fully resident plans keep their original copies. Async lookahead is excluded.

The fixed owner allocates one pinned64MiB buffer, shared by subsequent sweep
executors, instead of repeatedly pinning/unpinning the carry window. Exact CPU
gather/scatter respects the existing padded RAM-page layout. Each contiguous
CUDA copy retires before the slot is overwritten or its result is scattered;
original compute/copy events, lease/generation/extent checks, validation and
publication credit remain. The outer explicit host cap includes the slot;
existing device-delta admission, physical floor and paging stops remain.
No whole-bank pin is attempted; SINGLEPASS already observed that approach OOM.

The100K matched comparisons use the original Q4 GGUF, zero quality bias,
no YOCO/cone, F32 growable context and49,152+50,848 full sweeps. Certified FP4
index scoring is fixed on. Only the staging setting changes within each pair.
Tile-level event/host clocks are enabled on both arms; there is no per-node
synchronizing profiler. Prompt time includes inter-sweep admission and any
slot allocation there, excluding initial process/open and bulk export. TTFT
includes startup and first output, excluding lease queue and harness setup.

| 100K pair | Direct prompt s | Candidate prompt s | Direct/candidate prompt tok/s | Throughput gain | Direct/candidate TTFT s |
|---|---:|---:|---|---:|---|
| Forced1, control first | 65.561857 | 63.306814 | 1525.277 / 1579.609 | 3.56% | 69.704 / 67.563 |
| Auto, candidate first | 65.542031 | 63.379807 | 1525.739 / 1577.790 | 3.41% | 69.750 / 67.515 |

Auto leaves the full-resident49,152-row first sweep direct and stages only
the spilling50,848-row sweep. Its first allocation occurs in inter-sweep
admission, inside the prompt clock. Forced1 exercises reuse across both sweeps.

| Host stage,100K | Direct | Forced1 | Direct repeat | Auto |
|---|---:|---:|---:|---:|
| Bind s | 2.956314 | 0.000081 | 2.949941 | 1.244837 |
| Registration s, nested inside bind | 2.949839 | 0 | 2.943434 | 1.242096 |
| Upload s | 4.415126 | 3.021206 | 4.602526 | 3.769388 |
| Readback s | 0.785048 | 1.473242 | 0.786301 | 1.368913 |
| Initial inspect/admit/allocate s | 1.743946 | 1.851343 | 1.801923 | 1.745324 |
| Carry registration calls | 337 | 0 | 337 | 132 |
| Fixed staging allocations / executor binds | 0 / 0 | 1 / 2 | 0 / 0 | 1 / 1 |

Bind and registration are nested views of the same interval, not separate
2.95s savings. Upload decreases; CPU scatter/readback gets slower. Auto removes
205 registrations, leaving132 in the unmodified first sweep. Initial startup
allocation is not materially improved; graph reservation and construction
remain. Exposed weight waits and other work also vary, so host intervals must
not be added as independent GPU-stage savings.

All four accepted pairs, including2,049-token partial-tile checks, have17
bit-identical full-vocabulary rows each, KL0, top-1 agreement, identical natural
recall and matching ready/frontier/recall/forced checkpoints. At2,049 tokens,
forced staging changes prompt16.415526 to16.986354s (124.821 to120.626 prompt
tok/s), TTFT21.891 to19.719s; startup varies substantially. Auto instead witnesses
enabled0, no allocation and the same direct-copy path as control: prompt
16.077675/16.606487s (127.444/123.386 prompt tok/s), TTFT18.890/19.391s.
No short-prompt speedup is claimed. The auto choice prevents forced staging's
extra buffer/copy path there, rather than interpreting I/O noise as a benefit.

A numerically complete earlier auto repeat is excluded because its control's
outer guardian hit the pinned bootstrap-status receipt race. Its command
status later reads0 and cleanup is confirmed, but the guard verdict is FAILED;
it is not accepted performance evidence. A fresh same-day pair passes every
guard. The guard implementation and limits were not changed.

Reproduce using `scripts/perfexp_pair.py` and the settings/pins in
`examples/perfexp/e18/results.json`, then `analyze.py <receipt-root>
<label:count>...`. The analyzer requires successful guardian receipts and
exact numerical/checkpoint gates. Accepted arms close gracefully with
confirmed owned cleanup and healthy Studio/Remote Link. Host cap7.5GiB,
native plan6.25GiB growing to6.416GiB, floor0.5GiB, desktop reserve512MiB.
One forced candidate observes182.8MB global pagefile growth below the256MB
stop; bounded owned observation remains0 and minimum available RAM3.81GB.
This is not attribution to the model or proof of zero I/O; kernel ETW is
unavailable. The accepted auto repeat has no observed global paging growth.
