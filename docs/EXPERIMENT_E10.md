# E10: bounded CUDA HC graph islands

Adopt `DS4_CUDA_SINGLE_GPU_HC_GRAPHS=1` for CUDA single-GPU V4.1 decode.
Explicit0 retains eager execution. Every example and the shared profile overlay
uses1 while preserving an explicit local0. Metal/ROCm and unsupported graph
backends retain their prior path. Profiling defaults off.

The hypothesis is that position-independent HC normalization and input mixes
pay avoidable host launch cost. Only those operations are captured:38 islands
per token, excluding Engram layers1/14. Positions, KV writes, attention, router,
expert callbacks and weight streaming stay outside capture. Dense rebind/drop
invalidates captures; the existing graph state machine handles warmup,
capture failure and fallback. Full-step graphs and PDL are not implemented.

Both matched pairs use the same original Q4 GGUF,8,192 frozen tokens,
quality bias0,LFU recency HALF16 fixed,existing decode graphs enabled,
and the same binary/DLL. A32-sample warmup precedes128 natural samples;
127 actual evaluated tokens form each timed window. No EOS enters it.
Each timed token witnesses38 eager calls/0 replays in control and38/38
in candidate. Reverse the order in pair2.

| Pair | Eager decode tok/s | HC graphs tok/s | Change | HC host encode ms/token, eager -> graph |
|---|---:|---:|---:|---:|
| Control first | 5.841747 | 5.959057 | +2.01% | 1.991 -> 0.571 |
| Candidate first | 5.846415 | 5.948570 | +1.75% | 2.027 -> 0.577 |

The host interval measures encoding/dispatch wall, not GPU kernel duration.
HC savings total180.388/184.145ms per127 tokens. Total decode savings
428.074/372.042ms also include DirectStorage improvements273.397/200.391ms;
these overlap, so do not sum them or attribute the whole gain to launches.
Both arms stream exactly333,693,665,280 bytes, with6,661 L1 hits,7,056 L2
hits and16,763 misses in each timed window. The tested gain is small and
workload-specific; it is not a5-20% full-step-graph claim.

All34 full-vocabulary logit rows across the two pairs are bit-identical,
KL0,top-1 agrees,recall passes, and every natural/warm/decode prefix and
checkpoint matches. The conservative default follows the owner's rule for
an exact speedup. The final default-on DLL is separately qualified against
explicit0 at2K; the original paired DLL remains pinned in the JSON.

| Pair/arm | Prompt execution s | Prompt tok/s | End-to-end TTFT s |
|---|---:|---:|---:|
| 1 eager | 17.962859 | 456.052 | 20.969 |
| 1 graph | 16.859708 | 485.892 | 19.641 |
| 2 eager | 17.046260 | 480.575 | 19.797 |
| 2 graph | 17.746727 | 461.606 | 20.578 |

Prompt time is native embeddings through frontier logits across all chunks;
startup/model open and bulk export are excluded. TTFT includes startup and
first output, excluding lease queue and harness preparation. HC graphs affect
decode; the prefill variations are not a prefill improvement claim.

Rejected evidence: the initial presence-only switch enabled graphs for both
literal0 and1, so that pair cannot measure graph effect. A later control
failed after prefill because a JSON report was parsed with the config loader;
that arm is excluded. A reverse-order candidate completed cleanly but its
post-run Remote Link check failed; the harness stopped, waited for relay
recovery, and resumed only the completed arm with matching pins and fresh
passing health. Its control then completed normally. No Studio session or
foreign hold was changed.

Reproduce with `scripts/perfexp_pair.py`, the pinned settings and seed digest
in `examples/perfexp/e10/results.json`, then run `analyze.py <receipt-root>
E10-hc-fixed1 E10-hc-fixed3`. Guarded host cap6.5GiB, native6.25GiB plan,
physical floor0.5GiB and desktop reserve512MiB remain enforced. All accepted
arms close gracefully with confirmed owned cleanup. Zero bounded paging
observations do not prove zero disk I/O; ETW attribution was unavailable.
