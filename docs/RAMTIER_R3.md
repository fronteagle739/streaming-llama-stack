# Single-engine RAM experts: Round 3

The accepted Round 2 maximum-size upload, CPU and mix runs already used an
explicit **10240 MiB** setting, with a **10,729,635,840-byte** payload rounded
to 539 whole experts. Each completed its append and graceful shutdown under
the cooperative guard. Those performance points remain the measurements below;
Round 3 did not repeat them or change auto-sizing. There is no new capacity cap.

| RAM tier setting | Mode | Max quality | Max speed |
|---|---|---:|---:|
| Off | GPU | 6.87 tok/s | 18.26 tok/s |
| 3.5 GiB | Upload | 6.79 tok/s | 17.34 tok/s |
| 3.5 GiB | CPU | 6.81 tok/s | 14.58 tok/s |
| 10 GiB | Upload | 6.49 tok/s | 15.23 tok/s |
| 10 GiB | CPU | 6.93 tok/s | 13.22 tok/s |
| 10 GiB | Mix | 6.94 tok/s | 12.08 tok/s |

These are the original Q4_K/Q8_0 model, approximately 50K context, 1,000-token
WebUI replies and warm decode after the first 100 tokens. Hit rates and
per-token timings remain in [Round 2](RAMTIER_R2.md). CPU/mix free-running
sequences can choose different tokens and do not represent identical routed
work. The cache stays off by default; upload remains the default service for
an explicitly enabled tier.

## CPU correctness verdict

CPU and mix **do not pass the strict full-sequence Max quality screen** and
must remain experimental. The earlier first-16-row check missed later
differences. It cannot support a claim of sequence equivalence.

| 10 GiB setting, forced Max quality | Mean KL | Maximum KL | Matching top-1 |
|---|---:|---:|---:|
| CPU | 0.00567974 | 0.158810 | 252 / 256 |
| Mix | 0.00661200 | 0.217350 | 253 / 256 |

Those Round 2 tests held every input token fixed at bias 0. The conservative
screen was maximum KL below 1e-5 and at least 99% top-1 agreement; its numeric
thresholds were agent-selected, not an owner specification. Neither mode
passes. No CPU performance point is promoted to a qualified default.

Round 3 checks the whole forced sequence with an explicit 10 GiB tier. The
`reference` verification mode computes both the hybrid and all-GPU FFN on the
same activation, compares them, then forwards the GPU row. This prevents CPU
rounding from changing later layer inputs. It checks **4,049 hybrid rows /
4,864 CPU experts**, including **1,817 rows / 2,171 experts at Max quality**.

The maximum local FFN relative RMS error is **0.00109246**, rather than the
first-16 result of about 2e-7. **59 rows** exceed 1e-6 (30 at Max quality).
The maximum unquantized middle-row relative RMS difference on those rows is
**7.45493e-7**. The diagnostic records the Q8_K input and middle quantization
separately, including sign-normalized integer codes so equivalent opposite
scale-sign encodings are not mistaken for different quantized values.
All 59 discrepant rows have **identical sign-normalized input codes and
scales**, and **exactly one changed middle Q8_K integer code**. This identifies
middle activation quantization as the amplification point, rather than a
different input quantizer or a missing expert contribution.

Forwarding GPU rows restores **bit-for-bit identical full-vocabulary Max
quality logits across all 256 positions**, with KL zero and 256/256 matching
top-1. This localizes the quality difference to the hybrid FFN path. Small
middle-row rounding can cross Q8_K quantization boundaries before the down
projection, so local FP32 roundoff alone is an inadequate sequence tolerance.
The sequence verdict above remains the delivery criterion.

This is a correctness diagnostic, not another speed measurement. It performs
additional GPU FFNs and uploads, changing cache promotions. Its Max speed
residency-biased routing consequently differs from an ordinary CPU run; the
speed vectors are recorded but do not isolate CPU arithmetic.

## Reproduction and memory

Use the existing `ComboProducer.decode_forced` path with the original
49,553-token prompt, the same 256 forced tokens at each dial endpoint, bias
0 / 1.36125, a fixed 640-slot VRAM L2 and
`LLAMADECODE_HOST_CACHE_MB=10240`, `LLAMADECODE_HOST_MODE=cpu`,
`LLAMADECODE_HOST_CPU_VERIFY=reference`. Compare full-vocabulary vectors only
after verifying input-token identities. `all` instead forwards hybrid rows;
`1` retains the bounded first-16 diagnostic. Disable verification for timings.

All GPU diagnostics have a 13.25 GiB aggregate RSS budget, a 12.5 GiB native
producer ceiling, a 512 MiB runtime floor and an early independent RAM/paging
watcher. No paging limit was relaxed. Every completed diagnostic shut down
gracefully with exit 0 and zero native budget breaches. Binary/source hashes,
input identities, the complete discrepant-row counters and the final memory
observations are in the [machine-readable Round 3 record](../examples/ramtier/round3-summary.json).
The final diagnostic peaked at **11.39 GiB** aggregate RSS with **4.18 GiB**
available physical RAM remaining. Observed global pagefile growth was
**14,471,168 bytes**; this is not process attribution or a zero-paging claim.

The branch also incorporates current main's separately delivered cooperative
retirement and README changes. Round 3 did not author a README change.
