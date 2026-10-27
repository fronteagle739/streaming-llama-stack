# Measured performance

These are existing Flash 1017 Q4 measurements from October 24–27, 2026. Windows runs used an RTX 5090, 32 GB of system RAM, local NVMe and explicit host-memory budgets; Mac runs used an M3 Ultra with 512 GB unified memory. Rates depend on content, routing, cache warmth and context. ULTRA changes arithmetic relative to Accurate; YOCO additionally approximates prefill.

## ULTRA prefill

The fold-planner sweep curve used two runs per prefix of the same document. Rates include prefill setup, rather than timing only transformer layers.

| Prompt tokens | Mean prefill (s) | Prefill tok/s |
|---:|---:|---:|
| 28,241 | 28.66 | 985 |
| 32,234 | 31.10 | 1,036 |
| 39,864 | 35.27 | 1,130 |
| 48,520 | 41.70 | 1,163 |
| 51,129, with spill | 45.51 | 1,124 |

On the complete 103,189-token prompt, primary sweeps of 45,056 / 49,152 / 53,248 took 94.03 / 93.04 / 94.45 s, with identical payloads. The default is 49,152 with a 61,440 fold cap; `[49152,54037]` avoids a third small sweep for this prompt.

[Approximate YOCO140](ULTRAYOCO_QUALIFICATION.md) raised startup-plus-prefill throughput from 1,063.7 to 1,373.4 tok/s at 43,327 tokens (1.2912×), and 1,127.2 to 1,634.5 tok/s at 103,189 tokens (1.4501×). These paired comparisons are more appropriate than comparing unrelated historical runs. Earlier 32K/64K ULTRA records used a one-token-warm timing boundary and are not interchangeable with startup-inclusive rates.

## Single-engine decode and context

Overall decode includes the initial warmup; late 100-token windows can be much faster at Max speed.

| Prompt | Bias off (tok/s) | Max speed (tok/s) | Timing scope |
|---:|---:|---:|---|
| 14,859 | 6.03 | 12.93 | 500-token run overall; final Max-speed window 21.19 |
| 43,327 | 5.97 | 12.65 | 500-token run overall; final Max-speed window 19.55 |
| 237,703, 999,424 configured | 5.184013 | — | 52 evaluated tokens in 10.030840 s, including reasoning |

The last row returned the correct long-document marker. Native embedding/layers/head compute took 254.3255841 s; the complete request took 273.529 s. This establishes long recall with the 1M configuration, not a million-token prompt benchmark. Later short two-turn checks establish session continuation independently of this long recall.

An older 224,588-token recall reported 2.15 tok/s for a six-token reply, including one-time context/open/inspection overhead. Its token steps ran at 131–179 ms once warm (5.6–7.6 tok/s). It is not a comparable sustained-rate regression against the longer decode above.

## RTX prefill → Mac decode (measured before YOCO, plain ULTRA prefill)

Paired YOCO has no measured row yet. Its [release gate remains incomplete](YOCO_PAIRED_QUALIFICATION.md), including image transport, native sampler support and the live browser gate. CPU checks and a Mac build are not performance measurements.

The 184,432-token document used four RTX chunks. Native RTX compute totaled **173.2521194 s**, excluding the 16.1972473 s one-token warmup and other serialization/startup work. Four transfer/import/hash/ACK intervals totaled **7.662391 s**. Mac decode evaluated **97 tokens in 3.080 s = 31.4935 tok/s**, including reasoning. The Mac server reported **229.128 s** through prompt processing and **232.208 s** through generation.

A subsequent session used 21,398 tokens then a follow-up at 21,459 total; only 22 new prompt tokens were processed. Decode was 27.56 and 32.36 tok/s respectively. This is continuation evidence from a separate request, rather than a follow-up measurement on the 184K document.

Mac-only ds4 first-100 decode windows were 30.7–31.2 tok/s at approximately 9K and 31.26–31.47 tok/s at approximately 24K. Whole-response rates varied with the denominator; the first-100 windows provide a consistent comparison to the paired path. Mac prefill on real content was about 350–390 tok/s; an earlier roughly 489 tok/s synthetic-prompt result repeated system text and is not representative of the same workload.

## Quality/speed dial

Native routing measurements with 3K-token story and C++ workloads:

| Setting | Bias | Story decode tok/s | C++ decode tok/s |
|---|---:|---:|---:|
| Max quality | 0 | 5.8 | 5.0 |
| Balanced | 0.35 | 8.1 | 7.7 |
| Fast | 0.65 | 10.3 | 9.7 |
| Faster | 0.9 | 12.3 | 10.7 |
| Max speed | 1.36125 | 14.2 | 13.3 |

Faster introduced a code typo; Max speed introduced C++ syntax errors. These are different workloads from the single-engine 500-token runs. The routing dial is independent of prefill arithmetic and YOCO. [Live controls](LIVE_GENERATION.md)

## Memory

Single-engine launch presets use a 6.25 GiB producer base plan, 7.75 GiB producer maximum and 9.25 GiB controller-plus-producer budget, an 8 GiB desktop admission reserve and a 4 GiB runtime physical-RAM floor. The paired 184K run used a separate 8 GiB producer host budget. GPU private-commit backing has its own ceiling; it does not authorize that amount of physical host RAM.

Global pagefile counters in some historical runs rose slightly despite ample free RAM. Those counters cannot attribute paging to the producer. No machine-wide zero-paging claim follows from these measurements.
