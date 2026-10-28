# E8: recency-weighted L2 admission

Verdict: **adopt the opt-in LFU policy with a 16-token half life**. On one frozen
8K synthetic-record workload, it reduced routed expert-miss bytes by 11.7% in
both repetitions. Warm decode improved by 1.5-10.6%. Keep LRU as the default;
this is not a general speedup or a long-context qualification.

```text
LLAMADECODE_RECENCY_LFU=1
LLAMADECODE_LFU_HALF_LIFE=16
```

These are advanced settings. The policy defaults off, and the half-life control
defaults to 1 for reproducing the first arm. Scores affect L2 victim selection
and admission only. Routing, weights, arithmetic and L1 placement are unchanged.
The least-scored eligible L2 slot is selected, excluding any live source slot.
An incoming victim must exceed its score by 25%; otherwise it is discarded.
Scores halve at the selected token interval, with LRU breaking equal-score ties.

## Matched same-day results, 2026-10-28

Each pair used the same current Q4 model and producer binary, F32 context cache,
ULTRA arithmetic, 512 MiB desktop reserve and 0.5 GiB runtime floor. Only the
LFU enable setting varied. In the 16-token pairs, half life was fixed at 16 on
both arms. No shipping preset changed.

The input asks for the archive key followed by 100 numbered review notes. The
timed window contains 127 evaluated tokens after 62 natural recall tokens and
31 further warmup tokens. It precedes all forced tokens and never reaches EOS.
Input SHA256:
`c0a99bfa1e76f4801c914514469f9e430873b47690f515e80615676d1a05c453`.

| Half life | Order | Aggregate host cap | LRU tok/s | LFU tok/s | LFU/LRU |
|---|---|---:|---:|---:|---:|
| 1 token | control first | 8.5 GiB | 5.4906 | 5.5248 | 1.0062 |
| 1 token | candidate first | 8.5 GiB | 5.6082 | 5.6441 | 1.0064 |
| 16 tokens | control first | 8.5 GiB | 5.7624 | 6.3731 | 1.1060 |
| 16 tokens | control first | 6.5 GiB | 5.8081 | 5.8944 | 1.0149 |

Every pair passed original-key recall. The untouched frontier and 16 forced
full-vocabulary rows were bit-exact, KL 0 and top-1 100%. Natural recall, warmup
and measured token sequences also matched before comparing the forced rows.
The 1-token arm is effectively neutral; its miss-byte reduction was only 0.38%.

Both 16-token repetitions had these exact counters over the same 127 positions:

| Counter | LRU | LFU, half life 16 |
|---|---:|---:|
| L1 hits | 6661 | 6661 |
| L2 hits | 7056 | 9013 |
| Misses | 16763 | 14806 |
| Routed expert-miss bytes | 333,693,665,280 | 294,736,527,360 |

All 30,480 expert triples reconcile. Effective fetch times varied, so the byte
reduction is more stable than the wall-time gain. The first control also had an
anomalous 149.8 s startup/prefill versus approximately 20-22 s subsequently;
that is excluded from the decoder-policy conclusion. It is not attributed to LFU.

## Guard and implementation checks

Native CPU tests cover admission margin, decay, changed working sets, invalid
keys and the 16-token cadence. Builds ran under a 2 GiB host/commit cap. Across
eight successful GPU arms, peak tree RSS was 6,087,868,416 B and minimum available
RAM 4,214,685,696 B. All engines closed gracefully and their owned PIDs retired.
Bounded owned-page observations were zero; maximum global pagefile growth was
13,385,728 B, below the enabled stop threshold and not attributable to this engine.
These counters do not prove exhaustive zero pagefile I/O.

One 8.5 GiB attempt was refused before child creation when available RAM was
6.11 GiB. A tighter 6.5 GiB trial then hit the native executable's additional
1 GiB pre-allocation reserve and retired before bank allocation. The harness now
waits for both aggregate and native admission, including startup overhead, before
claiming the GPU. Neither memory reserve nor paging stop was weakened. The
successful tighter pair kept the same 6.25 GiB native bank budget as its control.

The half-life-1 binary SHA256 was
`51935cd6df20b975d2093af6ac9d34b2206ea8fba24b3b72c57903e11d40da73`;
the extended binary was
`c26ac911a076b52df820109df4feab07012c7ce116ca7897471776f1c9b51dae`.
Each pair used one binary, and the native diagnostic confirms the applied mode.
Studio was initially unloaded; post-run health and Remote Link checks passed.

This arm does not implement copy-engine swaps, refill bursts or context-loan
slots. They remain separate experiments, with no performance claim here.
