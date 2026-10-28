# Prefill defaults and optional experiments

Single-machine DSV4.1 launchers use `ultra-cone`: ULTRA arithmetic, a padded
2,541-row exact dependency cone and automatic routed reads. The matched
49,555-token text prompt improved from 36.250 to 25.375 seconds including
startup, with bit-identical logits and cache hashes. Whole-pass routed reads
apply through 2,048 tokens; longer prompts route only the decoder tail.
The 262K preset keeps F32 cache; the 1M preset keeps BF16 cache. The measured
cone throughput uses BF16. [Qualification and receipts](MULTICHUNK_PREFILL.md).

Use `examples/single-machine/dsv41-ultra-cone-webui.ps1` or the standard
`scripts/start-dsv41-single-262k.ps1`. Python callers can select
`scripts/dsv41_single.py --preset 262k --profile ultra-cone`. Explicit
`accurate`, `fast` and full `ultra` profiles remain available.

The older `ultra-yoco` single-machine filenames now default to exact cone too.
Add `-ApproximateYoco`, or select Python profile `ultra-yoco`, for approximate
keep-140 replay. CONEALL adds negotiated exact-cone state v2 and makes keep2541
the ULTRA paired and large-RAM preview default as well. The full mode3 owner
layout and raw history remain unchanged; v1-only peers refuse cone pairing.
[Protocol and live qualification](CONE_STATE.md).

Single-machine examples also enable E8's recency LFU admission with a 16-token
half life. On the matched 8K decode workload, both repetitions were exact and
faster: 5.7624 → 6.3731 and 5.8081 → 5.8944 tok/s. These are separate F32-cache
decode controls; the cone timings above do not measure a combined speed gain.

## Optional settings

Put environment toggles under `env_overrides` in your private config.

| Setting | Default | Measured result and decision |
|---|---|---|
| `LLAMADECODE_RECENCY_LFU` / `LLAMADECODE_LFU_HALF_LIFE` | `1` / `16` in single-machine examples | E8: warm decode +1.5–10.6%, miss bytes −11.7% in both repetitions; logits and generated prefixes exact, KL 0. Set LFU to `"0"` for LRU. One-token decay remains optional and was effectively neutral (~0.6%). |
| Approximate YOCO | Off; explicit keep140/v1 only | 1.26–1.45× prefill speedup in earlier controls; perplexity +0.56%, 381/384 top-1 agreement. Explicit toggle. |
| `LLAMADECODE_APPEND_DECODE_MAX` | `0` in single-machine examples | Set `"96"` for resident follow-ups. A 32-token append at 2K improved prefill 17.360 → 4.453 s, TTFT 17.375 → 4.485 s, but KL was 0.010868. At 8K/32K, KL was 0.001079/0.004664. All had 100% top-1 and passed recall; they remain approximate. |
| `LLAMADECODE_PREFILL_ROUTED` | Automatic, whole pass through 2K; cone tail beyond | Explicit `"0"`, `"1"`, `"tail"` overrides. Whole-pass source-text 2K: 19.172 → 15.234 s, exact; 8K: 20.578 → 21.391 s, so do not extend the automatic cutoff. |
| `producer_chunks_per_pass` | `1` | E5 grouping was exact but N=2 was 16.2%, 15.3%, 14.7% slower at 100K/200K/400K. Explicit `2`–`4` or `"auto"` requires host admission; no grouped default. |
| `LLAMADECODE_ENGRAM_READ_ALIGN` | `4096` | E13's `"512"` reader is geometry-checked and exact (230 full-vocabulary rows across the experiment), but diverse 8K prefill regressed 19.171 → 19.766 s and reversed-order 32K regressed 21.578 → 21.703 s. Keep opt-in. |
| `LLAMADECODE_ENGRAM_CACHE_MB` | `0` | Explicit `"64"` bounded row reuse reduced repeated 32K Engram reads to zero; the whole turn was 17.406 → 17.484 s. Keep opt-in. |
| `LLAMADECODE_ENGRAM_EARLY` | `auto` | Deferring reads showed no consistent end-to-end win; retain auto. |

[E13 measurements](EXPERIMENT_E13.md) ·
[E8 measurements](EXPERIMENT_E8.md) ·
[E2c resident-follow-up report](https://github.com/fronteagle739/streaming-llama-stack/blob/90a1dcb343232efad025ec9491f837ce229eceb2/docs/EXPERIMENT_E2C.md) ·
[YOCO quality](ULTRAYOCO_QUALIFICATION.md).

Only the completed E13 and E8 commits were imported from the active PERFEXP
branch. The published E8 checkpoint qualified its 16-token variant as exact
and faster in both repetitions, so it is now the example default. Other unfinished
PERFEXP experiments remain outside this release. No GPU runs were performed
for this integration; all performance figures above are attributed measured
controls, not fresh integration benchmarks.
