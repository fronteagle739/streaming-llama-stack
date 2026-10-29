# Performance

All RTX numbers come from one desktop: RTX 5090 (32 GB VRAM), 32 GB of system
RAM, Windows, NVMe RAID-0 at about 25 GB/s. The Mac is an M3 Ultra with 512 GB.
Every figure below was measured; each says what was timed.

## DeepSeek V4.1 Flash on one RTX 5090

Model: DeepSeek V4.1 Flash GGUF, Q4_K routed experts and Q8_0 for the rest
(483 GiB). Max quality uses unbiased routing on that GGUF. Experimental
packed FP4 and dense FP8 measurements use separately labelled model proxies.

### Prefill

Single-machine and paired ULTRA defaults use the exact 2,541-row dependency
cone. The current kernel stack includes certified FP4 index scoring, native-parity
fused Sinkhorn, TileLang attention, E18 automatic transfer staging, allocation-plan
reuse and the exact OTHER22 HC/CONCAT/QAT improvements. N=1 remains the default.

Latest 100,000-token measurements. The OTHER22 rows use exact cone, zero
quality bias, 131,072 context and 49,152/50,848-token sweeps. KERNELS2 supplies
a separate earlier Q4 default-retrieval control. Prompt-only covers embeddings through the frontier
head/synchronization, including inter-sweep admission. TTFT also includes
launch, model open, allocation and frontier certification; lease queue and
harness preparation are excluded.

| Build / configuration | Prompt-only time | Prompt-only rate | TTFT |
|---|---:|---:|---:|
| OTHER22 final source, packed FP4 proxy, graphs | **32.373785 s** | **3,088.92 tok/s** | **35.406 s** |
| OTHER22 rebuilt eager, packed FP4 proxy | 34.506991 s | 2,897.96 tok/s | 37.687 s |
| OTHER22 eager dense control, packed FP4 proxy | 32.615655 s | 3,066.01 tok/s | 35.797 s |
| OTHER22 eager FP8 dense proxy, packed FP4 proxy | 33.783610 s | 2,960.02 tok/s | 37.063 s |
| KERNELS2 default retrieval, Q4_K/Q8_0 | 38.368 s | 2,606.32 tok/s | 42.000 s |

[OTHER22 timing and graph conditions](../examples/other22/ROUND2_RESULTS.md) ·
[KERNELS2 Q4 default retrieval](../examples/kernels2/PERFORMANCE.md).

The 32.4 s result is the latest opt-in observation on the current kernel stack.
The graph experiment preserves the complete indexer certificate and BF16 fallback.
The production prefill preset retains `GGML_CUDA_DISABLE_GRAPHS=1`.

Earlier exact-cone defaults, before the combined FP4-index/fusion/attention/
allocation updates:

The 1M text preset keeps prefill carry in unused
physical space from the growable cache's original reservation, then returns
that space before decode or cache growth. Total device admission is unchanged.
These frozen recall-fixture runs use BF16 cache and include engine startup:

| Prompt tokens | Startup + prefill | Rate including startup | Peak guarded process-tree RAM |
|---:|---:|---:|---:|
| 50,000 | 24.203 s | 2,065.86 | 5.72 GiB |
| 100,000 | 43.063 s | 2,322.18 | 5.85 GiB |
| 200,000 | 92.407 s | 2,164.34 | 6.04 GiB |

The 100K time is the mean of 42.954 and 43.172 s; the other lengths are individual
runs. The corresponding N=1 controls were 25.344 / 45.141 / 93.141 s at
50K / 100K / 200K. Full checkpoint hashes and teacher-forced full-vocabulary
F32 logits match exactly, and all retrieval checks return the archive key.
The repeated background can affect routed weight reads; these numbers describe
this fixture. [Stage clocks, conditions and receipts](SINGLEPASS_PREFILL.md).

The earlier carry policy measured 25.375 s at 49,555 tokens, 45.484 s at 100K,
94.079 s at 200K, and 232.500 s (1,720.43 tok/s; 6.24 GiB engine RAM) at 400K.
At 49,555 tokens, the same-build full eager control took 36.250 s
(1,367.03 tok/s); the cone preserved every logit and cache checkpoint hash.
The 400K result has not been remeasured with the carry policy. The 262K launcher
retains growable F32 cache; the 1M launcher uses BF16. The paired cone image
and exact-cone one-million-token throughput remain unmeasured.

Earlier single-engine Q4_K/Q8_0, approximate YOCO140 controls from NATIVEFP4:

| Prompt tokens | Startup + prefill | Rate including startup |
|---:|---:|---:|
| 49,555 | 29.265 s | 1,693.32 tok/s |
| 200,000 | 96.984 s | 2,062.20 tok/s |
| 1,000,000 | 956.734 s | 1,045.22 tok/s |

The 1M run retained one session across twenty 49,152-token sweeps and a
16,960-token tail. [Frozen Q4 baseline receipts](../examples/native-fp4/q4-baselines.json).
These startup-inclusive rates are separate from the prompt-only 100K table.
[Earlier receipts](MULTICHUNK_QUALIFICATION.json) · [Defaults and toggles](MERGEPERF.md).

Earlier full ULTRA prefill, YOCO off, including startup:

| Prompt tokens | Prefill time | Prefill tok/s |
|---:|---:|---:|
| 28,241 | 28.7 s | 985 |
| 32,234 | 31.1 s | 1,036 |
| 39,864 | 35.3 s | 1,130 |
| 48,520 | 41.7 s | 1,163 |
| 103,189 | 91.5 s | 1,127 |

With the engine already loaded, one 49,152-token pass runs at **1,920 tok/s**.

**Earlier full-prefill one-million-token run.** A single session prefilled 1,000,000 tokens in
1,808 s (**553 tok/s**), in 49,152-token passes, without resetting the
context. Peak engine RAM was 5.7 GiB.

**Long recall.** In the 1M-context preset, a 237,703-token document returned the
correct hidden phrase.

**Image prompt.** A 198,288-token prompt with an image prefilled in 201.2 s
(**986 tok/s**) on one machine, before YOCO.

### YOCO

Approximate YOCO140 is opt-in; exact cone is the default. These earlier matched single-engine
controls report startup-inclusive prefill rates. YOCO skips the decoder layers' reads and compute on every pass except the last.
Each pair below is the same prompt with YOCO off and on.

| Prompt | YOCO off | YOCO on | Speed-up |
|---|---:|---:|---:|
| 43,327 tokens | 1,064 tok/s | 1,373 tok/s | 1.29× |
| 49,787 tokens with an image | 1,304 tok/s | 1,640 tok/s | 1.26× |
| 103,189 tokens | 1,127 tok/s | 1,635 tok/s | 1.45× |

On a fixed 384-token continuation, YOCO raised perplexity by 0.56% and agreed
with full prefill on 381 of 384 top-1 tokens. A 93,668-token recall test
returned the correct phrase. [Details](ULTRAYOCO_QUALIFICATION.md).

### Routed reads and exact replay

The complete weight stream is 315.83 GB: 305.76 GB of routed experts and
10.06 GB of other weights. Engram tables are already read on demand.
[Per-tensor inventory and settings](MULTICHUNK_PREFILL.md).

Matched Q4_K/Q8_0 controls on the same 49,555-token recall prompt, including
startup, with a 1 GiB prefill device admission reserve, the 512 MiB decode-cache
reserve, and a 0.5 GiB physical RAM floor:

| Configuration | Startup + prefill | Prompt tok/s |
|---|---:|---:|
| Full prefill, eager weight stream | 36.25 s | 1,367 |
| Padded exact cone, routed reads in layers 21–39 | 25.38 s | 1,953 |

Logits and cache checkpoint hashes were bit-identical. A separate cone run
returned the hidden recall key. An isolated 8K comparison with routed reads
in every layer was 14.94 s full versus 13.52 s cone, also bit-identical.

A later eager-cone control took 131.45 s, including an 89.33 s
inspect/admit/allocation startup outlier and 40.63 s prompt processing. Its
logits and cache also matched exactly. That run is retained in the receipts;
its elapsed-time difference cannot be attributed solely to replay. The adopted
cone profile uses routed tail reads, rather than this eager-cone combination.

On the repeated recall body, routed reads reduced 2K startup plus prefill from
25.06 to 10.34 s and 8K from 21.23 to 15.59 s. At 32K they streamed fewer bytes
but took 30.83 s versus 24.53 s, because they lost useful transfer overlap.
The automatic ULTRA policy therefore limits whole-pass demand reads to 2K and
uses eager encoder reads for longer prompts. On varied source text, 2K improved
from 19.17 to 15.23 s; 8K regressed from 20.58 to 21.39 s. This is why the
automatic range is narrower than the recall-body win.


### More chunks per weight pass

With VRAM carry, 100K in one weight pass improves from 52.235 to 44.633 s.
The optimized two-pass N=1 control is faster at 43.063 s, so grouping remains
explicit. At 200K, N=2 improves from 109.000 to 95.672 s; N=1 takes 92.407 s.
The old bank spilled many more encoder rows when grouped; registration and
carry transfers explain the regression. [Attribution and reproduction](SINGLEPASS_PREFILL.md).

Earlier routed exact-cone measurements, before VRAM carry, including startup.
Each cell is prompt tok/s:

| Prompt tokens | N=1 | N=2 | N=3 | N=4 |
|---:|---:|---:|---:|---:|
| 100,000 | 2,199 | 1,892 | 1,882 | 1,892 |
| 200,000 | 2,126 | 1,844 | RAM refused | RAM refused |
| 400,000 | 1,720 | 1,500 | RAM refused | RAM refused |

At 100K, N=2–4 use the same one-pass 100K geometry. They were run separately.
At 200K and 400K, N=3–4 were refused before engine launch because the measured
available physical RAM could not admit the worker, controller and reserve.

Peak producer RSS was 5.79 / 5.94 / 6.24 GiB for N=1 at 100K / 200K / 400K,
and 10.03 / 10.17 / 10.47 GiB for N=2. Board VRAM peaks were 27,211 / 27,513 /
28,123 MiB for both. Every completed grouped run had bit-identical logits and
cache checkpoint hashes against N=1. Grouping remains opt-in.

[Exact byte, memory and quality receipts](MULTICHUNK_QUALIFICATION.json).

### Decode

Latest calibrated Q4_K/Q8_0 controls at 49,555 prompt tokens. The warm rate
discards the first 100 evaluated steps and times the next 899:

| Setting | Bias | Warm decode |
|---|---:|---:|
| Max quality | 0 | **6.136593 tok/s** |
| Max speed, calibrated | 1.36125 | **20.657235 tok/s** |

[Frozen model/library and warm-window receipts](../examples/native-fp4/proxy-speed-comparison.json).
These controls use the earlier approximate YOCO140 prefill configuration.

E10 enables bounded CUDA HC input graph islands by default, with
`DS4_CUDA_SINGLE_GPU_HC_GRAPHS=0` retaining eager encoding. Matched 8K runs:

| Run order | Eager decode | HC graph decode | Gain |
|---|---:|---:|---:|
| Control first | 5.841747 tok/s | 5.959057 tok/s | 2.01% |
| Candidate first | 5.846415 tok/s | 5.948570 tok/s | 1.75% |

Both comparisons have exact full-vocabulary rows and generated tokens; each
candidate records 38 HC graph replays per measured token.
[E10 conditions and witnesses](EXPERIMENT_E10.md).

Single-machine examples enable recency LFU with a 16-token half life. On the
matched 8K F32-cache workload, warm decode improved 5.7624 → 6.3731 and
5.8081 → 5.8944 tok/s (+10.6% and +1.5%). Logits and generated prefixes were
exact, KL 0; expert-miss bytes fell 11.7% in both repetitions. These separate
controls do not establish the combined cone/LFU speed or a general 10.6% gain.
Set `LLAMADECODE_RECENCY_LFU=0` for the earlier LRU policy.
[E8 conditions](EXPERIMENT_E8.md).

Decode speed depends on how many routed experts are already in VRAM. The first
100 tokens fill the cache; after that the rate settles.

Earlier LRU single-engine controls, 500-token replies, rate per 100-token window:

| Prompt | Setting | Tokens 1–100 | 101–200 | 201–300 | 301–400 | 401–500 | Whole reply |
|---:|---|---:|---:|---:|---:|---:|---:|
| 14,859 | Max speed | 5.3 | 19.1 | 20.4 | 20.9 | **21.2** | 12.9 |
| 14,859 | Max quality | 6.0 | 6.2 | 5.9 | 6.1 | 6.0 | 6.0 |
| 43,327 | Max speed | 5.6 | 17.7 | 18.5 | 18.2 | **19.6** | 12.7 |
| 43,327 | Max quality | 5.9 | 6.0 | 6.0 | 6.1 | 5.9 | 6.0 |

At Max speed, 97% of expert lookups hit VRAM once warm, and the steady rate is
about 19.7 tok/s (51 ms per token).

Decode in the 1M-context preset, after a 237,703-token prompt: 5.2 tok/s at Max
quality over a 52-token reply.

### Experimental FP4 and FP8

The packed tensor-core-native FP4 prefill layout stores adjacent-nibble E2M1
weights as `[expert,N,K/2]`, followed by E8M0 scales in
`[expert,K/128,N,4]` order. The FlashInfer SM120 W4A8 path consumes those
bytes directly with MXFP8 activations. It retains the ordered expert combine
and avoids runtime expert-weight requantization or repacking.

Opt-in 100,000-token observations on their respective builds:

| Build / arm | Prompt-only time | Prompt-only rate | TTFT |
|---|---:|---:|---:|
| Packed layout, native-4k-matched | 36.490374 s | 2,740.45 tok/s | 39.797 s |
| KERNELS2 combination, native-4k-candidate-repeat | 34.934263 s | 2,862.52 tok/s | 38.313 s |
| Dense FP8 experiment, Q8 control | 43.308110 s | 2,309.04 tok/s | 46.594 s |
| Dense FP8 experiment, Q8-to-FP8 proxy | 43.821171 s | 2,282.00 tok/s | 47.063 s |

[Packed layout](../examples/native-fp4/PACKED_LAYOUT_RESULTS.md) ·
[Combined kernels](../examples/native-fp4/COMBINATION_RESULTS.md).

The FP4 timing model is a **lossy Q4-to-FP4 requant speed proxy**. It is
not shipped as a default. Conversion from the official safetensors at native
precision is coming; the converter and loader support exist, while complete
source-native model conversion and qualification remain unfinished.

Dense FP8 is also experimental and opt-in. Its native format accepts E4M3
weights and E8M0 scales in 32×32 blocks; timing above uses a lossy Q8-to-FP8
proxy. Explicit toggles retain the original paths.

### Host-RAM expert tier

The separate host-RAM tier experiment remains off by default and is not in
this release. Latest accepted Q4 measurements use a 49,553-token prompt and
reused-prefix follow-up; warm decode times 899 evaluated steps after discarding
the first 100:

| Run | RAM setting | Max quality | Max speed |
|---|---:|---:|---:|
| Tier off | 0 GiB | 6.87 tok/s | 18.26 tok/s |
| Upload to GPU | 3.49 GiB | 6.79 tok/s | 17.34 tok/s |
| CPU compute | 3.49 GiB | 6.81 tok/s | 14.58 tok/s |
| Mix of both | 3.49 GiB | 6.84 tok/s | 15.80 tok/s |
| Upload to GPU | 10 GiB | 6.49 tok/s | 15.23 tok/s |
| CPU compute | 10 GiB | 6.93 tok/s | 13.22 tok/s |
| Mix of both | 10 GiB | 6.94 tok/s | 12.08 tok/s |

[Round 3 results](https://github.com/fronteagle739/streaming-llama-stack/blob/a2371b6e84a09c8ef5f64ab67fc5230fc24aa95e/docs/RAMTIER_R3.md).

The speed settings already favour VRAM-resident experts, misses are spread
thinly, and at Max speed the GPU waits for the CPU. It is expected to matter
more on machines with much more RAM; that is untested and listed in the
[roadmap](ROADMAP.md).

### Quality/speed dial

The dial adds a bias toward experts already in VRAM. At 0 (Max quality) the
model routes exactly as released. Higher settings reroute some experts that
would otherwise be read from NVMe. This earlier sweep compares each setting
with unbiased routing on the same GGUF, over 384 generated tokens on a short
prompt. Speeds are after the
first 100 tokens, with the native ds4 decoder.

| Setting | Bias | Warm decode tok/s | KL vs full model | Top-1 agreement |
|---|---:|---:|---:|---:|
| Max quality | 0 | 7.3 | 0 | 100% |
| Balanced | 0.35 | 10.8 | 0.104 | 92.7% |
| Fast | 0.65 | 13.1 | 0.258 | 90.1% |
| Faster | 0.9 | 14.9 | 0.359 | 87.2% |
| Max speed | 1.36125 | 16.6 | 0.479 | 84.1% |

The same sweep shows the VRAM hit rate rising with the bias, and the cost in
quality rising with it:

| Bias | VRAM hit rate | NVMe reads per token | KL vs full model | Perplexity ratio |
|---:|---:|---:|---:|---:|
| 0.35 | 79.6% | 0.98 GB | 0.104 | 1.12 |
| 0.5 | 82.1% | 0.86 GB | 0.177 | 1.21 |
| 0.7 | 84.8% | 0.73 GB | 0.257 | 1.35 |
| 0.9 | 86.7% | 0.64 GB | 0.359 | 1.50 |
| 1.2 | 88.4% | 0.56 GB | 0.423 | 1.56 |
| 1.5 | 89.5% | 0.50 GB | 0.534 | 1.75 |
| 2.0 | 90.4% | 0.46 GB | 0.669 | 2.07 |

In C++ tests, Faster introduced a typo and Max speed introduced syntax errors.
Use Max quality or Balanced for code.

## More memory, cheaper speed

The dial only reroutes experts that would otherwise come from disk. The more of
the model fits in fast memory, the fewer experts need rerouting.

MiMo V2.6 Pro on the 512 GB Mac (ds4 decoder with expert streaming, 416 GiB
expert cache) already hits about 97% at bias 0. The same dial there:

| Bias | Warm decode tok/s | Expert misses | KL vs full model | Top-1 agreement |
|---:|---:|---:|---:|---:|
| 0 | 6.03 | 2.66% | 0 | 100% |
| 0.35 | 7.78 | 0.86% | 0.008 | 93.9% |
| 0.65 | 8.32 | 0.48% | 0.015 | 91.2% |
| 0.9 | 8.62 | 0.30% | 0.019 | 93.9% |
| 1.36 | 9.00 | 0.09% | 0.034 | 90.5% |

Warm rate is the last 128 of 256 tokens.

Side by side:

- **DeepSeek V4.1 on a 32 GB GPU:** about 75–90% expert hits. The dial trades
  real quality for real speed: KL 0.10–0.48, up to 2.3× faster.
- **MiMo V2.6 Pro on a 512 GB Mac:** 97%+ hits. The dial is nearly free: KL at
  most 0.034, up to 1.5× faster.

The more memory you have for the expert cache, the faster it runs and the smaller
the quality cost of the speed settings. Within the DeepSeek sweep above, the hit
rate and KL columns show the same trend.

## RTX prefill, Mac decode

Paired ULTRA defaults now use exact cone keep2541 and negotiated state v2.
The existing mode3 export/import retains owner hashes, token history and
compatibility checks. A matched actual-WebUI prompt contains 105,054 tokens:

| Route | WebUI prompt interval | Displayed prompt rate | Initial decode |
|---|---:|---:|---:|
| Single RTX exact cone | 46.282 s | 2,269.867 tok/s | 2.256 tok/s |
| RTX → Mac exact cone v2 | 59.005 s | 1,780.432 tok/s | 27.499 tok/s |

The paired interval includes startup, prefill, export and handoff; it is
separate from native prompt-only timing and TTFT. The measured mode3 transfer
is 691,633,076 bytes in 3.066 s, including verification/import acknowledgment.
Both routes and the follow-ups return the same retrieval answers.
[Live proof and timing components](CONE_STATE.md).

Earlier image measurements, approximate YOCO140/v1 or full prefill, with
startup and handoff included in the WebUI prompt interval:

| Prompt | YOCO | Prompt time | Prefill tok/s | Mac decode tok/s |
|---|---|---:|---:|---:|
| 198,287 tokens with an image | On | 118.8 s | 1,670 | 31.0 |
| 49,787 tokens with an image | On | 33.7 s | 1,477 | 31.8 |
| 49,787 tokens with an image | Off | 41.6 s | 1,197 | — |

The 198K run is 1.69× faster than the same-size image prompt on one machine
without YOCO (986 tok/s). The image was answered correctly in every run.

The handoff adds about 3.4 s: the 49,787-token prompt takes 30.4 s on the RTX
alone with YOCO, and 33.7 s with the Mac.

Prompts and follow-ups under 11,776 new tokens prefill on the Mac instead, at
about 350–390 tok/s. Mac decode runs at about 31 tok/s at 9K and 24K tokens.

## Memory

| Configuration | Budget | Measured peak |
|---|---:|---:|
| Single engine, engine process | 7.75 GiB | 5.7 GiB (1M-token run) |
| Single engine, engine and controller | 9.25 GiB | — |
| RTX side of the two-machine setup | 8.5 GiB | 6.3 GiB |

The single-engine example uses a 9.25 GiB host budget, 4 GiB admission floor,
512 MiB desktop reserve and 0.5 GiB runtime free-RAM floor. Experimental runs
declare their own host ceilings and retain paging stops. The model is never loaded into system RAM.

## Other models

Measured with the streaming engine on the same RTX 5090 desktop. These are
prefill rates timed by the engine; decode ran on the Mac.

| Model | Prompt | Prefill tok/s | Mac decode tok/s |
|---|---:|---:|---:|
| Step-3.5 Flash Q8_0 (194 GB) | about 25K | 2,294 | 25–33 |
| Step-3.5 Flash Q8_0 | 16K | 1,518 | — |
| MiniMax M2.5 UD-Q3_K_XL (100 GB) | 9K | 1,576 | 20–35 |
| MiniMax M2.5 UD-Q3_K_XL | 32K | 1,205 | — |
| MiniMax M2.5 UD-Q3_K_XL | 196K | 439 | — |
| MiniMax M2.5 UD-Q8_K_XL (200 GB) | 14.5K | 598 | 20–35 |
| GLM-5 UD-Q4_K_XL (400 GB) | 16K | 764 | — |
| GLM-5 UD-Q4_K_XL | 32K | 493 | — |
| GLM-5.1 UD-Q4_K_XL | 15.6K | 415 | — |
| GLM-5.3 UD-Q4_K_XL | short | — | 14.1 |
| Kimi K2.5 UD-Q3_K_XL (500 GB) | 16K | 652 | — |
| Kimi K2.5 UD-Q3_K_XL | 32K | 561 | — |
| Kimi K2.6 Q3_K_L (457 GB) | 23K | 576 | 24.3 |

Kimi K2.6 also reached about 690 tok/s with a 4,096-token microbatch; the
shipping Kimi settings use 2,048, which gave 576 tok/s at 23K.


## Limitations

- No 50K or 200K measurement uses the final OTHER22 100K build and options.
  The 1M figure and latest ~50K warm-decode controls use earlier YOCO140 defaults.
- The 32.373785 s headline is one opt-in proxy/graph observation. Same-build
  graph/eager frontier values match, but run variance prevents a reliable graph
  speedup claim. Binding-independent layer/sweep replay and CPU-loop removal
  remain unfinished; prefill graphs stay opt-in.
- The lossy FP4/FP8 proxies are not source-native model qualification. Packed
  FP4 measurements are prefill-only and do not establish decode or broad recall
  quality. Dense FP8 has no demonstrated repeatable whole-prompt speed gain.
- TileLang attention is not bit-exact. The tested retrieval answer matches;
  later forced continuations can diverge. Native attention remains selectable
  with `LLAMA_DSV41_TILELANG_ATTENTION=0`. [Quality evidence](../examples/kernels2/PERFORMANCE.md).
- The separate host-RAM tier has no consistent gain across dial settings.
  Its CPU/mix continuations can differ and fail strict sequence-equivalence checks.
- Historical dial sweeps, current warm-decode controls and E8/E10 comparisons
  use different fixtures and builds; their gains cannot be combined. HC graph
  islands are not full-step decode graphs or PDL.
- Paired cone validation covers the initial frontier and one forced row, plus
  actual three-turn WebUI recall. Multi-row continuation equivalence has not
  been established, and the older image speeds are approximate YOCO140 results.
- The 10,000 tok/s target at 100K remains unmet. [Remaining work](ROADMAP.md).
