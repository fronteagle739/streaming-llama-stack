# Performance

All RTX numbers come from one desktop: RTX 5090 (32 GB VRAM), 32 GB of system
RAM, Windows, NVMe RAID-0 at about 25 GB/s. The Mac is an M3 Ultra with 512 GB.
Every figure below was measured; each says what was timed.

[Any-GGUF Q4_K/Q8_0 smoke](ANYGGUF_GPU_SMOKE.md): 49.6K-token recall at **1,592 tok/s**; warm decode **6.8 tok/s** (Max quality) / **18.2 tok/s** (Max speed).

[Host expert tier trial](RAMTIER.md), same Q4_K/Q8_0 checkpoint and actual
WebUI, 49,553-token prompt and 1,000-token replies:

| Host tier | Warm Max quality | Warm Max speed |
|---|---:|---:|
| Off | 7.26 tok/s | 18.89 tok/s |
| 3.49 GiB (3.5 GiB requested) | 3.50 tok/s | 12.43 tok/s |
| 10.99 GiB (auto) | 3.36 tok/s | 11.32 tok/s |

The tier reduced SSD reads but slowed both settings, so it remains opt-in.
Auto reclaimed 4.26 GiB of prefill carry RAM, retained native state exports,
and passed a longer append with tier release/regrowth. Runtime RAM floor defaults
are now 512 MiB; the separate desktop admission reserve is still 512 MiB.
These are bounded single-run observations; see the trial for hit rates,
memory/paging observations, build differences and failed-attempt disclosures.

## DeepSeek V4.1 Flash on one RTX 5090

Model: DeepSeek V4.1 Flash GGUF, Q4_K routed experts and Q8_0 for the rest
(483 GiB). That matches DeepSeek's release precision (low-bit experts, FP8 for
the rest), so "Max quality" below is the full model.

### Prefill

ULTRA prefill, YOCO off. Times include engine startup.

| Prompt tokens | Prefill time | Prefill tok/s |
|---:|---:|---:|
| 28,241 | 28.7 s | 985 |
| 32,234 | 31.1 s | 1,036 |
| 39,864 | 35.3 s | 1,130 |
| 48,520 | 41.7 s | 1,163 |
| 103,189 | 91.5 s | 1,127 |

With the engine already loaded, one 49,152-token pass runs at **1,920 tok/s**.

**One million tokens.** A single session prefilled 1,000,000 tokens in
1,808 s (**553 tok/s**), in 49,152-token passes, without resetting the
context. Peak engine RAM was 5.7 GiB.

**Long recall.** In the 1M-context preset, a 237,703-token document returned the
correct hidden phrase.

**Image prompt.** A 198,288-token prompt with an image prefilled in 201.2 s
(**986 tok/s**) on one machine, before YOCO.

### YOCO

YOCO skips the decoder layers' reads and compute on every pass except the last.
Each pair below is the same prompt with YOCO off and on.

| Prompt | YOCO off | YOCO on | Speed-up |
|---|---:|---:|---:|
| 43,327 tokens | 1,064 tok/s | 1,373 tok/s | 1.29× |
| 49,787 tokens with an image | 1,304 tok/s | 1,640 tok/s | 1.26× |
| 103,189 tokens | 1,127 tok/s | 1,635 tok/s | 1.45× |

On a fixed 384-token continuation, YOCO raised perplexity by 0.56% and agreed
with full prefill on 381 of 384 top-1 tokens. A 93,668-token recall test
returned the correct phrase. [Details](ULTRAYOCO_QUALIFICATION.md).

### Decode

Decode speed depends on how many routed experts are already in VRAM. The first
100 tokens fill the cache; after that the rate settles.

Single engine, 500-token replies, rate per 100-token window:

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

### Quality/speed dial

The dial adds a bias toward experts already in VRAM. At 0 (Max quality) the
model routes exactly as released. Higher settings reroute some experts that
would otherwise be read from NVMe. The table compares each setting with the
full model, over 384 generated tokens on a short prompt. Speeds are after the
first 100 tokens, with the native ds4 decoder.

| Setting | Bias | Warm decode tok/s | KL vs full model | Top-1 agreement |
|---|---:|---:|---:|---:|
| Max quality | 0 | 7.3 | 0 | 100% |
| Balanced | 0.35 | 10.8 | 0.104 | 92.7% |
| Fast | 0.65 | 13.1 | 0.258 | 90.1% |
| Faster | 0.9 | 14.9 | 0.359 | 87.2% |
| Max speed | 1.36125 | 16.6 | 0.479 | 84.1% |

As a rough guide, quality runs from about Q8-quant-like at Max quality to about
IQ1-quant-like at Max speed.

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

The RTX runs the same single-engine prefill (ULTRA, 49,152-token passes) and
hands the finished state to the Mac once.

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

The launchers keep a 512 MiB desktop reserve and stop cleanly if free RAM falls
below 4 GiB. The model is never loaded into system RAM.

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
