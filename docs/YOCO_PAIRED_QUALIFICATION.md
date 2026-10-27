# Paired YOCO functional evidence; producer-alignment gate pending

The functional gate passed on October 27, 2026, with the actual stack WebUI in headless Mac Chromium. The 105,054-token recall and two follow-ups returned `AMBER OTTER 731`, the same phrase, and `731`. The uploaded-image conversation correctly identified a red square at 198,287 prompt tokens and remembered it on the next turn. Short initial and follow-up turns used Mac-local prefill at Studio's existing 11,776-new-token crossover.

The owner subsequently required a shared single-engine prefill plan, final-state-only handoff, a smaller measured host budget and 512 MiB admission reserve. Those changes and their new paired 50K/200K gates remain release blockers. The measurements below describe the earlier per-chunk mode3 profile.

The image run measured **1,058.088 prefill tok/s** over the WebUI's 187.401 s prompt interval, versus the supplied historical image reference of 985.74 tok/s. Its 104 answer tokens took 3.382 s, **30.755 Mac decode tok/s**. Native RTX prompt compute, excluding its one-token warmup, totaled **137.735 s**; five transfer/import/hash/acknowledgment intervals totaled **10.850 s**. These are different timing boundaries and must not be added to the historical compute row without their startup costs.

[Sanitized metrics and SSE](../examples/dual-machine/dsv41-ultra-yoco/proof/previous-profile/summary.json), [native sampler/routing evidence](../examples/dual-machine/dsv41-ultra-yoco/proof/previous-profile/native-applied-and-routing.log), and screenshots in that directory preserve the actual browser and engine observations. The sustained replies stream incrementally, live/final rates are numeric, and context usage updates. The visible accessible progress bar follows actual SSE counts. The single-engine cold-start diagnostic view now publishes before constructor completion; it no longer loses the entire initial prefill.

## Controlled image comparison near 50K

All four runs uploaded the same PNG and used the same text, SHA-256 `9ccb0c7b068baff15dc599634ee590358d0d1f1bba779d0cab97fd9fdd5ff6fc`, yielding **49,787 prompt tokens**, with no reused prompt tokens. Every answer correctly identified the image. The paired controls also passed their referring turns. Reported prompt intervals use each route's existing WebUI timing boundary.

| Route | YOCO140 seconds | YOCO tok/s | Full-evaluation seconds | Full tok/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Single RTX | 29.500 | 1,687.695 | 37.797 | 1,317.221 | 1.281× |
| RTX → Mac | 67.970 | 732.486 | 74.648 | 666.958 | 1.098× |

The single-engine result is consistent with the owner's roughly 1,600 tok/s expectation near this size. Paired throughput is lower because the current authenticated transport exports a decoder-ready checkpoint after each chunk. Even this 49,787-token prompt uses two paired chunks, 49,152 + 635; the single engine folds it into one logical prefill and skips the separate warmup through its existing fixed direct-open path.

## Why the earlier image result was only 1.04×

The earlier blocking transport reported 198,931 tokens in 194.248 s, or 1,024.109 tok/s. Serialized Mac owner hashing and catch-up acknowledgments also reduced decode to 11.003 tok/s. The candidate ports Studio's existing ordered catch-up worker and incremental hash chains; the qualified image decode is now about 30.8 tok/s.

The large-prompt prefill speedup is limited by the paired boundaries. Every exported chunk is a complete logical prefill: all its rows pass through layer 20, then its last 140 rows pass through layers 21–39. Decoder suffix weights are fetched again at each exportable boundary. The single-engine scheduler can instead use encoder-only internal sweeps before its one final suffix. At 198,287 tokens the paired route also pays for a small fifth chunk whose weight acquisition alone took 15.310 s.

| Paired chunk | Tokens | RTX prompt compute (s) | Exposed layer Engram (s) | Transfer/import/hash/ACK (s) |
|---|---:|---:|---:|---:|
| 1 | 49,152 | 26.013 | 0.001018 | 1.550 |
| 2 | 49,152 | 27.791 | 0.001055 | 2.070 |
| 3 | 49,152 | 32.057 | 0.015347 | 2.441 |
| 4 | 49,152 | 34.490 | 0.001022 | 2.848 |
| 5 | 1,679 | 17.385 | 0.000032 | 1.941 |

The initial RTX operation additionally includes a **17.788 s one-token warmup**. All RTX operation clocks, including startup, sum to 157.015 s. The WebUI prompt interval leaves 19.536 s beyond those clocks and the transfer intervals for export preparation, checkpoint/hash and receiver/control work; those remaining components are not independently timed. Exposed layer Engram sums to 0.018475 s. Prefetch overlaps other work, so this is not a complete Engram cost allocation and must not be added as another serial phase.

The real uploaded image occupies 184 projected rows at positions 198,045–198,228. All 184 receive full layer-20 publication, and 82 intersect the final 140-row decoder tail. The image does not force full evaluation of the whole 198K prompt. Repeated background text makes this a controlled synthetic workload; rates are content-dependent.

## Existing implementations and added integration

Negotiated YOCO v1 export, the mode3 state geometry, full checkpoint verification, projected-image transport and `combo_crossover.py` already existed. The paired route uses `native_broker`; the separate local `ComboProducer` snapshot restriction is retained. The integration connects the labelled launcher, explicit keep=140 negotiation, Mac native import, image identities/Engram masks and Studio's 11,776-row local crossover. Image YOCO needed the scheduler exclusion removed while retaining every layer-20 vision row and decoder-tail intersections.

Mac samplers and bias are ported from the existing Studio/Mac implementation. All GUI runs applied temperature .83, top-p .87, top-k 37, min-p .06 and bias .22; seed 123456 and max tokens 128 were requested. [Single-engine native evidence](../examples/dual-machine/dsv41-ultra-yoco/proof/previous-profile/single-applied-samplers.log) confirms the same core settings. CPU checks cover penalties, logit bias, sampler ordering, phase profiles and advanced settings, plus a greedy negative control. Model tool calls were disabled for this controlled gate.

The existing Studio asynchronous transport source was verified as SHA-256 `c21d89edb2390c89cc1304ccc2ed1559fc137572520bc6fad4f1df67528d1d1b`. Its ordered queue and incremental hash algorithm were ported while retaining negotiated YOCO, image transport and strict joins before RTX reuse or shutdown. Independent CPU owner hashes agree at even and odd positions; corruption fails and workers join cleanly.

## Quality, liveness and memory

The real-image numerical gate compares full evaluation with YOCO140. When the image intersects the replay tail, KL(full || YOCO) is 0.05814 nats and top-10 recall is 90%; before the tail, KL is 0.00733 and top-10 recall is 90%. Both retain the same leading color token. [Numerical receipts](YOCO_IMAGE_NUMERICAL.json). YOCO remains approximate and opt-in.

Long load, hash, prefill, handoff and decode operations use progress-based waits. The HTTP transfer view retains the producer's independent stall guard without a total-operation timer; native byte progress and authenticated producer status renew transport liveness. Live ancestor leases remain valid during progressing work beyond their reservation estimate. CPU fake engines and real HTTP/native CPU fixtures verify both slow progress and stalled failure.

The final focused/expanded CPU run passed **166 tests and 70 subtests**, with 2 environment/platform skips. Mac parser, sampler, image scheduling, incremental owner/hash and queue fixtures passed without opening a model. The Mac build used SHA-256-verified RTX source; 153 native-source entries remained unchanged after the final Python/UI synchronization.

Native RTX host cap remains 8 GiB, Mac cap 400 GiB, desktop admission reserve 2 GiB and runtime physical-RAM floor 4 GiB. No qualified native run breached a budget. The bounded owned-page scanner observed zero pagefile bytes; the system-wide counter grew 1,728,512 bytes in the image window, which is not process attribution and remained below the unchanged paging stop. [Memory receipt](../examples/dual-machine/dsv41-ultra-yoco/proof/previous-profile/owned-memory.json). This is not an exhaustive machine-wide zero-paging claim.
