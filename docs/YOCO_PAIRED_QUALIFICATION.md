# Paired streaming-llama YOCO qualification

The complete October 27, 2026 gate passes in the actual stack WebUI in headless Mac Chromium: 105,054-token recall plus two follow-ups, uploaded image at 198,287 tokens plus a referring turn, Mac-local short initial/follow-up turns, and all four matched 49,787-token image controls. The recall answers were `AMBER OTTER 731`, the same phrase, and `731`; every image answer identified the red square. Replies stream incrementally, the visible prefill bar advances, live/final rates are numerical and context usage updates. [Metrics and receipts](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/summary.json), [SSE progress](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/image-200k-sse.json), [progress screenshot](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/image-200k-progress-2.png), [streaming screenshot](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/image-200k-streaming.png), [final screenshot](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/image-200k-final.png).

The large image run used **118.766 seconds**, **1,669.559 prefill tok/s**, and **30.982 Mac decode tok/s** (104 answer tokens in 3.357 seconds). It is 1.694× the supplied historical image reference of 985.74 tok/s. That historical prompt differs slightly; the matched controls below compare identical inputs.

## Same producer, finished state, one handoff

Both routes execute the same streaming-llama producer, SHA-256 `e95af084559456750a50c8e75e03e2fa34d37311e08c218c689df33e972de9ce`. The earlier slow run also used this binary. `native_broker` and mode3 name the authenticated state format; they did not select ds4-win prefill. The Mac ds4 binary decodes the transferred state.

The shared `prefill_arguments` builder now selects the same ULTRA arithmetic, fold planner, sweep capacity, YOCO window, growable BF16 cache, DirectStorage settings and producer budgets. Both matched arms use context 1,000,001; direct-open enabled; one-token warmup disabled; primary sweep 49,152; fold cap 61,440; host base 6.25 GiB and maximum 7.75 GiB. Both have pair-FFN tiles enabled and fixed-resident-plan disabled. Their actual executed prefill arguments differ only in `capture-dspark` and negotiated `yoco-remote-state-version`. [Actual argument comparison](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/actual-argv-parity.json).

Mode3 export requires the **7,864,320-byte DSpark ring** and YOCO v1 provenance. The ring is captured within the same admission caps. Local snapshots use mode1 and omit that transfer-only ring. The growable exact mode3 window additionally needs 12 KiB of VMM padding; both paths reassign it from graph headroom without increasing the device cap.

The Mac sends the whole logical initial prompt, rather than truncating it into exportable 49,152-token requests. The producer completes its internal sweeps before one export/import/verified acknowledgment. Long remote appends retire the synchronized producer and replay the complete logical prompt through the same planner, preserving a single final handoff. Cleanup must be certified before replacement; no intermediate acknowledgment is fabricated. Small appends retain session reuse.

## Matched image comparison near 50K

All four runs uploaded the same PNG and text, SHA-256 `9ccb0c7b068baff15dc599634ee590358d0d1f1bba779d0cab97fd9fdd5ff6fc`, yielding **49,787 prompt tokens and zero reused tokens**. They use the same 1M BF16 growable cache and profile. Every image answer is correct; the paired controls also pass their referring turns. Timing uses each route's existing WebUI prompt interval.

| Route | YOCO140 seconds | YOCO tok/s | Full seconds | Full tok/s | YOCO speedup |
|---|---:|---:|---:|---:|---:|
| Single RTX | 30.359 | 1,639.942 | 38.172 | 1,304.281 | 1.257× |
| RTX → Mac | 33.716 | 1,476.679 | 41.607 | 1,196.596 | 1.234× |

Paired adds **3.357 s** with YOCO and **3.435 s** without it. Its native prompt compute is 26.815 / 34.685 s, versus single-engine 27.833 / 35.229 s. This is consistent with the same prefill work plus export/handoff overhead; the old roughly twofold paired slowdown is removed. These are individual controlled observations, not repeated statistical estimates.

| Paired mode | Startup + prefill | Compute within that | Export preparation | Transfer/import/hash/ACK | Save rewind | Other control | WebUI total |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOCO140 | 29.234 | 26.815 | 0.937 | 1.581 | 1.000 | 0.963 | 33.716 |
| Full | 37.344 | 34.685 | 0.937 | 1.461 | 0.937 | 0.928 | 41.607 |

Times are seconds. Compute is a subset of startup-plus-prefill, so those two columns must not be added. Transfer contains network, Mac import/hash and acknowledgment. Other control is the measured residual, not an independently timed phase. [YOCO SSE/bar samples](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/paired-yoco-sse.json), [full-evaluation SSE/bar samples](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/paired-full-sse.json), [single YOCO SSE/bar samples](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/single-yoco-sse.json).

## Large image phases and the earlier slowdown

The new 198,287-token request uses these four internal sweeps, then **one** final handoff:

| Sweep | Tokens | Layers evaluated | Prompt compute (s) |
|---|---:|---|---:|
| 1 | 49,152 | 0–20 | 18.810 |
| 2 | 49,152 | 0–20 | 19.590 |
| 3 | 49,152 | 0–20 | 24.085 |
| 4 | 50,831 | 0–39, last-140 decoder tail | 36.285 |

Compute totals **98.771 s**. Startup plus prefill is **102.344 s**; final export preparation **3.782 s**; transfer/import/hash/ACK **5.657 s**; saved rewind **4.422 s**; remaining control **2.561 s**. These sum to the 118.766 s WebUI prompt interval. Exposed layer Engram time sums to **0.003965 s**; its prefetch overlaps other work, so this is not a complete Engram cost allocation.

The image occupies 184 projected rows at positions 198,045–198,228. Every row receives layer-20 publication; 82 intersect the final 140-row decoder tail. The image does not force full evaluation of the whole document.

The previous profile measured 1,058.088 tok/s at this size: 137.735 s prompt compute, 17.788 s one-token warmup, five transfer/import/hash/ACK intervals totaling 10.850 s, and repeated export/checkpoint/control costs. Its tiny final 1,679-token chunk spent 15.310 s acquiring weights. The corrected run removes the warmup, folds that tail, skips decoder weight fetches on the first three sweeps, and reduces five handoffs to one. Its WebUI interval falls from **187.401 to 118.766 s**.

At 49,787 tokens the previous paired YOCO interval was 67.970 s versus single-engine 29.500 s. Logged extra work included a 16.411 s warmup and a separate 635-token prefill costing 16.772 s, of which 15.891 s was weight acquisition; two transfer intervals cost 2.229 s. The old paired 8 GiB host cap was above its measured 5.685 GiB native peak, so it was not the main 50K limit. The old single arm also used a different cache/context; the new matrix fixes that mismatch. [Previous-profile diagnostics](../examples/dual-machine/dsv41-ultra-yoco/proof/previous-profile/summary.json).

## Existing implementations, samplers and crossover

Negotiated YOCO v1 export, mode3 geometry and checkpoint verification, projected-image transport and `combo_crossover.py` already existed. Integration connects the labelled launcher, explicit YOCO negotiation, Mac import, image identities/Engram masks and Studio's existing **11,776-new-token strict-less-than crossover**. Short initial/follow-up requests prefill on the Mac without an RTX round trip; the long requests use RTX streaming prefill. [Native applied sampler and routing evidence](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/native-applied-and-routing.txt).

Image YOCO needed the scheduler exclusion removed while preserving every layer-20 vision row and every decoder-tail intersection. The real-image numerical gate reports KL(full || YOCO) **0.05814 nats** with the image intersecting the tail, and **0.00733** with it entirely before the tail; both have **90% top-10 recall** and the same leading color token. [Numerical receipts](YOCO_IMAGE_NUMERICAL.json). YOCO remains approximate and opt-in.

Mac sampler and bias support, plus ordered asynchronous catch-up and incremental owner hashes, were ported from existing Studio/Mac implementations. All GUI controls applied temperature **.83**, top-p **.87**, top-k **37**, min-p **.06**, bias **.22**, seed **123456** and max tokens **128**. [Single-engine applied values](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/single-yoco-applied-samplers.txt). CPU checks cover penalties, logit bias, advanced settings and phase profiles, including a greedy negative control. Model tool calls were disabled for this controlled gate. Strict joins precede RTX reuse and shutdown.

## Memory and liveness

The outer paired budget is **8.5 GiB**, reduced from 13.5; both comparison arms use it. The shared producer retains its **6.25 GiB base / 7.75 GiB maximum** and the outer budget allows **0.75 GiB** for controller, broker, guards and export metadata at the maximum plan. The measured paired tree peaked at **6,808,735,744 bytes (6.341 GiB)**. Component sampling observed producer working set up to **6,667,272,192 bytes**, broker **136,880,128 bytes**, controller **49,651,712 bytes**. Peaks are independent observations and need not be simultaneous. [Component RSS/private commitment and guard receipt](../examples/dual-machine/dsv41-ultra-yoco/proof/shared-profile/owned-memory.json).

There was no extra 4 GiB resident controller/export allocation. The previous large outer allowance was unused admission headroom. The 198K finished state is **1,288,705,656 bytes** and streams through pages of at most **64 KiB**, rather than being staged as a whole broker buffer. Raw producer private commitment reaches about **18.1 GB**, including GPU/driver backing; it is not physical host RAM or evidence of paging.

Desktop admission reserve is **512 MiB** on both hosts and in shipping defaults. The separate runtime free-physical-RAM floor is **512 MiB** (owner decision after qualification), the Mac cap **400 GiB**, and paging stops remain enabled. Qualified native receipts show zero budget breaches. The bounded owned-page scanner observed zero pagefile bytes and the paired guard's global pagefile growth was zero. Kernel ETW disk attribution was unavailable; these observations are not an exhaustive zero-I/O claim.

Long load, prefill, hash, handoff and decode operations wait while actual work progresses. Independent producer stall guards, native byte/hash progress and authenticated producer status enforce no-progress failure, rather than a total-operation deadline. CPU fake engines and actual HTTP/native fixtures cover slow progressing work beyond old caps and stalled failure. Final Python validation passed **183 tests and 61 subtests**, with **2 environment/platform skips**. Mac parser, sampler, image packing, owner/hash and queue fixtures passed without opening a model. Native Mac source was SHA-256 verified against RTX before building; the qualified binary is `bc5170e5f774c1e8f1c102d16b75cbd457ddf962cce596dbdc438400465498a3`.
