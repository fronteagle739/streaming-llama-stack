# Opt-in approximate YOCO: RTX qualification, 2026-10-25

`yoco_keep=140` is approximate and local to the single engine. Default `0`
preserves the existing execution path. The measured candidate is
`853246578bdd2cbdc73d6fdce97fbdb0c140a1a8`; executable SHA-256
`2b8fbf1d7b0fb3a80121b83710ec54b0ed56ecf331d904b74eae1ab72ef5d2d6`.
This document and its README link do not change native code or the binary.

## Flag-off identity

The reference is release `57af75e11016d9c5a2fe9adfa9b13aa7903e9c1c`.
Its native engine source is unchanged from the installed producer revision
`233a242d10c6b5a6e7e1a04c2fbe36f63a3b6dc9`, executable SHA-256
`71b2dbba39d3b8a0c3dae1426ecae4be4577583565d638b968673d07ab4682c1`.

At 43,327 tokens, the complete release and flag-off candidate exports both
contain **288,477,488 bytes**, SHA-256
`2e28505f7b2da2a54212facc7d57d2c06fb5aa7422303cb35df38a5a50a58eb9`.
Exports were streamed through a bounded RAM hash, without a disk KV payload.

## Timing and weight fetches

RTX 5090; original Q4 model; shipping ULTRA arithmetic and fold planner;
262,144 context with growable f32 cache. Host budget 9.25 GiB, producer base
6.25 GiB, maximum 7.75 GiB, desktop reserve 8 GiB, physical-RAM floor 4 GiB.
Every engine run had independent memory enforcement and build/test
contamination monitoring. No measured engine run was repeated.

Times include producer startup and prefill, excluding subsequent export or
decode. Each row ran in **exact / YOCO / exact / YOCO** order.

| Prompt | Exact runs (s) | YOCO runs (s) | Exact mean (s) | YOCO mean (s) | Time reduction |
|---|---|---|---:|---:|---:|
| 43,327 | 42.828, 38.640 | 31.391, 31.703 | 40.734 | 31.547 | 22.55% |
| 103,189 | 91.438, 91.656 | 63.296, 62.969 | 91.547 | 63.133 | 31.04% |

Mean rates are 1,063.7 to 1,373.4 tok/s at 43K, and 1,127.2 to 1,634.5
tok/s at 103K. These are startup-plus-prefill rates, not decode throughput.
The 103K sweep plan was `[49152, 54037]` in all four runs.

Both 103K YOCO logs show the non-final sweep fetching stages 0 through 21:
embedding and layers 0 through 20. Decoder layers 21 through 39 and the head
are absent. Fetcher-reported weight bytes are **166,382,548,664** for that
sweep, **52.68%** of a full pass's **315,825,603,008**. The final sweep fetches
the full stage set once. Total reported weight bytes are
**482,208,151,672** with YOCO versus **631,651,206,016** exact.
These are fetcher stage byte counts, not an independent physical DMA trace.

## G-b quality

384 teacher-forced decode rows on the existing 545-token G-b prompt and
native reference continuation; full-vocabulary KL. Exact logits retain the
qualified SHA-256 `d5bcee3663f1c02e48ba4eca19243b53c4762b204c111975cf1319cdadf47520`.

| Comparison | Mean KL | Top-1 agreement |
|---|---:|---:|
| Native reference to exact | 0.05494348 | 361/384 |
| Native reference to YOCO | 0.05668419 | 360/384 |
| Exact to YOCO | 0.00185296 | 381/384 |

Both arms meet the existing native-reference G-b limits: mean KL at most
0.06 and at least 355 top-1 matches. Exact-to-YOCO maximum row KL is
0.04797245. Mean exact-to-YOCO KL for continuation rows 0–127, 128–255 and
256–383 is respectively **0.00250268, 0.00147204, 0.00158414**.

PPL on this fixed native-generated continuation is **1.24889335 exact** and
**1.25590636 YOCO**, a **0.56%** increase. Native-reference PPL is 1.16414547.
This small continuation is not a general corpus perplexity evaluation.

## Recall and WebUI

The existing approximately 100K recall case renders to **93,668 tokens**.
YOCO returns the expected **amber-quartz-1147**. Prefill takes **60.343 s**,
using `[49152, 44516]` sweeps.

The actual `start-dsv41-single-1m-ultra-yoco.ps1 -WebUI` launcher serves HTML
and completes two API chat turns through its WebUI router: HTTP 200,
answers **12**, then **15**, in **22.062 s** and **3.829 s**. Launch metadata
confirms `yoco_keep=140`, context **999,424**, growable bf16 cache and the
4 GiB native runtime floor. Prompts are 593 and 611 tokens: this validates
the 1M configuration and follow-up path, not a million-token prefill.

All gate producers retire gracefully with exit code 0 and an empty owned
job. No runtime memory-floor breach or timing contamination is reported.
One 103K YOCO arm records a **20,627,456-byte (19.67 MiB)** rise in global
pagefile usage while retaining at least **10,200,412,160 bytes (9.50 GiB)**
available RAM. Other standalone gate arms record zero growth. This global
counter does not attribute paging to the producer; neither its 4 GiB floor
nor the unchanged 256,000,000-byte global paging guard tripped. The evidence
does not establish machine-wide zero paging.
After this check, the plain 1M WebUI is relaunched for the owner.

Images, verify, all-position scoring, decode and append-via-decode retain
full evaluation. Transport/export, persistence and rewind remain refused
while YOCO is enabled; edits re-prefill. No RTX-to-Mac or million-token
quality qualification is claimed.

## Image-prompt Engram read correction (2026-10-27)

A 23,717-token prompt containing six image spans spent 67.831 seconds in
Engram reads at layers 1 and 14, out of 89.450 seconds of native prefill.
The image path disabled the admitted whole-prompt cache and consequently
issued individual unbuffered row reads. This accounts for most of the
slowdown independently of the image exclusion from YOCO shortening.

The batched prompt cache now plans its row set with the exact image text
mask, including history resets and the following three text positions.
Image prompts now use the same asynchronous loader and per-table readiness
as text prompts. The worker retains the mask, checks it when adopting a cache,
and applies it to tile lookahead. Both live mask copies are charged before
sizing the IO arena. Masked loading starts at the sweep, when its exact image
positions and restored history are known. Cached reads use the existing
parallel row decoder and unchanged memory admission. Text-only speculative
loading and the image exclusion from YOCO shortening remain unchanged.

The native synthetic-file regression checks every masked cached row against
the original reader's hash and file bytes, including planner block boundaries,
asynchronous table publication and restored history.

Replaying the original 23,717 token IDs and six image embedding spans on the
RTX 5090 completes native prefill in **22.501 s** and **23.276 s** across the
overlap candidate and final build (**1,054.0** and **1,018.9 tokens/s**), versus
**89.450 s (265.1 tokens/s)** before the correction. Engram layer waits/decode
total **0.583 s** and **0.900 s**, versus **67.831 s**. This is native prefill timing; WebUI
request latency also includes producer startup and other request work. The
configured 6.25 GiB producer host budget, 4 GiB runtime floor and native
arithmetic settings are unchanged. The receipt reports full image evaluation;
YOCO shortening remains excluded. This measurement qualifies this replay,
not every image or context length.
The full **162,895,048-byte canonical state** hashes identically before and
after this correction, including the overlap variant, for this replay.
