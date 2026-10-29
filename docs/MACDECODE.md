# Mac decode comparison, 2026-10-29

This is a measured Q4 candidate, not a completed upstream performance match or
a production release. The candidate is opt-in while paired qualification and
the exact-model comparison remain incomplete. No preset was changed.

## Published upstream configuration

[Upstream README](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/f48c5e2e20b1aaae1c79735a8457225d2469ae73/README.md):
M3 Ultra, 80 GPU cores, 512 GB; October 17 MXFP4 target and MXFP4 DSpark
drafter; 393,216 context allocation; 8,192 prefill chunk. Decode restores
8K/131K prose or 8K code, warms the pipelines, then measures 128 greedy
tokens, reporting the mean of two passes. Drafter loaded for decode; absent
for cold-prefill measurements.

| Workload | Published upstream MXFP4 | Installed/retained Q4 | Q4 candidate |
|---|---:|---:|---:|
| Serial, 8K prose, tok/s | 41.4 | Studio 33.339; stack 33.242 | 34.001 |
| Serial, 131K prose, tok/s | 40.5 | Stack 32.617 | 33.353 |
| DSpark enabled, 8K prose, tok/s | 40.1 | Studio 32.628; stack 32.489 | 33.281 |
| DSpark, 8K code, tok/s | 58.8 | Not measured | Not measured |
| Cold prefill, 32K, seconds | 33.7 | Not measured | Not measured |
| Cold prefill, 131K, seconds | 138.8 | 375.513, different procedure | No prefill change |

The Q4 and MXFP4 columns are **different quantizations**, not a matched speed
claim. Our first prefill also includes pipeline warmup and has the drafter and
vision encoder loaded, unlike upstream's prefill procedure. The prose DSpark
controller accepted no draft cycles in these Q4 runs; it remained enabled.

See [upstream reproduction](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/f48c5e2e20b1aaae1c79735a8457225d2469ae73/docs/REPRODUCE-V41.md)
and [runtime controls](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/f48c5e2e20b1aaae1c79735a8457225d2469ae73/docs/RUNTIME-V41.md).

## Source audit and candidate

The verified upstream head is `f48c5e2e20b1aaae1c79735a8457225d2469ae73`,
408 commits beyond the vendored merge base
`b10cb40c1786d93aa2b4c819635a7b9e84b2b635`. Upstream now includes a DwarfStar
rebase, native MXFP4 drafting, grouped Metal dispatch controls, narrow weights,
matrix-unit prefill, bounded decoder replay, vision acceleration, keyed/full
distribution sampling, prompt/tool rendering fixes and revised KV stores.
Recent commits include `80d1986` (exact sampling), `476ef72` (GPU p/q
distributions), `1d28e1f` (tokenizer), `1addcde` (tool rendering), `b596064`
(KV eviction) and `ef55cbb`/`60e7ea3` (autorelease pools).

The existing Q4 branch already has counterparts for asynchronous Engram reads,
queued decode, rounding/HC/router/shared-expert fusions and depth selection.
A wholesale replacement would also require preserving its native import,
negotiated cone v2 and complete sampler/tool contracts; it was not performed.
Upstream's changed drafter names, target metadata, snapshots, numerics and
retired environment switches require explicit compatibility work.

Adapted upstream `a743c56d3c5816811f854f686b1e2919e3077eea`: submit after layer
zero, then every two layers. The existing two Engram buffers, joins, final
drain and output head remain intact. Enable with
`DS4_DS41_DECODE_QUEUE2=1` or `decode_queue2: 1` through the existing diagnostic
lever API. Zero retains the prior cadence. The environment inventory forwards
the option to the native Mac process.

At 8K: +2.28% serial; at 131K: +2.26%. Each timed configuration has two
interleaved runs after warmup. All 128 token IDs agree at each depth across
cadences and serial/DSpark modes. This is token parity, not a full-logit or
full-feature paired qualification. An upstream untracked-weight default was
also built and measured; it gave no benefit here and was not retained.

Build: Apple clang, `-O3 -ffast-math -g -mcpu=native`, C99, native-frontier
support enabled; Objective-C ARC; retained C++ samplers linked at `-O2`/C++17.
Every build overwrote one stable `ds4-server` path on the Mac. The final
candidate SHA256 is
`65e796778c6f711577a07c2061bc643f646d35fdf0f46c35322915287ba83c4a`.
The timing build was `8f6c97e14cae0c177f9293daf41157936d3ebeb15d384056bdb5f83e38174400`;
the final build removes the unsuccessful untracked-default experiment. Its
full timing and paired qualification have not been repeated.
All 1,114 delivered source files match their RTX-generated SHA256 manifest;
manifest SHA256
`6f195571778258b6ad1ecefe4b74f3a0ad4b5fb522781c1708a685ec56dbc0d1`.

## Qualification limits and remaining work

1. MXFP4 weights are absent from the checked model directories. The target is
   467.2 GiB and drafter 7.4 GiB; their pinned preparation needs substantial
   additional space. Download approval was requested; no weights were fetched,
   requantized, deleted or replaced. Q4 was not substituted as an exact match.
2. The installed production RTX producer rejects current stack options with
   `Unknown or repeated producer option`. Binary inspection also finds no
   `--carry-compact-record`, `--yoco-keep`, or
   `--yoco-remote-state-version` strings. The latter two are required for cone
   v2. The request failed before a successful RTX prefill/import; no paired
   speed is claimed. A compatible producer must be built and qualified.
3. No experimental merge, Studio runtime/preset update, or three-machine
   deployment was performed. These remain pending real paired qualification
   and the exact-model comparison. The candidate remains on its owned branch.

## Safety and restoration

Mac tests: explicit 400 GiB host cap, 16 GiB physical reserve, 494 GiB wired
ceiling, swap/compression monitoring. Builds: 4 GiB cap. All test listeners and
SSH forwards bound to `127.0.0.1`. The attempted paired test took the shared
exclusive RTX lease (8.5 GiB host cap, 0.75 GiB floor); it has been released.
No memory-budget violation or new system swap I/O was observed on the Mac.
RTX bounded paging observations and global pagefile growth were zero;
kernel ETW attribution was unavailable, so this is not an exhaustive no-I/O
claim. All owned engines stopped cooperatively.

MiMo was ejected through Studio on both hosts, then restored unchanged on the
Mac and verified with a sampled request returning `READY`. The RTX reload
route returned HTTP 401; the canonical desktop launcher subsequently adopted
the healthy Mac server without any token or security change. RTX/Mac
selections and live status were restored. Fleet reports all three machines
current with matching backend/frontend bytes; MacBook Remote Link remained
unavailable. No owner chat was created or modified.

## Capability reductions

None. No fields, controls, samplers, vision, tools, state import, cone options or
bias options were removed, ignored or forced off to make a test pass. The
failed paired request was retained as a failure. Unchanged features are not
claimed newly qualified by the token-parity measurements.

Machine-readable candidate measurements: [macdecode-measurements.json](macdecode-measurements.json).
