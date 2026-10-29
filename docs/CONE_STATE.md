# Negotiated exact-cone state

DSV4.1 ULTRA paired launchers and `slstack` single-engine/paired defaults select
`yoco_keep=2541`. Explicit full-prefill settings (`0`) and approximate YOCO140
(`140`) remain supported. The large-RAM preview selects cone too; its missing
CPU mode3 receiver still prevents launching that example.

| Keep | Negotiated state version | Meaning |
| --- | --- | --- |
| 0 | 0 | Full prefill; legacy peers omit both fields |
| 140 | 1 | Approximate bounded replay, explicit opt-in |
| 2541 | 2 | Exact dependency cone |

Version2 reuses the version1 mode3 export/import path. It does not truncate
token history or invent a second payload layout. Layers 0–20 publish every
compressed/index row. The padded cone preserves every final 128-row raw window,
the complete DSpark rings and final logits. The existing canonical native
payload, per-owner hashes, token/logit digests, delta and rewind checks apply
unchanged. Local snapshots retain their separate local-state identity.

The authenticated pair record fixes keep/version to the pinned producer and
receiver. Receiver health advertises supported versions `[1,2]`; a v1-only
receiver cannot accept cone. The Mac sends both request headers, the broker
compares both with its configuration, and the producer reports its negotiated
pair before export. Crossed pairs (`140/2`, `2541/1`), missing cone provenance,
unknown versions and boolean/malformed identities fail closed. No fallback to
approximate140 occurs.

Select `examples/dual-machine/dsv41-ultra-cone` for paired use. The existing
`dsv41-ultra` and legacy `dsv41-ultra-yoco` presets select cone too. To request
approximate140, pass `-ApproximateYoco` to the RTX PowerShell wrapper or
`--approximate-yoco` to `examples/run_preset.py`. Private direct configurations
can explicitly set `yoco_keep=140` or `0`.

CPU checks cover request/header validation, authenticated pairing, legacy v1
compatibility, rejected version mismatches, shared single/paired arguments,
launcher defaults and the native Mac negotiation helper. The transfer and
image qualification harnesses now default to cone; the image harness requires
identical full/cone logit hashes unless approximate mode is explicitly selected.

The historical [paired WebUI qualification](YOCO_PAIRED_QUALIFICATION.md)
measures approximate140/v1. The [CONEALL live proof](../examples/dual-machine/dsv41-ultra-cone/proof/summary.json)
passes on **105,054 native tokens** with keep2541/version2 through the actual
stack WebUI, headless Mac Chromium, RTX prefill and Mac decode. The identical
single-machine cone prompt and both follow-ups return exactly `AMBER OTTER 731`,
the same phrase, and `731`. Both interfaces show advancing prefill bars,
incremental replies and numeric rates/context usage.

| Route | Initial prompt seconds | Prompt tok/s | Initial decode tok/s |
| --- | ---: | ---: | ---: |
| Single RTX cone | 46.282 | 2,269.867 | 2.256 |
| RTX→Mac cone v2 | 59.005 | 1,780.432 | 27.499 |

These are one matched observation per route, not repeated estimates. The paired
prompt interval contains startup and prefill 49.141 s, export preparation 2.250 s,
transfer/import/hash/ACK 3.066 s, saved rewind 2.454 s and remaining control work.
The transferred complete mode3 state is 691,633,076 bytes. Mac-local 20/19-token
follow-ups retain the complete history and decode at 31.925/28.746 tok/s.

The [state parity gate](../examples/dual-machine/dsv41-ultra-cone/proof/state-parity.json)
reuses the actual WebUI token IDs. Every one of the 48 common owners and the
complete 129,280-float initial logit row are byte-identical between local mode1
and paired mode3/v2. The paired 49-owner checkpoint, including DSpark, also
exactly equals the real WebUI checkpoint that the Mac imported and verified
before acknowledgment. Full row SHA256 is
`ae6f051f883ff5ed22ad386cf71dd1f83111e4b976ebf835a133fbf678922bee`.
One forced ID captures that initial row without evaluating another token. A
separate three-ID non-bridge continuation attempt refused an existing exact
radix dispatch-coverage check; no multi-step continuation parity is claimed.

The proof uses producer source registration
`fc79ff19a94ae8d73d605f5ec31d4e19dd4cce9a`, qualified manual-build binary
`c97cc9de2f505366bbb1c4907a3942768a968d23867f3250e00aad7b14037715`, and Mac binary
`5734ca0ffdfb970e92c366d410338774970a0e905cf6f03aed391a4e022ffcdb`.
It does not assert that a full CMake/CUDA rebuild occurred. CPU checks and
native Mac negotiation/cone fixtures pass. Explicit physical-RAM/paging guards
remain active; the supplemental state gate reports zero owned pagefile bytes
and zero global pagefile growth, with kernel ETW attribution unavailable.

The candidate Mac server/browser retired gracefully. The original Mac server
was restored with exactly its saved command, environment, working directory
and binary; its model endpoint returns 200 and has no task-scratch dependency.
[Restoration receipt](../examples/dual-machine/dsv41-ultra-cone/proof/mac-restoration.json).
Studio restoration and final scratch retirement are recorded in CONEALL_RESULT.
