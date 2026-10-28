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
token history or invent a second payload layout. Layers0–20 publish every
compressed/index row. The padded cone preserves every final128 raw window,
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
measures approximate140/v1 and must not be interpreted as cone/v2 proof.
CONEALL's live release gate is a matched approximately100K recall run through
the actual stack WebUI on RTX→Mac and single-machine cone, verifying the
negotiated2541/2 identities, completed state import/hash acknowledgment and exact
recall answers, then restoring the prior Mac server and Studio. Live receipts
and timings belong in the CONEALL result; CPU checks alone do not establish
paired performance or numerical parity.
