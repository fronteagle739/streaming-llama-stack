# Streaming Llama Pack v1

`.slpack` is a section container for streaming models. The envelope does not
quantize tensors, change model arithmetic or select runtime features. Each
architecture defines its tensor layouts and validates them independently.
QFN's native routed expert payload retains every NVIDIA NVFP4 code, per 16
UE4M3 scale, FP32 projection global and calibration byte. This claim applies
to that relayout; it does not describe all dense or inherited auxiliary weights.

All integers are little-endian. The 64-byte header is:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | `SLPACK\0\0` (`53 4c 50 41 43 4b 00 00`) |
| 8 | u32 | Envelope version, 1 |
| 12 | u32 | Section count, 1..511 |
| 16 | u64 | Exact complete file length |
| 24 | 40 bytes | Zero, reserved |

The section table starts at byte 64. Each 128-byte descriptor contains:

| Offset within entry | Type | Meaning |
| --- | --- | --- |
| 0 | 64 bytes | UTF8 name, 1..63 encoded bytes, NUL terminated and zero padded |
| 64 | u64 | Absolute payload offset |
| 72 | u64 | Payload length |
| 80 | 32 bytes | SHA256 of the payload, including any internal tensor padding |
| 112 | 16 bytes | Zero, reserved |

The header and table reserve 64 KiB. Every section begins at a 2 MiB boundary;
gaps are zero filled. Descriptors are in physical order with unique names and
no overlap. An empty payload is permitted and has SHA256 of the empty string.
The file ends at the final payload byte, without trailing alignment padding.
Active unknown reserved fields, future versions, duplicates, overlap, invalid
names, truncated payloads and unadvertised file tails are rejected.

[`scripts/slpack.py`](../scripts/slpack.py) supplies an exclusive-create
deterministic writer, bounded file-range/part streams, strict reader and full
section/file hash verifier. Its verification writer consumes fully regenerated
payloads and checks them against a reference before calculating a new virtual
envelope hash; it creates no full payload copy. Readback for derived sections
requires matching generated and freshly authenticated reference section hashes.
Metadata reads default to a 32 MiB cap. Large tensor
payloads use 4 MiB streaming reads. Architecture readers may keep narrower
model-specific admission rules; envelope acceptance alone is not model support.

The QFN native reader accepts the new envelope and has an explicit historical
container compatibility path. Existing versions 1/2, original quantization
types, calibrated input scales, MTP, profile, vision and tokenizer data remain
available. New writers emit only the new envelope. Old metadata inside a
migrated file is retained as provenance; changing the envelope does not rewrite
its payload or erase its recorded source composition.

For QFN, `experts.layout` v7 identifies
`flashinfer-sm120-nvfp4-e16-expert-major-v1`. Each expert is 2,764,816 bytes:
gate/up/down projections each occupy 921,600 bytes, followed by 16 unchanged tail
bytes. Values use adjacent E2M1 nibbles; scales are ordered `[K/64,N,4]`.
`experts.input_scale` and its index retain the source calibrated activation
globals. The reader continues to validate these descriptors independently
of the envelope version.

The GLM importer must emit the same envelope with its own complete payload
inventory, shape constraints and source preservation certificate. A QFN
fixture does not qualify GLM's loader or a full GLM pack.

Run `python scripts/slpack.py MODEL.slpack` under the same explicit host-memory
budget and I/O admission used for model tools. `--allow-legacy` is required to
verify a historical container. Verification returns exact totalbytes,
per-section SHA256 and full-file SHA256. It does not start a model or listener.

The independent small fixtures in `tests/test_slpack.py` cover deterministic
envelopes, arbitrary preserved fields, MTP/vision payloads, part padding,
explicit legacy admission, malformed descriptors and source NVFP4 expert
conversion against a direct values/scales/tail/calibration oracle.
They also compare virtual/materialized file hashes, exercise main/MTP HC8
readback and reject a corrupt reference despite matching generated descriptors.
Set `SLPACK_FIXTURE_ROOT` to the caller's scratch directory; fixtures are retained.
These component tests do not replace a full-model quality or performance gate.
