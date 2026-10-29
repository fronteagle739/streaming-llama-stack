# Local standard-layout MXFP4 speed proxy

This follow-up uses existing local Q4 weights. No model or tool download is
needed. The target and DSpark routed experts are requantized from Q4_K to
standard GGML MXFP4. This is lossy and is a speed proxy for native MXFP4, not
the native-precision upstream checkpoint. Dense, Engram, tokenizer, vision and
other support tensors remain unchanged.

The converter now accepts the distinct `deepseek41-dspark` architecture. It
checks the MTP tensor roles and Q4 expert triplets explicitly. Main-model
validation remains unchanged. Kernel-native packed DSpark output is refused;
the supported drafter output is the standard layout used by the Mac loader.
Unknown tensor roles and types fail rather than being copied as guessed data.
Legacy drafter keys are retained and their current `dspark.*` aliases are
written from the original values. Target layers are read from the matching
main GGUF's embedded configuration; source URL/revision, stage count, expert
counts and layer bounds must agree. An ambiguous sibling requires an explicit
`--requant-target-gguf`. Neither checkpoint-family check is bypassed.

Build the existing CPU adapter with precise floating point and OpenMP. On
Apple clang this needs the already installed OpenMP headers and runtime;
`-ffp-contract=off -fno-fast-math` preserves the rounding oracle. Select one
stable library output path. Verify it with `test_dsv41_requant.py`, including
the independent float32 decoder/nearest-code oracle, unchanged dense payloads,
and interrupted/resumed main and DSpark conversions.

For each component, select a separate journal location inside the task's
single scratch root. Use the same adapter and explicit memory limits:

```sh
python scripts/convert_dsv41_gguf.py \
  --requant-gguf "$MODEL_DIR/source.gguf" \
  --out "$MODEL_DIR/standard-MXFP4-requant.gguf" \
  --runtime-root "$SCRATCH/component" \
  --hold-file "$SCRATCH/model-read-ledger.md" \
  --quants-library "$SCRATCH/requant_q4_mxfp4.dylib" \
  --threads 16 --expert-layout ggml-mxfp4 \
  --host-budget-gib 4 --runtime-floor-gib 16
```

Never convert or hash weights that a running engine holds. Retain the source
until the converted inventory, full stored-byte SHA readback and engine load
have passed. The converter keeps a 20 GiB output-volume reserve. Do not select
`--defer-readback` for a file that will immediately be timed in place.

The source quantization descriptions are retained as provenance; the added
`deepseek41.speed_proxy` marker identifies the lossy conversion, and routed
tensor descriptors identify the actual MXFP4 payloads (type 39). There is no
`deepseek41.expert_layout` packed-layout marker in standard output.

## Cadence qualification

`DS4_DS41_DECODE_QUEUE2` keeps its existing zero/one semantics. The optional
`logits_path` field on the serial `/debug/bench` diagnostic writes every
decision's complete raw float32 vocabulary row. Use separate diagnostic and
timing requests, restore the same fixture for both cadences, and compare all
bytes. Artifact I/O is excluded from reported throughput. DSpark and score
requests explicitly refuse this serial diagnostic field.

Throughput uses 128 greedy tokens, excludes the first token for serial runs,
and compares interleaved zero/one/one/zero arms after warmup. Prose uses the
upstream novel fixture at 8,192 and 131,072 tokens. Code uses the first 400,000
bytes of upstream `ds4.c` at the pinned comparison revision. Context allocation
is 393,216; prefill chunk size is 8,192. DSpark and vision stay loaded.

## Paired release gate

The existing transport binds state to the full selected model SHA. A locally
requantized Mac model has different weights and identity from the Q4 producer.
Changing a pathname, claiming the old SHA, removing the model check, or silently
falling back to local prefill cannot qualify this combination. Preserve the
current check and report its actual refusal. A production pair needs matching
weights on both ends, or a separately specified and qualified approximate
cross-quantization protocol.

## Measured candidates, 2026-10-29

All three controls remain explicit opt-ins pending the mixed-weight paired
release gate: `DS4_DS41_DECODE_QUEUE2=1`,
`DS4_DS41_MXFP4_FIXED_SHAPES=1`, `DS4_DS41_NARROW_WEIGHTS=1`.
Zero retains the corresponding existing reader/dispatch/cadence.

The fixed-shape kernels are adapted from upstream
`e06a0b7f28d7843d1084d4547250f27782e49115`. The exact narrow readers are
adapted from `f48c5e2e20b1aaae1c79735a8457225d2469ae73`: F32 router gates
whose low halves are zero become BF16, and exactly representable F16 Engram
and indexer blocks become E4M3/E8M0. All other values retain their original
readers. No sidecar or replacement weight values are introduced. The load
reports which copies were admitted. Eligible copies are built independently
of the runtime reader switch, so enabling it after load works too. Allocations
retain the existing wired-buffer and allocation-accounting controls.

| Decode workload, tokens/s | Published native MXFP4 | Upstream on our requant | Installed Studio on our requant | Candidate: all three on |
|---|---:|---:|---:|---:|
| Serial prose 8K | 41.4 | 38.90 | 32.963 | 34.638 |
| Serial prose 131K | 40.5 | 38.59 | 32.157 | 33.851 |
| DSpark prose 8K | 40.1 | 37.73 | 32.298 | 34.258 |
| DSpark code 8K | 58.8 | 51.74 | 46.188 | 45.749 |

These results do **not** establish an upstream performance match. Serial
decode is approximately 5% faster than the installed binary; DSpark code is
approximately 1% slower in the separate installed-control run. Within each
interleaved experiment the candidate was faster: fixed shapes about 2% and
narrow readers about 1.5–1.7% for serial decode. Cadence's 131K gain was 2.8%.
DSpark prose kept drafting enabled but its cost controller admitted no cycles;
code admitted 20 cycles. This is an observation, not a disabled drafter.

For each experiment, both arms produced byte-identical full logits at all
128 serial decisions for each of three fixtures: 49,643,520 float32 values
per experiment. The hashes also agree across experiments. This qualifies the
measured serial computations; it does not qualify every sampler, vision input
or imported-state combination. Those capabilities and checks remain intact.

The upstream control is the pinned `ds4-bench` engine, with harness-only
options for three runs restored from the same RAM snapshot and for keeping
the drafter resident in serial runs. Its genuine CLI has no vision option;
our server and installed control keep vision loaded. The published column
uses different, native-precision weights. The default upstream control and
our server also produce different first continuation tokens. Do not treat
their rates as a numerical-equivalence proof or as evidence that requantization
alone explains the remaining backend gap.

An additional explicit upstream `DS4_V41_EXACT=1` serial control measured
38.145 tokens/s at 8K and 31.950 at 131K. Its first continuation token still
differs from our server's. The default control's remaining speed advantage
therefore is not explained solely by its approximate groups. A complete
attribution of the residual gap is not established by these measurements.
The benchmark CLI refuses the server's `--exact` spelling; that refused
attempt is retained, and the documented environment control was used.

## Final paired check and release disposition

After all Mac-only work, the final test waited for the priority RTX prefill
lease and sufficient physical RAM. It paired cone v2 with `yoco_keep=2541`;
all three candidate controls, DSpark, vision and the complete existing
sampler/import contracts remained selected. The native eight-token control
completed. Paired qualification then returned HTTP 409 with
`native session configuration/model/build identity rejected`.

The selected Mac SHA is `cf05bdcc…`, while the native import contract still
requires the original RTX Q4 SHA `2cc6bc25…`. Its explicit model check remains
in `ds4_native_session_setup.inc`. No RTX prefill/import or paired recall/rate
is claimed. Both model processes retired cooperatively, the broker/tunnel
stopped and the RTX lease was released. The setup rebuild used the same
source/flags and stable server path; its SHA256 was
`538c47d5ca4b2e05bf0935a0f177b77e8b6b524bb221867b6a9fa765c3670ae7`.
All 1,312 delivered files still matched the pre-benchmark source manifest.

The exact optional changes and converter were merged into `experimental`
at `ae85e3346db755b785500f46fe480977c626445a`. Production preset replacement
and the three-machine runtime rollout are held at the actual paired gate.
Presets remain intact; no fallback disables paired import or changes it into
local prefill. The original Q4 remains available, and the verified requants
have a provenance receipt beside their model files. To release a production
pair, select matching weights on both ends and qualify the updated identity
manifest, or specify and separately qualify an approximate cross-quantization
state protocol. Merely allowing unequal model hashes is not that protocol.

MiMo was restored with its original preset and 460 GiB cap, under a watched
16 GiB floor, and answered `READY`. The first restore safely refused because
the new requant's clean file cache left too few immediately free pages.
The existing bounded per-file cache helper invalidated those two idle files
without payload reads or file changes; the retry succeeded without changing
MiMo settings. The RTX selection was restored through the canonical desktop
launcher and is live with the same preset. All three Studio machines report
the existing current revision; the MacBook Remote Link remains unavailable.
This is a restoration check, not a claim that the gated runtime update shipped.

Raw measurements, complete token sequences, logit hashes, model identities
and memory receipts are in [macdecode-r2](macdecode-r2/summary.json).
The candidate server SHA256 is
`356f8ce75839f3df9f57014c87613b8a6d986cbec1dfefe7a19b84d5f54f6de3`.
The main requant SHA256 is
`cf05bdccb86b823ce794eab2ca033a649b6a2f5c03cee8a4b177d86de017ae44`;
the final metadata-compatible drafter SHA256 is
`0e5d28ea5521ce3fcd0a8014ba243e77666a9b72722d292257bd7cc69370e120`.
The older drafter hashes in conversion/refusal receipts describe intermediate
metadata headers; the final receipt supersedes them without changing tensors.
