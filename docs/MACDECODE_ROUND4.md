# Mac decode upstream ports, Round 4

The remaining upstream execution paths are available without removing native
state import, cone, disk KV, vision, sampling, bias or tool support. Upstream
`v41-m3ultra` was fetched and still resolves to
`f48c5e2e20b1aaae1c79735a8457225d2469ae73`. Its
[published reference](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/f48c5e2e20b1aaae1c79735a8457225d2469ae73/README.md)
uses native MXFP4 weights; our existing Q4-to-MXFP4 target and drafter remain
explicitly lossy speed proxies. No model was downloaded.

## Execution paths and controls

| Port | Upstream source | Selection |
|---|---|---|
| Read-only model views without hazard tracking | da48ab1c | V4.1 default; `DS4_METAL_MODEL_TRACKED=1` retains tracking. Historical explicit `MODEL_UNTRACKED` semantics remain. |
| Aligned exact uint4 copies in the open compute encoder | 1df25c6f | Default for eligible disjoint ranges; `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1` retains blits. Bounds, misalignment, concurrent and overlap cases retain their original behavior. |
| Literal V4.1 attention low projection, including the fork's BF16 store | e06a0b7f | `DS4_DS41_ATTN_LOW_STATIC=1` |
| Shared down beside routed gate/up, retaining the combined router/shared/HC producer | adaptation of 555e258f | `DS4_DS41_CONCURRENT_DOWN=1` |
| Full BF16 shared/routed two-stage FFN | 555e258f | `DS4_DS41_CONCURRENT_FFN=1`; uses the retained separate HC/router producers so both stages can overlap. The original producer remains selectable. |
| Six-to-eight-row MXFP4 verify kernels | 04830a71 | `DS4_DS41_VERIFY_MXFP4=1`; only actual DSpark verify blocks gain the new admission. Ordinary batches retain their old arithmetic. |
| Distinct MXFP4 expert planning and shared reads | a5635c52 | `DS4_DS41_MXFP4_DEDUP=1`; two to eight fused rows, original per-pair arithmetic. The 4 KiB plan uses the owned/wired allocator (16 KiB physical page on this Mac). |
| Persistent per-row Engram workers | bb7e488f/current reader | `DS4_DS41_ENGRAM_PERSISTENT=1`; 24 bounded workers on Apple, both uncached tables, no table cache. Existing batch, prefill and synchronous paths remain. |

All six new graph levers start at zero and are exposed by the existing debug
lever mechanism. Their environment variables are registered and documented.
Internal schedule selection computes the same model operations; it does not
remove user capabilities. Read/worker failures fail explicitly. Reader cleanup
joins workers before table descriptors close, row views do not inherit reader
ownership, and actual completion timestamps preserve the existing diagnostics.

## Verification so far

`tests/native_metal_copy.m` executes the real Metal implementation and checks:

- Compute versus blit dispatch selection, exact copy bytes, views, disjoint
  ranges, alignment, zero length, bounds, concurrent fallback and ordering.
- Model-view policy controls, including historical override precedence.
- Static versus generic low projection for both BF16 modes.
- Down-only overlap, consumer ordering and abort cleanup.
- MXFP4 rows 2–8 versus independent scalar rows; dedup gate/up/middle/output
  bits for shared and disjoint experts; ordinary-batch behavior.
- Full shared/routed overlap versus serial middle/down/routed words with both
  fixed-shape settings, including early error cleanup.
- Persistent Engram generations versus serial rows, real timestamps,
  EINVAL/EIO/EDOM handling and cleanup during in-flight work.

The synthetic model is in owned wired RAM, with no multi-GB scratch file.
Run this test alone under an outer 20 GiB host budget and 16 GiB physical floor;
its allocator budget is 12 GiB. Set `DS4_TEST_SCRATCH` to the existing task
scratch directory, plus `DS4_METAL_WIRED_CORE_KV=1`,
`DS4_NATIVE_SESSION_HOST_BYTES=12884901888` and
`DS4_NATIVE_SESSION_HOST_FLOOR_BYTES=17179869184`. Compile it against the same
engine core objects and `ds4_image.o`, with Foundation/Metal, at one stable
test executable path. No actual model is loaded by the component test.

The eight small-port full-model proofs (all four arms at 8K and 131K), plus
four full-FFN/reader 8K proofs, match all 128 × 129280 baseline float32 decision
logits. Logit-capture requests are excluded from throughput means; the
verify/full-FFN sweeps explicitly record their enabled command-buffer and
Engram diagnostics. Forty Python compatibility, pairing,
memory, build-identity and environment checks plus the native routing/disk-seed
fixture pass in isolated scratch data directories. An initial environment
check correctly rejected stale Mac source references; SHA synchronization fixed
the mismatch without weakening the test.

## Measurement scope

The inherited standalone harness omitted the explicit owned-wired flag. Its
earlier Round 4 sweeps are matched comparative diagnostics, not live Studio
preset qualification. The corrected harness always enables owned-wired core/KV
and records that policy. No production preset or service rollout has occurred
in this round yet.

At 393216 allocated context and 128 greedy continuation tokens, the comparative
baseline measured 34.273 serial prose 8K, 33.308 serial prose 131K and 43.688
DSpark code 8K. Copies/view policy plus static low projection measured
34.601 / 33.794 / 45.812. Persistent readers reached 35.117 serial prose 8K;
full FFN and down-only overlap were slower than the existing fused producer
on this fixture. These are not a 40–60 tok/s release claim.

The command-buffer diagnostic attributes about 26.44 ms/token to GPU spans
and 2.43 ms/token between spans on 8K prose. The dispatch Engram reader exposes
about 2.53 ms at layer 1. Shader work therefore remains relevant even if every
host wait is hidden. All measured jobs exited gracefully within their explicit
budgets/floors and observed no new swap I/O. Final owned-wired, Studio GUI,
FP4-to-FP4 release and fleet qualification remain required.

## Capability reductions

The owner-selected target/drafter requants are lossy speed proxies. No other
capability reduction is introduced by these ports. No unsupported feature is
hidden, disabled or replaced with a no-op to make the model load or tests pass.
