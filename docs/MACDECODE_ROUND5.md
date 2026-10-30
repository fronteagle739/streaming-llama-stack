# Mac decode Round5 qualification

Mac-only qualification is complete. Final optimized paired qualification,
Studio rollout, fleet and recorded GUI acceptance remain pending.

Upstream is `IngeniousIdiocy/ds4-v41-m3ultra` branch `v41-m3ultra`, commit
`f48c5e2e20b1aaae1c79735a8457225d2469ae73`. Our origin remains `b10cb40`.
Both engines use the same existing owner Q4-to-standard-MXFP4 lossy target
and DSpark requants, plus the same vision sidecar. These are speed proxies;
no native-precision model download or published41/59 claim is made.
Upstream's current same-proxy control itself measures38.38/52.38 on
serial prose8K/DSpark code8K. Its published native-weight figures do not
describe this proxy campaign.

One warmup and two normal timed128-token samples per workload,393216
allocated context, RAM prefix fixtures, greedy sampling, no profiling:

|Workload|Upstream tok/s|Ours tok/s|Difference|
|---|---:|---:|---:|
|Serial prose8k|38.37955|38.17120|-0.543%|
|Serial prose131k|37.70740|36.86875|-2.224%|
|Serial code8k|38.43925|38.98990|+1.433%|
|DSpark prose8k|37.67465|37.42200|-0.671%|
|DSpark code8k|52.37835|52.65080|+0.520%|

The code128-token continuations match upstream exactly. Prose continuations
diverge despite identical prompt-token hashes. Full-logit equivalence is
therefore against our retained Round4 arithmetic/state, not an assertion of
identical prose prefill math between forks. Upstream uses ordinary shared
allocations; our required anonymous wired core/KV allocation policy remains.
The remaining serial prose gap is0.54% at8K and2.22% at131K. The measured
small-kernel attribution now matches upstream for HC join and short sort.
The remaining compact long-index adaptation failed full-model equality and
is explicitly not promoted. No unmeasured explanation is claimed as proven.

The qualified ports retain every output and original alternative:

- All-eligible scalar block-mask shortcut and checked BF16/F32 upstream
  wide index schedules (`index_schedule=0/1/2`). Authoritative F32 state and
  nonexact restored-key support remain.
- Query-store BF16/RoPE fused into the existing HC/KV producer, retaining
  query, mix, split, KV and every ring word; standalone paths remain.
- Independent deferred HC queued alongside complete shared/routed FFN
  products (`concurrent_ffn=2`); historical0/1 and boolean controls remain.
- Scalar top512 shuffle block sort with staged/fused prefix merge below
  16384 keys; generic, causal, batch and long radix paths remain.
- Literal5120x4 HC FFN join with the original ordered FMA tree, all optional
  add/in-place/threadgroup variants, and exact16-byte carry publication in
  the same dispatch. Generic join and original copy remain.
- Paired independent QA1280/KV512 norms retain each original row reduction,
  weights and BF16 boundary. The query writer consumes the normalized KV
  once; original raw-input and standalone norm paths remain.

Final stable engine SHA256:
`dcff74fec99047c45cb2f4098793f4118bb1730b2955246c7dc0a49d418fdc92`.
Upstream control SHA256:
`bfe27dedaf5137b8441a9cd045bab9731a7788a04d11bd2c527695a4c9f28f29`.
The final21 requests include six complete128x129280-logit artifacts,
all bit-identical to the retained Round4 schedule, forced-continuation KL0.
Peak RSS369204510720 bytes, minimum available169823731712 bytes,400GiB
cap/16GiB floor, no swap or memory guard violation. Final Metal components
pass under20GiB/16GiB, with the allocator capped at12GiB. CPU compatibility,
environment, sampler binding, FP4 gate, native import/crossover and runtime
checks:64 passed plus one existing checkout-dependent argv test skipped;
the actual Studio argv builder is required again during runtime staging.

Three compact masked candidate experiments remain opt-in diagnostics:
wide compact scoring, original-expression compact scoring, and retained
original scoring with selection-only compaction. All component threshold,
tail, mask, score and rank checks pass; all three fail the131K full-model
hash comparison. Their exactness assertions remain intact. No KL pass is
invented for an aborted proof. A one-shot live score capture independently
matches the final-layer512 indices against the old GPU selector and CPU
sort; this does not resolve the full-model mismatch. The release recipe
explicitly selects `index_candidates=0`. Both diagnostic APIs and the
original scorer/selector stay available; no existing capability is removed.

The paired broker binds initial requests and commands to the trusted
selected model SHA from its stat-tied full-SHA receipt. Q4 remains supported;
packed FP4 is never accepted by claiming the Q4 identity. Architecture,
tokenizer, logical shapes and state/YOCO compatibility remain authoritative.
Native qualification can use the actual long prompt while retaining the
same eight full-vocabulary continuation rows and numerical limits. Native
and paired requests must have the same semantic digest as well as observed
continuation decisions; cache routing IDs alone may differ.

The primary portable paired recipe is
`examples/dual-machine/dsv41-ultra-cone-fp4/preset.json`: packed FP4 requant,
ULTRA exact cone2541/v2, automatic single-stream fold planning, the qualified
native FFN bridge and the current192MiB VRAM headroom. Set every `SET_`
field in a private local config. The Q4 recipes remain available and inherit
the qualified Mac schedules. For single RTX, use the existing single-engine
launcher with the same packed-FP4 local model/environment. Runtime staging
now carries selected native FFN/dense DLLs and relocates their paths and
temporary/cache paths to the permanent installed root; no task scratch
dependency is accepted.

See [measurements](macdecode-r5/measurements.json) for all eleven campaigns,
individual rates, full-logit hashes, failures, applied levers, source/binary
pins, matched diagnostic counter attribution and memory observations.

Capability reductions: none. The new compact candidate optimization is
not promoted because it failed qualification. Quality, all sampler/bias/tool
fields, vision, thinking/history, native import, cone/state versions and disk
KV capabilities retain their original routes. Deployment and final paired
acceptance are not yet claimed.
