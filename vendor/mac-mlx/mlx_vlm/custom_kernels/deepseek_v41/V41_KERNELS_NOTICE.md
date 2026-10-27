# V4.1 fork kernel provenance

The `v41_attention.metal`, `v41_index.metal`, and `v41_topk_{sort,merge}.metal`
ports derive from [IngeniousIdiocy/ds4-v41-m3ultra at c8aa547](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/tree/c8aa547e03ff14839b32763045a2c86f2e8362b5),
branch `v41-m3ultra`. The repository's full MIT copyright and permission notice
accompanies this port as `V41_KERNELS_MIT_LICENSE.txt`; SPDX identifiers are
preserved in the derived source and adapters.

Sources:

- `metal/dsv4_misc.metal:5784-5882,6116-6286`: half-row online attention
  helpers, sink update, selected-ID mask helper and rb16/lean kernel.
- `metal/dsv4_misc.metal:2226-2357`: tiled index scores, fp32 Q/K staging,
  fp32 matrix accumulation and fp32 weighted head reduction.
- `metal/argsort.metal:51-315`: descending bitonic sort and sorted-run merge.
- `ds4_metal.m:20077-20324,32457-32704,38210-38323,38455-38463`: dispatch.
  `ds4.c:41008-41083` confirms V4.1 reuses the GLM-named fp32 score tile on
  pre-M5 hardware. The older DeepSeek-named half tile is not this V4.1 route.

Local adaptations:

- MLX supplies typed buffers and template dimensions instead of byte-strided
  argument structs; Metal threadgroup arrays replace dynamic allocations.
- Attention uses the upstream lean geometry, eight heads and sixteen rows,
  with the dot/online-softmax/sink helpers unchanged. One threadgroup handles
  one query and eight heads; queries are parallel, not shared within a group.
- Explicit signed IDs replace the chronological ring-window scan and sorted
  compressed-only list. Negative/out-of-range IDs are skipped without ending
  the scan; duplicates and valid row order are preserved. Bank offset is NW.
- Attention promotes no persistent caches: each call stages Q and both banks
  as fp16 once; fp32 output is cast back to Q dtype. Bank concatenation is
  unnecessary. Index Q/keys are promoted to fp32 once and staged as float inside
  the original tile; projection, weights and accumulated scores remain fp32.
  The fork BF16-rounds its weight projection; our existing FP32 projection is
  retained. Its fixed scale 1/64 becomes the caller's scale (also 1/64 at H=32).
- Index visibility uses caller-supplied compressed lengths, not absolute
  positions/ratio. Existing candidate-source and candidate-consumer logic stays
  between scoring and selection. Generic noncausal sort/merge handles the
  already masked scores, including empty visibility. Final MLX ID sorting and
  masking add the caller's bank offset and emit -1 for unreachable positions.
- The generic fork top-k network is ported, not its causal shuffle/prefix or
  decode-only radix/indirect-dispatch accelerators. Sort uses 256 threads at
  most; intermediate runs retain their original full scratch width.
- Separate default-off `DSV41_PREFILL_V41_ATTN=1` and
  `DSV41_PREFILL_V41_INDEX=1` require N>=64. DSpark verification excludes both
  through an explicit context guard; decode retains its existing route.

Attention staging and reduction order differ from the existing fp32 three-stage
path; the index scorer changes FP32 reduction/scale order. FP16 has a narrower
exponent range than BF16;
large or tiny values may overflow or underflow. The Mac attention gate reports
both paths against fp32 and fails above exactly 2x baseline error. No tolerance
relaxation, speedup, bit-exactness, or sampled-equivalence claim is made.
