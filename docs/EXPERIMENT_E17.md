# E17: packed FP4 KV codec qualification

Verdict: **needs-native**. The codec qualifies on captured model values;
there is no integrated packed cache allocation/read/write path. Shipping
presets retain their existing KV types. The experiment is behind
`LLAMADECODE_PERFEXP_KV_SAMPLES=1`, default0, plus the standalone codec's
packed/eight-bit arm selection. Neither flag selects a shipping KV format.

At a100,077-token native frontier, capture the last128 rows of all four
compressed-KV owners and all four index owners:327,680 actual values.
Compressed values use E2M1 with one E4M3 scale per16; index values use E2M1
with one E8M0 scale per32. Those are the model's existing QAT formats.
The codec rejects nonfinite/noncanonical inputs and any block that cannot
restore every original bit. Signed zeros are preserved.

| Captured values | Expanded8-bit value codes + same scales | Packed4-bit codes + same scales | Saved |
|---|---:|---:|---:|
| All8 owner samples | 346,112 B | 182,272 B | 163,840 B (47.34%) |

The8-bit comparator stores each existing E2M1 code in a whole byte; it is
not an arbitrary FP8 quantizer or a native8-bit engine cache. CPU decode
and CUDA decode restore all samples bit for bit against canonical values
and the expanded-code comparator. CUDA timing replicates each source slice
to32,768 rows; the two arms use the same output buffer and kernel shape.
Three warmups then100 iterations, mean GPU-event milliseconds:

| Owner | Kind | Expanded8-bit-code ms | Packed4-bit ms |
|---|---|---:|---:|
| 0 | compressed | 0.042059 | 0.042150 |
| 1 | compressed | 0.042106 | 0.042044 |
| 2 | compressed | 0.041439 | 0.041729 |
| 3 | compressed | 0.041712 | 0.041776 |
| 0 | index | 0.013167 | 0.013016 |
| 1 | index | 0.013534 | 0.013283 |
| 2 | index | 0.013180 | 0.013498 |
| 3 | index | 0.013509 | 0.013178 |

No robust component speed gain is established. This times unpack to F32,
excluding CPU packing, uploads and attention/index consumption. The current
component allocates its input arena at the larger comparator size, so the
serialized byte reduction is not an observed engine VRAM-peak reduction.
CUDA memcheck on all actual sample rows reports0 errors. Boundary tests
cover signed zero and refusal of NaN, infinity and noncanonical values.

For the existing ownership ratios (sources2/8/14 at1:2 and20 at1:1), packed
compressed data is720B per prompt token and index data170B:890B/token.
Expanded8-bit codes plus scales total1,690B/token, BF16 canonical storage
3,200B/token and F32 canonical storage6,400B/token. All retain the F32 raw
rings/pool tails (10,498,048B fixed). These are layout arithmetic, excluding
allocator alignment, metadata and workspaces; no new native layout is built.

| Hypothetical cache layout | Bytes at1,048,576 tokens | Tokens within1GiB, capped at model maximum |
|---|---:|---:|
| Packed | 943,730,688 | 1,048,576 |
| Expanded8-bit codes | 1,782,591,488 | 629,138 |
| BF16 | 3,365,941,248 | 332,263 |
| F32 | 6,721,384,448 | 166,131 |

The packed-versus-expanded-code difference at1M is800MiB. Larger supported
sweeps/context and end-to-end memory/speed gains are **unmeasured**: a real
packed owner allocator, append/publication rules, canonical checkpoint/wire
handling, and fused sparse reads/dequantization are still required. Merely
unpacking whole owners into F32 would consume the memory being saved.

The read-only capture is separately checked against sampling disabled, both
with the unchanged original Q4 model and existing F32 cache,100K prompt:

| Arm | Prompt execution s | Prompt tok/s | End-to-end TTFT s |
|---|---:|---:|---:|
| Sampling disabled | 65.176960 | 1,534.285 | 69.421 |
| Sampling enabled after quality checks | 65.269230 | 1,532.116 | 69.469 |

Both retrieve the long-context key,17 full-vocabulary logit rows match
bitwise (KL0,top-1 all), and the capture does not mutate canonical state.
Those model gates qualify the sampler, **not** an integrated packed-cache
model path. Full-model KL/top-1 and long-context recall against8-bit KV
remain unmeasured. Prompt time excludes startup/open and bulk export;
TTFT includes startup and first output, excluding lease queue/harness setup.

Source, all numerical results, pins, memory formulas and guardian receipts
are in `examples/perfexp/e17/`. Compile the CUDA driver with CUDA13.2,SM120a,
and the existing vendor/source include roots; compile the CPU fixture with
C++17. Run under shared exclusive GPU lease, explicit3GiB host budget,
0.5GiB physical floor,512MiB desktop reserve and paging stops. `--check`
uses only actual sample rows for memcheck; the default benchmarks replicated
rows. Accepted runs close gracefully with confirmed cleanup and healthy
Studio/Remote Link. Bounded page observations are not a zero-I/O proof.

The first capture was refused by the Python session operation contract before
native sampling; preserved/excluded. The strict bounded local command was
then added and the new same-day pair passed. Native code still requires its
explicit default-off setting and an absolute new output path.
