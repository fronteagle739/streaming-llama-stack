# Reading fewer weights during prefill

DeepSeek V4.1 prefill reads **315,825,603,008 bytes** for a complete Q4_K/Q8_0
weight pass, rather than the GGUF's entire 518.6 GB. The two Engram tables stay
on disk; the existing loader gathers the prompt's selected rows into its bounded
host cache. A non-final encoder pass reads **166,382,548,664 bytes**.

[The tensor inventory](DSV41_STREAM_BYTES.json) gives each tensor's physical
file offset, payload size and read count in either kind of pass. The inventory
was reconciled against every observed stage of a four-pass 200K prefill.

| Complete pass component | Bytes |
|---|---:|
| Routed experts | 305,764,761,600 |
| Dense and other layer weights | 8,033,710,528 |
| Embedding | 1,323,827,200 |
| Output head and norm | 703,303,680 |

The measured 49,555-token control spent 8.179 s acquiring layers and 17.746 s
inside the executor. Transfer durations summed to 20.007 s, but much of that
overlapped computation. Adding transfer duration to executor time would count
the overlap twice.

## Routed reads

The single-engine launcher selects router-driven reads for ULTRA prompts of at
most 2,048 tokens. It runs the existing graph through its real top-six router,
reads the IDs, fetches missing expert extents into their original layer-bank
addresses, and runs the rest of the graph. No expert IDs, weights, products or
rounding operations are changed. Consecutive missing experts share one extent;
an expert is fetched at most once while that layer remains leased.

For longer full-layer passes the launcher keeps eager reads of the next layer,
because the measured 8K source-text and 32K recall-body routed experiments lost useful transfer overlap. Set
`env_overrides.LLAMADECODE_PREFILL_ROUTED` to `0` or `1` to compare both paths.
`tail` applies routed reads only to layers 21–39.

## Exact replay coverage

Use `--profile ultra-cone` with `scripts/dsv41_single.py`, or set
`producer_profile` to `ultra` and `yoco_keep` to `2541` in a local single-engine
configuration. Layers 0–20 still process every row and publish the complete
compressed KV, index keys and candidates. Layers 21–39 reuse that shared state
and replay only the tail needed to construct their private 128-row windows.

The dependency bound is 128 + 19 × 127 = 2,541 rows. This implementation uses a
rectangle padded to the existing 4,096-row tile boundaries, retaining the full
prefill's GEMM shapes. The first ragged rectangle had the correct dependency
coverage but changed floating-point kernel shapes. Padding restored bit-identical
logits and cache checkpoint hashes in the 8K and 49,555-token controls. The
49,555-token recall check also returned the hidden key correctly.

The automatic read policy uses `tail` for these longer cone prefills. Local
snapshots retain the replay setting in their provenance. Paired export of this
new mode is refused until a corresponding receiver state version is qualified.
The existing `yoco_keep=140` setting remains the approximate alternative.

## More chunks per weight pass

`producer_chunks_per_pass` accepts `1`, `2`, `3`, `4`, or `"auto"`; the launcher
also exposes `--chunks-per-pass`. It groups the existing 49,152-token units into
one layer-major pass, preserving increasing causal row order within each layer.
Only the final logical pass runs the decoder layers under replay. The 61,440-row
legacy fold limit becomes N × 49,152 + 12,288 for an N-unit group.

The graph still works on at most 4,096 rows at once. Its device hidden bank keeps
a resident prefix and one 4,096-row spill slot. Remaining hidden rows live in the
existing admitted RAM bank. The planner measures available physical RAM, keeps
at least 512 MiB free, respects the enclosing process budget and fails explicit
counts that cannot fit. `auto` chooses a fitting count. Paging guards remain on.

The canonical host carry uses VirtualLocked RAM with the current CUDA tile
registered for copies. It stores four F32 hyper-connection streams, previous-pre
and IDs: 92,176 logical bytes per row, or 92,288 physical bytes per row after
32-row page alignment, plus a 4,096-byte header. The prompt Engram allocation is
another 12,864 bytes per token plus 4 MiB. Device spill traffic is 40,976 bytes
per row in each direction, plus required H2D IDs. Shared KV stays with its four
device owners; DSpark capture uses its fixed 7.5 MiB ring when requested.

The measured BF16, growable 1M-context plan admitted these device ceilings:

| Allocation | GiB |
|---|---:|
| Weight provider, including the two streaming banks | 16.001 |
| Graph workspace, including 0.75 GiB backend temporary allowance | 6.992 |
| Context cache, including compressed state and private raw windows | 3.004 |
| Hidden bank and IDs | 2.348 |
| Weight casts | 0.438 |
| Indexer cuBLAS workspace | 0.063 |
| Pinned-copy device scratch | 0.063 |

Total admitted device bytes were 31,377,412,860, with another 1 GiB kept free
at prefill admission. These are allocation ceilings, rather than simultaneous
physical high-water measurements; the observed board peak at 49,555 tokens was
27,061 MiB. The decode expert cache separately keeps its 512 MiB reserve.

An in-engine probe copied the actual registered 4K carry lease while layer 1's
weight requests were in flight. GPU events measured 34–39 GB/s H2D and 50–55 GB/s
D2H. Its first H2D also paid host-side setup, so neither a 70 GB/s RAM estimate
nor isolated copy bandwidth is a substitute for measured swap time.

The current grouped path reuses the existing synchronized spill copies. A new
three-way asynchronous carry pipeline is not claimed. Prefill banks also replace
decode's resident dense weights and L1/L2 expert cache; this branch does not claim
additional residency across separate prefills. Retaining all 10.06 GB of
non-routed weights alongside the existing two-bank prefill allocation exceeds
the observed free VRAM. Any smaller residency experiment needs an explicit
device budget and a matched overlap measurement.

## Diagnostics and references

Set `LLAMADECODE_BYTE_LEDGER=1` for native per-tensor receipts, then run:

```sh
python -B scripts/analyze_dsv41_prefill.py stderr.log --model model.gguf --output ledger.json
```

The summary separates Engram row I/O, layer acquisition, router-driven fetch
wait and overlapped transfer duration. Optional CUDA node profiling separates
expert GEMMs, attention/indexer work and other GPU nodes; use separate diagnostic
runs because profiling changes timing.

[Machine-readable qualification](MULTICHUNK_QUALIFICATION.json) contains exact
bytes, per-pass walls, peak memory, input and executable digests, and numerical
comparisons. It also preserves separate instrumented GPU-stage and carry-copy
measurements.

The [reference engine overview](https://github.com/SL-x-TnT/DSV4.1-Flash-RTX-Pro-6000-Overview)
describes the dependency cone, selected-expert streaming and native W4A8 kernels.
Its private fork is not imported here. The upstream
[b12x implementation](https://github.com/local-inference-lab/b12x) is Apache-2.0
and now lives in FlashInfer. Its stock epilogues need a numerical gate before
porting: the reference changes the activation rounding and route-weight placement.
[TileLang](https://github.com/tile-ai/tilelang/blob/main/LICENSE) and DeepSeek's
[pinned official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE)
are MIT licensed. W4A8 and TileLang sparse
attention remain kernel-comparison candidates; this change imports neither.
