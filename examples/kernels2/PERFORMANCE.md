# E19: measured imports, defaults and remaining gap

2026-10-29; RTX5090 SM120,32GiB VRAM,33.7GiB physical host memory. Base
`ccd310061ac702b9642942df9180f53767da3034`; branch `agent/kernels2`.
Original Q4 model bytes and placement are unchanged. Prompt-only timing spans
embeddings through the required frontier head/sync; TTFT also includes producer
initialization and the first returned token, and excludes GPU-lease queue time.
Process open, weight preload and JIT are excluded from component timers.

The final embedded-image build runs the real100K retrieval prompt in38.368s
(2606.32tok/s), TTFT42.000s. Its direct answer-probe run takes37.901s
(2638.43tok/s), TTFT41.719s. This is26.4% of the10,000tok/s target; the prompt
budget would have to fall to10.000s, another27.901s below that measured run.
The answer-probe control TTFT48.813s has extra startup latency; its prompt time
39.405s remains comparable. Do not attribute all of that TTFT difference to kernels.

## E19: official layer and same-resident component comparison

Pinned official inference revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT license, unmodified source
hash manifest in `upstream/`. The real layer0 source shard was fully SHA256
verified (`41611994fc9b27829b757d72257570d65a976fe65b8c6df8641adcd139d65ded`).
It supplies2333 parameters,7423022552 bytes preloaded into VRAM. Input is
reproducible synthetic BF16 residual, seed19, rather than captured model state.
Layer0 has neither an indexer nor Engram; their component tests are separate.

| Rows | Official whole layer (ms) | Norm/HC | Dense | Attention | Router | Experts | Shared | Other wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2541 | 213.684 | 7.504 | 3.250 | 1.137 | 0.334 | 128.232 | 1.506 | 71.721 |
| 65536 | 966.466 | 208.207 | 79.618 | 30.922 | 7.513 | 433.725 | 37.494 | 168.988 |

Leaf CUDA event scopes are disjoint; other wall includes launch gaps and
unclassified work. The64K run peaks at24.034GB Torch allocated. Two explicit
adapters are16-head attention groups and1024-row HC-post tiles, avoiding a
21.5GB broadcast temporary. The HC adapter is bit-exact at2541 rows. A94%
Torch ceiling admits64K while leaving GPU headroom. Smaller90% ceilings refused
allocation, and those attempts remain failed evidence. Windows expandable
segments were unsupported. The earlier2541 observation inside an overall failed
two-size job is superseded by the clean single-size run above.

| Same resident operands | Rows | Official (ms) | Native (ms) | Scope / decision |
|---|---:|---:|---:|---|
| Routed MoE + shared |2541|215.209|19.531|Same original FP4 bytes; existing FlashInfer ABI4 wins|
| Routed MoE + shared |65536|604.666|256.318|Native16K chunks; not one64K call|
| RMS norm |2541|0.178|0.116|Keep native|
| RMS norm |65536|8.756|3.540|Keep native|
| HC split/Sinkhorn |2541|0.027|0.390|Import fused arithmetic, qualify native rounding|
| HC split/Sinkhorn |65536|0.086|8.161|Whole official batch versus sum of native4K tiles|
| BF16 dense GEMM |2541|0.242|0.273|Same BF16 weights; excludes FP8 conversion|
| BF16 dense GEMM |65536|3.754|5.775|Do not infer full-model FP8 speed from this|
| FP4-rounded index score |2541|66.273|36.424|Official128-row score-table tiles; keep native|
| FP4-rounded index score |65536|549.439|294.068|Sum of independent microtiles; keep native|
| Sparse attention |2541|3.870|5.891|Official BF16 kernel motivates adapter repair|
| Sparse attention |65536|97.086|149.266|Sum of independently timed4K query tiles|

The index fixture now applies the same1/64 score factor on both sides. The
original omitted it in the official comparator and is rejected for numerical
comparison; raw artifact digest is retained. Corrected relative L2 is about
0.00183 because official BF16 reductions and native F32 head reduction differ.
Historical attention metadata said stages1, while the pinned official source
used2; published metadata is corrected without changing measured times.

The MoE call includes router, gather, sort, quantization, scatter, clamps,
route weights, shared expert and expert-id-ordered combine. Compilation and
weight packing precede timing. Relative L2 is4.93e-5/5.26e-5, max abs
0.03125/0.125 at2541/65536 rows; this is not a model-logit qualification.
ABI4 workspace is3646768128 bytes;64K peak Torch allocation is27.551GB.

**Expert-layout coordination:** retain the FlashInfer W4A8 family. ABI4 accepts
`[expert,N,K/32,17]`: byte0 E8M0 scale, followed by16 packed bytes with codes
0..15 in low nibbles and16..31 in high nibbles. Gate/up have
`E=384,N=2304,K=5120`; down has `E=384,N=5120,K=2304`. It repacks internally
to adjacent-nibble E2M1 weights and `[expert,K/128,N,4]` E8M0 scales. NATIVEFP4's
new direct-packed ABI should own the next on-disk layout. No competing model
file or modification of NATIVEFP4-owned adapter code was made.

The requested three-way comparison on captured100K routing is **not complete**.
The reference repository's pinned tree contains only a README; its custom b12x
epilogues and dense kernels are described, not published with reusable source.
Q4/HYPER does not consume the same original FP4 bytes, so its existing full-model
time cannot honestly establish a same-weight winner. DeepInfer is separately
Apache-2.0, Linux/SM100/Ascend950 oriented; its SM100-only kernels were skipped.
The official complete layer is not a demonstrated faster replacement for our
streaming layer. Selected C++ imports keep the loader, banks, cone/state export,
sampler and decode handoff intact; no complete PyTorch layer is inserted.

## 1. Unpadded cone: measured, still opt-in

The optional path executes2541 logical replay rows instead of5792
(4096+1696) for layers21..39. It saves1.428s/1.722s in forward/reverse pairs,
but shape-dependent floating-point changes fail the strict E4 bit-exact condition.
Default remains0. Logical dependency coverage passes the C++ contract oracle.
Weather full-vocabulary frontier plus16 forced rows: mean forced KL1.873e-5,
max0.000141916, all top-1 and natural continuation tokens match. That weather
fixture is not a retrieval test; no actual-recall qualification is claimed here.

| Experiment | Control prompt / TTFT (s) | Candidate prompt / TTFT (s) |
|---|---:|---:|
| E19-unpadded | 39.341 / 42.875 | 37.914 / 41.454 |
| E19-unpadded-reverse | 39.346 / 42.938 | 37.624 / 41.218 |

| Arm | Experts | Attention | Indexer | Dense | Norm/HC | Other GPU | Host prepare | Node invocations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 9.582 | 4.713 | 2.195 | 3.692 | 2.373 | 5.338 | 1.991 | 75532 |
| candidate | 9.311 | 4.573 | 2.060 | 3.625 | 2.285 | 5.175 | 1.938 | 72997 |

Stage diagnostics use synchronized CUDA events and are slower than the clean
timing pairs. Event sums are not GPU kernel union and cannot be added to host
wall clocks. There are61 executed layer passes across49152/50848-token sweeps.

## 2. TileLang attention: default2, off toggle0, accuracy caveat

The repaired adapter stages the full BF16 query tile and raw+compressed KV once,
builds IDs once, then launches one `(queries,4)` grid:16 heads/group, d512,
block64,256 threads, one pipeline stage,86016 dynamic shared bytes, sm_120,
TileLang0.1.15, Torch2.11.0+cu128, CUDA13.2, C++20/O3, no fast-math flag.
The legacy128-query/head loops repeatedly copied query/output and relaunched.
No compile/JIT, Python runner, per-call transpose or output-copy loop remains
in the producer. Staging plus existing F16 mirrors stays below640MiB inside
the already admitted768MiB pool. Contexts beyond the bounded compressed-row
contract continue through native attention.

The AOT image is embedded and pinned by SHA256
`30bbd5acf05fe2b46dd8655d71e6fc6a92f2e37aed5fc7ff910138baa2514181`.
`LLAMA_DSV41_TILELANG_ATTENTION=0` disables it;1 keeps the legacy experiment
and requires its legacy image. Mode2 can take an explicit matching CUBIN
override but needs no external file by default. Default applies both through
the launcher table and when the C++ producer sees an unset variable.

| Experiment | Control prompt / TTFT (s) | Candidate prompt / TTFT (s) |
|---|---:|---:|
| E19-attention-v3 | 39.333 / 42.968 | 37.934 / 41.500 |
| E19-attention-reverse | 39.459 / 43.266 | 38.498 / 42.110 |
| E19-answer-probe | 39.405 / 48.813 | 37.901 / 41.719 |
| E19-default-retrieval | 39.778 / 43.563 | 38.368 / 42.000 |

| Arm | Experts | Attention | Indexer | Dense | Norm/HC | Other GPU | Host prepare | Node invocations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 9.520 | 4.690 | 2.184 | 3.677 | 2.374 | 5.327 | 2.006 | 75532 |
| candidate | 9.524 | 3.202 | 2.193 | 3.682 | 2.354 | 5.335 | 2.032 | 75532 |

Weather frontier+16 forced rows: max KL0.001882246, mean0.000207541,
top-1 agreement100%, continuation identical. Those continuations were initially
mislabeled recall; the timing seed only contains weather text. The correction
is explicit in results.json. A new100K chat prompt plants the archive key near
the beginning and asks for it at the end. Both native and TileLang return
exactly `QUARTZ-7429-LANTERN` before EOS. The direct teacher-forced10-token
answer+EOS probe, starting at the prompt frontier, has max KL1.54858e-6,
mean2.32314e-7, top-1 agreement100%. This and the owner instruction qualify
default2 for these fixtures. The final embedded-image natural-answer run passes.

**Retained negative result:** unrestricted post-EOS generations diverge. Even
with EOS stopping and identical histories, forcing the final question/chat
markers after the completed answer gives max KL1.41912 and top-1 agreement
87.5%. Without EOS stopping it gives max KL15.2308 on different histories,
which is not a valid same-history quality comparison. These are not hidden as
passes. Attention is not bit-exact, and the available fixtures do not establish
a universal KL bound. "Make it exact" is on ROADMAP.md. The default is the
owner-requested tolerance choice, with its off toggle retained.

New packed attention component outputs match the pinned official BF16 kernel.
Memcheck reports0 errors. Racecheck restricted to the new AOT entry reports
0 hazards/errors/warnings. The unfiltered racecheck includes existing native
baseline warp-WAR reports and hits its3GiB host cap; it is not a clean whole
backend racecheck. The guard stops it at3.343GB with12.36GB still available;
6GiB scoped retry peaks at2.093GB. Failed missing-image attempts remain recorded.

## 3. Fewer kernels: native-parity HC default2; graph replay/PDL pending

Imported DeepSeek Sinkhorn specialization: hc4,20 iterations, epsilon1e-6,
one64-thread kernel and one allocation for pre/post/comb. Other HC settings
keep the original graph. Mode2 preserves native separate affine rounding,
fast exponential/division and softmax reciprocal-multiply order. It is exact
against original native outputs on rows1,3,7,8,9,127,128,129,2541,4096 and
tested100K frontier+16 forced logits. The real retrieval answer also matches.
Mode1's official rounding misses tolerance at2K (max KL0.01023); it is rejected
as a default. Mode0 selects the original graph.

| Experiment | Control prompt / TTFT (s) | Candidate prompt / TTFT (s) |
|---|---:|---:|
| E19-hc-parity | 41.036 / 48.000 | 39.891 / 43.703 |
| E19-hc-parity-reverse | 42.833 / 46.438 | 40.037 / 43.640 |
| E19-recall-hc | 40.282 / 43.937 | 39.288 / 42.937 |

| Arm | Experts | Attention | Indexer | Dense | Norm/HC | Other GPU | Host prepare | Node invocations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 9.559 | 4.755 | 2.192 | 3.715 | 2.384 | 5.325 | 2.769 | 90170 |
| candidate | 9.586 | 4.714 | 2.181 | 3.721 | 2.355 | 5.333 | 1.936 | 75532 |

The profile removes14638 measured node invocations (16.2%) and862 distinct
node-profile keys. These counts cover the profiled two-sweep execution, not
the earlier Nsight trace's unique graph-node count. Existing ggml fusion
already reduces device HC work; much of this saving is graph/allocation work.
No whole layer-sweep graph capture/replay or programmatic dependent launch was
implemented. Current allocator-only bank/cast/routing guards disallow reused
layer graphs, and async inputs require CUDA graphs disabled. They remain intact.

## 4. Host work: less graph metadata; no new overlap worker

The lightweight per-tile timeline, exported as layer histograms and normalized
fetch/dispatch/host event CSVs, shows:

| Host/stream scope | HC control (s) | Fused HC (s) |
|---|---:|---:|
| Clean prompt wall |40.520|39.564|
| Prepare phase |2.700|1.996|
| Host graph build |0.274|0.171|
| Host reserve/plan |1.138|0.482|
| Registration |0.663|0.687|
| Upload phase |2.989|3.202|
| Host launch |5.157|5.015|
| Host sync envelope |26.924|26.551|
| CUDA stream elapsed |32.028|31.515|

These scopes overlap or nest; host sync is not idle CPU work to subtract, and
CUDA stream elapsed includes waits. The host metadata reduction is measured;
there is no new prepare-next-layer, staging or bookkeeping overlap pipeline.
The prior11.5s outside kernel union cannot be claimed recovered.

## Ranked next work, before further integration

The pinned reference tree has no custom kernel sources. None of these estimates
is a measured integration win. The planning column assumes a50% reduction of
the current event total, with complete elimination as the upper bound; the
numbers are not additive to wall time or guaranteed overlap savings.

| Rank | Candidate | Current event cost (s) | 50% planning saving (s) | Absolute event upper bound (s) |
|---:|---|---:|---:|---:|
|1|Dense/projection FP8 kernels, owned by NATIVEFP4|3.820|1.910|3.820|
|2|Remaining norm/HC expand/reduce fusion|2.384|1.192|2.384|
|3|CONCAT + copy elimination|1.645|0.823|1.645|
|4|QAT/round-trip fusion|1.256|0.628|1.256|
|5|TOP_K selection fusion|0.810|0.405|0.810|

The baseline operator grouping differs slightly from the disjoint stage grouping
(dense includes HC projection). HC and dense estimates therefore must not be
summed. Original coneall's2.2s QAT/1.6s TOP_K/3s copy estimates are not silently
substituted for these fresh event totals. The official norm/index kernels are
already slower here; no unsupported projection/TOP_K/copy swap was attempted.
Single-pass100K and large expert chunks still require NATIVEFP4's VRAM/lifetime
restructuring and completed original native-FP4 source/model qualification.

## Validation, rejected attempts and resource ownership

612 Python tests pass,12 skipped. C++ HC contract and exact-cone structural
fixtures pass. Native custom-kernel component tests, full-vocabulary paired
probes, memcheck and scoped racecheck are above. Fresh CPU units, CUDA dispatcher,
token-tile unit and allocator were compiled; unchanged backend objects/libs and
the decode DLL were reused with digests in results.json. This is not a fresh
full-CUDA rebuild or a qualified native-FP4 full-model result.

Every engine/component used exclusive shared-ledger GPU ownership, explicit
aggregate host RSS/Windows commit budgets, physical-memory admission and runtime
floors,512MiB desktop reserve and paging stops. Typical engine RSS cap8.5GiB,
native host plan7.75GiB, runtime floor0.5GiB; component caps3–5GiB. The unfiltered
sanitizer cap refusal was retained rather than raising the cap during that run.
Windows kernel ETW was refused. Bounded owned-page scans observed no pages;
small global pagefile growth6.95–20.01MB appeared on three historical receipts
and is not attributed to this task. Zero observations do not prove zero I/O.

All149 receipts collected at the evidence snapshot confirm owned cleanup. The
list includes34 failed/rejected/synthetic-test attempts: initial build/setup
errors, explicit allocation ceilings, false fixture labels, missing-image and
environment-contract refusals, component-buffer repair, diagnostic cap refusal,
bootstrap/receipt races, and the negative quality probes. Successful reruns
supersede them; no failed timing is represented as accepted. Unit-test shutdown
error messages are intentional mocked fixtures. Publication/cleanup checks and
final receipt count are recorded separately in the task RESULT.

STATUS: PARTIAL — selected imports measured and defaulted;10Ktok/s, captured-routing three-way expert comparison, strict bit-exact unpadding, full graph replay/PDL and host overlap remain incomplete.
