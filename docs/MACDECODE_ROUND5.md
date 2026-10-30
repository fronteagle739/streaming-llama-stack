# Mac decode Round 5

Upstream remains `f48c5e2e20b1aaae1c79735a8457225d2469ae73`.
The existing target and drafter are owner-selected lossy Q4-to-MXFP4
speed proxies. No model was downloaded. Vision and drafter remain loaded;
context allocation is393216 and decode restores RAM fixtures. Normal timing
uses128 greedy tokens after warmup, two samples in reverse arm order.
Owned anonymous wired core/KV storage and400GiB/16GiB memory guards remain.

| Workload, tok/s | Upstream, same proxy (Round4) | Round5 index0 | index1 | index2 |
|---|---:|---:|---:|---:|
| Serial prose8K |38.559|36.679|36.799|36.875|
| Serial prose131K |37.753|35.623|35.545|35.673|
| Serial code8K |38.412|36.531|36.674|36.688|
| DSpark prose8K |37.696|35.972|36.194|36.155|
| DSpark code8K |53.113|50.755|50.886|50.653|

The separate query-store sweep measures serial code37.101 without and37.468
with query-store RoPE at index1. DSpark code measures52.183/51.897; its
verify timing varies between sessions even with the same selected levers,
accepted tokens and cycle counts. These are comparative engine measurements,
not a final Studio/paired release or a published-native-weight40-60 claim.
The prose continuation differs from upstream after seven tokens; the code
continuation matches all128. Prefix-token hashes agree.

`DS4_DS41_INDEX_SCHEDULE` retains the original schedule at0. Mode1 folds
all-eligible block selection to its zero mask and avoids masked planning
where no block can be excluded. Mode2 adds upstream scalar wide/split BF16
key scoring; checked nonexact restored keys keep complete F32 support.
The original F32 cache is authoritative and all state information is retained.
Upstream wide sums have a different order, so mode2 is explicit even though
all actual-model proofs in this campaign match.

`DS4_DS41_QUERY_ROPE=1` adapts upstream QKV_FUSION's query-store rotation to
the retained exact HC/KV producer. It preserves every query, HC mix/split,
KV and ring word and the original standalone rotation. Custom geometry,
quality, streaming, TP, ordinary batches and all existing schedules remain.
Both controls default to the existing path until preset qualification.

The later complete-FFN schedule adds `concurrent_ffn=2`: independent HC
mix/split work runs in the same concurrent encoder as the shared/routed
products. Modes0/1 retain their original behavior and environment spelling.
Its56-request sweep passes eight full-vocabulary exact proofs; serial prose
measures37.189/37.387 with the original/new schedule, code37.259/37.222,
and DSpark code52.413/52.461. It is a modest scheduling gain, not a uniform
speedup across workloads.

`topk_short=1` imports upstream shuffle block sorting and staged/fused
prefix merging for scalar top512 below16384 keys. Generic/causal/block
sorting and the long radix path stay supported, as do the existing shuffle
and prefix controls. `hc_join_static=1` imports literal5120x4 HC join
dimensions while retaining every block/HC output and the ordered FMA tree.
Both default to0. The84-request sweep includes12 full128x129280-logit
proofs: all bit-identical, forced-continuation KL0. Both controls on with
HC geometry256 measure37.787 prose8K,37.834 code8K,37.117 DSpark prose
and51.757 DSpark code. Corresponding controls-off means are37.117,
37.168,36.498 and51.548. Geometry32/64 alternatives remain available.

Actual Metal coverage includes every selected index at512/513/1023/1024/
1025/4095/8192/8193/16383/16384 keys, tied values, signed zero, infinities,
all three short-sort dispatch witnesses, retained long/block/causal paths,
and every HC/block word with/without addition and in-place residuals.
All43 isolated compatibility/environment/build/guard regressions and the
native crossover fixture pass after these ports. The final stable binary is
`bf52468d1872ffadef9944bb59edab20ecd2af0d12a8468d8d705044eb1ab0cf`.
Peak RSS356185260032 bytes and minimum available182006874112 bytes;
no new swap, host-cap/floor or wired-ceiling violation.

A matched diagnostic with pre-encoding off on both engines captures all40
layers. Before the two small-kernel ports, scalar encoder kernels sum to
26.152ms/token versus25.299 upstream. The concurrent FFN kernels are
9.625/9.601; HC join0.510/0.258; short sort plus merge0.556/0.295.
These instrumented counter sums describe attribution, not normal throughput
or non-overlapping GPU busy time. Production pre-encoding remains supported.

The index campaign completes51 requests and six full128x129280-logit
proofs at8K/131K; all hash-identical. Wide-mode forced-continuation KL is0.
The query campaign completes42 requests and six full code proofs; all
hash-identical with forced-continuation KL0. Actual Metal tests cover all
index thresholds/tails, signed zero, all four dispatched score variants,
retained short scores, every query/HC/KV output at0/8K/131K/1048575 in both
frequency modes, bounds, null refusal and control ranges. Reviewed sources
also pass43 isolated compatibility/environment/build/guard regressions and
the native crossover fixture. No new swap, physical-floor, host-cap or
wired-ceiling violation was observed. Evidence includes binary/source hashes,
individual timings, complete applied levers and guard observations in
[the measurement index](macdecode-r5/measurements.json).

Capability reductions: none. The existing lossy speed-proxy weights remain
labelled; optional matrix/wide reductions retain their original alternatives.
Final matched control/profiling, Studio presets, optimized FP4 paired proof,
release gate, fleet deployment, GUI proof, MiMo restoration and retirement
remain pending. The paired proof waits for the RAM-tier agent.
