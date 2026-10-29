# Direct-packed FP4 and FP8 dense experiments

The packed FP4 speed proxy reaches **2740.45 prompt tokens/s at100K**, compared
with2278.46 for original Q4. The10000 target is unmet. Q4-to-MXFP4 and
Q8-to-FP8 are explicitly lossy speed proxies and stay opt-in. Source-native
full-model conversion, timing and quality await the existing download.

[`packed-native-comparison.json`](packed-native-comparison.json) binds results
to model, producer, bridge, input and raw-receipt hashes. All headline runs use
the same fixed100000-token input, bias0, exact cone keep2541, context131072 and
49152/50848 sweeps. Prompt time excludes startup/decode; TTFT includes launch,
model open, allocation and frontier certification.

|Arm|Prompt seconds|Prompt tokens/s|TTFT seconds|10K target|
|---|---:|---:|---:|---:|
|Original Q4 experts/Q8 dense|43.889|2278.46|47.297|22.8%|
|Packed FP4 proxy,4K chunks|36.490|2740.45|39.797|27.4%|
|Packed FP4 proxy,16K chunks|48.443|2064.29|55.390|20.6%|
|Q4, BF16 dense-store control|43.308|2309.04|46.594|23.1%|
|Q4, Q8-to-FP8 dense proxy|43.821|2282.00|47.063|22.8%|

The first three share producer3e2d787e72409391491ca7c7f220e4f0a5df4e7174a755be44d92fe0c0c4fc77.
The dense pair shares a later producer containing dense tail/admission fixes,
whose hash is in the JSON. A single pair does not establish a small speed
difference: dense FP8 demonstrates no prompt gain here.4K remains the chunk
choice, with FP4 and FP8 toggles off by default.

## Layout, memory and remaining batching work

KERNELS2 selected the existing SM120 FlashInfer W4A8 family; its reference's
custom b12x fork is unpublished. No captured-routing three-way winner is
claimed. [`NATIVE_LAYOUT.md`](../../engines/streaming-llama/flashinfer-sm120/NATIVE_LAYOUT.md)
defines the exact adjacent-nibble E2M1/E8M0 layout and mandatory GGUF flag.
DirectStorage fills kernel banks directly; canonical/decode loaders refuse it.
The verified file is501609136128 bytes; its SHA256 is in the evidence JSON.

The fused FFN removes the384-expert repack arena and routed F32 tensors. One
sorted map serves three products per chunk; one gather also quantizes input.
Gate/up use BF16; weighted SwiGLU/F16 rounding fuses with down-input packing.
Actual before/after backend counters prove zero runtime weight requant/repack.
The4K/16K arenas are381702144/1526255616 bytes.16K fits by returning only
unreachable default million-context cache rows to graph storage, retaining
exact VMM-rounded owner charges at131072. Total device31377412860 bytes,
reserves,8.5GiB outer host cap,0.75GiB physical floor and paging stop remain.

Million-context16K graph admission and the current100K single-sweep host plan
still refuse. Failed launches contribute no accepted timing. Whole-layer
routing and separate64K attention/FFN lifetimes remain unimplemented.64K
component support does not qualify a64K full-model graph.

## Separate synchronized profiles

|Arm|Profile prompt s|Stream wait|Expert|Attention|Indexer|Other|
|---|---:|---:|---:|---:|---:|---:|
|Q4|47.064|4.690|9.547|4.671|2.934|25.222|
|Packed FP4,4K|40.280|5.623|2.925|4.681|2.925|24.126|
|Packed FP4,16K|49.777|5.228|2.815|4.662|2.908|34.164|
|Q4 dense control|46.401|4.826|9.557|4.677|2.914|24.428|
|Q4 FP8 dense proxy|47.538|5.235|9.581|4.676|2.913|25.134|

Profiles synchronize nodes and are diagnostic. Packed Expert includes sorting,
gather/quantization, three products, nonlinearities and combine; Q4 Expert
covers routed products. Independent useful expert work938548333117440 FLOPs
gives98.31/320.87/333.40 TFLOPS for Q4/packed4K/16K. Mixed MXFP8-by-MXFP4
fused throughput is not utilization of the nominal1676TFLOPS dense FP4 roof.
The16K upload/readback sums grow to8.010/5.710 seconds from4K's3.164/1.135.

Packed streams456303081592 bytes across61 passes; Q4 streams482208151672.
Engram tables are demand-read. Packed currently requires full-stage expert
reads, including cone layers, without layout-aware sparse expert extents.
Even25GB/s gives an18.25-second bound for its two-sweep bytes.10K requires
fewer passes/bytes and less remaining graph/host work. Later KERNELS2 attention,
fusion and host improvements are separate from the merged ccd3100 baseline.

## Dense FP8 and correctness limits

[`DENSE_FP8.md`](../../engines/streaming-llama/flashinfer-sm120/DENSE_FP8.md)
documents the pinned BSD/Apache SM120 kernel, seven shapes, native E4M3/E8M0
32x32 encoding, converter and refusing loaders. Native operand values are
preserved; the Q8 proxy includes GPU weight requantization in timings. Private
zero padding handles2541-row cones. Remaining converter tensors retain the
reference recipe; this is not a fully native checkpoint.

Both dense formats pass28 synthetic sampled double/BF16 and full-output
finiteness/written checks at1/128/2541/4096 rows. All seven1-row cases and the
large2541-row case pass memcheck with zero errors. Instrumented timings are
not performance.3941 selected dense products take2.282s for Q8 versus2.864s
for the FP8 proxy, including packing. This subset is already included in
Other; never add it again. FP8 remains off; no native-source model claim.

Packed4K/16K match all129280 untouched-frontier float bit patterns. Against
Q4 this one row has KL1.4170e-8, same top1, maximum absolute difference2.96322.
Dense FP8 versus its control has KL1.5389e-9, same top1, maximum difference
2.75762. These narrow probes establish no broad quality, recall, decode or
lossless-proxy claim.

The4K/16K synthetic FFN oracle matches20971520/83886080 output bit patterns
against separate qualified ABI4;16K memcheck has zero errors.64K checks
finiteness/writes without an independent reference. Warm balanced component
times6.424/19.793/77.694ms give270.77/351.54/358.22 usefulTFLOPS for4K/16K/64K.
They exclude map construction and streaming and are not captured100K routing.

Reproduce with `speed_proxy.py --prefill-only`, the validated receipt, explicit
native bridge and documented toggles. An optional one-token forced probe saves
only the untouched frontier. Use fresh output paths and a separate
`--profile-stages` run. Decode, bias changes and long continuations are refused
in this mode. No production preset changes.
