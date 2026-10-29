# Matched 100K after KERNELS2 defaults

All FP4 rows are **Q4 to FP4 requant (lossy), speed proxy for native FP4**. FP8 rows additionally use a **Q8_0 to FP8 lossy dense proxy**. Both remain opt-in. Main `f5d35f1` is merged into agent/native-fp4 as `6fe2f0e`. All producer C++ units, allocator, CUDA dispatcher and token-tile unit were rebuilt; other qualified CUDA objects, static libraries and the decode DLL were reused. The embedded TileLang path is observed in every accepted trace. No full clean CUDA-library rebuild is claimed.

The fixed 100000-token prompt, bias 0, exact cone 2541 and context 131072 are matched. The primary comparison uses 49152/50848 sweeps. Prompt-only time starts at prompt execution and ends at frontier logits; TTFT includes launch/model open/allocation. Separate synchronized profiles are diagnostic. Every throughput below uses prompt-only seconds.

|Arm|Prompt s|Prompt tok/s|TTFT s|Fraction of 10K|
|---|---:|---:|---:|---:|
|q4|46.941|2130.35|72.281|21.30%|
|q4-repeat|41.552|2406.61|44.922|24.07%|
|native-4k|37.695|2652.84|41.265|26.53%|
|native-4k-repeat|36.261|2757.82|39.547|27.58%|
|native-4k-dense-control|36.992|2703.26|40.860|27.03%|
|native-4k-dense-control-repeat|35.569|2811.44|38.797|28.11%|
|native-4k-fp8|34.694|2882.37|38.016|28.82%|
|native-4k-fp8-repeat|37.067|2697.80|40.282|26.98%|
|native-4k-candidate|38.538|2594.83|42.093|25.95%|
|native-4k-candidate-repeat|34.934|2862.52|38.313|28.63%|
|native-16k-control|44.260|2259.36|47.391|22.59%|
|native-16k-candidate|43.786|2283.81|47.312|22.84%|

The initial Q4 launch has an unusually long startup; both its prompt-only and full TTFT clocks are retained. Repeated results are reported separately. These samples do not establish a small FP8 speed gain. BF16 dense-store controls, where present, isolate that companion setting. The 10K target remains unmet.

|Profile|Prompt s|Stream wait|Expert|Attention|Indexer|Other|Expert TFLOPS|
|---|---:|---:|---:|---:|---:|---:|---:|
|q4-profile|44.616|5.653|9.666|3.205|2.902|23.191|97.10|
|native-4k-profile|37.320|5.755|2.946|3.205|2.926|22.488|318.56|
|native-4k-fp8-profile|38.206|6.169|2.944|3.269|2.924|22.900|318.78|
|native-16k-candidate-profile|46.138|5.521|2.757|3.183|2.882|31.794|340.43|

Expert for the packed path includes route-map construction, gather/quantization, all three GEMMs, SwiGLU and ordered combine. Q4 Expert covers the routed products. Useful expert FLOPs are independently derived from completed graph shapes. These mixed MXFP8-by-MXFP4 rates are not utilization of the nominal pure-FP4 roof. Dense is a subset of Other and is never added twice.

Packed two-sweep profiles fetch 456303081592 bytes across 61 layer passes. Engram tables are read on demand. At 25 GB/s those bytes alone take18.25seconds; the10-second target needs fewer passes/bytes plus less remaining work. 16K saves expert time but increases RAM carry transfers. Separate attention/FFN graph lifetimes and sparse packed expert extents remain needed for a full 64K model batch.

## Scale-index follow-up

The candidate computes the padded scale-row index once per route and reuses the existing late inverse-map allocation. Gather and SwiGLU/down-input packing read that map; ordered same-stream work replaces it with inverse routes only after down GEMM. Weight layout, arithmetic, workspace size and ABI remain unchanged. 4K/16K full outputs are checked against the separately built qualified ABI4 adapter. 64K full outputs are checked against the retained pre-optimization ABI1 bridge, which is a differential reference, not an independent scalar 64K oracle. Balanced and sparse/empty-expert cases are distinguished in the JSON.

|Balanced resident component|Control median ms|Candidate median ms|Candidate TFLOPS|
|---|---:|---:|---:|
|4096|6.449|6.336|274.55|
|16384|19.524|19.112|364.05|
|65536|78.015|75.298|369.62|

These are three alternating control/candidate process repetitions, five CUDA-event calls per process, balanced synthetic routing and preloaded weights. They include activation packing and combine, exclude CPU routing-map construction and disk streaming, and are not captured 100K routing. Full-model follow-up rows above use the candidate only when explicitly named.

Retain the bit-exact scale-index optimization in the fused bridge: median component reductions are 1.76%/2.11%/3.48% at 4K/16K/64K. The small component gain does not establish a full-prompt speedup. 4K remains the model chunk choice; neither lossy FP4 nor FP8 becomes a production default. 16K control/candidate frontier vectors match every float bit, as do the4K candidate/control vectors. The candidate 16K profile still spends far more time uploading/reading back RAM carry than 4K. 64K graph lifetime separation remains unfinished.

## Correctness and resource limits

The raw manifests, binary/bridge/model/input identities, component checks and guarded cleanup receipts are bound in [combination-results.json](combination-results.json). Frontier comparisons cover one 129280-logit row only; small KL/top1 agreement does not establish broad quality. Repeated same-arm rows must remain bit-identical. Zero runtime weight repack/requant is witnessed by the existing before/after backend counters.

Model runs retain the 31377412860-byte device ceiling,512 MiB desktop reserve, explicit host caps, physical RAM floor 0.75 GiB and paging stops. Single-sweep trials use separately requested 11.5/11.75 GiB outer budgets; ordinary runs use 8.5 GiB. The single-sweep attempts refuse aggregate or physical-RAM admission before prompt execution. Refused attempts are retained and excluded from accepted timings. No production preset, original model, foreign branch or Altium tree changed. Existing dlwatch continues the sole source download. Native-source qualification and the original task remain unfinished.

Reproduce primary rows with `speed_proxy.py --prefill-only`, the pinned model receipt, ULTRA, cone 2541, context 131072, sweep 49152/fold 61440, the producer hash in the JSON, and explicit opt-in native-FFN/dense-FP8 DLLs. Add `--profile-stages` only in a separate run. For component checks build `native_ffn_check.cu`, then pass candidate DLL, qualified ABI4 DLL, source-row count, and retained ABI1 DLL; append `sparse` for six active experts and 378 empty experts. All engine/component commands require the shared GPU lease and host-memory guardian.
