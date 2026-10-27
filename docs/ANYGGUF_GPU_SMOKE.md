# Any-GGUF Q4/Q8 GPU regression smoke

2026-10-27; tested source `5ec948606934f425bf8e4271657a61e7d219d661` on one RTX 5090. **PASS for the requested smoke**, using the full warm decode interval after 100 tokens. No main branch was changed.

The actual stack WebUI ran in headless Chromium. Single engine: streaming-llama ULTRA prefill, YOCO 140, ds4-decode.dll, context 1,000,001 with growable BF16 cache, 49,152 primary sweep/61,440 fold cap, eight DirectStorage split reads, and the existing Q4_K/Q8_0 GGUF. No model download or conversion occurred. The complete producer source was rebuilt with the reference native MSVC recipe against its unchanged, hash-recorded GGML libraries. The changed ds4 external-decode DLL was freshly built with windows_build.ps1; all nine CUDA images target sm_120a. Native profile preflight passed.

| Check | Observation | Reference |
|---|---:|---:|
| Fresh 49,555-token recall prefill | 31.125 s; **1,592.13 tok/s** | ~1,617 tok/s; 1.54% lower |
| Warm Max quality, 899 steps after 100 | **6.8118 tok/s** | 6.0 tok/s; faster |
| Warm Max speed, 899 steps after 100 | **18.1550 tok/s** | 19-21 tok/s; 4.45% below lower figure |

Recall returned **AMBER OTTER 731** with a coherent explanation. The follow-up correctly returned 731 and reused 49,624 tokens. Both tests also passed on the initial run. All four original WebUI turns completed; each long decode generated 1,000 tokens. Native sampler evidence confirms bias 0/1.36125, temperature .6, top_p .95, top_k 40, min_p .05. The screenshots show the actual dial endpoint, numeric rates and context use. Captured SSE confirms incremental text; the visible prefill bar advanced.

The initial 49,553-token cold run was slower: 39.312 s/**1,260.51 tok/s**. Native opening/allocation took 9.585 s versus 1.985 s on the repeat; prompt compute was 28.053 s versus 27.531 s. The earlier qualified image run's compute was 27.833 s. The extra cold opening cost did not persist; its exact system cause was not established. Both observations are retained. Max speed's shorter 100-500 window was 17.8530 tok/s, while the entire post-100 interval was 18.1550; quality's 100-500 window was 6.8106. This is a bounded smoke comparison across different prompt/content lengths, not an exact replication or statistical performance qualification.

The host ceiling was 8.5 GiB, with an independent 4 GiB available-RAM floor. Recorded aggregate peak RSS was 6,380,183,552 B; minimum available RAM was 8,162,062,336 B. Both native cleanup receipts show zero budget breaches, zero observed pagefile growth, graceful shutdown and confirmed process removal. Observed paging growth is not exhaustive paging-I/O attribution. Studio was gracefully paused, restored by its canonical launcher, and verified healthy with Remote Link serving and end-to-end OK. The other WebUIs and Mac engine were left running as found.

IQ2_XXS/Q2_K: **CPU-verified only**. MXFP4: **CPU-verified only**. This smoke makes no GPU claim for those unavailable quants.

[Machine-readable observations and binary/library hashes](../examples/anygguf-gpu-smoke/summary.json). [Prefill progress](../examples/anygguf-gpu-smoke/prefill-progress.png), [recall](../examples/anygguf-gpu-smoke/recall.png), [Max quality](../examples/anygguf-gpu-smoke/max-quality.png), [Max speed](../examples/anygguf-gpu-smoke/max-speed.png). Full raw SSE and native per-token logs remain in the recoverably recycled task artifacts recorded in the delegation result.
