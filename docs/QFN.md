# Qwen3.8 Flash Next streaming engine

The Windows CUDA engine streams expert weights from NVMe through DirectStorage
while keeping the dense core in VRAM. Its native packed FP4 prefill and decode
paths share expert bytes, retain persistent scratch, and reuse CUDA graphs.
It includes the tokenizer and chat-template path, prefix reuse, reasoning and
tool-call parsing, cancellation, and the complete sampler: temperature, top-k,
top-p, min-p, typical sampling, repetition/presence/frequency penalties, seed,
stop strings, logit bias and token limits. MTP and the host-RAM expert tier remain
available; the qualified defaults keep both off because their measured arms did
not improve throughput. DFlash requires compatible assistant weights and a CUDA
implementation; it is not claimed delivered.

## Build and run

Use Windows, an SM120 CUDA GPU, CUDA 13 or newer, a compatible MSVC toolchain,
the DirectStorage SDK, CMake, and the Python dependencies already declared by
the stack. Configure `engines/streaming-llama/prefill_llama.cpp` with
`LLAMA_QFN_ENGINE=ON` and `GGML_CUDA=ON`, then build `qfn-engine`. If needed,
set `QFN_JSON_INCLUDE_DIR`, `QFN_DSTORAGE_INCLUDE_DIR`, and
`QFN_DSTORAGE_LIBRARY` to the installed dependencies. The stable output is
`bin/qfn-engine.exe`; its companion bridge DLL remains adjacent.

Run CMake from an x64 Native Tools console. Set `DS_SDK_DIR` to the complete
DirectStorage SDK package root, including headers, import libraries and runtime
DLLs. Use the official `Microsoft.Direct3D.DirectStorage` NuGet package pinned
to **1.3.0**; see the [installation command](../README.md#install-build-and-run).
The QFN include/library defaults and the shared llama target use that
same SDK root. Headers alone are insufficient. For example, under the RAM guard:

```text
cmake -S engines/streaming-llama/prefill_llama.cpp -B BUILD_DIR -G Ninja -DLLAMA_QFN_ENGINE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DDS_SDK_DIR=SDK_ROOT
cmake --build BUILD_DIR --target qfn-engine --parallel 2
```

The native grouped kernel exporter under `engines/streaming-llama/flashinfer-sm120`
requires the pinned CUTLASS DSL 4.8.1 and the checked-in FlashInfer source closure.
Pass explicit workspace, DSL-site, AOT directory and CUTLASS runtime arguments
to its exporter and build script. Use the existing physical-memory guardian for
all compilation and engine runs. The CMake target includes its native tokenizer,
vision and packed-kernel dependencies. The incremental build helper separately
accepts an explicit prior Ninja build and native link recipe; it does not replace
the clean CMake build above.

Copy `examples/single-machine/qfn.example.json` to a local config, provide the
model pack, complete tokenizer GGUF, GPU identity and explicit physical-memory
budget, then run a named launcher with that config. All paths are supplied by
the user. Test listeners must bind only to `127.0.0.1`.

## Named modes and calibration

`qfn-max-quality.ps1`, `qfn-balanced.ps1`, and `qfn-max-speed.ps1` use
`qfn.settings.json`. Selected bias values are 0, 0, and 0.68; Balanced remains
the conservative unbiased choice until a local sweep qualifies another point.
The numeric bias capability retains its full range of 0 through 8. The settings
also record router spreads, cache seed, half-life, tile, eight-split reads, and
prefill/decode/indexer modes. DSV4.1 has its own separate settings and launcher.
See `examples/single-machine/CALIBRATION.md` for the calibration script, which
reuses the existing full-vocabulary gate and warm timing protocol and measures
KL, top-1 agreement, throughput, cache hits and natural generation stops.

See the [performance table](../README.md#performance-and-quality) for measured
prompt lengths, biases, prefill/decode rates and available KL/top-1 results.
Use the [exact 100K protocol](QFN_100K_REPRO.md) to reconstruct its fixture and
run the timing sequence. The portable wrapper's native qualification remains
pending. Numeric intermediate values remain supported independently of named
modes; the calibration workflow measures a machine/model/quant combination.

## Format and source provenance

The shared format is named Streaming Llama Pack and uses `.slpack`. Packing
preserves the NVIDIA expert E2M1 values and UE4M3 scales through a byte-layout
change; this does not establish that every other section in an existing model
pack came losslessly from the NVIDIA download. The original reference pack
also contains converted dense/Engram tensors and auxiliary sections inherited
from an earlier pack. Reproduction must declare those dependencies explicitly;
do not describe the entire pack as an end-to-end lossless NVIDIA relayout.
Use the pinned model recipe and its verification hashes when those prerequisites
are available. Complete all recipe prerequisites before claiming reproduction.

Upstream kernel attribution is retained in the vendored licenses and provenance
manifests. Legacy kernel ABI, DLL and indexer option identifiers are retained for
compatibility; changing the format name does not remove those capabilities.

## Remaining work

The prefill target of 10,000 tokens/s remains open. Shared gate/up input
quantization and other unmeasured kernel candidates remain available in source
behind their explicit default-off switches; they are not qualified defaults.
DFlash needs compatible assistant weights and its CUDA port. Full-model GLM
qualification and the recipe output identities listed as pending remain open.
