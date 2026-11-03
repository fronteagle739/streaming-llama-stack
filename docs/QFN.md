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

The native grouped kernel exporter under `engines/streaming-llama/flashinfer-sm120`
requires the pinned CUTLASS DSL 4.8.1 and the checked-in FlashInfer source closure.
Pass explicit workspace, DSL-site, AOT directory and CUTLASS runtime arguments
to its exporter and build script. Use the existing physical-memory guardian for
all compilation and engine runs. A fresh source build of this publication
snapshot has not yet been qualified; the incremental build helper accepts an
explicit prior Ninja build and native link recipe and is not a fresh-build claim.

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

The final accepted quiet installed control at exactly100,000 prompt tokens
measured4,439.86 tokens/s prefill,25.2482 unbiased decode and46.2537 decode at
explicit bias0.35, with100 warmup and128 measured tokens per mode on the same
sequence, MTP off/eight read parts, recall and guarded cleanup PASS.
The historical accepted checkpoint measured4,573.37/26.7441/46.7450 and used a
different executable. The regression investigation is closed by owner decision;
the remaining roughly3% prefill read-path difference is unexplained and roadmap-only.

Selected Max speed0.68 uses historical preinstalled-build evidence:50.1246tok/s,
mean KL0.137320748 and top-1 agreement88.28125% over256 full-vocabulary rows.
No new installed-build0.68 inference was run. Current installed-build KL/top-1
are unmeasured; do not attach historical metrics to its timing row. Numeric0.35
and its historical sampled natural-stop evidence remain available. See the
[complete evidence table](../README.md#performance-and-quality-evidence) and
[portable exact100K protocol](QFN_100K_REPRO.md). These are measured reference
points, not a promise for another machine or a fresh source build.

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
