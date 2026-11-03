# QFN native NVFP4 streaming

Qualified defaults use `dsv-native` mode6 for prefill and decode, the refined
tensor-core prefill indexer, eight-split expert reads and MTP off. Packed expert
bytes, persistent activation storage and a CUDA graph per token remain intact.
The model format is Streaming Llama Pack (`.slpack`); see [format](SLPACK.md),
[pinned recipes](SLPACK_RECIPES.md) and [exact100K reproduction](QFN_100K_REPRO.md).

Named modes select0/0/0.68. The accepted installed0.35 control is a separately
requested intermediate point, not a new0.68 measurement. At exact100K with100
discarded warmup and128 measured tokens per mode on the same sequence, the final
quiet installed control measured4,439.8570 prefill/25.2482 quality/46.2537tok/s
at bias0.35, recall and guard PASS. Current installed KL/top-1 remain unmeasured.
Historical0.68 measured50.1246tok/s, KL0.137320748/top-1 88.28125%, before the
installed build. No new0.68 inference was run. [README evidence](../README.md)
separates all checkpoints and the owner-accepted regression closure.

All arithmetic aliases, numeric bias values, samplers, stop behavior, MTP,
RAM expert tiers and other runtime capabilities remain supported. Workspace12
remains the default; the optional32MiB setting is unqualified. Fresh source
build and full three-model reproduction qualification remain incomplete.

## Retained reference and experimental arithmetic

This path retains the existing DirectStorage banks, cache/miss/replay lifecycle, router
weights, sequential top-k sums, shared experts, HC, GDN/QSA, sampler and append/MTP state paths.
Main experts remain raw NVFP4 bytes, including all sixteen FP32 scale-tail bytes. The copied
drafter uses its separate Q2_0 descriptor and existing Q2_0 relayout/execution.

`--nvfp4-mode dsv-native|native-q8|auto|w4a4|mmq|calibrated-w4a4` selects **main-expert prefill** arithmetic.
`native-q8` is the retained numerical reference. The independent dynamic W4A4
experiments below do not replace the qualified mode 6 default. `auto` uses upstream native q8_1 below1024
caller rows and W4A4 from1024; `w4a4` explicitly requests W4A4 even for a short final bank
chunk. The native-q8 decode reference supports arbitrary physical cache slots.
The independent `--nvfp4-decode-mode dsv-native|native-q8|calibrated-w4a4` optionally selects calibrated
target decode, including MTP target verification; the copied Q2 drafter retains its original arithmetic.
The front end propagates explicit `qfn_nvfp4_mode` and `qfn_nvfp4_decode_mode` settings;
no sampler/MTP setting is changed. Either explicit mode for a non-NVFP4 main model fails loudly.

W4A4 internally uses bounded16384-row windows while preserving the original caller phase,
logical attention tile and selected separate-MoE chunk. This is not a changed public tile.
The native q8_1, dynamic W4A4 and MMQ paths use their pinned upstream numerical policies.
Explicit `mmq` selects the retained MMQ/F32-q8 candidate for main prefill, including final
short windows. It does not promote that arithmetic to the default.

Explicit `calibrated-w4a4` consumes NVIDIA's raw exported `input_scale=amax/(6*448)` for
each layer/expert/gate, up and down projection. Unequal gate/up globals remain independent;
the down activation scale applies to original H before the down weight scale. The opaque
`experts.input_scale` section holds LE F32 `[layer,expert,3]`; its checkpoint descriptor,
exact bytes, SHA256 and every finite positive scalar are validated. The original provenance
index and checkpoint config/generation/quantization bytes remain present with SHA validation.
These fields never masquerade as ordinary GGUF tensors or overwrite the expert blob tails.
Legacy packs remain valid for their original modes; an explicit calibrated request requires
the complete calibration metadata and independent calibrated ABI1 exports before allocation.

The scalar array remains F32 on a stable GPU allocation admitted before cache sizing;
each calibrated custom node depends on the actual layer `[3,512]` view. Optional calibrated
decode retains separate logical calibration IDs and physical cache IDs. Its stream gathers every
selected raw blob and calibration triplet into one compact route bank per row, preserving all16 tail
bytes, output order and source calibrated gate/up/down quantization. The independent decode ABI
and complete persistent workspace are required before cache admission; unsupported requests fail.
Discarded replay/timeout attempts may use repeated safe physical slot-zero placeholders while
their logical selections remain distinct. No cache policy, capacity request or RAM/MTP feature
is reduced, and no failed calibrated request selects native-q8. The phase workspace query
reserves the full requested chunk (calibrated:47,542,528B atT1,339,202,560B atT2048,
2,382,369,280B atT16384). No activation field is dropped or replaced by a dynamic fallback.
No model-matched full-engine numeric or performance claim follows from a successful build.

Set front-end `qfn_vocab_gguf` to an explicit model-specific cache path when selecting a
different tokenizer. Its absent default remains `<runtime_root>/qfn-vocab.gguf`; native reuse
validates the complete tokenizer fingerprint and never overwrites an incompatible old cache.

Authenticated source generation JSON retains its exact raw bytes. Sampling defaults follow the
converter policy: legacy defaults < checkpoint sampling fallback < canonical GGUF defaults < explicit
base `meta.sampling` < explicit CLI < request. All existing samplers and repetition aliases remain
supported; unknown active metadata controls fail. Source BOS/PAD roles and the exact source EOS
integer/list are independent from sampler-default precedence. Source-absent EOS preserves legacy
behavior. `--do-sample 0|1` and request `do_sample` use the same ordinary/speculative greedy
predicate without discarding temperature or filters. Actual NVIDIA add_bos=false adds no BOS.
Global EOS token IDs remain independent from per-request detokenized stop strings.

The custom CUDA node returns unweighted F32 `[2560,K,T]` rows in router order. Its raw-I8
source has2764816-byte stride, with all three921600-byte matrices and the16-byte tail.
The adjacent `qfn-basalt-nvfp4.dll` is resolved by absolute executable path and must expose
ABI1 workspace/prepare/run. DLL/ABI/format admission fails before GPU/model allocation.
Kernel preparation precedes graph capture; missing support never selects CPU or another format.

Decode scratch is engine-owned and stable across graph replay. Prefill scratch is queried
for the actual attention tile/chosen MoE chunk; exact extra bytes enter MoE admission.
It is released before the original admitted cache capacity is restored/reseeded. A cold
slot-zero placeholder receives an actual full expert blob through DirectStorage, with
nonresident metadata retained, for discarded record-only replay attempts. Its read bytes
and time count toward cold load/cache-restore TTFT. Duplicate short physical IDs are valid
for these discarded attempts and hot/repeated entries.

Source pin, original license notices, complete vendored closure and C ABI are recorded in
`engines/qfn-basalt-nvfp4/PROVENANCE.json` and its README. The integration follows the
existing DSV4.1 native-FFN custom bridge and keeps ggml type40 reserved.

## Portable build prerequisites


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

The incremental native-build helper additionally requires explicit
`--workspace`, `--input-build`, `--native-recipe`, `--vcvars` and
`--native-vcvars` arguments. A completed guarded object closure supports
`--relink-only`. Stable test target `bin/qfn-op-test.exe` remains separate from
the production `bin/qfn-engine.exe`; variants use flags rather than extra
executable names. The companion DLL remains a dependency of both CMake targets.

CPU-only checks use the same executable: `--nvfp4-node-self-test`,
`--sampling-self-test`, and `--pack-check --model FILE`. CUDA parity and full-model gates
require the coordinator's exclusive GPU lease, memory/paging guard and explicit admission.
The unfiltered stable `qfn-op-test.exe` retains all prior Q2_0, GDN, QSA, HC and V3 checks
and adds actual NVFP4 CUSTOM dispatch at4 native-q8 rows and1024 W4A4 rows. It compares
unweighted output with the same pinned DLL, validates external router multiply/sequential
top-k reduction, and requires static CUDA capture/replay with stable inputs and scratch.
These wiring checks supplement independent DLL arithmetic parity; they do not replace
matched full-model numeric, MTP/append/state/recall or100K timing gates.
