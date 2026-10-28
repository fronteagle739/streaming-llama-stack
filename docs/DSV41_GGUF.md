# DeepSeek V4.1 Flash GGUFs

The combined streaming-prefill/ds4-decode engine accepts native `deepseek41`
Flash models with the existing 40-layer geometry and Engram row format. File
names and the measured Q4 file's SHA are not restrictions. A completed SHA-256
receipt binds whichever file is selected to its size, file ID, volume and write
time. Those physical checks remain active while streaming; caches and local
session messages carry that selected file's digest.

| Tensor role | Supported representation |
| --- | --- |
| Routed gate/up/down | Q4_K / Q4_K / Q4_K |
| Routed gate/up/down | IQ2_XXS / IQ2_XXS / Q2_K |
| Routed gate/up/down | MXFP4 / MXFP4 / MXFP4 |
| Attention, shared experts, output head | Q8_0, Q4_K, Q4_0 |
| Embedding and auxiliary projections | Architecture-required F16 |
| Norms, routers, biases, HC scales/base | Architecture-required F32 |
| Engram table | Original packed I8 row264 |

Layers may use different supported expert triplets. The cache admits its largest
slot stride and invalidates both cache tiers when changing representation. The
Q4-specific native-half kernels remain selected only for Q4 tensors; other
expert formats use the existing GGML kernels. Attention arithmetic follows the
selected profile. Q8-only dense optimizations apply where Q8 matrices exist;
other dense types use their normal products. Explicit Q4-to-FP4 HYPER packing
is a separate Q4-specific technique, not a requirement to load a model.

Unsupported tensor types or gate/up/down combinations fail with tensor names
and types, before a full-file verification hash or CUDA launch. The native
loader independently rechecks types, shapes, metadata and verified file extents.

## Existing public GGUFs

The [public antirez release](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/9dcee51)
includes an IQ2_XXS/Q2_K GGUF and nine standard MXFP4 shards. Use an existing
single GGUF directly. **Split files must first be losslessly assembled** using
antirez's pinned `gguf-tools/merge_deepseek41_shards.py`; its `--help` describes
the required original byte count and SHA. It preserves tensor bytes, retains
the inputs, restores 16 KiB alignment and verifies the assembled file before
publication. It currently requires POSIX `fcntl` (Linux/macOS). This is a file
assembly prerequisite, not a new quantization. Allow space for the assembled
file plus the inputs; this release does not directly stream a shard set.

Use the same complete file on both machines for an exact paired handoff. This
change's new quant coverage is for the combined single-engine path; it does
not qualify additional paired model identities or speed/quality measurements.

## Convert an existing raw checkpoint

[`scripts/convert_dsv41_gguf.py`](../scripts/convert_dsv41_gguf.py) wraps
[antirez/ds4 at ee7e8405232c8971000cc7ba2b945b00bf6d20d3](https://github.com/antirez/ds4/tree/ee7e8405232c8971000cc7ba2b945b00bf6d20d3/gguf-tools).
Every imported converter file is checked against the SHA-256 manifest in
[`scripts/dsv41/antirez-ds4.json`](../scripts/dsv41/antirez-ds4.json).
It reuses upstream planners, native quantizers and streaming writers; it does
not implement another quantizer. `--fetch-tools` downloads only these small
source files, never model weights or a quantizer binary.

The input is an **already available** raw
[deepseek-ai/DeepSeek-V4.1-Flash checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170),
about 475 GiB of safetensors. The measured files record source revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. No download command is supplied.
Full conversion also needs the source tokenizer, NumPy, tokenizers and a locally
built upstream `libds4quants` shared library. The fetched source includes its
Makefile, `quants.c` and `quants.h`; run `make -C <converter-source-dir>
quants-shared` on Linux/macOS (C compiler and pthreads required). A native
Windows build needs a compatible pthread-enabled toolchain. Use local storage with room
for the raw input, roughly 483 GiB main output, 7.8 GiB DSpark, 0.91 GiB Vision,
and upstream's free-space reserve. Conversion is CPU-only and can take a long
time. It was not performed again for this release.

An optional `--expert-format native-fp4` recipe preserves the main routed
experts in their released E2M1/E8M0 MXFP4 representation, keeping the same Q8_0
dense recipe. See the [source audit and candidate limits](NATIVEFP4.md).
Its lossless expert repack and small-file writer are CPU-tested; full conversion
and native GPU performance/quality qualification remain pending.

The committed [reference tensor map](../scripts/dsv41/reference-layout.json)
was extracted from the existing RTX main, DSpark and Vision GGUF **headers
only**. It records header hashes, shapes and actual per-tensor type IDs. Main
uses Q4_K routed experts and Q8_0 dense matrices; DSpark uses that same quantizer
recipe with F16/F32 auxiliaries, while Vision copies BF16 tensors unchanged.
This specifies a tensor layout, not a promise of byte-identical quantized values
or a particular resulting whole-file SHA. The measured quality baseline is the
existing GGUF, not a claim of bitwise identity to raw safetensors.

### Plan and verify without weights

```powershell
python -B scripts/convert_dsv41_gguf.py --source <existing-config-and-index-dir> --ds4-tools <converter-source-dir> --fetch-tools --dry-run --reference-gguf <existing-main.gguf>
python -B scripts/convert_dsv41_gguf.py --source <existing-config-and-index-dir> --ds4-tools <converter-source-dir> --component dspark --dry-run --reference-gguf <existing-dspark.gguf>
python -B scripts/convert_dsv41_gguf.py --source <existing-config-and-index-dir> --ds4-tools <converter-source-dir> --component vision --dry-run --reference-gguf <existing-vision.gguf>
```

Dry-run opens only `config.json`, `model.safetensors.index.json` and, if given,
the reference GGUF header. It does not open safetensors, load a quantizer or
read GGUF tensor payloads. `--plan <new-plan.json>` saves the planned inventory.
For planning only, a narrow AST adapter replaces upstream's shard-dependent
`claim` validation with source-index name assertions; upstream's config-based
name and shape expressions remain unchanged. Types come from the recorded
header recipe. The source index has no shapes or dtypes, so dry-run cannot
verify the raw shards' contents. Real conversion uses upstream's original
`SourceDB` and planner to check those source headers too.

The release dry-run matched every name/type/shape in all three actual headers:
**1,046 main tensors, 81 DSpark tensors and 266 Vision tensors**. Verification
does not download or convert another copy of the model.

### Run conversion when the raw checkpoint and output space already exist

```powershell
python -B scripts/convert_dsv41_gguf.py --source <existing-raw-checkpoint-dir> --ds4-tools <converter-source-dir> --execute --out <new-main.gguf> --quants-library <built-libds4quants> --host-budget-gib 8 --threads 1
```

Use `--component dspark` with the same arguments for DSpark; use `--component
vision` without `--quants-library` for Vision. Keep output paths distinct. The
main/DSpark writer supports upstream's `--resume`. Existing complete outputs
are never overwritten. The selected host budget must fit available physical
RAM plus the default 4 GiB free-RAM floor. The supervisor enforces an aggregate
Windows Job commitment limit and watches tree RSS/available RAM; failures are
loud. Its 600-second timeout is a renewable **no-progress** window, not a total
conversion deadline. On POSIX the existing supervisor supplies RSS/floor checks.
Logs and the memory receipt are beside the output. The resulting header is
checked against the plan before success is reported.

## Validation scope

The Q4_K-routed/Q8_0-dense build is GPU-tested, including a
[49.6K-token recall and decode smoke](ANYGGUF_GPU_SMOKE.md). The IQ2_XXS/Q2_K
and MXFP4 builds are CPU-tested only; their large-model GPU execution and
performance remain untested.

The retained [CPU/header receipt](DSV41_GGUF_VALIDATION.json) records the exact
header hashes, tensor counts, numerical errors and validation limits.

CPU numerical tests round-trip tiny Q4_K, IQ2_XXS, Q2_K and MXFP4 GGUFs and
compare actual MMID products to independent dequantized dot products. Python
tests cover unsupported types before hashing, stale receipts, two simultaneous
session identities, missing source tensors, changed config geometry and all
three metadata-only plans. Native CPU build and CUDA host compilation also pass.
No large-model load, model download, full conversion or new GPU benchmark was
used to establish these format and header checks.
