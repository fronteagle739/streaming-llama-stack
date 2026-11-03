# Reproducing model files

Every recipe records its original Hugging Face repository, immutable40hex
revision, full snapshot inventory, expected byte counts and SHA256 or Git-blob
SHA1 identities. Model licensing and gated-source access remain the original
source's terms. The source links are pinned, rather than links to a moving
default branch.

| Recipe | Pinned source | Snapshot |
| --- | --- | --- |
| [QFN](../recipes/qfn-nvfp4.json) | [NVIDIA Qwen3.8 Flash Next NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47) | 25files,132,734,506,847bytes |
| [DSV4.1](../recipes/dsv41-flash.json) | [DeepSeek V4.1 Flash1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | 83files,510,311,045,295bytes |
| [GLM](../recipes/glm53f-nvfp4.json) | [Huihui GLM5.3 Flash NVFP4](https://huggingface.co/edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4/tree/25a335075327eb8edf863a2a8f779c32f0095808) | 47files,204,476,333,629bytes |

Download only when a source is wanted and space is available:

```text
python scripts/download_slpack_source.py --recipe recipes/qfn-nvfp4.json --out SOURCE_DIR
python scripts/download_slpack_source.py --recipe recipes/qfn-nvfp4.json --out SOURCE_DIR --verify-only
```

All source files are retained and authenticated, including the source metadata.
The downloader never overwrites a completed file; it verifies it. Interrupted
`.partial` files remain available and require explicit `--resume`. Hash failure
retains the file and fails loudly. Source names cannot escape the requested
destination. Recipe choice changes the model; no local machine path is baked in.

## QFN: source plus explicit auxiliary pack

The delivered QFN file combines several representations. Routed experts use a
lossless NVIDIA NVFP4 relayout. Dense BF16 projections were converted once to
Q6_K where rows allow it and otherwise Q8_0; the embedding/output projections
use Q8_0. The FP8 ngram table was converted to Q8_0. MTP, vision and profile
sections were inherited from a compatible earlier QFN pack, together with its
explicit sampler defaults. The new converter retains this composition and
checks compatibility. It does not substitute raw NVIDIA MTP/vision weights
for the delivered auxiliaries.

```text
python scripts/pack_slpack.py --source SOURCE_DIR --sections-from AUXILIARY.slpack --out QFN.slpack --scratch PACK_SCRATCH --host-budget-gib 6 --runtime-floor-gib 4
python scripts/slpack.py QFN.slpack
qfn-engine --pack-check --model QFN.slpack
```

Use an explicit host budget that fits available physical RAM above the floor.
The existing process-tree guardian monitors memory and refuses an inadmissible
budget; the job commit cap is explicit. The importer uses one worker and a
bounded expert queue, inverse-checks each emitted expert and fully hashes the
finished output. No intermediate full model copy is written. Tokenizer temp
files, failed partials, profile sidecar, log and memory receipt remain in the
caller-supplied scratch/output locations for review. Heavy packing or hashing
must be admitted by the machine's scheduler and paused for timed benchmarks.

`--allow-legacy-base` explicitly admits a historical auxiliary container.
`--dense-q8-0`, `--mtp-dense`, `--name` and the source/auxiliary paths are
explicit inputs, and alter the resulting identity. An unsupported base section,
active sampler default or source tensor fails instead of being omitted.

The source importer and quantizers are vendored under
[`scripts/slpack_qfn_source`](../scripts/slpack_qfn_source/UPSTREAM.json), with
their original MIT notice and source hashes. Their required Python dependencies
are NumPy, `regex` and the existing guardian's `psutil` dependency. The format
and default provenance strings identify the new tool. Profile references in
new metadata use relative output names.

The reference native file is131,730,218,667bytes. Its recipe records all26
payload-section hashes and the original container's historical whole-file
hash. The expected new-envelope SHA256 is
`0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe`,
projected from a complete original-file hash with only its12-byte envelope
prefix changed. Its deployed readback is still pending. A portable fresh import has different provenance
metadata, so its identity must be recorded from its own verified output;
neither the old hash nor the migrated reference hash is asserted for it.

**Reproduction limitation:** the precise inherited auxiliary pack currently has
no pinned distributable download recipe. Its required section sizes/hashes
are recorded, and the converter requires it explicitly. Without that input,
NVIDIA's download alone cannot reproduce the historical full file. No
raw-checkpoint-only full-file reproducibility claim is made. Resolving that
missing public auxiliary source recipe is marked `BLOCKED-OWNER` in the QFN
recipe, rather than inventing a URL, weights or output hash.

For a locally archived original NVFP4 container, a complete native relayout can
be regenerated and checked without writing another full copy:

```text
python scripts/verify_slpack_relayout.py --source ARCHIVED_ORIGINAL --target QFN.slpack --receipt REPRODUCTION.json
```

Run this under the existing memory guardian and scheduler. It regenerates every
expert and replacement metadata byte, checks inverse equality, compares all
payload bytes, validates zero gaps and the exact new header, and calculates the
full target SHA256 in the same read. Source and target identities must remain
unchanged. The receipt explicitly labels its source as an archived composed
container. It does not reinterpret that proof as raw-source-only reproduction.

For runtime settings use the named-mode QFN examples and the current calibrated
mode table. An artifact hash proves storage identity; normal sampler, quality,
natural-stop chat and performance qualification still apply to the runtime.

## DSV4.1

DSV4.1's pinned upstream checkpoint is FP8, rather than an NVIDIA NVFP4 snapshot.
Its existing [`convert_dsv41_gguf.py`](../scripts/convert_dsv41_gguf.py) route
creates GGUFs consumed by the DSV engine. Preserve the main, DSpark/MTP and
vision components; the recipe does not silently change this established route
to a different container or drop its sidecars. The converter's source tools
are checked by [`scripts/dsv41/antirez-ds4.json`](../scripts/dsv41/antirez-ds4.json).
See [the existing conversion documentation](DSV41_GGUF.md) for build prerequisites,
dry-run, native-FP4 expert/native-FP8 dense options, source pin and validation.

The recipe gives the measured Q4 reference identity separately:
518,596,067,328bytes,
`2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`.
The native-FP4 conversion's full output hash is unmeasured and remains null;
the Q4 reference is not presented as its result. All supported original dense
and expert formats remain available through the existing converter options.

## GLM

The GLM recipe pins every raw source file. Its separate native importer must
retain the entire source inventory and use the same `.slpack` envelope, with
the GLM-specific layout and source-preservation certificate. Its new full pack
has not yet been produced, so expected output size/hash remain explicitly
pending. No guessed output identity or QFN tensor geometry is supplied. GLM's
native reader, format fixtures, model quality and final runtime qualification
remain separate deliverables.
