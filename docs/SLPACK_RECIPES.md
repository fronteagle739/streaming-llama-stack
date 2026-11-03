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

The exact default auxiliary input is now available through a separate
[pinned dependency recipe](../recipes/qfn-auxiliary.json). Its
[upstream source attribution](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/tree/a229744b82e18c3e12057ab944062c0c713c7006)
identifies the unmodified historical container. Its69,685,970,297-byte file has
SHA256 `dee4ae542817d73e4f7a56c448ef97cb7bb47ef92b8b2c5b7721ec0e8c25f76c`.
That immutable upstream LFS identity matches the independently full-hashed
auxiliary input. The current moving branch has a different artifact; use the
recipe's exact revision. Pin discovery downloaded metadata only, not weights.
The recipe selects the one required container, rather than alternative model
files in the upstream repository. MTP, vision, profiles and sampler defaults
remain byte-preserved or explicitly resolved under the documented precedence.

Use Python 3.12.10 with NumPy 2.5.0, `regex` 2026.5.9 and `psutil` 7.2.2 to
reproduce the recorded environment. Install the pinned
[packing requirements](../requirements-pack.txt) in an isolated environment:

```text
python -m venv .venv-pack
.venv-pack/Scripts/python -m pip install -r requirements-pack.txt
```

On POSIX use `.venv-pack/bin/python` for the install command. For every
subsequent Python example, use this environment's interpreter in place of
`python`; activation or changes to the runtime environment are unnecessary.
Dependency versions are semantic inputs to the expected pack identity.

```text
python scripts/download_slpack_source.py --recipe recipes/qfn-auxiliary.json --out AUXILIARY_DIR
python scripts/download_slpack_source.py --recipe recipes/qfn-auxiliary.json --out AUXILIARY_DIR --verify-only
python scripts/pack_slpack.py --source SOURCE_DIR --sections-from AUXILIARY_FILE --allow-legacy-base --out QFN.slpack --scratch PACK_SCRATCH --host-budget-gib 6 --runtime-floor-gib 4
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
`--dense-q8-0`, `--mtp-dense`, `--name` and source/auxiliary contents are
explicit inputs, and alter the resulting identity. An unsupported base section,
active sampler default or source tensor fails instead of being omitted.

The source importer and quantizers are vendored under
[`scripts/slpack_qfn_source`](../scripts/slpack_qfn_source/UPSTREAM.json), with
their original MIT notice and source hashes. The format and default provenance
strings identify the new tool. Profile references in
new metadata use a relative content identity. The original `OUTPUT.slpack.profile`
sidecar is retained alongside `profiles/PROFILE_SHA256.profile`; existing exact
copies are reusable, and differing files are never overwritten.

The reference native file is131,730,218,667bytes. Its recipe records all26
payload-section hashes and the original container's historical whole-file
hash. The independently verified new-envelope SHA256 is
`0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe`,
confirmed by a complete fresh file readback after only its12-byte envelope
prefix changed. The native reader's metadata, tokenizer and MTP admission check
also passed. The complete fresh raw-import verification passed on 2026-11-03:
all 25 pinned source files were authenticated again, all 26 sections
were regenerated, all expert inverse and HC8 checks passed, and the entire
reference was independently read back. Its canonical output identity is:

| Output scope | Bytes | SHA256 |
| --- | --- | --- |
| Default fresh import, canonical provenance policy1 | 131,730,218,740 | `195df4842abf2dd335ad706be6518611b2a433e806d98c4edcc8fe5e2b309a92` |
| Historical-provenance reference, new envelope | 131,730,218,667 | `0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe` |

Compare the completed output verifier's byte count and SHA256 with
`expected.portable_import` in the recipe. This expected identity applies to
the pinned source and auxiliary, default model name/dense encoding and no MTP
replacement. All nonmetadata payloads match the historical reference's section
inventory. Canonical metadata is 8,101,620 bytes with SHA256
`d9de7e38aed8db5b44236365f7d40f95cc70f8f9c78397a86dc405069b2cf4d1`.
Its 73 additional bytes account for the output-length difference; the changed
metadata digest and length also change the envelope. No second full model was
materialized for this proof.

Both required weight inputs are pinned: the original NVIDIA snapshot and the
historical auxiliary artifact. The former missing auxiliary source dependency
is resolved. NVIDIA's download alone does not contain the inherited tuned
auxiliaries, so no raw-checkpoint-only full-file reproduction claim is made.

**Determinism boundary:** new imports use canonical provenance policy1. Every
metadata field remains present. Format/version/tool identify the new importer;
`checkpoint.dir` is the logical relative label `source`, and the auxiliary
path is `auxiliary/sha256-METADATA_SHA256`. Its size, version, fingerprint,
copied-section hashes, sampler defaults and other provenance remain intact.
Profile export and runtime arguments refer to `profiles/PROFILE_SHA256.profile`.
Source filenames remain the pinned filenames. Physical source, auxiliary and
output basenames therefore do not alter pack bytes. Model name, dense encoding,
an optional MTP replacement, source contents and tool/dependency versions
remain semantic inputs. Unknown profile runtime arguments fail explicitly.

The historical reference retains its original provenance payload. Fresh
canonical metadata changes exactly seven fields: format, version, converter
tool and the four documented source/profile path fields. No fields were
removed. The recipe keeps the reference identity and canonical import identity
separate, including their different metadata-section hashes.

The complete raw importer can calculate its exact new identity without a second
full model copy:

```text
python scripts/pack_slpack.py --source SOURCE_DIR --sections-from AUXILIARY_FILE --allow-legacy-base --out PLANNED.slpack --verify-against REFERENCE.slpack --scratch EMPTY_PACK_SCRATCH --host-budget-gib 6 --runtime-floor-gib 4
```

This mode authenticates every pinned source file, runs the same tokenizer,
dense/PLE quantizers, native expert conversion and inverse checks, copied
auxiliaries, calibration, sampler resolution and HC8 generator as normal
packing. Each generated nonmetadata section must match the reference's order,
offset, size and SHA256. HC8 reads reference dense bytes only after the generated
section and a fresh whole-reference-section hash agree. Full regenerated
metadata must equal the reference after only the declared canonical provenance
changes; additional differences fail with field names. No fields are dropped.

After regeneration, one full reference pass checks its exact header, zero
alignment gaps and every original payload digest. Hashing substitutes the new
header and actual regenerated metadata to calculate the exact would-be output
size/SHA256. Source, auxiliary and reference file identities must remain
unchanged. A successful `PLANNED.slpack.raw-import.verify.json` receipt certifies
this complete import; `PLANNED.slpack` itself is not created. Tokenizer output,
both small profile aliases, logs and receipts remain available. Any section or
metadata mismatch prevents a successful receipt and expected identity.

The small fixture validates the virtual hash against a materialized canonical
file, both main/MTP HC8 readback and corrupt-reference rejection. The complete
real verification passed with 6 GiB host/commit caps and a 4 GiB free-physical-RAM
floor. Both nested guards exited successfully and confirmed cleanup; the outer
tree's paging telemetry closed. The inner guard recorded a 4,049,231,872-byte
peak RSS and 6,699,986,944-byte minimum available RAM. Bounded outer-tree paging
observation peaked at 4 KiB under the unchanged 64 MiB limit; kernel ETW attribution
was unavailable, so this does not establish zero physical pagefile I/O.

Admit a 6 GiB job only with at least 10 GiB initially available physical RAM and
retain the 4 GiB runtime floor. The observed peak is evidence for that run, not a
guarantee for other machines. Plan roughly 133 GB source authentication reads, another
full source conversion read, the inherited auxiliary reads and fingerprints,
about2.3GB HC8 readback authentication, and one132GB reference pass. A fresh
full auxiliary download verification adds70GB of reads. No full-size model
write occurs in this mode.

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
shared-envelope writer and native-consumer format port have been prepared
separately, with 33 CPU fixtures passing. Their integration adoption, native
build, full-model quality and final runtime qualification remain separate
deliverables; that separate engine code is not included in this QFN package.

The separate GLM source handoff has33 CPU fixtures. Its packer command refers
to that separate source branch; `scripts/glm53f_pack.py` is absent from this
QFN review tree and is not a working executable route here. Native adoption,
build, full pack/inverse/runtime qualification and expected output size/hash
remain pending. DSV4.1 native main plus DSpark/MTP and vision output identities
also remain unmeasured; its pinned Q4 reference is main-only. See the exact
qualification limits in each recipe rather than substituting that reference.
