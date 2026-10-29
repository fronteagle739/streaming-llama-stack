# E19 DeepSeek imports on SM120

`PERFORMANCE.md` reports the measured results and limitations; `results.json`
contains component results, paired timings/quality, stage totals, guard receipts
and build-asset digests. `raw-evidence-manifest.json` pins the retained local
logs and artifacts without publishing machine configuration or job specs.

`upstream/` contains unmodified MIT inference sources at the pinned checkpoint
revision, with a byte manifest. `official_layer.py` preloads a real layer-0
shard using bounded tensor reads and runs synthetic BF16 residuals at2541/65536
rows. It adapts only attention head grouping and HC-post row tiling. Its source
receipt requires a full previously verified SHA256, byte count, mtime and revision.

`benchmark.py` compares identical resident inputs with the native bridge
`e19_native.cpp`. Compile that source together with `llama-dsv41-graph.cpp` and
`boundary-capture.cpp`, include the producer's src/ggml headers, and link the
same CUDA-enabled ggml backend used by the producer. The measured bridge used
MSVC C++17 /O2 /MD, CUDA13.2 SM120a backend objects; unchanged objects/libraries
were reused and hashed. Components requiring4K native tiles report sums of
independently timed tiles. Official index scores use128-row query tiles to
bound their otherwise128GiB head-score table. Neither timing implies one64K
integrated streaming graph.

`moe_compare.py` calls the existing ABI4 FlashInfer adapter on the same original
FP4 bytes. Router, gathers, sorting, quantization, scatter, clamp/weight ordering
and shared expert are timed; initial packing is excluded. Native expert chunks
are16K or smaller. NATIVEFP4 owns that adapter and the next direct-packed layout.

The production imports are native-parity fused HC mode2 and packed TileLang
attention mode2. Both default to2; their corresponding environment variables
accept0 to disable. Attention's pinned AOT image is embedded in C++, avoiding
runtime compilation and external-image deployment requirements. The unpadded
cone remains an opt-in experiment because it changes floating-point bits.
Attention itself is lossy; read the retained negative post-EOS result before
interpreting the passing weather and answer probes as general accuracy.

All hardware commands must use shared GPU ownership and explicit host-memory,
commit, runtime-floor and paging guards. `run_guarded.py` wraps OwnedProcess;
its config must put runtime_root, cache, dependencies and temporary files under
the one task scratch directory. Run from a configured compiler/Python environment.
For example (replace placeholders before running):

```text
python -B examples/kernels2/run_guarded.py --config CONFIG --host-gib 5 --commit-gib 32 --name official-layer -- PYTHON -B examples/kernels2/official_layer.py --scratch-root SCRATCH --source-receipt RECEIPT --rows 2541 65536 --output SCRATCH/layer.json
python -B examples/kernels2/run_guarded.py --config CONFIG --host-gib 3 --commit-gib 16 --name components -- PYTHON -B examples/kernels2/benchmark.py --scratch-root SCRATCH --native BRIDGE --rows 2541 65536 --output SCRATCH/components.json
```

Build a new image ahead of inference with `build_sparse.py --scratch-root
SCRATCH --output SCRATCH/sparse.cubin` under that guard. It uses
`sparse_native.py`, TileLang0.1.15, Torch2.11.0+cu128 and nvcc13.2 with C++20/O3.
Then validate it before promotion. `embed_sparse.py --cubin IMAGE --output HEADER`
requires the qualified digest from `native-sparse-image.json`; never silently
replace that manifest with an untested build. CUBIN byte identity can depend on
the compiler environment; new bytes require requalification.

Generate the actual retrieval fixture with guarded `make_recall.py --model
MODEL --output SCRATCH/recall.le32`. The receipt records the planted key, EOS
and token SHA256. Run `scripts/perfexp_pair.py` with `--seed-file` that fixture,
`--count 100000 --recall-stop-token 1 --recall-key QUARTZ-7429-LANTERN` for
natural retrieval. Use a separate pair with `--probe-answer QUARTZ-7429-LANTERN`
for forced answer+EOS logits directly from the prompt frontier. Compare full
F32 vocabulary rows (129280 entries), normalized KL, bit patterns and top-1.
Do not treat the earlier repeated-weather seed or generation after EOS as recall.
