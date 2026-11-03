# Exact QFN 100K benchmark protocol

`scripts/build_qfn_100k_fixture.py` reconstructs the archived measured fixture
from pinned public text and a synthetic password/essay instruction. It produces
exactly 100,000 little-endian signed 32-bit token IDs: 400,000 bytes, SHA256
`87fb9af861237f91724d56ca9c1a29f77c0e4aadce673ff939abe8500fa6c936`.
This is an audited deterministic reconstruction; the original generator is
unknown. The novel text and a private archived fixture are not bundled.

The text is Alessandro Manzoni's *I Promessi Sposi*, from
[antirez/ds4](https://github.com/antirez/ds4/blob/c97964b8aa77034a8e17afb10ca7a686c40ab511/speed-bench/promessi_sposi.txt)
at revision `c97964b8aa77034a8e17afb10ca7a686c40ab511`. Its normalized LF UTF-8
SHA256 is `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
The upstream repository carries its own MIT software license. Preserve upstream
attribution and consult the source for the text's provenance; this package does
not relabel the novel as newly authored software.

The tokenizer comes from
[NVIDIA's pinned model revision](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47),
with SHA256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Use the upstream model card and licensing terms. Reconstruction was verified
with `tokenizers==0.22.2`; the builder rejects any resulting token hash mismatch.
The header and novel use CRLF, while the synthetic footer uses LF. Those bytes
are deliberate parts of the measured fixture.

## Reconstruct the fixture

Run from the repository root in an environment with the repository requirements
and `tokenizers==0.22.2`. Set `QFN_BENCH_SCRATCH` to your existing scratch root;
the commands below write only there. On Windows, select the intended fast data
volume and keep caches and temporary files on that volume as well.

```powershell
$Scratch = (Resolve-Path $env:QFN_BENCH_SCRATCH).Path
curl.exe --fail --location --output "$Scratch/promessi_sposi.txt" https://raw.githubusercontent.com/antirez/ds4/c97964b8aa77034a8e17afb10ca7a686c40ab511/speed-bench/promessi_sposi.txt
curl.exe --fail --location --output "$Scratch/tokenizer.json" https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/fc694b54fb0174e0913e6adf86691ef85a4ead47/tokenizer.json
python -B scripts/build_qfn_100k_fixture.py --corpus "$Scratch/promessi_sposi.txt" --tokenizer "$Scratch/tokenizer.json" --output "$Scratch/qfn-100k.tokens.le32" --report "$Scratch/qfn-100k.fixture.json" --scratch "$Scratch" --host-budget-gib 1 --runtime-floor-gib 4
```

The builder supervises its tokenizer worker with an explicit RSS/commit budget
and free physical RAM floor. Existing outputs are preserved: choose a fresh
output/report path for a repeat. No native engine or GPU is used.

## Run the exact timing sequence

`examples/single-machine/benchmark_qfn_100k.py` reuses the existing native stdio
protocol and the repository memory guardian, GPU lease and compiler admission
checks. It prefills the exact 100,000 tokens once, then continues the **same
sequence** through quality warmup 100, quality measurement 128, speed warmup 100,
and speed measurement 128. Default quality bias is zero, temperature zero and
seed one; the speed bias is explicit. The native context allocation is 100,480
for this sequence, tile 2,048, MTP off and eight read parts.

Provide a complete model, its complete compatible tokenizer GGUF, and the exact
engine/DLL build under test. Set `QFN_MODEL`, `QFN_MODEL_SHA256`, `QFN_ENGINE`,
and `QFN_VOCAB_GGUF` accordingly. Put a machine-local runtime configuration at
`$Scratch/local.json`, based on `config.example.json`, with your GPU UUID,
priority/lease settings and required model environment. Keep test hosts at
`127.0.0.1`. This harness itself uses stdio and opens no listener.

```powershell
python -B examples/single-machine/benchmark_qfn_100k.py qfn-100k-quality-speed --repo-root . --runtime-root "$Scratch/runtime" --config "$Scratch/local.json" --model "$env:QFN_MODEL" --model-sha256 "$env:QFN_MODEL_SHA256" --engine "$env:QFN_ENGINE" --vocab-gguf "$env:QFN_VOCAB_GGUF" --fixture "$Scratch/qfn-100k.tokens.le32" --output-prefix "$Scratch/qfn-100k-run" --model-certificate "$Scratch/qfn-model.readback.json" --certify-model --speed-bias 0.35 --warmup-tokens 100 --tile 2048 --host-rss 3 --commit-budget-gib 40 --min-available-gib 6 --env QFN_MTP=0 --env DS4_DS_READ_PARTS=8
```

`--certify-model` performs a complete pack section/hash readback under the CPU
guardian before GPU admission. A later run may omit that flag and reuse the
certificate only while its exact file identity and requested SHA256 still
match. The commit cap includes Windows WDDM accounting; RSS and available
physical RAM remain separately guarded. Adjust budgets explicitly for the
machine, without allowing physical RAM to exhaust into paging.

The example uses the retained intermediate bias `0.35`. The selected Max speed
mode uses `0.68`; changing `--speed-bias` changes the requested mode, and does
not establish a new measured result. Engine aliases, optional cache slots and
environment overrides remain available. `--engine-arg` forwards additional
native argv tokens; use the explicit harness arguments for conflicting model,
context, tile and protocol settings. `--sampler-json` forwards every additional
generation field unchanged, including supported reasoning/tool/stop fields.
It rejects conflicts with the benchmark's operation, window length and bias.
Nondefault sampling or warmup is recorded as a protocol variant.

Natural EOG and stop behavior stays active. An early warmup stop fails the arm;
an early measured stop fails matched-window qualification. No answer is
extended to reach the target. The quality continuation must recall the
synthetic password. Request/event logs, all native statistics, exact build and
tokenizer identities, actual process settings and guardian cleanup are retained
under the output prefix. Generated observations can contain machine-local
paths/environment; keep those receipts private unless separately audited.

## Evidence scope

The reconstructed fixture matched all 400,000 archived bytes. Three CPU-only
in-memory protocol checks covered the exact same-sequence windows, additive
sampler forwarding, and early-EOG failure without extension. These are unit
protocol checks: they do not prove native acceptance of arbitrary added fields,
model inference, rates, or quality. Native qualification of this portable
wrapper is pending a reserved benchmark slot. Match engine/DLL/model hashes and
all effective settings to a measured receipt before comparing performance.

This timing/recall harness does not replace full-vocabulary KL/top-1 evaluation
or natural sampled chat tests. Keep using the existing
[calibration workflow](../examples/single-machine/CALIBRATION.md) for those independent capabilities.
