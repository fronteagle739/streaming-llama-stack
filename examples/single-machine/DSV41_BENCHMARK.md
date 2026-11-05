# Reproduce the DeepSeek V4.1 tables

Use the packed FP4 speed proxy, its compatible DSpark/vision files, and the
matching tokenizer. This protocol does not compare with full-precision weights.
Hardware is an RTX 5090 with 32 GB VRAM, 32 GB RAM and DirectStorage NVMe.
Temperature is 0, seed 1, MTP off, eight read parts, bias warmup/ramp 0.
Natural EOS remains active; early stop fails the matched-window test.

Copy the single-machine config and provide every local model/build/GPU path.
The full-performance profile uses 10.25 GiB aggregate, 10 GiB worker,
6.25 GiB base producer, 7.75 GiB producer maximum and 4 GiB physical reserve.
At least 14.25 GiB available physical memory is required. Automatic queue policy
uses four Engram queues for a single sweep and one for multiple sweeps.
The launcher records all resolved settings; alternative caps remain selectable
with an explicit experiment flag. No model download is performed by the scripts.

Supply the exact little-endian signed int32 real-text fixture:

| Prompt tokens | Fixture SHA256 |
| ---: | --- |
| 100,000 | `221c96d2ea5397c3f250e10d973ded38dd2d90537689d59de7b92abd1921bbb1` |
| 199,952 | `f8e345b7848fca2fdbaa8b239050ec4d3a9549ebddd2d70355af042df4608821` |

Fixtures contain different real-text prefixes. A different token hash is a new
workload and cannot reproduce these rows. The fixture files are external inputs.
This script checks the supplied hash before it creates a native producer.

From the checkout root, validate the config and fixture without loading a model:

```text
python -B examples/single-machine/benchmark_dsv41.py --config LOCAL.json --fixture PROMPT_LE32 --fixture-sha256 FIXTURE_SHA256 --output OUTPUT.json --mode timing --bias 0 --validate-only
```

For each prompt length, run the three timing arms under the RAM guard with
different output names. Repeat the command at biases 0, 0.35 and 1.36125:

```text
python -B scripts/qfn_dual_guard.py --root RUNTIME_ROOT --label table-timing --rss-gib 10.75 --commit-gib 50.75 --floor-gib 4 --seconds 900 --progress-log PROGRESS_LOG -- python -B examples/single-machine/benchmark_dsv41.py --config LOCAL.json --fixture PROMPT_LE32 --fixture-sha256 FIXTURE_SHA256 --output TIMING.json --mode timing --bias 0
```

Prefill counts the native embedding, 40-layer and head phase, excluding load.
Decode samples 229 tokens, discards 100 evaluated steps and times the next 128.
The bias-zero timing saves `TIMING.reference.le32`. For quality, use those same
IDs at every bias, starting a fresh producer with the same exact prompt:

```text
python -B scripts/qfn_dual_guard.py --root RUNTIME_ROOT --label table-reference --rss-gib 10.75 --commit-gib 50.75 --floor-gib 4 --seconds 900 --progress-log PROGRESS_LOG -- python -B examples/single-machine/benchmark_dsv41.py --config LOCAL.json --fixture PROMPT_LE32 --fixture-sha256 FIXTURE_SHA256 --output REFERENCE.json --mode quality --bias 0 --reference TIMING.reference.le32
python -B scripts/qfn_dual_guard.py --root RUNTIME_ROOT --label table-candidate --rss-gib 10.75 --commit-gib 50.75 --floor-gib 4 --seconds 900 --progress-log PROGRESS_LOG -- python -B examples/single-machine/benchmark_dsv41.py --config LOCAL.json --fixture PROMPT_LE32 --fixture-sha256 FIXTURE_SHA256 --output CANDIDATE.json --mode quality --bias 0.35 --reference TIMING.reference.le32 --reference-logits REFERENCE.f32
```

Repeat the candidate at bias 1.36125. Validate all 229 rows of 129,280 logits;
score only predictive rows 100 through 227. KL(reference bias 0 || candidate)
is in nats, and top-1 is argmax agreement on those 128 rows. Do not crop the
vocabulary, change reference IDs or extend an early-stopped sequence.

The script uses the shared producer, sampler metadata, GPU lease, native memory
guard and certified retirement. Keep its settings and cleanup receipts with the
result. Generated receipts contain local paths and should remain private.
