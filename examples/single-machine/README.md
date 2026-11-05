# Single-machine inference

Run from the checkout root with a local configuration containing model, engine,
vision, GPU and storage paths. All launchers accept `-DryRun` to validate the
resolved configuration without loading a model or changing a preset.

## DeepSeek V4.1

Copy `examples/single-machine/config.example.json` to a local file. It selects
the packed FP4 streaming engine, exact cone prefill, eight read parts and an
explicit RAM/sweep plan. The 262K context uses an F32 growable context cache;
the 999,424-token context uses BF16. Both keep KV at least eight bits.

```powershell
powershell -File examples/single-machine/dsv41-modes.ps1 -Config LOCAL.json -Mode max_quality -Context 262k -DryRun
powershell -File examples/single-machine/dsv41-modes.ps1 -Config LOCAL.json -Mode balanced -Context 262k -DryRun
powershell -File examples/single-machine/dsv41-modes.ps1 -Config LOCAL.json -Mode max_speed -Context 1m -DryRun
```

Remove `-DryRun` to serve. The full-performance defaults use 10.25 GiB aggregate
host budget, 10 GiB worker, 6.25 GiB base producer, 7.75 GiB producer maximum and
a 4 GiB physical reserve. Automatic Engram queue policy uses four queues for a
single sweep and one for multiple sweeps. Budgets that cannot fit available
physical RAM refuse work; they do not use the page file as extra capacity.
Explicit alternate profiles remain available with `--performance-experiment`.

The accurate, fast, ultra, ultra-cone and approximate ultra-yoco launchers, with
API-only and web UI variants, remain available. Approximate YOCO requires its
explicit flag. Resident sweep append, CPU expert lanes, RAM cache, MTP and
writer controls remain selectable through their documented flags.

See [performance and quality](../../README.md#performance-and-quality),
[table reproduction](DSV41_BENCHMARK.md), [model conversion](../../docs/DSV41_GGUF.md)
and [backend setup](../../docs/BACKEND_SETUP.md).

## Qwen3.8 Flash Next

Copy `qfn.example.json` and provide the complete `.slpack`, stable CUDA engine,
compatible vision companion and GPU identity. Named mode settings are in
`qfn.settings.json`; all performance controls are explicit flags or settings.

```powershell
powershell -File examples/single-machine/qfn-max-quality.ps1 -Config LOCAL.json -DryRun
powershell -File examples/single-machine/qfn-balanced.ps1 -Config LOCAL.json -DryRun
powershell -File examples/single-machine/qfn-max-speed.ps1 -Config LOCAL.json -DryRun
```

Max quality and Balanced use unbiased routing; Max speed uses bias 0.68.
Numeric biases, including 0.35, retain the full supported range. When changing
the Max speed bias, set the same value in the local settings file and explicit
`-MaxSpeedBias` flag. Bias changes routing and the measured quality tradeoff;
it does not remove sampler controls. MTP and host-RAM caching remain opt-in for
single-machine experiments. Prefill/decode use `dsv-native`, the refined
tensor-core indexer, tile 2048 and eight read parts.

The native API retains streaming, thinking, tools, temperature, top-k/top-p/min-p,
typical sampling, penalties, seed, stop strings, logit bias, token limits,
cancellation, prefix reuse and guarded lazy image encoding. See
[QFN setup](../../docs/QFN.md), [100K reproduction](../../docs/QFN_100K_REPRO.md),
[calibration](CALIBRATION.md), and [paired Q4/Q8 recipes](../dual-machine/qfn/README.md).
