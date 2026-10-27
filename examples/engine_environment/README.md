# Environment behavior examples

Launcher flags and local `env_overrides` control child processes; inherited engine variables do not silently override them. [Variables](../../docs/VARIABLES.md) records scopes, defaults and source consumers.

The image-worker sample admits a 5 GB free-memory floor and refuses an impossible 64 GB floor under an independent 1 GiB RSS budget and 4 GiB physical reserve. The native CPU sample accepts zero routing bias and reaches its missing-model error; nonzero bias requires CUDA. These are startup examples rather than model-throughput measurements.

Use local paths and explicit budgets when running the example scripts:

```text
python -B tests/prove_image_environment.py --config <config.json> --expect admitted --receipt <output.json> --mem-guard-min-free-gb 5
python -B tests/prove_native_environment.py --config <config.json> --expect model-open --receipt <output.json> --ds4-cache-affinity-bias 0
```

[RTX vector casts](rtx-vector-cast.json) compare the complete 67-token state with vector dispatch on/off. [Mac Metal budget](mac-metal-budget.json) exercises the owned wired-buffer factory with explicit allowance/refusal cases. These bounded samples do not establish inference behavior for every variable.
