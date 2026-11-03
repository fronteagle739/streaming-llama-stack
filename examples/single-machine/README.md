# Single RTX machine: DSV4.1

Each quality setting has an API-only launcher and a `-webui` launcher. Both run
the same single engine: streaming-llama prefill followed by modified ds4 decode.

Short follow-ups reuse resident experts through the sweep graph by default.
The measured 32-token append took 2.7-2.9 s, with small numerical differences
(KL 0.0000235-0.00250 at 2K-32K; recall/top-1 passed). Set
`"LLAMADECODE_APPEND_RESIDENT_SWEEP": "0"` in `env_overrides` to use full sweeps.
See the [measured tradeoff](../../docs/EXPERIMENT_E2C.md).

| Scripts | Prefill setting |
| --- | --- |
| `dsv41-accurate*.ps1` | Accurate arithmetic; YOCO off |
| `dsv41-fast*.ps1` | Fast arithmetic; YOCO off |
| `dsv41-ultra*.ps1` | ULTRA arithmetic; YOCO off |
| `dsv41-ultra-cone*.ps1` | Recommended: exact ULTRA dependency cone, keep 2541 |
| `dsv41-ultra-yoco*.ps1` | Exact cone by default; `-ApproximateYoco` selects approximate keep 140 |

Copy `config.example.json` to `rtx.local.json` and replace the model placeholder
with the original supported DSV4.1 1017 Q4 GGUF. Set paths for the built producer
and native runtime if they differ from the stack defaults. Install/build the
stack first; these examples do not download model weights or select arbitrary
replacement quantisations.

From the stack root:

```powershell
.\examples\single-machine\dsv41-ultra-cone-webui.ps1 -Config .\rtx.local.json -DryRun
.\examples\single-machine\dsv41-ultra-cone-webui.ps1 -Config .\rtx.local.json
.\examples\single-machine\dsv41-accurate.ps1 -Config .\rtx.local.json
```

`-Context 262k` is the default. `-Context 1m` selects 999,424 configured tokens
with the growable BF16 context cache; this is capacity, not a measured one-million
token prefill result. `-DryRun` starts nothing and writes nothing. Stop a running
foreground launcher with Ctrl+C; its existing guardian owns the engine tree.

The exact cone is the default for `scripts/start-dsv41-single-*.ps1` and the
legacy YOCO examples. Select `--single ultra-yoco` in `examples/run_preset.py`
or add `-ApproximateYoco` to a legacy launcher to opt into approximate replay.
The cone is bit-exact against the same full-prefill control at 8K and 49,555
text tokens. Real-image numerical qualification remains separate.

YOCO is approximate. Use an
appropriate vision-enabled local config and producer for image prompts; do not
advertise the text-only YOCO speed as image throughput. Prefill quality is separate
from the live decode bias slider and phase sampler controls.

Exact resident sweep follow-ups are enabled with `LLAMADECODE_APPEND_RESIDENT_SWEEP=1`.
The per-token append path changes logits and remains opt-in (`LLAMADECODE_APPEND_DECODE_MAX=96`).
LFU admission defaults to 1 with half life 16; explicit 0 keeps LRU.
Certified FP4 index scoring is enabled by default on supported SM120 BF16-index
paths. The 100K comparison is bit-identical and faster; unsupported inputs use
the existing BF16 scorer. Set `LLAMA_DSV41_INDEX_FP4` to `"0"` in local
`env_overrides` to disable it. See [E6 measurements](../../docs/EXPERIMENT_E6.md).

## Qwen3.8 Flash Next (RTX 5090, native packed NVFP4)

Copy [qfn.example.json](qfn.example.json) to `my.local.json`, set `model` and `gpu_uuid`, then run
`qfn-max-quality.ps1`, `qfn-balanced.ps1` or `qfn-max-speed.ps1`. The engine's code defaults are the fastest qualified path:
packed-vector FP4 decode on the bytes prefill streams, one model graph and one service graph per token,
the prefill carry in the cache tail, eight-split expert reads, MTP off. Exact 100K prompt, warm decode:

| Setting | Decode | Against Max quality |
|---|---:|---:|
| Max quality | 26.74 tok/s | byte-identical to the native Q8 reference |
| Balanced, initial reference | same unbiased routing | conservative until a machine-local sweep selects an intermediate value |
| Explicit bias 0.35, historical qualified build | 46.74 tok/s | deliberate routing tradeoff: KL 0.0372, top-1 92.58% on 256 forced rows |
| Explicit bias 0.35, accepted quiet installed control | 46.2537 tok/s | exact 100K recall/guard PASS; installed-build KL/top-1 unmeasured |
| Max speed, bias 0.68, historical pre-install build | 50.12 tok/s | deliberate routing tradeoff: KL 0.137, top-1 88.3%; no installed-build GUI/timing test at this bias |

The historical default-only 100K run measured prefill 4,573 tok/s; the three-run combined
candidate median was 4,625 tok/s. Final accepted quiet installed control measured
4,439.86 prefill and 25.2482 Max quality tok/s; the regression investigation is
closed, with roughly 3% residual prefill read-path difference
unexplained and roadmap-only. The earlier installed r2 remains historical.
The delivered Max speed selection is 0.68 as of 2026-11-03;
no new 0.68 inference tests were run.
The 0.68 historical row predates the installed build. Three historical GUI checks at bias
0.35 stopped naturally; this is sampled evidence, not a guarantee for every prompt. Keep the pack
on a fast NVMe volume. MTP and the RAM expert tier remain available but are off
in these measured defaults because their tested configurations lost throughput.

Each model carries its own settings in [qfn.settings.json](qfn.settings.json) or
[dsv41.settings.json](dsv41.settings.json). These record all three bias values,
the valid numeric range, router-spread reference, cache seed/half-life and
tile/read settings. The QFN named launchers accept `-Settings your.settings.json`;
`dsv41-modes.ps1 -Config your.local.json -Mode balanced -Settings your.settings.json`
selects the DSV4.1 decode mode independently of its exact-cone prefill. Both
support `-DryRun`. Explicit local engine options and sampler fields remain available.
The QFN example uses the qualified 3 GiB physical host budget and 6 GiB
free-memory floor. Its 40 GiB Windows commit cap includes WDDM accounting;
the physical-RAM guardian still applies and page-file capacity is never used.
DSV4.1 retains its delivered 0 / 0.35 / 1.36125 points and QFN starts at
0 / 0 / 0.68; the initial QFN Balanced value is intentionally identified as
uncalibrated. Numeric QFN requests still accept the full [0,8] range.

Settings marked `reference-settings` are measured reference choices, not an
automatic calibration of the current machine. Recalibrate after changing
hardware, storage, model weights, quantisation, context, cache or kernels; see
[CALIBRATION.md](CALIBRATION.md). A new machine-local calibration sweep must report KL/top-1, warm throughput,
cache hits and natural generation stops. The shipped 0.68 Max speed selection is
an explicit selection from historical 100K evidence, not a new machine calibration.
