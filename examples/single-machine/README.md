# Single RTX machine: DSV4.1

Each quality setting has an API-only launcher and a `-webui` launcher. Both run
the same single engine: streaming-llama prefill followed by modified ds4 decode.

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

Short resident follow-ups are off by default because they change logits. Set
`env_overrides.LLAMADECODE_APPEND_DECODE_MAX` to `"96"` in your private config
to enable the measured faster path. [Measurements and other toggles](../../docs/MERGEPERF.md).
