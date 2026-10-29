# Paired launchers

The runnable Mac preset directories contain `mac-big-memory.sh`, `rtx-fast-gpu.ps1`, and
`preset.json`. Scripts resolve the stack relative to themselves. Private model
paths, interface addresses and tokens belong in separate local config files;
they override the recipe's placeholder settings. Dry runs list unresolved fields
and start nothing. Normal launches refuse unresolved placeholders.

| Preset directories | Route |
| --- | --- |
| `dsv41-accurate`, `dsv41-fast`, `dsv41-ultra`, `dsv41-ultra-yoco` | Original DSV4.1 GGUF: RTX streaming producer and native Mac ds4 state receiver. The selected directory fixes the RTX prefill tier and YOCO choice. |
| `dsv4-flash-peer-streaming`, `dsv4-flash-153g-peer-streaming` | Earlier DS4P peer protocol: RTX streams GGUF weights and prefills, Mac decodes. These use `ds4-peer`, distinct from the DSV4.1 native state broker. |
| `kimi-k2.7-streaming` | Studio's optimised Kimi K2.7 streaming settings: RTX GGUF prefill and Mac MLX receiver, 65,536 RTX / 262,144 Mac configured context, 24,576 batch, 2,048 microbatch, F16 KV, MTP off. |
| `kimi-k2.7-standard-streaming` | Standard Kimi K2.7 settings, using the same weights and receiver with a 512-token RTX microbatch. Replaces the retired Kimi K2.6 weight selection. |
| `glm-5.1-streaming`, `glm-5.2-streaming`, `glm-5.3-streaming` | Full GLM DSA route: RTX GGUF prefill and Mac MLX receiver, 262,144 configured context, full sparse DSA, 512 microbatch, F16 KV, MTP off. |

The `dsv41-ultra-yoco` route is integrated and measured with negotiated YOCO v1
and mode3 state. Round 4 reached 1,670 prefill tok/s on a 198,287-token image
prompt and 31.0 Mac decode tok/s. The matched 49,787-token image prompt reached
1,477 prefill tok/s with YOCO and 1,197 without it.
[Timings and conditions](../../docs/YOCO_PAIRED_QUALIFICATION.md).

Paired presets retain YOCO keep 140. The single-machine exact cone (keep 2541)
has no negotiated remote state version: its export is refused before launch.
The same gap applies to the large-RAM preview. [Defaults and limits](../../docs/MERGEPERF.md).

The [large-RAM ds4-win untested example](dsv41-large-ram-untested/README.md)
contains a portable preview launcher and config pair for a Windows/Linux
RAM-rich decode host. Its missing CPU mode3/YOCO receiver is documented;
the launcher refuses to start either half.

The GLM recipes share the retained full-DSA route and current Studio streaming
defaults; they are not separate new benchmark qualifications. GLM 5.3 Flash uses
a different architecture and is not represented by the full GLM 5.3 recipe.
The Kimi recipe copies configuration, not a newly verified 700/30 tok/s result.
See `docs/BACKEND_SOURCE_AUDIT.md` for measured results and remaining backend limits.

For Kimi or full GLM, a minimal RTX private config contains:

```json
{
  "model": "/path/to/first-model-shard.gguf",
  "general": {"mac": {"host": "MAC_INTERFACE_ADDRESS"}}
}
```

The matching Mac private config contains:

```json
{
  "decode": {
    "model_path": "/path/to/compatible-mlx-model",
    "bind_address": "MAC_INTERFACE_ADDRESS"
  },
  "mac_host_budget_gib": 400
}
```

Replace every placeholder. The example 400 GiB Mac budget requires a machine
with sufficient available memory, such as the measured 512 GB M3 Ultra host;
choose a suitable budget and model for other hardware. GGUF/MLX model families,
tokenizers, decoder layer counts and KV layouts must match. The RTX launcher reads
the GGUF's architecture and layer count, including separate MTP layer accounting.

On the Mac, install Python 3.12+, the Xcode/Metal toolchain, and run from the root:

```sh
sh examples/dual-machine/kimi-k2.7-streaming/mac-big-memory.sh --config mac.local.json --setup
sh examples/dual-machine/kimi-k2.7-streaming/mac-big-memory.sh --config mac.local.json
```

`--setup --wheelhouse /path/to/wheels` is available offline. The retained customised
MLX package and native GLM kernels are installed into the selected runtime's decode
environment. Use the same runtime for setup and serving. Changing a backend to
oMLX requires its separate environment and setup; an HTTP backend does its own
prefill and cannot be substituted for the KV receiver in this streaming recipe.

On the RTX machine, build/install its streaming producer and run:

```powershell
.\examples\dual-machine\kimi-k2.7-streaming\rtx-fast-gpu.ps1 -Config .\rtx.local.json -DryRun
.\examples\dual-machine\kimi-k2.7-streaming\rtx-fast-gpu.ps1 -Config .\rtx.local.json
```

The RTX half serves WebUI/API on the configured loopback port. Open that URL in
the RTX browser. Allow the Mac receiver ports 9100/9101 on the chosen trusted
interface and start its receiver before submitting a prompt.

For DSV4.1 native pairing, use the corresponding directory and separate configs.
The RTX file needs `model`, `mac` (the Mac receiver's HTTP URL, normally port 8180),
and `pair_token`. The Mac file needs `model`, `mtp_model` (matching DSpark support
GGUF), and the same private `pair_token`. Use a random 16–128 character URL-safe
token. `--setup` builds the retained native Mac engine. Its original-model preset
checks the wired memory limit and prints the required system setting if needed.
Neither script changes that system setting automatically. This route uses the
existing authenticated state broker rather than the generic MLX KV receiver.

The DS4P examples require a matching Mac GGUF path in both private configs:
`general.mac.model_path` on RTX and `decode.model_path` on Mac. Start the Mac
listener first and permit its peer port 49173 on the selected interface. Its
HTTP endpoint stays on loopback. RTX uses `--peer-role decode`, two cache budgets
and no batched-session flag, matching Studio's peer launch path. The two recipes
differ in weight selection; choose compatible quantizations and adjust the host
budget for the selected GGUF. The Studio preset with an optional small fast
model uses this same large-model peer path; its separate automatic fast-model
switch is not implemented by these examples.
