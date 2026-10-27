# Two machines: RTX prefill, Mac decode

The RTX 5090 does what it is fast at: streaming prefill of long prompts. The Mac
Studio does what it is fast at: decode with the whole model in 512 GB of unified
memory. They share one live session.

## Speed

DeepSeek V4.1 Flash, RTX 5090 desktop plus M3 Ultra (512 GB), measured in the
stack's web UI:

| Prompt | YOCO | Prompt time | Prefill tok/s | Mac decode tok/s |
|---|---|---:|---:|---:|
| 198,287 tokens with an image | On | 118.8 s | 1,670 | 31.0 |
| 49,787 tokens with an image | On | 33.7 s | 1,477 | 31.8 |
| 49,787 tokens with an image | Off | 41.6 s | 1,197 | — |

The 198K image prompt runs 1.69× faster than on one machine without YOCO
(986 tok/s). The handoff itself adds about 3.4 s.
[Phase-by-phase timing](YOCO_PAIRED_QUALIFICATION.md).

## How a session flows

1. The Mac receives the whole prompt.
2. Under 11,776 new tokens, the Mac prefills it itself (about 350–390 tok/s).
   No round trip.
3. Longer prompts go to the RTX. It runs the same prefill as the single-machine
   setup, then sends the finished state to the Mac once.
4. The Mac decodes and keeps the session. Short follow-ups stay on the Mac;
   long additions go back to the RTX.

Images travel with the prompt. Pairing checks that both sides run the same model,
tokenizer and engine versions.

## Memory

The RTX side has an 8.5 GiB RAM budget and peaked at 6.3 GiB. The Mac runs with
a 400 GiB budget.

## Setup

Install on both machines and use the same model file. On the Mac:

```sh
./bin/slstack install --all
./bin/slstack mac --model '<model_dir>/<deepseek-v4.1-flash>.gguf' --host-budget-gib 400 --min-available-gib 0.5
```

The Mac prints its endpoint, a pairing token and the Windows command. On Windows:

```powershell
./slstack.ps1 rtx --config my.local.json --mac <mac-endpoint> --pair-token <printed-token> --webui
```

Keep the pairing token in local configuration. Stop with Ctrl-C.

Follow-ups reuse the session; edits to earlier messages rewind it.
[Prefix reuse](PREFIX_REUSE.md) · [State transfer and images](STACK_MEDIA_AND_STATE.md)

## Other models

The handoff is general. Kimi K2.7 and GLM-5.1/5.2/5.3 use RTX prefill with an MLX
receiver on the Mac. Presets are in
[examples/dual-machine](../examples/dual-machine/README.md).

## Planned: a large-RAM decode box

An example configuration for an RTX prefill machine paired with a large-RAM
second machine (for example a DDR4 server) running ds4-win decode is planned.
