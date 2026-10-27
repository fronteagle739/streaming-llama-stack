# RTX prefill and Mac decode

The paired backend uses the RTX for long prompt prefill and an M3 Ultra for decode. It is a supported session handoff used through Studio. It does not divide every transformer layer between machines.

Install the stack on both hosts and configure the identical Flash 1017 Q4 model. The Mac needs enough unified memory for its native core, context and safety reserve; the measured system has 512 GB. The RTX retains its own explicit host budget and available-RAM floor.

On the Mac:

```sh
./bin/slstack install --all
./bin/slstack mac --model '<model_dir>/DeepSeek-V4.1-Flash-1017-Q4-v41.gguf' --host-budget-gib 400 --min-available-gib 8
```

Use the receiver's printed endpoint, pairing token and Windows command:

```powershell
./slstack.ps1 rtx --config my.local.json --mac <mac-endpoint> --pair-token <printed-token> --webui
```

Pairing checks model, tokenizer/renderer, engine identity, state format and memory admission. Keep tokens in local runtime configuration. Both hosts must be able to reach their announced transport endpoints. Stop foreground launchers with Ctrl-C and allow native shutdown to complete.

Initial or incremental state moves from the RTX to the Mac. Decode and small local appends update the live session; changes flow back to the RTX before another offloaded append. A strict join prevents a new RTX turn from using stale state. Short follow-ups can remain local instead of replaying a long document. Edits, reasoning-history changes and incompatible checkpoints can require rewind or fresh prefill. [Prefix reuse](PREFIX_REUSE.md)

The recorded case was measured before YOCO (plain ULTRA prefill). The 184,432-token case used 173.252 s RTX compute, 7.662 s transfer/import and 31.494 Mac decode tok/s; the full prompt interval was 229.128 s. [Timing details](PERFORMANCE.md)

Negotiated YOCO state v1 export exists. Mac decode with YOCO has not been benchmarked. Approximate YOCO transport requires explicit negotiation of compatible state support on both peers. See [media/state behavior](STACK_MEDIA_AND_STATE.md) for the YOCO format and checkpoint requirements.

The paired candidate accepts the default WebUI phase profile and explicit
sampler overrides. Toolbar changes apply on the next turn. Its Mac launcher
selects the compatible existing vision encoder beside the language model, or
uses `vision_encoder`/`ds4_vision`. Projected image spans and identities travel
over the authenticated session channel for initial and appended prompts.
These paths have CPU/browser-fixture coverage; the required live image and
long-context YOCO gate is still incomplete.
