# RTX prefill and a large-RAM decode box — UNTESTED example

Use a fast RTX machine for streaming-llama ULTRA + YOCO prefill, then keep the
model in RAM on a second Windows/Linux machine running ds4-win decode. A DDR4
server is one possible host. **UNTESTED example: the required CPU receiver is
missing. The launcher prints a proposal and refuses to start either machine.**

The RTX uses the same streaming producer as the single-engine route, with the
existing paired state export enabled. It exports one completed mode3 state,
including the DSpark ring and YOCO v1 provenance. The decode box would import
that state and retain the session for replies and short follow-ups. This reuses
the [RTX-to-Mac design](../../../docs/YOCO_PAIRED_QUALIFICATION.md); no second
export format or conversion is proposed.

## Hardware and memory

- RTX: Windows, RTX 50-series and fast DirectStorage NVMe, as in the existing
  single-machine setup. The example keeps the paired 8.5 GiB outer RAM budget,
  6.25–7.75 GiB producer allowance, 512 MiB admission reserve and 4 GiB runtime
  free-RAM floor.
- Decode: enough physical RAM for the complete GGUF, session state, runtime
  and operating system. The measured Q4/Q8 file alone is 483 GiB. A 768 GiB
  machine with at least 604 GiB currently available is an illustrative starting
  point for the 600 GiB host budget and 4 GiB free-RAM floor in this proposal.
  Size for the actual model and context; these are not measured requirements.
- Both machines need identical model/tokenizer identities and compatible state
  layouts. No decode speed, RAM residency or Linux operation is qualified here.
  The RTX producer remains Windows-only because it uses DirectStorage.

## Inspect the configuration

Copy [config.example.json](config.example.json) to `large-ram.local.json`. Set
both model paths, the decode address and a shared private pairing token. `rtx`
uses existing broker config keys; its `mac` key is the existing peer URL field.
`decode_plan` contains proposed receiver settings, **not ds4 CLI options**.
Choose a budget that fits available physical RAM, with no page-file allowance.
A future receiver must admit and monitor that budget and free-RAM floor before
loading weights; this example does not load them.

From the repository root, on either machine with Python 3.12+:

```sh
python -B examples/dual-machine/dsv41-large-ram-untested/launch.py --machine rtx --config large-ram.local.json --dry-run
python -B examples/dual-machine/dsv41-large-ram-untested/launch.py --machine decode --config large-ram.local.json --dry-run
```

The RTX preview shows the existing broker argv and export settings. It never
writes the effective config mentioned in that argv. The decode preview has no
argv because there is no compatible CPU receiver to invoke. Omitting
`--dry-run` exits with an error before reading config or starting a process.

## Existing import and the remaining gap

ds4-win already has native session import:
[`ds4_session_load_payload` and `ds41_load_payload`](../../../engines/ds4-win/ds4.c)
accept V4.1 GPU snapshots with mode 1 (target only) or mode 2 (draft). Its
[`ds4_session_native_restore`](../../../engines/ds4-win/ds4_native_session.inc)
adapter requires a local CUDA V4.1 graph and mode1 geometry.

The qualified Mac path instead accepts mode3 DSpark capture, negotiates YOCO v1,
and verifies the authenticated state contract through its
[native session transport](../../../engines/ds4-mac/ds4_native_session_transport.inc).
The RTX [existing argument builder](../../../slstack/combo_producer.py)
enables `--capture-dspark 1` and `--yoco-remote-state-version 1` for that export.

The Windows/Linux CPU RAM-resident decode backend does not implement this
mode3/YOCO receiver and broker integration. Existing ds4-win import is therefore
insufficient for this recipe. Switching the Mac launcher to `--cpu`, or sending
the export to the earlier DS4P listener, does not complete this route. This
release documents the gap; it does not port the receiver or claim a working pair.
