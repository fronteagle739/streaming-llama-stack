# Two-process single-RTX coordinator

This advanced coordinator alternates a streaming ULTRA producer and a native Windows ds4 decoder on one GPU. For ordinary chat use the [single-engine launcher](../README.md#quickstart), which runs prefill and decode in one producer process.

## State contract

The mode-1 adapter provides bounded visit, delta apply, full restore and mutable-frontier rewind. Owners are state regions identified by kind and index: 48 at even positions and 54 at odd positions. Mode 3 remains the producer's separate session format.

The contract binds the original model, native tokenizer/template, RoPE inputs, F32 KV, context, position, epoch, generation, token/logit hashes and every region hash. Transfers use 64 KiB pages in budgeted locked RAM. A failed import invalidates the epoch and cannot publish stale bytes.

Before switching, the coordinator requires an empty process Job, a signaled process, a retired PID and recovered physical VRAM. It restores the expert cache and uses the real new suffix size for follow-up routing. [Prefix reuse](PREFIX_REUSE.md)

## Calibration and inspection

`combo_calibrate` compares end-to-end routes, including preparation and the first 100 decoded tokens. TTFT is separate from steady decode speed. Calibration binds model, binaries, provider and workspace identity; it cannot be reused for mismatched inputs.

`combo_measure` records streaming TTFT, 100/500/1000-token milestones, warm decode windows, transfer legs and prefix reuse. `combo_alternating` compares a six-turn transcript with native-only replay and complete logit vectors. These tools require explicit budgets and independent GPU admission. Automatic routing requires a compatible measured crossover rather than a guessed threshold.

The immutable-runtime builder stages matching executable dependencies after its parity and conversation checks. Staging itself does not install a preset. Quality controls use authenticated native endpoints and preserve independent reasoning, tool, answer and code settings.
