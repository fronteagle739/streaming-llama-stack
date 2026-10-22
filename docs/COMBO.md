# Single-RTX ULTRA prefill and native decode

STATUS: IN PROGRESS. Source/CPU checkpoint only. GPU state parity, end-to-end
measurements and Studio acceptance are required before enabling a normal preset.

The Windows C adapter reuses the existing native span/delta codec in target-only
mode1. Owners are identified by kind and index:48 at even positions and54 at
odd positions. It provides bounded visit, delta apply, full restore and mutable
frontier rewind. A failed partial import remains invalid. Canonical state is
F32; model weights remain the original1017 GGUF.

The producer can consume a pinned, versioned native checkpoint over its existing
private pipe before binding the next executor. Mode1 commands and replies bind
the same positive owner epoch. Legacy mode3 is the unchanged default. The native
server adds authenticated `/decode/setup` ownership and control over the same
public state adapter. Bulk state lives in a parent-owned anonymous RAM section,
physically locked until transfer completion; it is never a checkpoint file.

The checkpoint contract validates model, tokenizer, built-in template, compiled
RoPE inputs, KV dtype, context, position, request/owner epoch, generation, token
and logit hashes and every owner hash. All state transfers use64KiB pages. Native
and producer processes must retire with an empty Job, signaled process and gone
PID before the next GPU owner can start. Handoff refusal is visible.

CPU evidence so far:1,070 valid planner cases /432 refusals against the independent
C++ producer planner;12 complete mode1/mode3 packets byte-exact; actual C adapter
host fixture for restore/delta/rewind/corruption/cancellation;13 new and legacy
contract checks; shared locked-RAM checks including a separate reader process;
state-byte corruption checks; four native Job/epoch startup/failure checks; and
the existing GPU admission/environment checks. These are not model parity proof.

The native source incorporates the frozen035c04d decoder delivery, preserving the
stack's existing output-adapter and frozen-prefix training changes. All native
CUDA units and all producer C++ units build. The producer reuses SHA-verified
existing backend libraries; this is not a claim of a new backend-kernel build.

Pending acceptance: native roundtrip and rewind byte equality for384 complete
logit vectors; native-prefill versus ULTRA-prefill/import KL/top1 for384 steps;
measured crossover and50K/reused-prefix TTFT, switch and decode rates; then the
new Studio preset, live quality/speed tool, two-turn TEST chat and screenshot.
