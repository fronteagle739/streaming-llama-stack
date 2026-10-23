# Single-RTX ULTRA prefill and native decode

This candidate reuses the original ULTRA producer and the Windows ds4 decoder.
The CUDA native core has passed a real 384-vector byte-exact checkpoint reopen
and saved-frontier rewind test. ULTRA import parity, crossover, long-context
performance and normal Studio acceptance are still pending. Source and CPU
tests do not qualify the combined preset.

The versioned mode-1 contract binds the original model, native tokenizer and
template, both RoPE implementations, F32 KV, context, position, token/logit
hashes and every owner by kind/index. Mode 1 has 48 even-position owners or
54 odd-position owners. The existing mode-3 producer default is retained.

Each request uses one positive owner epoch. The coordinator exports into
budgeted locked RAM, retires the native owned Job, certifies recovered VRAM,
starts ULTRA, imports the base and prefills only the changed suffix. It then
exports, certifies producer retirement and restores the actual native decoder.
Every release requires an empty Job, a signaled process, a gone PID and
recovered physical VRAM. A failed transfer poisons the epoch; it cannot fall
back or publish stale bytes.

`combo_calibrate` reuses Kimi's crossover equation with measured end-to-end
costs. Its decision includes preparation and the first 100 decode tokens to
charge cold-cache refill. TTFT is recorded separately. Each admission adds at
most one measured pair, retaining completed receipts. Both bracket endpoints
must repeat with reversed route order before automatic routing qualifies.
Only matching model, engine, provider and workspace evidence can be reused.

Short follow-ups use new suffix size rather than total context. When the
normal reasoning-history policy requires bounded rewind, the coordinator
logs replayed history separately and uses an additional native render to
identify the new user text. That policy keeps a short follow-up native even
when its retained document is large. No second tokenizer is introduced.

`combo_measure` records actual OpenAI streaming TTFT, native 100/500/1000-token
milestones, the 500-to-1000 warm rate, individual transfer legs and exact prefix
reuse. `combo_alternating` tests six turns with native-only replay of the same
actual transcript, full-vocabulary 384-vector comparisons, document-fact
checks and both reasoning-history settings. Its TTFT explicitly includes
qualification probes; the separate performance campaign omits those probes.

All hardware commands require the shared independent `slstack.guard`, an
aggregate physical budget and available-RAM floor, and separate GPU windows
of at most 19 minutes. Qualification uses locked checkpoint/reference RAM and
the original GGUF in place. No model conversion or disk checkpoint is used.

`scripts/build_combo_runtime.py` stages an immutable runtime only after the
mandatory parity, three 50K quality settings and both six-turn campaigns pass.
It reuses Studio's actual pure native-argv builder and emits a normal preset.
Staging does not install files or mutate settings. The installed launcher
uses the normal native Bearer token and quality/speed API. It is CPU-only
while idle; each chat acquires its own independently guarded GPU window.
Native streaming bytes are relayed unchanged, with terminal metadata held
until checkpoint publication and certified owner retirement complete.

The new preset and desktop/phone GUI evidence must be produced after those
hardware receipts. No combined-preset throughput or GUI result is claimed yet.
