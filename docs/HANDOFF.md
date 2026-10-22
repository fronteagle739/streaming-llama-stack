# Windows ULTRA crossover gate (phase 4 is not implemented)

Dependencies inspected: DS4WINBIAS2_RESULT.md, section "One RTX preset:
per-message ultra prefill and native decode"; pinned streaming engine c5ee9ec9
and native engine 058ae091. Existing Kimi crossover and `/decode/setup` provide
ownership patterns, not a compatible DeepSeek V4 state receiver.

The current span/delta/rewind adapter requires Metal, mode3, a DSpark frontier.
Windows native target-only mode1 has 48/54 owners; mode3 has 49/55. A new target
adapter must map state by kind/index, not by positional shifting. Removing a
Metal guard is insufficient. The router refuses `long_prompt=true` until this
adapter exists and the following qualification passes:

1. Validate model, tokenizer/template, RoPE, precision, KV >= 8 bit, context,
   generation epoch, position and exact prefix/state hashes before mutation.
2. Restore history-off rewind before counting newly required prompt tokens.
3. Admit a single RAM checkpoint plus bounded delta/ring journal, incoming tokens
   and logits against actual free physical memory minus the guard floor.
4. Bootstrap base0 through native full snapshots; use acknowledged chunks <=49152
   tokens and bounded 64 KiB transfers. Reject partial/stale imports.
5. Export native, certify its GPU allocator released, run original-weight ULTRA,
   certify producer released, reload native and validate before publishing state.
6. Demonstrate next-logit/top1/KL behavior against identical native-only input;
   short native path, ~50K crossover, second-turn prefix reuse, history on/off,
   stale epoch, cancellation and failed-import fallback all need real proof.
7. Choose crossover with repeated end-to-end timings including both reloads,
   checkpoint/import/export and incremental cold-cache refill. Keep bias0 and
   adaptive/confidence/promotion off for initial parity. Add hysteresis.

No fabricated crossover, logit parity or second-turn reuse is reported by this
repository. Missing phase 4 does not qualify phases 2/3 as complete.
