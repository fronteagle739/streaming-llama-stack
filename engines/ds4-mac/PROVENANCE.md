# Mac ds4 provenance

The current development core and Metal kernels are rebased onto
[IngeniousIdiocy/ds4-v41-m3ultra, v41-m3ultra, f48c5e2e20b1aaae1c79735a8457225d2469ae73](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/commit/f48c5e2e20b1aaae1c79735a8457225d2469ae73).
The backend derives from [antirez/ds4](https://github.com/antirez/ds4), MIT.
Original licenses and third-party attribution remain in the source.
The previous fork's merge-base was `b10cb40c1786d93aa2b4c819635a7b9e84b2b635`;
its historical individual ports are documented in the Round4 and pre-rebase
Round5 reports at the stack root.

Local integration retains native RTX state import and cone v2, logical
model/tokenizer compatibility, bounded native ring conversion, checkpoint
cadence, anonymous wired storage and host-memory admission, projected vision
transport, KV-store pins, and the sampler/bias/tools server handler. Batched
DSpark requests use the same model ownership and inference mutex as ordinary
requests. The retained llama sampler implementation links through
`ds4_sampling_bridge.cpp` without opening a second model; DRY uses the resident
vocabulary. The final draw keeps ds4's seeded 64-bit RNG. Selection-only cache
bias retains the fully resident expert semantics; streaming-expert bias is
not implemented on this resident native-session lane.

The owner's Round6 decision retires 98 nonwinning experimental tuning controls
without upstream equivalents. Explicit selection fails loudly. The remaining
22 controls have actual upstream adapters or retained diagnostics, including
measured Markov phase profiling. No sampler, bias, tools, vision, checkpoints,
native save, cone, wired allocation, state import or Studio adapter is retired.
See `docs/MACDECODE_ROUND6.md` and its complete control manifest for names,
qualification, and capability limits. Earlier executable timings must not be
attributed to this rebased source.
