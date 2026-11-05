# Mac ds4 provenance

The current development core and Metal kernels are rebased onto
[IngeniousIdiocy/ds4-v41-m3ultra, v41-m3ultra, f48c5e2e20b1aaae1c79735a8457225d2469ae73](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/commit/f48c5e2e20b1aaae1c79735a8457225d2469ae73).
The backend derives from [antirez/ds4](https://github.com/antirez/ds4), MIT.
Original licenses and third-party attribution remain in the source.

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

Explicit tuning controls without an implementation fail clearly. Native
samplers, tools, images, checkpoints, state import, wired allocation and paired
serving retain their documented implementations. See ds4-server --help for the
supported native controls and their limits.
