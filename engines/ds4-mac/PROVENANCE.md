# ds4-mac

Derived from a private fork at `37e261d984a82e7c0446165e9fd976887338475f`. Minimal build and run source export. Upstream: antirez/ds4 and IngeniousIdiocy/stack-backend (MIT); Windows also derives from mishal-harbi/ds4-win-NovaStar. Upstream licenses and third-party notices are retained. No model weights or private history are included.

Owned anonymous/wired core, tensor/KV and common scratch allocation candidate: `8076c7317c5050c079c4df6edf471750b2c097e5`. Allocation ownership follows the native MiMo wiring pattern at `c427e423702145aeb8b9e14d05f1c60cee31d9b9`; no shader math or native state format changes. The stack requires it for native Mac inference. Real region/copy/budget tests and full-model residency/parity are distinct qualification gates.

Full-model startup now reads the registered model descriptors directly into owned wired storage in bounded chunks, avoiding a second resident file mapping. Wired views are capped at 64 GiB with the existing tensor overlap preserved. EOF page padding matches mmap semantics; failed reads, unregistered descriptors and memory admission failures refuse startup. No tensor arithmetic or state encoding changes.

Stack-only native build passed in 10.39 s (peak RSS 1,034,469,376 B). The owned/wired variant rebuilt in 3.77 s (454,787,072 B). Its real Metal fixture passed exact copying, anonymous region identity, positive OS wired count, zero swapped region pages, release, and insufficient-budget refusal. These small-allocation results do not qualify full-model residency or parity.

Complete standalone native-session inclusion and source closure, matching this export: `0710258ca185c4804295d882a1389104ba2c9571`. This keeps the private comparison build on an owned branch and uses the same memory protection.

The bounded file-read factory passed an actual 96 MiB offset/chunk-boundary test, anonymous wired-region inspection, EOF padding, refusal and release checks. The receiver rebuilt in 9.59 s (peak RSS 802,816,000 B), under 2 GiB RSS and a 32 GiB physical reserve. See `examples/engine_environment/mac-wired-file.json`.

The full owned-wired model subsequently loaded and completed its native smoke request at one-million-token context. Peak native RSS was 353,347,649,536 B under the 384 GiB cap, with at least 60,210,151,424 B available against a 32 GiB reserve. The enclosing native process ran for 56.91 s, including startup, the request and certified teardown; startup alone was not separately timed. See `examples/engine_environment/mac-full-startup.json`. Pairing stopped at the subsequent memory admission after teardown; paired parity, long-context inference and second-turn reuse remain pending.
