# ds4-win

Derived from a private fork at `058ae09159a18fd4d0947d26bb1a7056771f5c8f`. Minimal build and run source export. Upstream: antirez/ds4 and IngeniousIdiocy/stack-backend (MIT); Windows also derives from mishal-harbi/ds4-win-NovaStar. Upstream licenses and third-party notices are retained. No model weights or private history are included.

Live-control update: `c65136f30f8c9a5536d929165dcbb2abca2e7a6f`, five native control units, reusing the delivered token-boundary setter and authentication.

Dense-output adapter and authenticated native frozen-prefix training source: `6f6a5f6a6cd483b9179ec829674f419c164d19f0`. It supports a separate once-per-load FP16 VRAM head and preserves the quantized base mapping. Real identity/held-out/rainbow, graceful shutdown, unchanged-base full hashing and crash/restart acceptance passed on the 1017 Q4 model; see examples/train_jeff_lora/LORA_CARD.txt.

Fork merge: the owner fork range c65136f30f8c9a5536d929165dcbb2abca2e7a6f..e1e45374ddc307c75bd7fc0e4b8dccce6b0af6e2 was ported by a recorded 3-way merge (see the DS4COMBO fork-port commit), on top of the DS4COMBO growable-context and layer-prefill work.
