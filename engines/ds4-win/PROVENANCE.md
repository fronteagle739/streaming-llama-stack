# ds4-win

Derived from a private fork at `058ae09159a18fd4d0947d26bb1a7056771f5c8f`. Minimal build and run source export. Upstream: antirez/ds4 and IngeniousIdiocy/stack-backend (MIT); Windows also derives from mishal-harbi/ds4-win-NovaStar. Upstream licenses and third-party notices are retained. No model weights or private history are included.

Live-control update: `c65136f30f8c9a5536d929165dcbb2abca2e7a6f`, five native control units, reusing the delivered token-boundary setter and authentication.

Dense-output adapter and authenticated native frozen-prefix training source: `6f6a5f6a6cd483b9179ec829674f419c164d19f0`. It supports a separate once-per-load FP16 VRAM head and preserves the quantized base mapping. Real identity/held-out/rainbow, graceful shutdown, unchanged-base full hashing and crash/restart acceptance passed on the 1017 Q4 model; see examples/train_jeff_lora/LORA_CARD.txt.

Four-phase quality provider: `035c04d2b17024132a9ee37932f7e95b72b07822`, merged with output-adapter support on the owned source branch at `8b57e80214476a367cd6b5b0aaafe82de69408cf`. Five names, range 0..1.36125, reasoning/tool_call/answer/code. Stack portability edits remain applied. Native build/live qualification is recorded separately.

The fresh-source build exposed three omitted CUDA include dependencies. They are now included byte-for-byte from that same source merge: ds4_dspark_aux_graph.cuh, ds4_cuda_dspark_lowbit.cuh and ds4_cuda_dspark_aux.cuh. Optional DSpark paths remain off by default; this inclusion does not qualify their runtime behavior.

Queued four-phase request preservation: `5ada2a7bf1ad12aebf4fc749121d49c4da78fb46`. Four independent settings survive the existing chat queue; legacy three-phase maps still inherit answer for code at parsing. Compiled production transport regression includes the failing three-slot negative control. CUDA arithmetic is unchanged.
