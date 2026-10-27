# Mac ds4 provenance

Derived from [IngeniousIdiocy/ds4-v41-m3ultra at `b10cb40c1786d93aa2b4c819635a7b9e84b2b635`](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/commit/b10cb40c1786d93aa2b4c819635a7b9e84b2b635), the merge-base of the exported Mac source and the upstream V4.1 branch, MIT. The backend itself derives from [antirez/ds4](https://github.com/antirez/ds4). The historical `IngeniousIdiocy/stack-backend` name could not be resolved to a separate verifiable base.

Changes: bounded direct reads into anonymous wired core/tensor/KV storage, host-memory admission, native RTX session transfer with negotiated decode deltas, checkpoint/frontier handling and portable stack integration. Original licences and third-party attribution remain in the engine source.

Paired WebUI sampling links the retained Streaming llama `llama-sampler.cpp`
implementations through `ds4_sampling_bridge.cpp`; it does not open a second
model. DRY uses the resident ds4 vocabulary's token pieces. Phase/code tracking
is shared with the retained Windows ds4 implementation. The final categorical
draw retains ds4's64-bit seeded RNG. The Mac MiMo selection-only cache-bias
policy is preserved for the admitted fully resident expert set: there are no
uncached candidates to penalize, so nonzero bias leaves its routing unchanged.
Streaming-expert bias on this DSV4.1 native-session lane is not implemented.
