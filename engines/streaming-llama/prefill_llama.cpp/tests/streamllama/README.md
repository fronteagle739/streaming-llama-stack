# streamllama port acceptance tests

Tiny random-weight models of a ported arch are run in the fork and in upstream llama.cpp on the same GGUF and
prompt (64 tokens + 8 single-token decode steps), single-ubatch and multi-ubatch, flash attention off and on:

    tests/streamllama/run_acceptance.sh glm5-next build-cpu/bin <upstream-src> <upstream-build>/bin <work-dir>

(`<upstream-build>` is a CPU build of upstream's `llama` target.) `load_check.cpp` loads a real GGUF with
`no_alloc` to check tensor mapping. The fork still reads and allocates global
embedding/output tensors in that mode, so a real-model load check needs an
explicit host-memory guard.

For the standalone CMake graph tools and a guarded Windows-compatible matrix
including cleared-cache state restore, see `docs/GLM53_FLASH.md` at the stack
root. These synthetic graph tests do not qualify ULTRA streaming or ds4 decode.
