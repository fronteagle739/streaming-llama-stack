# GLM-5.3 Flash: experimental graph port

The retained streaming-llama sources now recognize `glm5-next`, including the
ds4 GGUF layout. **The single-machine ULTRA-prefill/ds4-decode engine is not
implemented or qualified for this architecture.** The stack refuses to pass a
GLM file through its DSV4.1 single-engine runtime.

The expected original model is `GLM-5.3-Flash-Q4_K.gguf`, 190,875,526,464 bytes,
SHA-256 `c7a0d950363238dd7804782c88340d737775aba53a15f8d4fdcc34e984f25221`.
This identifies the historical receipt; it does not certify a current model file.
No weights, SDK libraries, generated GGUFs or binaries are committed.

## Reused source

The paused GLM port contributes the hybrid attention/recurrent/indexer memory,
KDA/MLA model graph, loader aliases, mHC operations and gated delta-net kernels.
The CUDA launch compatibility shim is retained. Existing DSV4.1 operation IDs
remain unchanged; the three new operations are appended.

The new integration synchronizes indexer and attention cache dimensions when
reserving a layer-major graph. The CPU build also exposes two existing assert
bugs: an unavailable power-of-two helper and reading the sequence count of an
empty KV cell. These are fixed without disabling assertions.

The graph tools cover full/shared indexers, the upstream and ds4 tensor naming
layouts, 64/320-token prompts, 64/16-token microbatches, and eight continuation
tokens. State restore clears the original caches before restoring, exercising
the indexer and recurrent state as well as attention. Tests use F32 KV.
An independent scalar HC check covers the legacy weighted pre operation, the
new gated pre operation and identity post mixing with both four and twelve
residual streams.

## Reproduce CPU checks

Configure the retained llama CMake project with `LLAMA_BUILD_GLM53_TESTS=ON`,
`LLAMA_BUILD_COMMON=OFF`, `LLAMA_BUILD_TOOLS=OFF`, `LLAMA_BUILD_TESTS=OFF`,
`GGML_CUDA=OFF` and `BUILD_SHARED_LIBS=OFF`; build `glm53-logits` and
`glm53-load-check` and `glm53-hc-check`. On Windows the retained optional DirectStorage headers
require a locally provided SDK import library. Run all builds through the stack
guardian with explicit host and commit limits.

Using an existing local `gguf` Python package and NumPy, generate tiny fixtures:

```sh
python -B tests/streamllama/glm5-next/make_tiny_glm5next.py "$FIXTURES"
```

From the retained llama source directory, run the checked-in matrix:

```sh
python -B tests/streamllama/run_glm53_cpu.py --config "$CONFIG" \
  --models "$FIXTURES" --binary "$CPU_LOGITS_BINARY" --output "$RESULTS"
```

`CONFIG` is a stack config. Keep its runtime, temporary files and results inside
one task scratch root; set equal host/commit caps and a physical-RAM floor.
If a shared engine hold ledger is used, supply it as `hold_file`. The runner
yields to active holds. It performs CPU checks and acquires no GPU lease.

The original `run_acceptance.sh` separately supports comparisons against an
already available upstream CPU build. Same-fork layout/microbatch checks do
not replace that independent comparison or real-model acceptance.

## Remaining single-engine work

1. Generalize the ULTRA producer's model receipt, sweep and weight provider.
   Its carry sizing, Engram handling and state views currently describe V4.1.
   Keep DirectStorage lookup tied to original GGUF offsets after tensor aliases.
2. Extend ds4's external decode adapter with the existing GLM graph kernels.
   Bind KDA convolution/recurrent state, MLA latents, pooled indexer keys and
   raw key/gate tails in the correct layouts, preserving ownership across turns.
3. Use GLM's tokenizer, chat template, tool parser and actual stop IDs. Reuse
   the process/session controller after these architecture-specific contracts
   are supplied. Do not use V4.1's hard-coded EOS, renderer or checkpoint ABI.
4. Derive fold/sweep memory admission from GLM shapes. Preserve the 512 MiB
   desktop reserve, 4 GiB runtime floor, explicit host cap and progress guards.
   Qualify quality/speed calibration for this model. V4.1 YOCO is unavailable
   until an architecture-specific equivalence proof exists.
5. Verify the real file SHA, then qualify a coherent multi-turn WebUI chat,
   roughly 16K/50K prefill and warm decode on fast storage. Only then publish
   262K and measured maximum-context WebUI/Studio presets.

Two-machine serving is outside this change. No presets are installed, no
production services are restarted, and no benchmark numbers are inferred from
the synthetic models.
