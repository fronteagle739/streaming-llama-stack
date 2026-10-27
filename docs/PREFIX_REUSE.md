# Follow-up prefix reuse

Append-only turns reuse the live decoded frontier and prefill only new tokens. The two-process single-RTX coordinator and paired session transport also validate saved state against the new conversation render.

## Sampled and canonical tokens

The model's sampled token IDs and a fresh BPE tokenization can represent identical text with different splits. Comparing only IDs would discard a valid prefix and re-prefill the earlier answer.

The native render endpoint therefore uses three outcomes:

- `memory-token`: the live IDs prefix the rendered tokens, so reuse them directly.
- `memory-text`: live token bytes prefix the new rendered text; retain those IDs and tokenize only the suffix.
- `canonical-miss`: the text also diverges; use canonical tokens and report the divergence.

Routing uses actual new-prefill tokens. The expert cache is restored before native prefill, and eviction-protected VRAM slots are cleared before seeding.

## History changes and rewind

Turning off previous reasoning changes rendered history before the latest frontier. If neither checkpoint nor anchor matches, the renderer reports `canonical-miss`, and the divergence is older than every saved frontier, the coordinator releases the current native process and prefills the complete new render.

`DS4COMBO_FRESH_PREFILL` and `metrics.fresh_prefill` expose the reason, divergence position, both frontiers and surrounding token IDs. A mismatch at or after an available anchor still refuses inconsistent history. The fallback costs a native restart plus full prefill on that turn.

A per-turn anchor journal could limit replay to the changed suffix. The current adapter shares its single journal between anchor and rewind; retaining multiple frontiers needs extra engine support and locked RAM.
