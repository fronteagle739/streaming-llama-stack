# E14:128-token prefix boundary snapshots

Verdict: **needs native boundary capture**. No cached-turn speedup or completed
128-token cache is claimed from the existing single rewind anchor.

The existing native capture/restore API admits one mutable anchor up to24MiB,
checks token/logit/owner hashes, and relies on the same owner's immutable
compressed/index prefix. It is a frontier rewind facility. After layer-major
prefill, old128-row raw rings and intermediate pool tails have been overwritten;
capturing more idle anchors cannot recover those earlier boundaries.

A correct implementation must capture each retained boundary while each
layer's projected rows and pooling inputs remain live. It also needs the
boundary's final residual/pre and exact head logits, Engram tail, selected
IDs/candidates, prefix identity, cache precision and owner generation.
Generating128-token full-weight sweeps to manufacture anchors would add
hundreds of fixed stream passes and is not an acceptable first-pass cost.

The bounded design admits eight snapshots, at most192MiB, inside both native
and parent host plans. Snapshots may borrow immutable prefix storage only
while its owning allocation/generation is unchanged. Different models, owner
transfer, overwritten prefixes, precision changes and arena eviction invalidate
them. Restore plus suffix execution must be one publication transaction.

Qualification must compare disabled/enabled repeated32K prompts, divergences
at127/128/129 and around the last retained boundary, changed-prefix and stale
generation rejection. Measure first-pass capture cost, hit/eviction counts,
arena bytes and cached TTFT; require exact owner/frontier/logit state and
retrieval. The existing one-anchor interface does not satisfy those tests,
so no shipping prefix-cache toggle or preset was added.
