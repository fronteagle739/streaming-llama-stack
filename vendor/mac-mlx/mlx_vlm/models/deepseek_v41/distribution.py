"""CPU-readable distribution preflight; no weights or collectives are changed.

Planning is not execution support. See docs/deepseek-v41-distribution-constraint.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelinePlan:
    # Execution order, not RPC rank order (RPC rank zero owns the final stage).
    stages: tuple[tuple[int, int], ...]
    dependencies: tuple[tuple[int, int, str], ...]


def sharing_dependencies(config):
    """Trace the producers actually read by Attention._compressed_indices.

    Index selections and candidate masks have lifetimes independent of KV.
    A window-only layer does not reset any of these shared pointers.
    """
    kv_owner = key_owner = selection_owner = candidate_owner = None
    edges = []
    for layer in range(config.num_hidden_layers):
        if not config.compress_ratios[layer]:
            continue
        if layer in config.kv_source_layer_ids:
            kv_owner = layer
        if layer in config.index_source_layer_ids:
            if layer in config.kv_source_layer_ids:
                key_owner = layer
            if key_owner is None:
                raise ValueError(f"Layer {layer} has no preceding index-key owner")
            edges.append((key_owner, layer, "index keys"))
            if layer == config.candidate_source_layer_id:
                candidate_owner = layer
            elif candidate_owner is not None:
                edges.append((candidate_owner, layer, "candidate mask"))
            selection_owner = layer
        if kv_owner is None or selection_owner is None:
            raise ValueError(f"Layer {layer} has no preceding compressed KV/selection owner")
        edges.extend(((kv_owner, layer, "compressed KV"),
                      (selection_owner, layer, "selected indices")))
    return tuple(edge for edge in edges if edge[0] != edge[1])


def plan_pipeline(config, counts):
    """Validate stage counts in execution order, rejecting every remote read."""
    counts = tuple(counts)
    if not counts or any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("Pipeline counts must be positive integers")
    if sum(counts) != config.num_hidden_layers:
        raise ValueError("Pipeline counts must sum to the backbone layer count")
    edges = sharing_dependencies(config)
    stages, start = [], 0
    for count in counts:
        stop = start + count
        for source, consumer, state in edges:
            if source < stop <= consumer:
                raise ValueError(
                    f"Unsafe cut before layer {stop}: {state} from layer {source} "
                    f"is consumed by layer {consumer}"
                )
        stages.append((start, stop))
        start = stop
    return PipelinePlan(tuple(stages), edges)


def require_single_rank(group, operation):
    """Fail before slicing parameters or assigning any distributed state."""
    size, rank = group.size(), group.rank()
    if type(size) is not int or size < 1 or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("Invalid distribution group size/rank")
    if size == 1:
        return
    if operation == "pipeline":
        reason = (
            "The release has a safe 20/20 KV split, but stages must carry fp32 "
            "pre_mix alongside activations and use V4.1 conversation caches, "
            "Engram history and staggered final collapse."
        )
    else:
        reason = (
            "Attention and MoE need validated partial-output all_sum hooks "
            "before tensor weights can be sliced."
        )
    raise NotImplementedError(
        f"DeepSeek-V4.1 {operation} with {size} ranks is not implemented. {reason} "
        "See docs/deepseek-v41-distribution-constraint.md."
    )


def reject_rpc_pipeline():
    raise NotImplementedError(
        "DeepSeek-V4.1 RPC pipeline requires a typed (hidden, fp32 pre_mix) "
        "boundary and a conversation-cache stage runner; the V4 layer runner "
        "and hc_expand are incompatible. "
        "See docs/deepseek-v41-distribution-constraint.md."
    )
