"""Seed-preserving target sampling over a single DSpark verify forward.

This is target-sample prefix matching, not vLLM's probability-ratio block
rejection sampler. Each emitted token uses the same target RNG draw and causal
logits as ordinary decoding, provided the target's batched forward is numerically
equivalent to its single-token forward. Discarded speculative rows restore the RNG to the
last emitted token. Thus acceptance may be lower than probability-ratio
rejection, but fixed-seed output is directly comparable to ordinary decoding.

Only sampling is position-wise: the transformer is called ONCE for the block.
There are no per-position eval/item/tolist calls. Stateful custom samplers must
provide their own position-aware protocol; this helper supports MLX RNG samplers.
The real checkpoint currently fails batched-logit equivalence; deployment stays
opt-in until that target-model gate passes (see docs/dsv41_s1_report.md).
"""

import mlx.core as mx

def _copy_rng_state():
    return [mx.array(state) for state in mx.random.state]


def _restore_rng_state(state):
    for i, value in enumerate(state):
        if hasattr(mx.random.state, "__setitem__"):
            mx.random.state[i] = value
        else:
            # MLX 0.32.2 exposes a thread-local _RandomState sentinel rather
            # than a list. Its current key is still a mutable array object.
            key = mx.random.state[i]
            key[:] = value


class DSparkSamplerRNG:
    """Preview target draws for drafting, then restore the target RNG unchanged.

    Common random draws improve agreement when draft and target logits are
    close. This coupling is safe specifically because acceptance emits only the
    target's own sampled prefix, not a probability-ratio rejection decision.
    """

    def draft_tokens(self, fn, *args, **kwargs):
        target = _copy_rng_state()
        try:
            result = fn(*args, **kwargs)
            mx.async_eval(result, mx.random.state)
            return result
        finally:
            _restore_rng_state(target)

    def target_sampled(self):
        pass  # The next draft call snapshots the current committed target state.


class DSparkTargetSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.states = []
        self.position = 1  # The caller already emitted target position zero.

    def __call__(self, logits):
        if logits.shape[0] != 1:
            raise ValueError("V4.1 DSpark supports one conversation")
        self.states = []
        tokens = []
        for position in range(logits.shape[1]):
            row = logits[:, position].astype(mx.float32)
            logprobs = (row if getattr(self.sampler, "expects_raw_logits", False)
                        else row - mx.logsumexp(row, axis=-1, keepdims=True))
            positioned = getattr(self.sampler, "sample_target", None)
            token = (positioned(logprobs, row_ids=[0], positions=[self.position + position])
                     if callable(positioned) else self.sampler(logprobs))
            tokens.append(token.reshape(-1))
            self.states.append(_copy_rng_state())
        return mx.stack(tokens, axis=1)

    def commit(self, emitted):
        if not 1 <= emitted <= len(self.states):
            raise ValueError("Invalid number of committed target samples")
        _restore_rng_state(self.states[emitted - 1])
        self.position += emitted

    def commit_prefix(self, emitted, previously_committed):
        """Undo unconsumed draws if the streaming generator closes mid-block."""
        self.position -= previously_committed
        self.commit(emitted)
