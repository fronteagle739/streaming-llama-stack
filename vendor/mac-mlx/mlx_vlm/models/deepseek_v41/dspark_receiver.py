"""Opt-in adapter for Studio's Mac-local text prefill/decode receiver.

Enable with preset mac.env.DSV41_DSPARK=1; 0 retains ordinary decoding.
The receiver supplies its existing sampler and streaming emitter. Resident
weights are shared; classic MTP mode is never used.
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

from ...speculative.dspark import _dspark_rounds, dspark_context_from_captures


def enabled():
    if os.environ.get("DSV41_DSPARK", "0") != "1":
        return False
    # Optional local diagnostic control permits A/B on the same loaded weights.
    # The preset's off switch always wins. No network request controls this file.
    control = os.environ.get("DSV41_DSPARK_MODE_FILE")
    return not control or bool(json.loads(Path(control).read_text())["enabled"])


class _ReceiverSampler:
    expects_raw_logits = True

    def __init__(self, sampler):
        self.sampler = sampler
        # The receiver sampler is stateless apart from its numeric profile.
        # Reuse the trace across requests with identical sampling settings.
        self.dspark_compile_key = ("studio", tuple(sorted(sampler.profile.items())))

    def __call__(self, logits):
        if self.sampler.greedy:
            return mx.argmax(logits, axis=-1)
        # The receiver's sampler accepts 2-D rows and drops sequence dimensions.
        # DSparkTargetSampler supplies one row per target position for sampling.
        return self.sampler.sample_array(logits)


class DSparkReceiver:
    def __init__(self, language_model, sampler):
        from .dspark_sampling import _copy_rng_state

        self._rng_before = (_copy_rng_state()
                            if os.environ.get("DSV41_DSPARK_STATS") else None)
        self.model = language_model
        self.drafter = language_model.make_dspark_drafter()
        if os.environ.get("DSV41_DSPARK_TIMING", "0") == "1":
            from ...speculative.dspark_timing import DSparkTiming
            self.drafter.phase_timing = DSparkTiming()
        if not getattr(language_model, "_dspark_materialized", False):
            # Studio's existing loader explicitly materializes backbone layers.
            # Finish the restored draft expert stacks on CPU before generation.
            with mx.stream(mx.cpu):
                mx.eval(language_model.dspark.parameters())
            language_model._dspark_materialized = True
        self._rng_after_materialize = (_copy_rng_state()
                                      if self._rng_before is not None else None)
        self.sampler = _ReceiverSampler(sampler)
        self.greedy = sampler.greedy
        self.hidden = None
        self.forward_start = language_model._dspark_verify_calls
        self.shape_start = dict(language_model._dspark_verify_shapes)

    def capture(self, ids, cache, skip_logits=False):
        if not cache[0].is_trimmable():
            raise ValueError("DSpark requires rollback-capable caches; set DSV41_BOUNDED_CACHE=0")
        out = self.model(ids, cache=cache, capture_layer_ids=self.drafter.capture_layer_ids,
                         skip_logits=skip_logits)
        h = dspark_context_from_captures(out.hidden_states)
        captured = int(h.shape[1])
        if self.hidden is not None:
            h = mx.concatenate([self.hidden, h], axis=1)
        self.hidden = h[:, -self.drafter.window_size:]
        if int(ids.shape[1]) > 1:
            # Say what the drafter was ACTUALLY warmed with, every prefill. Under
            # DSV41_PREFILL_CONE the target produces DSV41_PREFILL_CONE_TAIL rows, not
            # one per prompt position, and a history length that short must be visible
            # in the log rather than implied by a switch somewhere else.
            sys.stderr.write(
                f"[DSV41] DSpark warm: {captured} capture row(s) from "
                f"{int(ids.shape[1])} prompt token(s), history now "
                f"{int(self.hidden.shape[1])} of window {self.drafter.window_size}\n")
            sys.stderr.flush()
        return out

    def tokens(self, cache, first, max_tokens):
        if max_tokens <= 0:
            return
        if self.hidden is None:
            raise RuntimeError("DSpark requires captures from local prefill")
        control = os.environ.get("DSV41_DSPARK_MODE_FILE")
        if control and json.loads(Path(control).read_text()).get("audit", False):
            self.audit(cache, first)
        yield first
        rounds = _dspark_rounds(self.model, self.drafter, cache, self.hidden,
                               first_bonus=first, max_tokens=max_tokens,
                               sampler=self.sampler, greedy_sampling=self.greedy,
                               draft_block_size=self.model.config.dspark_block_size)
        try:
            for token, _ in rounds:
                yield token
        finally:
            rounds.close()

    def audit(self, cache, first):
        """Separate diagnostic: teacher-force one proposal batched vs serial.

        This deliberately incurs six extra target calls and is never a timing
        run. Fresh independent cache storage prevents modifying the live cache.
        """
        from .dspark_sampling import _copy_rng_state, _restore_rng_state

        before = _copy_rng_state()
        try:
            proposal = self.drafter.draft_block(mx.array([first]), self.hidden,
                cache[0].offset, sampler=self.sampler, greedy=self.greedy,
                block_size=self.model.config.dspark_block_size)
            ids = mx.concatenate([mx.array([[first]]),proposal['draft_tokens']],axis=1)
            state = [mx.array(x) for x in cache[0].state]
            parallel, serial = self.model.make_cache(), self.model.make_cache()
            parallel[0].state = state
            serial[0].state = state
            out = self.model(ids,cache=parallel,capture_layer_ids=self.drafter.capture_layer_ids)
            logits = []
            captures = []
            for i in range(ids.shape[1]):
                row = self.model(ids[:,i:i+1],cache=serial,capture_layer_ids=self.drafter.capture_layer_ids)
                logits.append(row.logits)
                captures.append(dspark_context_from_captures(row.hidden_states))
            reference = mx.concatenate(logits,axis=1)
            actual_h = dspark_context_from_captures(out.hidden_states)
            reference_h = mx.concatenate(captures,axis=1)
            mx.eval(out.logits,reference,actual_h,reference_h)
            mx.synchronize()
            result = dict(diagnostic_only=True, verify_shape=list(ids.shape),
                logits_max_abs=float(mx.max(mx.abs(out.logits-reference)).item()),
                logits_norm_ratio=float((mx.linalg.norm(out.logits)/mx.linalg.norm(reference)).item()),
                capture_max_abs=float(mx.max(mx.abs(actual_h.astype(mx.float32)-reference_h.astype(mx.float32))).item()),
                capture_norm_ratio=float((mx.linalg.norm(actual_h.astype(mx.float32))/mx.linalg.norm(reference_h.astype(mx.float32))).item()),
                argmax_equal=bool(mx.array_equal(mx.argmax(out.logits,axis=-1),mx.argmax(reference,axis=-1)).item()))
            destination = os.environ.get('DSV41_DSPARK_STATS','/tmp/dsv41_s1_stats.json')+'.audit.json'
            Path(destination).write_text(json.dumps(result,indent=2)+'\n')
        finally:
            _restore_rng_state(before)

    def stats(self):
        accepts = self.drafter.accept_lens
        proposals = self.drafter.draft_lens
        calls = self.model._dspark_verify_calls - self.forward_start
        emitted = sum(a + 1 for a in accepts)
        result = dict(method="dspark", acceptance_rule="seed_preserving_target_prefix",
                    draft_calls=self.drafter.draft_calls, verify_calls=calls,
                    verify_shapes={str(k): v-self.shape_start.get(k,0)
                                   for k, v in self.model._dspark_verify_shapes.items()
                                   if v>self.shape_start.get(k,0)},
                    accepted_drafts_mean=sum(accepts)/max(1,len(accepts)),
                    accepted_drafts_range=[min(accepts,default=0), max(accepts,default=0)],
                    committed_tokens_per_round_mean=emitted/max(1,len(accepts)),
                    committed_tokens_per_round_range=([min(accepts)+1,max(accepts)+1]
                                                       if accepts else [0,0]),
                    acceptance_rate=sum(accepts)/max(1,sum(proposals)),
                    target_calls_per_committed_token=calls/max(1,emitted),
                    target_calls_per_accepted_draft=calls/sum(accepts) if sum(accepts) else None,
                    target_and_draft_calls_per_committed_token=(calls+self.drafter.draft_calls)/max(1,emitted))
        destination = os.environ.get("DSV41_DSPARK_STATS")
        timing = getattr(self.drafter, "phase_timing", None)
        if timing is not None:
            result["phase_timing"] = timing.summary()
        controller = getattr(self.drafter, "cost_controller", None)
        if controller is not None:
            result["cost_controller"] = controller.summary()
            print("[dspark-adaptive] " + json.dumps(result["cost_controller"]), file=sys.stderr, flush=True)
        if destination:
            if self._rng_before is not None:
                result['rng'] = dict(before_materialization=[x.tolist() for x in self._rng_before],
                    after_materialization=[x.tolist() for x in self._rng_after_materialize],
                    final=[x.tolist() for x in mx.random.state])
            Path(destination).write_text(json.dumps(result,indent=2)+'\n')
        return result
