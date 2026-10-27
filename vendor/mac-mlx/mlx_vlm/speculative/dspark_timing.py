"""Opt-in synchronized DSpark phase accounting; no MLX import or device policy.

These diagnostic fences change scheduling. Never compare instrumented throughput
with an unfenced baseline. Prefill, first bonus and consumer backpressure are excluded.
"""
from time import perf_counter


class DSparkTiming:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.rows = []

    def start(self):
        self.started = self.previous = self.clock()
        self.current = {}

    def phase(self, name):
        now = self.clock()
        self.current[name + "_ms"] = (now - self.previous) * 1000
        self.previous = now

    def finish(self, accepted, committed):
        self.phase("bookkeeping")
        self.rows.append(dict(self.current, accepted_drafts=accepted,
                             committed_tokens=committed))

    def summary(self):
        totals = {k: sum(r[k] for r in self.rows)
                  for k in ("draft_ms", "verify_ms", "bookkeeping_ms")}
        total = sum(totals.values())
        accepted = sum(r["accepted_drafts"] for r in self.rows)
        committed = sum(r["committed_tokens"] for r in self.rows)
        fraction = totals["draft_ms"] / total if total else None
        return dict(synchronized_diagnostic=True, rounds=self.rows, totals=totals,
                    accepted_drafts=accepted, committed_tokens=committed,
                    ms_per_accepted_draft={k: v / accepted if accepted else None
                                           for k, v in totals.items()},
                    ms_per_committed_token={k: v / committed if committed else None
                                            for k, v in totals.items()},
                    draft_fraction=fraction,
                    ideal_zero_draft_speedup=(1 / (1 - fraction)
                        if fraction is not None and fraction < 1 else None))
