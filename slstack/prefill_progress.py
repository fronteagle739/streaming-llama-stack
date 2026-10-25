"""Prefill progress for streaming clients (owner: show the user prefill percent complete).

The ULTRA producer logs one line when each layer of a weight sweep begins and completes:
    [dsv41-sweep] layer=N begin tokens=T
    [dsv41-sweep] layer=N complete
A prompt longer than the producer's prefill reservation takes several sweeps, and each sweep
walks every layer. PrefillProgress turns those lines into a token-weighted fraction, so the
percentage moves smoothly across sweeps, with elapsed time and an ETA from the measured rate.

Wire format (an OpenAI-compatible extension): a chat.completion.chunk with an empty `choices`
list, like the usage chunk, plus a `prefill_progress` object. Clients that do not know the field
skip it, because the chunk carries no delta.
    {"object": "chat.completion.chunk", "choices": [], "prefill_progress": {
        "tokens_total": 40354, "tokens_done": 20177, "percent": 50.0,
        "sweep": 1, "sweeps": 1, "layer": 20, "layers": 40,
        "elapsed_s": 27.1, "eta_s": 27.1}}
"""
from __future__ import annotations
import math
import re

LINE = re.compile(rb'\[dsv41-sweep\] layer=(\d+) (begin tokens=(\d+)|complete)')


class PrefillProgress:
    def __init__(self, tokens_total, chunk_tokens, layers=40, started=0.0):
        self.total = max(1, int(tokens_total));self.chunk = max(1, int(chunk_tokens));self.layers = int(layers)
        self.sweeps = max(1, math.ceil(self.total / self.chunk))
        self.started = started;self.sweep = 0;self.sweep_tokens = 0;self.done_before = 0;self.layer_done = 0
        self.tail = b''

    def feed(self, data):
        """Consume new producer diagnostic bytes (any split; partial lines are kept)."""
        text = self.tail + bytes(data);lines = text.split(b'\n');self.tail = lines.pop()[-512:]
        for line in lines:
            m = LINE.search(line)
            if not m:continue
            layer = int(m.group(1))
            if m.group(3) is not None:
                if layer == 0:   # a new sweep begins
                    if self.sweep:self.done_before += self.sweep_tokens
                    self.sweep += 1;self.sweep_tokens = int(m.group(3));self.layer_done = 0
            elif self.sweep:
                self.layer_done = max(self.layer_done, layer + 1)

    def tokens_done(self):
        within = self.sweep_tokens * min(self.layer_done, self.layers) / self.layers if self.layers else 0
        return min(self.total, self.done_before + within)

    def snapshot(self, now):
        done = self.tokens_done();fraction = done / self.total;elapsed = max(0.0, now - self.started)
        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else None
        return dict(tokens_total=self.total, tokens_done=int(done), percent=round(100.0 * fraction, 1),
                    sweep=self.sweep, sweeps=max(self.sweeps, self.sweep), layer=self.layer_done, layers=self.layers,
                    elapsed_s=round(elapsed, 1), eta_s=None if eta is None else round(eta, 1))


def chunk(cid, model, created, progress):
    return dict(id=cid, object='chat.completion.chunk', created=created, model=model, choices=[], prefill_progress=progress)
