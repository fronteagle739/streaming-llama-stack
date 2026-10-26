# Live generation controls

Single-engine WebUI launches include a quality/speed slider and a Phases popover
in the existing toolbar. The controls follow the WebUI's light/dark theme and
remain usable on a phone. Settings apply to the serving session and persist in
the configured runtime's `generation/live.json`.

The native decoder reads an atomically published, validated revision at every
token boundary, independently of its occupied session pipe. Slider changes
apply to the next forward token; they do not wait for the decode RPC to finish.
Changing the main slider selects one bias for every phase. The phase preset
uses reasoning bias 1.36125 / temperature 1, answer .35 / .6, tools 0 / .1,
and code 0 / .2. Each phase also exposes top-p, top-k and min-p.

The implementation reuses the vendored ds4 quality-phase tracker, including
actual special token IDs, DSML tool boundaries and streamed code fences. The
sampler is a narrow export of the native ds4 implementation; its source hash
and refresh script are recorded in `studio-sampling.c`. Temperature zero keeps
stable greedy selection. Seeded sampling uses ds4's RNG and filtering order.

Explicit OpenAI sampler parameters override the profile for that API turn.
WebUI requests use the toolbar settings rather than the stock UI's implicit
sampler defaults. Old producer executables without generation-control version
1 are refused by the new serving path.

`GET /slstack/generation` returns saved settings and the latest native applied
revision, phase, bias, sampler values and token position. POST accepts partial
updates for `value`, `per_phase`, `sampler`, and `phases`.

Validation: 53 Python tests passed, 2 skipped, plus 8 subtests. The CPU native
regression exercises real special-token/DSML/code boundaries, seeded sampling,
top-k/top-p/min-p, next-token revision changes and partial-publication recovery.
The compiled UltraYOCO model returned the expected short answer. Headless
desktop light/dark and 390-pixel phone checks passed without page errors or
horizontal overflow. During the owner's live use, native logs recorded bias
and temperature revision changes within one uninterrupted decode, including
temperature revisions 106, 108 and 109 at token positions 2104, 2105 and 2106.
The owner's live session was preserved; no separate controlled latency
benchmark is claimed.
