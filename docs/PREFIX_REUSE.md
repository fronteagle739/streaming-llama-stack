# DS4COMBO follow-up prefix reuse

A follow-up turn should prefill only its new tokens on top of the saved
post-decode frontier. Before this work, it re-prefilled the whole previous
answer instead.

## Symptom: the 346 s second turn

`m50k-ultra-maxspeed` (qualified native `b16abd1d...`) turn 1:

| | prompt | reused | re-prefilled | native prefill | TTFT | wall |
| --- | --- | --- | --- | --- | --- | --- |
| before | 44 449 | 43 327 (`rewound_to` 43 327) | 1 122 | 333.1 s | 346.7 s | 397.2 s |

The 1 100 decoded tokens were thrown away. The session rewound to the
pre-decode anchor (43 327) and replayed them through native prefill.

## Root cause: sampled token IDs are not canonical BPE

After decode, the live frontier holds the token IDs the model sampled. The
next turn re-renders the conversation as text and tokenizes it canonically.
The same bytes can be split differently. On the long case the first mismatch
is at token 325:

| side | token IDs at the divergence (hex bytes) |
| --- | --- |
| live (sampled) | `85530` ` parasit` (`2070617261736974`), `1290` `als` (`616c73`), then `339` `.\n\n`, `372` `##`, `223` ` `, `20` `2` |
| canonical (re-render) | `21526` ` paras` (`207061726173`), `16823` `itals` (`6974616c73`), then `339`, `372`, `223`, `20` |

The text is identical; only the split differs. The frontier digest
(`prefix_digest` over the restored position) therefore failed. The session
fell back to the pre-decode anchor and re-prefilled everything after it.

## Fix: keep the live IDs when their text prefixes the new prompt

Commit `b0152f1` gives `/decode/render` in `engines/ds4-win/ds4_combo_api.inc`
the HTTP chat path's memory-text rule:

- If the live tokens are a token prefix of the render, they are reused as-is
  (`prefix_source: memory-token`).
- If not, but the live tokens' bytes prefix the rendered text, the live IDs
  are kept exactly and only the suffix is tokenized (`memory-text`).
- Otherwise canonical tokens are returned (`canonical-miss`). The report
  includes the divergence index and six token IDs and bytes from each side.

Routing uses the real prefill count. The expert cache is restored before
prefill. The slot-clear change (in `ds4.c`, `ds4_cuda.cu` and `ds4_gpu.h`,
committed with this note and built into the prefix4 binary) empties the eviction-protected VRAM slots before seeding. That
raised the follow-up restore from 234 to 301 of 301 experts.

## After: native binary `combo-native-prefix4.exe` (`34c484d1...`)

Performance figures come from run7, before the option 1 change. Run7 is
the set designated for performance. Idleness was not independently
re-verified for it. That change only affects the
re-prefill path, which no run7 turn reached.

| case | turn | prompt | reused | new | prefix source | native prefill | TTFT | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chat | 0 | 23 | 0 | 23 | empty | 13.5 s | 16.8 s | 25.9 s |
| chat | 1 | 64 | 53 | 11 | memory-token | 3.4 s | 8.5 s | 20.0 s |
| chat | 2 | 116 | 104 | 12 | memory-token | 8.8 s | 16.2 s | 19.1 s |
| long | 0 | 103 | 0 | 103 | empty | 31.9 s | 37.8 s | 130.2 s |
| long | 1 | 1220 | 1203 | 17 | memory-text | 5.1 s | 10.4 s | 22.2 s |
| long | 2 | 1271 | 1260 | 11 | memory-text | 7.5 s | 14.8 s | 18.8 s |
| think | 0 | 52 | 0 | 52 | empty | 15.3 s | 20.2 s | 46.4 s |
| think | 1 | 159 | 148 | 11 | memory-token | 3.4 s | 9.5 s | 21.5 s |
| think | 2 | FAIL: `ValueError('Rendered history changed outside the saved rewind frontier')` (the frontier limit below) | | | | | | |

Run8, after the option 1 change, is **correctness only**. The RAM-teardown
agent was active on the same machine, so its timings are contended and are
not proof numbers. Its pass/fail and token counts are:

| case | turn | prompt | reused | new | prefix source | fresh prefill | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| think | 0 | 52 | 0 | 52 | empty | - | PASS |
| think | 1 | 159 | 148 | 11 | memory-token | - | PASS |
| think | 2 | 84 | 0 | 84 | empty | divergence 52, include_previous_reasoning toggled off | PASS |
| long | 0 | 103 | 0 | 103 | empty | - | PASS |
| long | 1 | 1220 | 1203 | 17 | memory-text | - | PASS |
| long | 2 | 1271 | 1260 | 11 | memory-text | - | PASS |

Chat was not re-run: its turns never reach the changed path. It passed in run7.
The run8 long run's `run_direct` leak check failed: unowned RAM went
10 249→10 621 MB. This is the machine-wide leak tracked in `FAST_SWITCH.md`,
not a prefix-reuse result.

The long case decodes 1 100 tokens in turn 0, the same replay that cost 333 s
above. Its follow-ups now prefill 17 and 11 tokens. That is a much smaller
context than the 43 k case. A like-for-like 43 k repeat
(`cfg-50k-ultra.json` with only the native binary swapped,
`case-50k-maxspeed.json`, route ultra, script `prefix/run9.sh`) is **pending
a reboot**. Host RAM was near the 10 GB stop line (10.48 GB at 60 s after run8),
with about 1.75 GB left unreleased per think/long pair (see the RAM-teardown work), so it was not run. Until it
has run, the 346 s case is fixed in mechanism only. Its after time has not
been measured.

## Limit: history changed before every saved frontier

The session keeps two saved points: the restored post-decode checkpoint and
one pre-decode anchor in a single 24 MB journal RAM block. With
`include_previous_reasoning: false` on a later turn, the template drops
*every* earlier thinking span. On `case-think` turn 2, that moves the
divergence to token 52, the end of turn 0's prompt. There the live state has the sampled thinking `We need answer` (`2581 1309 3287`), and the new render has `</think>Three common` (`128822 22298 2953`). This is before the anchor (159) and
the restored frontier (207). Neither saved point is a prefix of the new
prompt, so the run failed with
`ValueError('Rendered history changed outside the saved rewind frontier')`.

### Option 1 (implemented): full re-prefill

The change is in `slstack/combo_session.py` (`ComboSession.prepare`); there
are no engine or `combo_native.py` changes. A fresh prefill happens only
when all of the following hold:

- neither the restored checkpoint nor the anchor matches the new render
- the engine reports `canonical-miss`
- the divergence index is below the oldest saved frontier

In that case the session releases the restored native owner, starts an empty
one and prefills the whole new prompt. The switch is logged as a
`DS4COMBO_FRESH_PREFILL` stderr line and stored in `metrics.fresh_prefill`.
That record holds the reason (for example `include_previous_reasoning
toggled off`), the divergence token, both frontiers and the tokens on each
side. Any other mismatch, such as one at or after the anchor that still
fails its digest, is still refused as an inconsistent history. The tests are
in `tests/test_combo_sequence.py`
(`test_divergence_before_every_frontier_prefills_fresh` and
`test_canonical_miss_after_the_anchor_still_refuses`).

Cost: one extra native release and launch plus a prefill of the full prompt,
on the turn where reasoning history is switched off.

### Option 2 (deferred, not started): one anchor per turn

The idea is to save an anchor at the start of every turn and rewind to the
latest one that still matches. Here that would be turn 0's prompt end
(52), so only the changed tail would be prefilled. It needs engine work,
because `anchor` and `rewind` share the single 24 MB journal mapping, and it
costs one more anchor of locked RAM per turn kept. This is a recorded,
deferred decision; revisit it only if switching reasoning off mid-conversation
needs to be fast.
