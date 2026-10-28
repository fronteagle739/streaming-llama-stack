# E2c: short-prompt plan and prior-turn residency

A 32-token follow-up used the existing resident decoder in 4.45-4.72 s instead
of a 17.36-18.22 s full weight sweep, after matched 2K/8K/32K initial prompts.
Recall and first-token top-1 checks passed. Follow-up logits were numerically
different, with KL 0.00108-0.01087; this is not a bit-exact substitution.
Verdict: **adopt the resident follow-up as an opt-in toggle with its documented
numerical difference; do not enable whole-prompt demand reads at 8K/32K**.
The owner's subsequent correction requires faster-but-lossy paths off by default.
`LLAMADECODE_APPEND_DECODE_MAX` now defaults to `0` in code, producer environment
and the example config; explicitly set `96` to reproduce the measured toggle.
The original measurements below used explicit `0`/`96`, so this default change
does not alter them. The accuracy attribution remains in progress.

## Cold prompt plan

The separate E2/E4 qualification supplies the cold strategy. Those experiments
own the provider and exact-cone implementation; E2c does not duplicate or alter it.

| Situation | Plan | Evidence and limit |
|---|---|---|
| ULTRA, up to 2048 new tokens | Real routed-expert reads | Source-code 2K: 19.172 to 15.234 s, 315.826 to 263.073 GB requested |
| ULTRA, 8192 or more new tokens | Eager encoder reads | Source-code 8K demand reads regressed: 20.578 to 21.391 s despite fewer bytes |
| Qualified exact-cone profile | Eager layers 0-20; routed tail reads in layers 21-39 | Padded 2541-row dependency coverage preserves existing GEMM shapes; 8K and 49,555-token logits/cache hashes match full execution |
| Existing session, short follow-up | Preserve resident decoder and its L1/L2 experts | Fresh E2c matrix below measures 32 input tokens after each requested context length |
| New full sweep after decode | Repurpose banks under the normal budget | Do not assume decode L1/L2 survives a full prefill bank rebuild |

The initial repetitive-body E2 test improved at both 2K and 8K, but its source-code
check narrowed the automatic cutoff to 2K. The superseded ragged cone and an
unmatched 49K input control are excluded. The qualified cone has separate local
snapshot provenance; its unnegotiated remote export remains refused.

The E2/E4 timings above are attributed controls from the peer's qualification,
using its BF16 growable context cache. They are not new E2c cold TTFT measurements
or directly comparable to this experiment's F32 cache. In particular, no new
32K cold-cone TTFT value is inferred from the 8K/49K results.

## Matched resident-follow-up experiment, 2026-10-28

One setting varied: `LLAMADECODE_APPEND_DECODE_MAX=0` forces a full sweep;
`96` permits the existing resident path. All initial prompts used full execution
(`yoco_keep=0`) and identical IDs. After the same natural recall and forced prefix,
both arms received exactly the same canonical 32-token user turn asking for the
original archive key. EOS/user/assistant markers and the question are preserved.
First output is timed before the frontier readback.

| Initial context | Full sweep prefill s | Resident prefill s | Full sweep TTFT s | Resident TTFT s | Full/resident input tok/s | Follow-up KL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 17.360 | 4.453 | 17.375 | 4.485 | 1.843 / 7.186 | 0.010867756 | 100% |
| 8192 | 17.531 | 4.531 | 17.547 | 4.547 | 1.825 / 7.062 | 0.001078951 | 100% |
| 32768 | 18.218 | 4.719 | 18.234 | 4.734 | 1.757 / 6.781 | 0.004663514 | 100% |

Both initial and follow-up recall passed at every length. The 17 initial/forced
rows per pair were bit-exact; the changed follow-up path compares one untouched
full-vocabulary frontier per pair. Report that limited numerical sample explicitly.
No whole-trajectory or bit-exact follow-up claim is made. Raw diagnostic generation
can run beyond EOS; recall is assessed only on the answer before the first EOS.

The native method receipt confirms full sweep on the control and
`append_via_decode=true` on the candidate. Each candidate processed all 32 input
tokens, retaining already-resident dense weights and useful L1/L2 expert entries.

| Initial context | Routed expert-miss B during append | L1 hits | L2 hits | Misses |
|---|---:|---:|---:|---:|
| 2048 | 58,525,286,400 | 2519 | 2221 | 2940 |
| 8192 | 58,385,940,480 | 2572 | 2175 | 2933 |
| 32768 | 60,237,250,560 | 2424 | 2230 | 3026 |

Each row reconciles all 7680 routed expert triples (32 tokens x 40 layers x 6).
The peer ledger's full weight pass is 315,825,603,008 B. Candidate miss bytes are
an attributed sub-counter, not a whole-operation I/O total; Engram and any
post-prefill decoder promotion reads must not be silently included or excluded
when comparing total bytes.

## Admission and reproducibility

The model and native producer match E13's recorded identities. F32 context cache,
ULTRA arithmetic, disabled RAM row LRU, 512 MiB desktop reserve, 0.5 GiB physical/
runtime floors, paging stop and the 8.5 GiB aggregate host RSS cap were identical
across each pair. Every run used an exclusive shared GPU lease, shut down
gracefully, and checked Studio health and Remote Link afterwards.
Across six arms, peak guarded tree RSS was 6,182,653,952 B and minimum available
RAM was 10,453,217,280 B. Bounded owned-page and global-growth observations were
zero; the bounded observation does not establish exhaustive zero pagefile I/O.

The comparison harness accepts a separate scratch-local config and records
append input identity, method, timing, first output, frontier logits and recall:

```text
python -B scripts/perfexp_pair.py --root <scratch> --config-name E2c-config.json --experiment E2c-residency --candidate LLAMADECODE_APPEND_DECODE_MAX=96 --count 8192 --append 32
```

Set the control config's append limit to 0. This measures one 32-token follow-up;
it does not recalibrate the existing 96-token crossover or justify arbitrary
CPU/GPU hybrid thresholds. Retaining an additional 10 GB dense bank across a full
sweep is not admitted by the observed spare VRAM; a selective extra residency
tier requires its own explicit budget and experiment.
