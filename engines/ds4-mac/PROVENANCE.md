# Mac ds4 provenance

Derived from [IngeniousIdiocy/ds4-v41-m3ultra at `b10cb40c1786d93aa2b4c819635a7b9e84b2b635`](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/commit/b10cb40c1786d93aa2b4c819635a7b9e84b2b635), the merge-base of the exported Mac source and the upstream V4.1 branch, MIT. The backend itself derives from [antirez/ds4](https://github.com/antirez/ds4). The historical `IngeniousIdiocy/stack-backend` name could not be resolved to a separate verifiable base.

Changes: bounded direct reads into anonymous wired core/tensor/KV storage, host-memory admission, native RTX session transfer with negotiated decode deltas, checkpoint/frontier handling and portable stack integration. Original licences and third-party attribution remain in the engine source.

Paired WebUI sampling links the retained Streaming llama `llama-sampler.cpp`
implementations through `ds4_sampling_bridge.cpp`; it does not open a second
model. DRY uses the resident ds4 vocabulary's token pieces. Phase/code tracking
is shared with the retained Windows ds4 implementation. The final categorical
draw retains ds4's64-bit seeded RNG. The Mac MiMo selection-only cache-bias
policy is preserved for the admitted fully resident expert set: there are no
uncached candidates to penalize, so nonzero bias leaves its routing unchanged.
Streaming-expert bias on this DSV4.1 native-session lane is not implemented.

MACDECODE candidate: adapted upstream `a743c56d3c5816811f854f686b1e2919e3077eea`
command-buffer cadence behind `DS4_DS41_DECODE_QUEUE2=1`; existing cadence is
retained at zero. Q4 serial decode improves about 2.3% at 8K and 131K with
identical 128-token outputs. Paired qualification and the MXFP4 comparison are
pending; this is not an upstream rebase. See `docs/MACDECODE.md` at stack root.

Round 4 ports from the still-current upstream
`f48c5e2e20b1aaae1c79735a8457225d2469ae73`:
read-only weight hazard policy (`da48ab1c`), compute tensor copies (`1df25c6f`),
fixed low projection (`e06a0b7f`), full BF16 shared/routed FFN overlap
(`555e258f`), MXFP4 verify-row admission (`04830a71`), exact expert-read
deduplication (`a5635c52`), and persistent per-row Engram workers (`bb7e488f`
and the current reader implementation). The down-only overlap adaptation
retains the fork's combined router/shared/HC producer as a separate option.
All original paths, samplers, vision, native import, disk KV and cone support
remain. The new graph levers are opt-in; weight views and eligible aligned
copies adopt upstream defaults with explicit controls. The owned anonymous
wired allocation/admission policy is unchanged. See `docs/MACDECODE_ROUND4.md`
for exactness checks, measured scope and remaining release qualification.

Further Round 4 ports: next-token first-buffer pre-encoding (`67dcb001`),
exact per-step RoPE tables and transforming attention stores (`b1ba3164`),
literal QA/KV and attention/HC dimensions (`e06a0b7f`), and Unicode JoyAI
pre-tokenization (`1d28e1ff`, including its generated Unicode class table).
Held buffers retain generation/position/host-state/lever checks; the fork's
producer epilogues and numerical controls are preserved. The tokenizer's
three splits are checked directly against the existing released tokenizer JSON.

Further exact ports include Engram code/scale format lookup (`962ac191`),
index-query table RoPE/quantization (`b1ba3164`), the shared-down/HC/carry join
(`0ba56e9c`, `73f1bdb1`, `e06a0b7f`), segmented Engram gating (`dbe5d51d`)
and Q8 row weight reuse (`7924b68a`). Optional DSpark Q8 matrix rows
(`f8e36424`) preserve the upstream changed reduction order, bounded to DSpark
steps with failure-safe scope restoration; original row reductions remain.

Later verification ports adapt indexed batch attention and small-row scoring
(`74d56235`), exact derived BF16 index-key views (`18439adb`), and attention
low/output weight reuse with projection/HC and shared-down/HC row joins
(`fbbe5c10`). Derived views are budgeted, rebuilt after native import and
checked for exact representability; original F32 keys remain authoritative.
Every low, projection, shared, block and carry output is retained. The new
`verify_batch_core=2` and `verify_output=1/2` schedules are optional; historical
values and all quality, streaming, TP, sampler and native-state paths remain.

Round5 adds optional scalar index schedules from upstream's wide/BF16 family
and the all-eligible block mask shortcut, plus QKV_FUSION's query-store RoPE
adapted to the existing HC/KV producer. Original schedules and authoritative
F32 state remain. See `docs/MACDECODE_ROUND5.md` for measurement scope,
full-logit/forced-continuation checks and remaining release qualification.

Round5 also queues the retained deferred HC producer alongside complete
shared/routed FFN products (`concurrent_ffn=2`), imports the upstream scalar
shuffle/prefix top512 sort below16384 keys (`topk_short`), and specializes
the5120x4 FFN HC join (`hc_join_static`). Every original output, ordered
FMA, generic shape, optional add, disable control and original schedule is
retained. Sources are adapted from the same pinned upstream `f48c5e2e`.

Round5 also retains the original HC carry words in the static join and
adapts upstream independent QA/KV normalization to the retained query writer.
Every original reduction, state/output word and shape/control alternative
remains. Compact masked candidate scoring/selection stays opt-in diagnostic:
components pass but long full-model equality fails, so the release uses the
original selector. Final exact six-artifact qualification, failed experiments
and normal same-proxy rates are recorded in docs/MACDECODE_ROUND5.md.
