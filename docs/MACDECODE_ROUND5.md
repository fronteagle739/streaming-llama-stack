# Mac decode Round5 upstream rebase

Round6's owner decision and current qualification supersede the development
control status below; see [Round6](MACDECODE_ROUND6.md).

The development engine now uses upstream `f48c5e2e20b1aaae1c79735a8457225d2469ae73`
as its core and Metal implementation. This replaces the earlier approach of
porting individual kernels into the old `b10cb40` base. The previous campaign
and its measurements are preserved in [the pre-rebase report](MACDECODE_ROUND5_PRE_REBASE.md).
Its qualification does not qualify this new executable.

The retained integration includes native state/cone v2 import, architecture
and tokenizer compatibility, sampler/bias/tools handling, anonymous wired
allocations and host-memory admission, checkpoint cadence, native vision
transport, KV-store pins, and the server's batched inference ownership.
DSpark now runs through the batched executor instead of being excluded when
the normal server uses `--batched-session`. All ordinary and Qwen decode
requests retain their existing executor routes.

Native mode3 retains every RAW F32 capture word. Bounded conversion scatters
and gathers the upstream ring representation; BF16 conversion occurs at the
drafter-input boundary. Ordinary mode2 snapshots retain upstream's BF16
values without overwriting the authoritative native RAW ring. Native import
also repairs GPU logits, selector-derived data and drafter state. The fixture
checks full words, wrapping, deltas, rollback, save/load, and nonfinite refusal
under the actual fast-math compiler flags.

The executor fixture compiles the actual extracted server functions and checks
mixed ordinary/Qwen/DSpark requests, sampled RNG and policy, EOS handling,
17-token capacity, cancellation and the nonbatched route. Both focused
fixtures pass with a 2 GiB cap, a 16 GiB available-memory floor and no swap.
Object compilation passes under a 4 GiB cap and the same floor. Full-model
qualification and production deployment are separate gates.

## Capability reductions

This is a development-only candidate, blocked from release. Of 120 legacy
tuning names, 21 have real adapters and 99 currently lack equivalent
implementations in the upstream graph. Their exact names, environment
settings and evidence are in [the control status](macdecode-r5/upstream-controls.json).
Explicit requests for unavailable controls fail loudly; diagnostic output
must report their unsupported status rather than fabricate values. The
deployed runtime has not been replaced by this candidate. Retiring these
controls requires an explicit owner scope decision; retaining them requires
implementing their behavior before release.

The matched comparison uses only the existing owner Q4-to-MXFP4 lossy
target/drafter requants and vision sidecar. They remain labeled speed proxies.
No model is downloaded. The reduced qualification is three eight-row
full-vocabulary proof pairs (8K prose, 131K prose, 8K code), plus one 128-token
timing per arm and workload. Code timing enables DSpark. No repeated lever
sweep is planned. The optimized FP4-to-FP4 paired import proof, real GUI rates,
fleet deployment, and final MiMo restoration remain pending.
