# Paired YOCO release gate: incomplete

This candidate is not qualified for release. The measured two-machine row still
describes plain ULTRA. No paired YOCO performance or image-quality result is
claimed.

The paired route uses `native_broker`, not `ComboProducer`. Pairing already
negotiates keep=140/state version 1; the native Mac importer checks both against
its configured peer before accepting the existing mode3 payload. The local
`ComboProducer` restriction protects a separate snapshot contract and is retained.
Launcher configuration now validates keep=0/140 and permits the explicit Mac
route. The labelled YOCO example retains the 4 GiB runtime floor and 2 GiB
desktop reserve, including native broker and Mac admission.

The existing image fixes handle projected embeddings, Engram masks and image
checkpoint identity. They did not remove the YOCO image exclusion. This candidate
removes that exclusion: all rows, including projected image rows, run through
layer 20; decoder layers retain image rows intersecting the last 140 positions.
The scheduler/carry fixture passes 42,170 checks on Windows and Apple Silicon,
including image spans before, across and within the replay tail. These checks
do not establish numerical equivalence or KL/recall quality.

The subsequent RTX numerical image gate compares full evaluation with keep=140
using an SHA-256-verified real image projection. With the image intersecting
the replay tail, KL(full || YOCO) is 0.05814 nats and top-10 recall is 90%; with
the image entirely before the tail, KL is 0.00733 nats and top-10 recall is 90%.
Both arms retain the same leading color token. These are frontier distributions,
not full generated image answers or Mac decode. All four arms retain the native
8 GiB host cap, 2 GiB desktop admission reserve, 4 GiB runtime floor and paging
stop; receipts show no budget breach or paging growth. See
[numerical evidence](YOCO_IMAGE_NUMERICAL.json) and `tests/yoco_image_gate.py`.

The live single-engine headless WebUI also answered its arithmetic check
correctly. Its actual native log applied temperature .83, top-p .87, top-k 37,
min-p .06 and bias .22. This does not replace paired Mac decode qualification.

The RTX previously forwarded chat request fields but omitted the separate WebUI
generation profile. Both streaming and buffered requests now carry that complete
profile. Core Mac sampler defaults are resolved from it, with explicit request
values taking precedence. Regression tests cover non-default temperature, top-p,
top-k, min-p, seed and token limit; a greedy substitution fails the same checks.
A CPU test includes the actual Mac server parser, with tokenization stubbed so
that no model is opened. A separate single-engine regression checks its native
generation command still receives explicit sampler overrides.

The continuation candidate accepts the default phase profile and scalar bias.
Native decode selects phase-specific sampler values at token boundaries;
explicit request values win. Presence/frequency/repetition penalties and logit
bias change the actual sampled logits. WebUI sampler order, typical sampling,
dynamic temperature, XTC and DRY reuse the retained llama.cpp implementations.
Seeded final selection uses ds4's64-bit RNG. `-1` token/history limits preserve
their sentinel meaning instead of being clamped to zero by the old parser.

Quality/speed is stored in the decoding session and logged from that session.
The Mac MiMo policy penalizes uncached expert selections without changing their
combine probabilities. This DSV4.1 lane admits fully resident Metal experts, so
its uncached set is empty and a nonzero bias leaves routing unchanged. This is
not qualification of streaming-expert bias or a Mac bias speedup.

Paired multimodal prefill now carries SLVP v1 projected spans and SHA-256-bound
SLVI embeddings. The broker materializes immutable owned files, rejects remote
filesystem paths and checks historical image identities. Initial and appended
prefill use the projections, original image routing and Engram masks. Native
import restores image fingerprints. A compatible existing Mac vision encoder
is selected automatically beside the language model.

A real headless browser CPU fixture exercises the shipped WebUI and paired
relay with temperature .83, top-p .87, top-k37, min-p .06 and bias .22. It also
checks Send during a pending toolbar save. The old fractional-step validation
dropped visible temperature/top-p edits; arbitrary valid fractions now persist.
These responses are synthetic and no model is opened. They do not satisfy the
live decode gate. Toolbar changes on the paired path apply on the next turn.

Round 4 subsequently completed the real headless WebUI conversation checks
with the blocking catch-up implementation: a 105K-token recall prompt and two
follow-ups, and an actual uploaded image in a 198,931-token prompt followed by
a referring turn. All answers were correct. Browser evidence shows incremental
content, advancing prefill progress, numeric live/final rates and updated
context usage. Mac-native logs retain temperature .83, top-p .87, top-k 37,
min-p .06, bias .22 and seed 123456. The short prompt and short follow-up use
Mac-local prefill at Studio's 11,776-new-token threshold.

The image prompt's startup-through-prefill interval was 194.248 s, or
1,024.109 tok/s, above the supplied 985.74 tok/s image baseline. Its blocking
Mac decode rate was only 11.003 tok/s. Logs identify serialized full-owner
hashing and RTX acknowledgment on the decode thread as the cause; the ordinary
Mac-local WebUI turns measured about 32 tok/s. These are diagnostic measurements
of the blocking implementation, not qualification of the current candidate.

The current candidate ports Studio's ordered catch-up worker and incremental
hash chains. It preserves the existing YOCO negotiation, projected images,
full checkpoint verification, and strict joins before RTX reuse or shutdown.
CPU checks compare incremental snapshots with independent full-state hashes at
even and odd positions, reject a changed compressed prefix, and verify that a
corrupt queued base fails and its worker shuts down. The sampler parser,
penalty/order checks and greedy negative control still pass.

The following work remains for this asynchronous candidate:

- Requalify the candidate transport and samplers with real Mac decode and images.
- Repeat the real headless browser gate with at least 100K prompt tokens, two further
  Mac-decoded turns, an uploaded image and a referring follow-up. Verify applied
  sampler settings in native decode evidence, confirm the improved decode rate,
  and check single-engine behavior.
- Only then add measured timings, remove the unbenchmarked wording, and qualify
  the candidate for release.

The Mac candidate builds from SHA-256-verified RTX source without replacing the
active engine. A successful build and CPU parser test do not prove live import,
Mac decode, image support or WebUI qualification.
