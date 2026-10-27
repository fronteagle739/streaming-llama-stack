# Paired YOCO release gate: incomplete

This candidate is not qualified for release. The measured two-machine row still
describes plain ULTRA. No paired YOCO performance or image-quality result is
claimed.

The paired route uses `native_broker`, not `ComboProducer`. Pairing already
negotiates keep=140/state version 1; the native Mac importer checks both against
its configured peer before accepting the existing mode3 payload. The local
`ComboProducer` restriction protects a separate snapshot contract and is retained.
Launcher configuration now validates keep=0/140 and permits the explicit Mac
route. The labelled YOCO example retains the 4 GiB runtime floor and 8 GiB
desktop reserve, including native broker and Mac admission.

The existing image fixes handle projected embeddings, Engram masks and image
checkpoint identity. They did not remove the YOCO image exclusion. This candidate
removes that exclusion: all rows, including projected image rows, run through
layer 20; decoder layers retain image rows intersecting the last 140 positions.
The scheduler/carry fixture passes 42,170 checks on Windows and Apple Silicon,
including image spans before, across and within the replay tail. These checks
do not establish numerical equivalence or KL/recall quality.

The RTX previously forwarded chat request fields but omitted the separate WebUI
generation profile. Both streaming and buffered requests now carry that complete
profile. Core Mac sampler defaults are resolved from it, with explicit request
values taking precedence. Regression tests cover non-default temperature, top-p,
top-k, min-p, seed and token limit; a greedy substitution fails the same checks.
A CPU test includes the actual Mac server parser, with tokenization stubbed so
that no model is opened. A separate single-engine regression checks its native
generation command still receives explicit sampler overrides.

The following work remains:

- Mac ds4 has no quality/speed bias control, phase-specific sampler profile or
  repetition/presence/frequency penalty implementation. The receiver now refuses
  unsupported active settings before loading the decoder. This is fail-closed
  admission, not implementation of those controls. The default WebUI phase
  profile is consequently refused; this branch must not ship as a working paired
  WebUI release.
- Paired Mac multimodal prefill explicitly rejects images. The mode3 protocol
  currently carries token IDs rather than projected image spans. Implement and
  authenticate image-span transport and preserve image identity on continuation.
- Run image KL/recall comparisons against full evaluation, including images
  before and within the replay tail.
- Run a real headless browser gate with at least 100K prompt tokens, two further
  Mac-decoded turns, an uploaded image and a referring follow-up. Verify applied
  sampler settings in native decode evidence and check single-engine behavior.
- Only then add measured timings, remove the unbenchmarked wording, and qualify
  the candidate for release.

The Mac candidate builds from SHA-256-verified RTX source without replacing the
active engine. A successful build and CPU parser test do not prove live import,
Mac decode, image support or WebUI qualification.
