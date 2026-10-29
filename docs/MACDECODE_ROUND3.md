# Architecture-compatible Q4 to MXFP4 native state import

The RTX original Q4 producer and the Mac standard-layout MXFP4 requant now
qualify by checkpoint architecture, tokenizer, logical tensor names/shapes and
native state configuration. Their model SHA256 values remain separately pinned.
Different weight quantization is supported normally. An incompatible checkpoint,
tokenizer, state version or cone configuration still fails explicitly.

The compatibility description hashes all checkpoint architecture metadata except
explicit storage provenance, every tokenizer entry, and the complete logical
tensor inventory. This deployment uses native-C state v2.1, wire2/mode3,
context1000001 and cone-v2 keep2541. Existing producer-to-Mac state conversion
remains active. No model download or new weight conversion occurred this round.

Both receiver pairing and its native startup notice report the independent
endpoint identities. Pairing logs include each endpoint's quant type inventory.
Pinned producer binary/source, receiver binary/renderer, verified physical model
file and authenticated transport checks remain in force. The receiver also
retains lazy disk-KV seeding before a remote session starts; an active native
frontier cannot be overwritten by a disk restore.

The final Q4 to MXFP4 check recalled `MAPLE-CEDAR-7421` after an 8,941-token
RTX prefill. The Mac log measured **34.15 tokens/s over all128 generated tokens**.
The password occurs in the reasoning continuation; generation stopped at the
128-token limit before a final answer. A separate common-prefix continuation
compared all129280 raw, pre-bias vocabulary logits at eight positions:
KL(Mac-only || paired) mean **0.0410169028 nats**, maximum **0.100558174**,
inside declared mean<=0.1/max<=0.5 limits. The common eight tokens were selected
with the same explicit bias in both arms, while the observer captured logits
before that bias. This is a measured numerical difference, not exact parity.
Same-weight qualification retains its exact greedy parity requirement.

The tested Mac receiver SHA is
`90756b687a5560e6600a936ebeb3aff0128e0eac226bc159ddb9bf2c282d8a45`.
The canonical full-source installation subsequently rebuilt the stable path and
installed SHA `1f171f9896209907f502d6f2cb4278c410d9bf65de296b56f9e22442b4be0469`.
That new installed hash still needs its live preset qualification before rollout
is claimed. Source is `e0442804c05e9dd16202de747466f354a00c1a8b`;
producer source is `8e78f530d9c392ca512c986b0dbe27bf9555abed`, binary
`0822075d76ca6195ea081e9031ec54935f7d25975e3d02a15b0e9535e6358153`.
The full-source build guard passed at peak RSS1.23GB with no observed swap growth,
and confirmed all owned processes exited. Model tests used explicit400GiB/16GiB
Mac and8.5GiB/0.75GiB RTX physical budgets. The separate48GiB Windows WDDM
commit allowance does not authorize extra physical host RAM or pagefile use.

## Matched-weight performance baseline

These are **Round2** default measurements, not a new Round3 sweep. Native
published numbers use different, native-precision MXFP4 weights. Our local
Q4-to-MXFP4 target AND drafter are an explicitly lossy speed proxy.

| Decode workload | Published native MXFP4 | Upstream on our requant | Installed on requant | Our Round2 candidate |
|---|---:|---:|---:|---:|
| 8K prose serial |41.4|38.900|32.963|34.638|
| 131K prose serial |40.5|38.590|32.157|33.851|
| 8K prose DSpark |40.1|37.730|32.298|34.258|
| 8K code DSpark |58.8|51.735|46.188|45.749|

Units are tokens/s. Upstream reference is
`f48c5e2e20b1aaae1c79735a8457225d2469ae73`. Its benchmark CLI does not load
vision; our server controls retain vision. Separate upstream exact-mode serial
controls measured38.145/31.950. That distinction and different first logits mean
the default gap cannot all be attributed to exact scheduling optimizations.

Existing optional exact ports are two-layer decode submission, fixed routed
MXFP4 shapes, and narrow local weight readers. Remaining reviewed upstream
candidates include untracked read-only model resources and aligned compute
copies, then compatible shared/routed FFN overlap and next-token pre-encoding.
They have not been applied or measured in Round3 yet. No speed match is claimed.

## Two-machine configuration

Credential-free overlays and the measurement receipt are in
[examples/macdecode-r3](../examples/macdecode-r3/measurement.json).
The Mac overlay selects both verified requants, retains the original vision
encoder, cone-v2, the three measured exact controls, and explicit RAM limits.
The RTX overlay retains original Q4 weights and the shared priority lease.
Supply the freshly built/pinned producer through the existing configuration
and keep its runtime dependencies pinned. Pairing generates private credentials
and qualification files; none are included in these examples.

For the owner's Mac, run the normal `mac-setup` and `mac` commands with
`--config examples/macdecode-r3/mac-mxfp4.json --backend ds4`. The receiver binds
127.0.0.1:18081 and the engine127.0.0.1:18080. Establish authenticated loopback
SSH forwards before pairing from the RTX; forward its loopback broker18090
back to the Mac. Before using the RTX overlay, replace its explicit
`<ABSOLUTE_EXISTING_SHARED_GPU_LEDGER>` value with the existing shared lease
ledger in a private local config. The published template deliberately uses an
invalid Windows pathname so forgetting this cannot silently create a separate
ledger. On the RTX, use that configured overlay with
`--config examples/macdecode-r3/rtx-q4.json pair --mac http://127.0.0.1:18081`
with the private token using the normal pairing workflow, then the normal
`start` command. Wait for the priority RTX lease; never preempt the prefill job.
Existing launcher options, samplers, bias, tools, vision, disk KV, session
counts and custom techniques stay available.

## Deployment state

The full Mac receiver source/runtime is installed but no production preset or
native-service store has been changed. Ten complete-preserving preset patches
and a fully pinned producer-service generation are prepared. Studio's source
pin must be integrated only when its service data is ready; otherwise a daemon
restart would correctly refuse the mismatched binary.

The original MiMo preset was restored unchanged and answered `READY`. Its first
reload safely refused on insufficient immediately free pages. The existing
bounded clean-cache helper released only the two idle requant files, reading no
payload and changing no files; the exact retry passed under460GiB/16GiB guards.
The three existing Studio daemons are healthy on the prior revision. Fleet
reports RTX Remote Link serving, Mac Studio client tunnel verified and MacBook
client tunnel unavailable. Those are restoration/status checks, not a new
three-machine deployment claim.

The owner instructions prohibit all M: writes and also require M: backups before
production D: store edits. The backup location needs clarification before those
writes. The staged rollout and then the remaining upstream gains remain pending.

## Capability reductions

- The owner-selected target/drafter requantization is lossy; this is a speed proxy.
- No sampler, field, bias, tool, vision, import, disk-cache or cone capability was
  removed, hidden, disabled or stubbed to qualify the mixed pair.
- The upstream no-vision reference and explicit exact-mode numerical controls are
  comparison limitations; they do not replace the owner's production features.
