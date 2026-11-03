# Calibrating named router modes

`calibrate-router.py` measures a local bias/settings sweep, prints the measured
KL, top-1 agreement, prefill and decode throughput, expert-cache hit rate and
natural-stop results, and updates a **machine-local copy** of a model's settings.
The example launchers consume that same JSON through `model_settings`.
Numeric bias requests retain the model's full `bias.min` / `bias.max` range;
the measured range is recorded separately and never clamps that capability.

The QFN adapter reuses the archived `recall_100k.py` warm timing protocol,
`native_nvfp4_forced_feed.py` full-vocabulary feed mechanism, and the existing
bounded rowwise `compare_qfn_logits.py` calculation. It uses the production
`QfnEngine.command()` and `model_settings.apply()` functions. Inference, chat
templates, samplers, stopping, cache policy and speculation stay in the existing
engine. This adds a portable command adapter, not another inference engine.

## QFN on a new machine or quant

1. Copy `examples/single-machine/qfn.settings.json` to a local settings JSON. In
   your local stack config set `model_settings` to that copy. Set the model,
   stable engine executable, tokenizer, runtime directory, GPU identity and
   host-memory guard configuration for your machine. Keep model weights,
   caches and outputs on storage appropriate for the measurements.
2. Copy `qfn-calibration.example.json` locally. Replace its `inputs` placeholders
   with your complete tokenizer GGUF, exact-context LE32 prompt fixture, pinned
   reference continuation IDs and expected recall answer. The prompt must
   contain exactly `context` signed little-endian int32 token IDs. The pinned
   continuation must contain 128–1024 IDs from an unbiased reference using the
   same model/tokenizer; every sweep arm feeds those same IDs. A completed
   reference timing's `ids` array can be encoded directly as LE32; do not create
   a new continuation for each bias. Set `context`, `quant` and `vocab` to match
   the model. Change both the manifest's `vocab` and the adapter command's
   `--vocab` when calibrating a different vocabulary.
3. Review the declared quality policy. The example Balanced gate is KL ≤0.02
   and top-1 ≥95%; its deliberate Max speed policy is KL ≤0.5 and top-1 ≥75%.
   These are explicit tradeoff limits, **not measured results or the near-exact
   kernel adoption gate**. Every candidate must also finish all three sampled
   real-chat cases naturally. Tighten these limits if your application requires
   closer routing. The unbiased setting profile must always pass KL ≤0.002
   and top-1 ≥99% against the original unbiased profile.
4. Run from the stack root with an explicit physical-memory budget and reserve.
   The output parent directory must already exist. Use a fresh output prefix
   and label for every campaign; previous evidence is never overwritten.

```powershell
python examples/single-machine/calibrate-router.py `
  --settings my-qfn.settings.json --manifest my-qfn.calibration.json `
  --config my.local.json --model MODEL_PACK --quant native-packed-nvfp4 `
  --context 100000 --host-budget-gib 5 --min-available-gib 6 `
  --output-prefix OUTPUT_PREFIX --progress-log PROGRESS_LOG --label local-calibration `
  --plan-only
```

`--plan-only` validates the input structure and starts nothing. Remove it to
execute the sweep, after arranging an exclusive measurement slot. The RAM
numbers above are examples: choose a budget that actually fits the machine.
The existing guardian enforces process-tree physical RAM, available-memory
reserve, paging telemetry and the configured Windows commit cap. Commit is not
a substitute for physical RAM. Each GPU arm acquires the existing GPU lease;
the adapter uses stdio, so it opens no listener. A custom harness with a listener
must bind exclusively to `127.0.0.1` and declare that in its manifest.

The example sweeps 0, 0.175, 0.35 and 0.68. Add only the candidates worth spending
measurement time on. `setting_profiles` can sweep tile/read/cache settings, for
example a second entry with `engine: {"tile": 1024}` or
`cache: {"half_life": 16}`. Keep all other fields in that entry empty to inherit
the original settings. Explicit local config/environment overrides that
conflict with requested profile settings fail loudly; resolve them before
measuring. Every profile is compared to the **original unbiased settings**,
so changing tile or cache cannot quietly change the numerical reference.

The supplied timing preserves the existing sequence: exact-context prefill,
128 unbiased recall tokens, a 100-token candidate warm window, then 128 measured
candidate tokens. It records the raw engine events and full cache counters.
Natural end tokens remain active. A natural stop that interrupts the matched
timing window makes that timing incomplete; the script never extends the
completion or masks end tokens. The separate sampled chats use actual native
`op=chat` requests with distinct prompts/seeds and their full original sampler
fields. A length-limited, empty or nonterminating answer rejects that candidate
for adoption. Stopping reliably is necessary, but these checks do not certify
answer quality for every workload; inspect the saved answers as well.

## Selection and saved evidence

The chosen engine/cache profile is the one with the fastest passing Max speed
candidate and a passing unbiased baseline. All three named modes use this one
profile. Max quality stays bias zero. Balanced and Max speed choose the fastest
measured bias in that profile within their separate declared quality gates and
natural-stop checks. The report explains each choice. No candidate is called
calibrated until all required measurements exist.

The JSON report includes the entire measured table, quality policy, selected
profile, command/guardian receipts, hashes of the engine and harness sources,
full F32 vocabulary dumps and forced positions, generation request hashes,
actual answers, expert-cache counters, host/GPU/storage-volume identity,
model SHA-256, quant label and context. The final settings record the machine
fingerprint, model/quant/context binding and report hash. Recalibrate after a
hardware/storage change, quant/model change, cache allocation change or a new
engine profile. A result at a shorter context does not establish 100K quality.

Settings publication preserves unrelated fields and the full valid bias range.
It writes a timestamp/campaign-labelled backup before an atomic replacement.
An incomplete campaign leaves the original settings in place and records an
`.blocked.json` diagnostic with completed evidence. No files are deleted.
Every stage and measurement emits progress, with a heartbeat every 20 minutes
while a command runs. Reports explicitly record `capability_reductions: []`;
the adapter forwards the complete generation request and uses production
engine settings rather than reducing a sampler or feature to get a pass.

Router-spread references are preserved and bound in the settings, including
`reference_only` metadata when the production engine uses its qualified
embedded constants. This script does not invent a spread reference for a new
architecture/quant. Supply the appropriate measured reference first. The
existing QFN `QFN_AFFINITY_CALIBRATE` collector requires segmented decode; this
script does not silently turn off production graph execution to collect one.

## Other models, including DSV4.1

The orchestrator's manifest interface accepts existing model-specific harness
commands and JSON pointers. The three mandatory stages are full-vocabulary
quality, warm timing with recall/cache counters, and sampled natural generation.
Each command receives the explicit budget or generated guard config. Declare
whether the existing harness or the orchestrator owns its GPU lease to avoid
nested admission. Fields missing from a receipt are errors, never defaults.

For DSV4.1, `slstack.combo_measure` already supplies selected-context native
stream timing and generation finish information, but does not by itself supply
all full-vocabulary bias-sweep/cache-counter evidence. Connect its existing
qualification/full-vector harness and cache receipts through a local manifest
before using this orchestrator for DSV4.1. The shipped DSV settings are reference
values; adding this script does not make them newly calibrated on this machine.

The portable QFN path passed a real three-bias execution proof at 2,048 prompt
tokens: complete 128-row full-vocabulary comparisons, 100 warm plus 128 measured
tokens per arm, recall PASS and three sampled natural-stop chats per bias. It
printed its measured table and reasons, then published machine-bound local
settings with a preserved backup, choosing 0 / 0 / 0.35. All 15 guarded stages
completed. This short-context proof establishes that the calibration workflow
runs; it is not a new 100K performance or calibration claim. Run at the intended
context before replacing a production configuration. The [exact100K guide](../../docs/QFN_100K_REPRO.md) distinguishes its
CPU protocol checks from pending native wrapper qualification.


Current shipped QFN named modes are 0 / 0 / 0.68 for the native-packed NVFP4
model/quant profile. The owner selected Max speed 0.68 on 2026-11-03 and requested
no new 0.68 GUI or 100K tests. Its historical 100K evidence is 50.12 tok/s,
top-1 88.3%, KL 0.137 and predates the installed build. The short calibration
workflow proof above selected 0.35 and remains historical evidence of the
workflow; its measured table is unchanged. Continuous numeric bias values,
including 0.35, remain available. Future explicit calibration still runs and
reports its own complete gates and sampled-stop evidence; none are disabled.
