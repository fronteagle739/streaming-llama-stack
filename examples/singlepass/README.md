# Reproduce the carry comparison

Build this branch's producer and pin its SHA-256 in your local config. Use the
same original Q4_K/Q8_0 model, binary, DLL, context and token file in each pair.
The config supplies the shared GPU hold ledger and explicit memory budgets.
The measured profile is ULTRA with `yoco_keep=2541`, growable BF16 cache and
the existing 49,152-token sweep / 61,440-token fold policy.

Create the exact frozen inputs without loading the model:

```powershell
python -B examples/singlepass/fixture.py --count 100000 --output runtime/recall-100k.tokens.le32
```

The fixture preserves the original leading archive key and final question,
repeating only the neutral background. `fixture.json` contains the exact
token fragments for the qualified 129,280-token vocabulary.

For the N=1 control, set:

```json
{
  "producer_chunks_per_pass": 1,
  "env_overrides": {"LLAMADECODE_SINGLEPASS_VRAM": "0"}
}
```

Measure it through the exclusive GPU and host-memory guardian:

```powershell
python -B examples/singlepass/measure.py --config control.local.json --tokens runtime/recall-100k.tokens.le32 --output runtime/control-100k.json --recall AMBER OTTER 731
```

Repeat with `LLAMADECODE_SINGLEPASS_VRAM=1`. The complete initial logits and
owner/cache hashes in `checkpoint` must match. Both runs must pass recall and
close cleanly. Timings include startup; run another pair in reverse order.
Repeat at 50,000 and 200,000 tokens using fresh output names.

For grouped N=2, use `producer_chunks_per_pass=2`, an aggregate host cap of
12 GiB or more, and `producer_host_max_gib=11.75`. Keep the desktop reserve
and runtime physical floor at least 0.5 GiB. Admission refuses a group that
does not fit physical RAM. Grouping is explicit because optimized N=1 was
faster in the qualified matrix.

For stage attribution, run a separate pair with `--profile`, then:

```powershell
python -B scripts/analyze_singlepass.py runtime/control-100k.trace.log --output runtime/control-100k.stages.json
```

The diagnostic records per-layer cumulative CUDA snapshots and host phases.
It changes scheduling and is excluded from throughput claims. Registration,
copy waits, GPU execution and synchronization overlap; the parser preserves
those relationships. [Measurements and lifetime checks](../../docs/SINGLEPASS_PREFILL.md).
