# Resident QFN: RTX NVFP4 prefill, Mac Q4 or Q8 decode

The production entry point is `native_pair.py`, or `qfn-q4.ps1` / `qfn-q8.ps1`.
It keeps the existing Mac `ds4-server` resident with its OpenAI API. The native
RTX engine streams a complete QFN state directly into the native-session
socket reader and existing session loader. There is no checkpoint file,
Python payload relay or decoder subprocess in this production link.

The resident route retains text, streaming, all native samplers, reasoning,
tools, cancel, follow-up turns and lazy RTX image prefill. Both compact Q4 and
Q8 Mac formats use 16-bit KV. Session reuse includes bidirectional owner deltas,
paired rewind anchors, ordered asynchronous catch-up, checkpoint acknowledgement
and visible peer-failure recovery. Local append allows up to 11,776 suffix
tokens; longer warm suffixes prefill only the new rows on the RTX after importing
the actual sampled Mac state. The recipe can run independently of a GUI.

RTX I/O tuning has explicit `ple_submit_low_water` (0 preserves the shared
policy), `ple_thread_priority` (`normal` or `above-normal`), `ple_slot_threads`
(1Ã¢â‚¬â€œ32), `ple_io_workers` (1Ã¢â‚¬â€œ8 disjoint row-ID ranges) and `ple_poll_us` (0Ã¢â‚¬â€œ5000
microseconds). All remain real controls with bounds and memory/cancellation
checks. The supplied preset selects watermark 0, normal reader priority, one
slot thread, one I/O worker and polling 0. These flags are rejected on the Mac, including explicit defaults.
`--qsa-attention native|basalt` exposes the existing RTX attention choice.
The supplied preset selects native attention.
`--process-qos inherit|high` controls only the owned Windows engine's execution
policy. The preset explicitly selects `high`, verifies it, and fails visibly if
unsupported. It changes no system power/security settings.

`--trim-host-heap-cache 0|1` requests release of unused Windows heap-cache
memory before residency pinning. It applies to the RTX serving process and
fails visibly if the operation is unavailable. Both supplied profiles set 0.

`--populate-host-heaps 0|1` admits and locks all committed readable CPU heap
pages, including demand-zero and standby pages. Both profiles set 0. The
complete prospective host total must fit the explicit budget before population;
an already-paged readable heap page refuses work. Protected pages and model
mappings keep their original protections. The independent paging guard remains
active.

### Measured performance

Exact 100K prompt, 1,024 outputs, temperature 1, 16-bit KV, direct private TCP, and state hashing off. Each row uses a fresh RTX process; two runs per format.

| Mac experts | RTX prefill median tok/s (range) | Handover median s | Warm TTFT median s | Decode median tok/s (range) |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 4531.9 (4521.6-4542.1) | 4.434 | 27.252 | 40.970 (40.851-41.089) |
| Q8 | 4498.0 (4469.0-4527.0) | 4.994 | 27.840 | 39.742 (39.603-39.881) |

Actual MTP depth 2; accepted/proposed drafts are 301/446 for Q4 and 308/447 for Q8. Both formats pass sampled recall and API/image checks. Two measurements describe observed spread and do not establish a format ranking. API prompt throughput includes handover and differs from pure RTX prefill throughput.

## Weights and conversion

The source is the already downloaded NVIDIA Qwen3.8-Flash-Next NVFP4 checkpoint,
imported with `scripts/pack_slpack.py` and `recipes/qfn-nvfp4.json`. This recipe
never downloads a model. The measured source pack SHA256 is
`0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe`.
Its original Hugging Face/config/calibration metadata and companion assets are
retained in the pack and conversion receipts. Conversion decodes original
NVFP4 scales directly, then independently writes either main-expert Q4_0 or
Q8_0 with new compact offsets. Q4 needs neither the Q8 file nor APFS cloning.
Both formats retain Q8 MTP/PLE, F16 dense tensors and FP32 norms. The vision
companion is converted separately by exact BF16-to-F32 widening. All K/V are
16-bit and recurrent/convolution state is FP32. This does not create 16-bit
original language-model weights.

Check physical disk free space before conversion (`df -h /path/to/models`). Keep
the verified source plus approximately 134 GB for compact Q4 and 195 GB for
Q8, companions, and conversion headroom. Both outputs coexist; neither
replaces the other. The compact Q4 file occupies 133,789,359,104 bytes.

Use an independent Python environment. `bootstrap_python.py --target PATH
--source-site-packages PATH --package psutil --package numpy` creates a clean
venv from locally installed distributions and records versions/source. On the
Windows test controller also install/copy Pillow. This does not import another
project's site-packages at runtime. Alternatively provision these packages in
an independent venv using the normal package installer. Native production
scripts require psutil; conversion requires numpy; image codec normalization requires Pillow.

For each bit width, run the converter's `plan`, `write`, and `verify` modes with
the same flags, under the existing memory guard and `nice -n 10`. For example,
with explicit local paths (repeat for `--expert-bits 8` and its own output):

```sh
PY=/path/to/qfn-runtime/.venv/bin/python
ROOT=/path/to/qfn-runtime
"$PY" -B "$ROOT/scripts/qfn_dual_guard.py" --root "$ROOT" \
  --label convert-q4-write --rss-gib 4 --commit-gib 4 --floor-gib 16 \
  --seconds 1140 --blas-threads 1 --progress-log "$ROOT/progress.log" -- \
  nice -n 10 "$PY" -B "$ROOT/scripts/qfn_mac_convert.py" write \
  --source /path/to/models/source.slpack \
  --source-sha256 0f6291a714839961007f0d130aa715b53d82f8d1e73cef9785700e83bce74fbe \
  --out /path/to/models/QFN-Metal-Q4-compact.gguf \
  --expert-bits 4 --workers 1 --priority below-normal \
  --progress-log "$ROOT/progress.log"
```

Use a suitable bounded guard duration for the measured conversion; partial
outputs are retained and require explicit `--resume`. Verification reads every tensor/companion and records the full
model identity.

## Build and deploy

Build `qfn-engine` with its existing Windows CUDA/DirectStorage CMake target.
Keep a single executable location, for example `RUNTIME_ROOT/bin/qfn-engine.exe`,
and its compatible adjacent DLLs. Build the Mac's existing `ds4-server` target
with `DS4_NATIVE_FRONTIER_ENABLE=1`; retain the full sampler bridge sources and
llama sampler library. Use one stable Mac executable path. Build jobs also go
through the RAM guard. `qfn-dual-decode` remains a test-only target.

Build the packed expert bridge with vector policies 3 and 4 for the two installation profiles. Use
`engines/streaming-llama/flashinfer-sm120/export_qfn_nvfp4.py` for its guarded AOT
export, then `build_qfn_nvfp4.ps1 -TaskRoot <runtime-root> -BinDirectory <stable-bin>
-AotDirectory <existing-AOT-directory> -CutlassRuntimeLibrary <runtime-library>`
in the existing VS x64/CUDA13.2 build environment. Its stable output is
`qfn-native-nvfp4.dll`, adjacent to the one `qfn-engine.exe`. Keep all compiler
outputs and caches under the explicit runtime root. Existing verified AOT inputs
may be read without copying or modifying them.

Run `scripts/qfn_pinned_build.py --help` for the guarded executable build wrapper.
Pass an explicit JSON compiler/build argv and an explicit JSON list of all
linked library/object/DLL inputs, plus the stable executable, manifest,
runtime directory, working directory and memory/floor/duration flags. The wrapper checks
actual source and input bytes before/after compilation and requires successful
closed guard certificates before recording a build. The launcher requires that
manifest through `--producer-build-manifest`; source or binary drift refuses
startup. External libraries and vision objects may be reused when their bytes are
pinned in the manifest; rebuild them separately if their source changes.

Copy `deployment.example.json` to a private local deployment config and fill
its verified model receipt, source SHA, GPU UUID, token path, SSH alias and
Mac private-link IPv4 address (`mac-private-host`).
Generate a private random 256-bit hex token once and store it outside the checkout. Never commit it. The launcher
provisions the same private token on the Mac. Both endpoint model identities and
the independently derived QFN configuration/tokenizer/shape contract are checked.
Unknown settings fail. Production settings live in `native.settings.json`,
which also loads `examples/single-machine/qfn.settings.json`. No performance
or correctness environment-variable setup is needed: advanced options are
explicit flags through the retained option registry and recorded per run.

```powershell
powershell -File examples/dual-machine/qfn/qfn-q4.ps1 -Python PYTHON_EXE -Config LOCAL.json --dry-run
powershell -File examples/dual-machine/qfn/qfn-q8.ps1 -Python PYTHON_EXE -Config LOCAL.json --dry-run
```

Use a fresh output directory for each launch. Dry run validates
the real native CLI and the weight/vision receipts without loading a model.
Remove `--dry-run` to start a resident pair. Its API is
`http://127.0.0.1:18795/v1/chat/completions` (also `/v1/completions`). Ctrl-C
requests retirement of both independently guarded peers. Use `benchmark.py`
when borrowing a Studio-owned Mac: it saves the complete preset/store/native
argv, verifies 60 idle queue seconds, uses normal eject, and restores the prior
preset in its outermost cleanup even on child errors. A direct launcher does
not itself eject an unrelated production server.

The shipped profile admits RTX aggregate RSS 4.25 GiB, native host budget 3 GiB,
physical floor 6 GiB and Windows commit allowance 44.25 GiB (including device
accounting); it enforces Studio's shared 64 MiB owned-paging allowance,
sustained hard-fault and global limits. Native working-set
protection and host heap/stack residency locking are explicit flags, not a
relaxed guard. Locking records actual resident bytes and refuses excess.
Mac RSS/commit 320 GiB and
floor 16 GiB are explicit deployment flags. Adjust both budgets to the physical
machine before admission. Guards fail loudly instead of relying on paging.

## User behavior and limits

Requests below 11,776 tokens stay on the Mac. Larger changed conversations
prefill on the RTX and stream state. Local append (up to 11,776 new tokens)
continues using the native Mac state. Longer matching suffixes use RTX delta
prefill; a changed prefix may require a full rebuild. The profile allocates RTX and Mac context 131,072; native
flags allow different capacities subject to real memory admission. The receiver payload budget is explicitly 16 GiB. Capacity
is finite and requests beyond the selected RTX context are refused, not truncated.

The real server retains streaming, reasoning, tools, stop sequences, sampler
chains, follow-ups, cancel and RTX vision.
All non-default penalties/DRY/typical/dynamic-temperature/XTC/top-sigma and order
are applied by the retained llama sampler implementations. An explicit order
that omits an enabled sampler is rejected. Unsupported sampler extensions are
rejected explicitly. Advanced filter callbacks can use the existing serial
native decode path, so do not assume MTP speed for every sampler combination.
MTP depth is `auto`, `2` or `3`; automatic exact temperature-1 sampling uses
depth 2. Explicit depth 3 applies exact point-proposal/residual sampling
successively to both drafts.
Greedy also supports fixed depth 3. Actual verifier depth is reported.

Windows listeners and the Mac OpenAI API bind 127.0.0.1. The default state
route is direct TCP: `--transport direct --mac-private-host PRIVATE_IPV4`.
The existing Mac server opens a listener bound only to that private interface
for each request, and the RTX connects outbound. Control, image encoding and
the API forwarding still use SSH. No Windows non-loopback listener, payload
file or Python data relay is created. A request nonce, pairing-token check and
model/settings/layout identities bind the state connection to its request.

Direct state TCP does **not** encrypt the state or authenticate every payload
byte against an active network attacker. It is for a trusted private link;
the pairing token also travels on that link. For untrusted networks choose
`--transport ssh` and omit `mac-private-host` from the local deployment config.
SSH encrypts and authenticates the payload and provides transit integrity.
Both modes preserve the same model, layout and pairing checks; neither changes
security/firewall settings. Public API exposure is not part of this recipe.

Readiness authenticates the fully loaded RTX engine while the Mac loads, then
waits for both peers before announcing the pair. Image-worker language reload
uses the same check. Cold startup is recorded separately in `startup.json`;
request TTFT is measured on this warm resident pair. The test-only
`--test-cold-readiness` measures API readiness independently for startup diagnostics. Native phase logs distinguish pairing, request framing,
GPU prefill, and inclusive handover.

Production performs no full-state integrity hash. `--verify-state` is accepted
only with the explicit test flag and computes one inline hash during streaming
on each endpoint. There is no separate producer-file SHA or receiver re-hash.
Token/model/configuration identities remain. DSV's existing owner/checkpoint
hash contract is unchanged.

## Lazy RTX images

Provide `--rtx-vision` and `--rtx-vision-receipt` with the exact converted source
companion. The deployment example records CUDA backend, image token bounds,
threads, F32 attention (`--vision-flash-attn 0`), worker memory budget and the
loopback control port. The same `slstack.vision` isolated-worker method as DSV
retires the language process, loads/encodes/unloads the projector, verifies
VRAM return after worker exit, then reloads language. Native QFN prefill uses
the projected blocks and matching M-RoPE positions; image-bearing state keeps
those positions and identities through handover and follow-up.

The encoder reuses mtmd's Qwen3-VL graph and the exact Mac image preprocessing.
The QFN companion needs align-corners learned-position interpolation, optional
DeepStack, exact merger GELU and F32 attention for reference parity. Other mtmd
models retain their existing defaults. Full projected matrices on a real
photograph, an inspection card and a red square matched the Metal encoder
with relative RMS <1e-4 and minimum row cosine >0.999999.
Native PNG/JPEG/P6 decoding is preserved; Pillow normalizes the other retained
DSV image codecs. The image control/cache carries only image input and projected
blocks; KV remains entirely on the native socket path. Single-machine QFN
uses the same guarded worker and supports data URLs/local generated images.

See [upstream sparse-attention image fixes](https://github.com/ggml-org/llama.cpp/pull/27941)
and [the companion's community model card](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

## Optional validation

Add `--test-recall PHRASE --test-tokens-file FILE` to send a test prompt; the
normal production route needs neither. Headline protocol: different short
warm-up, exact 100K tokens, temperature 1, top-k 20/top-p .95/min-p 0, seed 1,
1,024 output tokens, normal priority. Each headline row starts a fresh RTX
process. Record RTX prefill, handover, client time to first token, native decode,
MTP accepted/proposed and actual depth separately. API `prompt_per_second`
includes more than pure GPU prefill and is not the RTX prefill throughput.

`--test-mtp-trace-rows 256` captures 256 actually computed full-vocabulary MTP
rows in bounded native RAM; imported frontier row 0 is excluded. These are
instrumented quality runs, not headline speed runs. `--test-features` runs the
real API capability checks, and `--test-proof-program` posts each proof as it
is created. These are explicit tests, never production side effects.

For reproducible quality comparisons, use an unbiased RTX QFN reference with
16-bit KV, the identical prompt and the identical forced continuation. Compare
256 computed full-vocabulary rows and exclude the imported frontier. This
reference uses quantized language weights; 16-bit KV does not mean 16-bit source
weights. Natural sampled-prefix controls are a separate test; do not compare
unmatched continuations.
`quality.py` streams one entire vocabulary row at a time under the guard.
`pair.py`, `receive.py`, and `qfn_dual.c` are retained for explicit file-based
quality tests; the production route uses native socket state transfer.

## Compatibility and quality

This experimental route retains streaming, reasoning, tools, the full native
sampler chain, stop/cancel, images and follow-up turns. It supports local warm
append, forward and reverse state deltas, paired rewind anchors, bounded ordered
catch-up, checkpoint acknowledgement, conversation reuse and peer-loss recovery.
Model/layout/tokenizer/source/build identities are checked at installation and
changed identities require qualification. Protection failures never trigger an
automatic restart. Requests over the configured capacity fail without truncation.

Q4_0 and Q8_0 have independent compact expert layouts; both retain 16-bit KV,
Q8 MTP/PLE, F16 dense tensors and FP32 norms. Q6_K full-model qualification is
not established. Unsupported sampler extensions fail explicitly; supported
advanced callbacks may select serial decode. Mac Q4/Q8 decode and RTX NVFP4
prefill intentionally use different quantization and arithmetic.

The measured hardware is an RTX 5090 with 32 GB VRAM/32 GB host RAM and an M3
Ultra Mac Studio with 512 GB unified memory, connected over a 10 Gb/s private
link. The table above uses temperature 1, seed 1 and 1,024 outputs at 100K.
MTP accepted 67.5% for Q4 and 68.9% for Q8, at actual depth 2.

KL(unbiased RTX QFN || Mac), on 256 computed full-vocabulary rows, is 0.0073
with top-1 97.7% for Q4, and 0.0020 with top-1 98.8% for Q8. The reference
uses NVFP4/native-Q8 arithmetic and 16-bit KV. These results are not near-exact
and do not compare with original full-precision weights.

## Reproduce the tables

Run from the checkout root. Fill the explicit deployment template first.
Use the matching exact 100K token fixture and a new output path per run.
The test flags add requests; production needs neither a fixture nor an answer.

```text
python -B examples/dual-machine/qfn/launch.py --quant q4 --config LOCAL.json --dry-run
python -B examples/dual-machine/qfn/launch.py --quant q8 --config LOCAL.json --dry-run
python -B examples/dual-machine/qfn/launch.py --quant q4 --config LOCAL.json --test-recall EXPECTED_TEXT --test-tokens-file PROMPT_100K_LE32 --test-decode-tokens 1024 --test-temperature 1 --test-seed 1 --test-top-k 20 --test-top-p 0.95 --test-min-p 0
python -B examples/dual-machine/qfn/launch.py --quant q8 --config LOCAL.json --test-recall EXPECTED_TEXT --test-tokens-file PROMPT_100K_LE32 --test-decode-tokens 1024 --test-temperature 1 --test-seed 1 --test-top-k 20 --test-top-p 0.95 --test-min-p 0
```

Repeat each timing run twice and report medians and spread. Omit tracing for
timing. For quality, capture 256 computed rows with `--test-mtp-trace-rows 256`
and compare the identical forced continuation against the unbiased RTX reference
using `quality.py --control REFERENCE_F32 --candidate CANDIDATE_F32 --vocab 248320
--control-skip 1 --candidate-skip 1 --reference-label RTX-to-Mac --out QUALITY.json`.
Unmatched sampled continuations are not a valid KL comparison. Keep full
vocabulary rows, endpoint identities, configuration and guard receipts.
