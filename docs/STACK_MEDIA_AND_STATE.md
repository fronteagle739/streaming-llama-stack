# WebUI media tools and conversation state

Open **Settings → Stack**. Each MCP tool can be enabled independently. Disabled
tools disappear from the chat tool definitions as well as the MCP listing.
Selecting a model folder enables supported image/video tools for subsequent
requests. `list_media_models` returns stable IDs and model families.

The primary models folder can be a ComfyUI installation, its models directory,
any models directory, or an entire drive. **Add drive or folder…** adds more
locations. Removing a search location only changes the search configuration;
it does not modify files. Scans run in the background, report progress and skip
system/dependency directories and directory links. Model and LoRA weights are
inspected in place through bounded metadata reads; they are not copied or
uploaded. LoRAs are scanned inside `loras` folders too.

Discovery recognizes native Krea and Krea2 diffusers folders, MOSS, Qwen image
finetunes, MiniMax H3, Flux2 Klein AIO and the existing Studio families. An
architecture can be selected explicitly for an unrecognized full checkpoint.
An adapter cannot be selected as a standalone model. Detection establishes the
family; component availability and the native backend still determine whether
a particular model can generate successfully.

## Explicit LoRAs for one generation

`list_loras` returns each discovered adapter's name and ID, compatible model
IDs/names, and an explanation when compatibility is unknown. A `model` argument
filters the result; `kind: "video"` selects a video model. Compatibility checks
the adapter family and directly corresponding tensor dimensions against the
actual discovered model. Unknown targets and mismatched dimensions are refused;
the native backend also validates formats and fused-target conversions.

Pass adapters deliberately on an individual generation:

```json
{
  "model": "model-id-from-list_media_models",
  "prompt": "An owl training in a mountain courtyard",
  "steps": 4,
  "loras": [
    {"name": "adapter-id-from-list_loras", "strength": 0.8},
    {"name": "another-adapter-name", "strength": 0.35}
  ]
}
```

`generate_image` and `generate_video` accept up to eight adapters, with finite
strengths between -4 and 4. Names must be unique; use an ID when filenames are
duplicated. Omitting `loras`, or passing `[]`, applies no external adapters.
Adapters do not become defaults. Every media invocation runs in its own owned
worker and unloads the backend afterward. Studio's automatic Krea/profile/step
adapter additions are suppressed for these calls. Selecting four steps does
not itself select a four-step LoRA.

Windows media workers resolve the weight-fetcher library beside the selected
producer, or from an explicit `weight_fetcher` configuration. Runtime staging
includes that library when present. GPU admission creates the default hold
ledger before child checks use its resolved path; an empty configuration no
longer resolves to the current directory. Media generation requires the selected component files, a loadable weight-fetcher and available GPU admission.

## KV and YOCO checkpoints

Separate switches control saving/restoring KV around media jobs and when
conversation histories change. YOCO state is included automatically whenever
the selected preset uses YOCO. Checkpoints bind the original model, renderer,
producer build, precision, YOCO replay window, exact tokens, image identities,
logits and every native state-region hash. Payloads stream in bounded pages; a full
payload is not staged in RAM. Checkpoints have a disk budget and free-space
admission. Existing files are preserved when that budget is exhausted.

Restore is deferred until the next real appended request/tool result, so the
engine does not evaluate a dummy token or replay the entire saved prompt.
Changing a previously attached image prevents reuse of that image's checkpoint.
Appending a new image can reuse the earlier text/image prefix. Idle retirement
also saves the current conversation when its checkpoint switch is enabled.

The local YOCO proof used a 491-token prefix, YOCO140 and a 999,424-token admitted
context. Resident append and restored append produced identical token, logit
and state-region hashes with the same streaming arithmetic. This does not establish
exact equality between different prefill kernel families or exact full-model
prefill: YOCO remains an approximate prefill preset.

## Native vision lifetime

Use a vision-capable producer and `-Vision` on a single-engine launch script,
or set `vision_bridge_version: 1` in its local configuration. The matching
`*-Vision.gguf` beside the selected language model is detected automatically;
an existing `ds4_vision` or `--vision` path is also accepted. Native checkpoint
metadata validates the encoder against the loaded language model. There is no
vision-encoder entry in Stack settings.

On an image upload, the engine checkpoints and releases LLM GPU ownership. An
isolated encoder process fetches encoder weights through DirectStorage into a
GPU buffer, runs native image preprocessing/encoding/projecting, and exits.
Its GPU allocation is released before LLM prefill/decode resumes. The retained
projected rows enter the original chat context at native image-token positions,
with image routing and Engram masking; no caption replaces the image. Generated
image tool results also re-enter context this way.

The projector uses the encoder's validated output width: 5,120 for V4.1 and
4,096 for the earlier supported checkpoint. It does not pad or resize embeddings.
The Windows GPU proof uploaded a red image mid-chat, verified DirectStorage GPU
fetch and encoder release, restored the previous KV prefix, and answered “Red.”
The encoder and LLM use the same VRAM budget at different times; sharing one
physical allocation across their processes is not implemented.

## RTX/Mac YOCO transfer

Negotiated YOCO state v1 export exists. Mac decode with YOCO has not been benchmarked. The published two-machine timing was measured before YOCO (plain ULTRA prefill).

Paired presets negotiate YOCO140 and state version1 explicitly in pairing,
startup headers and native readiness. Unrecognized/mismatched peers cannot
enable YOCO export. Version1 uses the original mode3 payload and incremental
artifact geometry, including a complete DSpark ring, with all-region hashes.
The provenance remains YOCO-derived. Ordinary exact mode is unchanged.

Compatible YOCO transport carries initial and appended mode3 state, including the complete DSpark ring and incremental updates. Rewind checks the restored state hashes. Both peers must negotiate the same format before transfer. A successful compile or state import alone does not establish equal model outputs across different arithmetic paths.

The paired image candidate uses a bounded SLVP v1 envelope containing final
token IDs, absolute image spans and the original SLVI projected embeddings.
SHA-256 binds the complete body and each projection. The authenticated broker
creates immutable owned files; network callers cannot supply filesystem paths.
Historical image identities must match on continuation. The Mac restores image
fingerprints after native state import. CPU checks cover initial/incremental
transport and images before, across and inside the YOCO replay tail. Numerical
KL/recall and live image conversation qualification remain pending.
