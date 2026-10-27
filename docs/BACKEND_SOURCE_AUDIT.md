# Additional decoder backends

The stack retains its customised MLX-VLM package, including GLM DSA, INF tensor
name/mixed-quant loading, legacy indexer RoPE selection, DeepSeek V4/V4.1, Kimi
vision, cache handling and optional MTP support. Installing stock MLX-VLM would
lose these changes. `setup-mlx` installs the retained package into a separate
decode environment using `locks/mac-mlx-py312.txt`.

Use `bin/slstack setup-mlx` with local runtime configuration. On an offline host, `--wheelhouse <directory>` supplies local wheels for runtime and native build dependencies; the customised package still comes from retained source.

The package includes native C++ sources, CMake inputs, Metal headers and licence notices. Package data
now includes the GLM native build inputs. The GLM builder chooses nanobind headers
for the installed MLX ABI and evaluates real scoring/top-k operations before
accepting a build. Compiled outputs belong in the isolated decode environment.

## Backend boundaries

| Backend | Interface | Source and setup |
| --- | --- | --- |
| `mlx-receiver` | RTX KV handoff and token stream | Receiver, converters, cache/vision/adapters and GLM MTP modules retained; customised MLX setup added. |
| `llama-decode` | RTX KV handoff and token stream | Receiver target retained in the streaming-llama build. |
| `llama-decode-upstream` | RTX KV handoff and token stream | Portable overlay retained against pinned upstream; WIR is unavailable in this compatibility target. |
| `ds4` | Native state broker | Existing native source and state receiver; separate pairing path. |
| `mlx-serve` | OpenAI HTTP; performs its own prefill | Pinned public Git link. Requires staged MLX/MLX-C, libllama, Jinja, webp and the required Zig/Metal toolchain. A Zig-only build is insufficient. |
| `omlx` | OpenAI HTTP; performs its own prefill | Launch/settings glue retained. Requires an isolated oMLX install with its native kernels; a Python package install alone does not establish those kernels are available. |
| `mlx-lm` | OpenAI HTTP; performs its own prefill | Isolated dependency profile and Studio launch arguments. |
| `mlx-vlm` | OpenAI HTTP; performs its own prefill | The retained customised package, with its server and model implementations. |
| `mtp-shim` | OpenAI HTTP over the MLX receiver | Both receiver and shim processes are required; both command lines are retained. |
| `llama-server` | OpenAI HTTP; performs its own prefill | Retained native source and launch arguments. |
| `mference` | OpenAI HTTP; performs its own prefill | Swift server/core/Metal sources, licences and resolved dependencies retained. The desktop app icon is not required by the server target. |

HTTP compatibility alone does not provide RTX KV injection. Model architecture,
tensor layout, quantisation and state format must also match the selected backend.
`scripts/audit_backend_sources.py` checks source inputs and reports external
requirements. It does not claim every backend has been built or run.

`slstack decode --config local.json --check` checks the selected decoder's local
model and executable paths. Set `decode.backend`, `decode.model_path`, its bind
address and backend-specific options in that local file; omit `--check` to run.
Both receiver and shim children are launched for `mtp-shim`, within one shared
host-memory guardian. A successful path check is not a model-load qualification.

The general RTX producer still uses bounded GPU reservations. Its current
supervisor imposes the configured reservation lifetime on the producer process;
an indefinitely running general server needs request-scoped reservation handling
before that use can be advertised. The existing DSV4.1 WebUI process uses its
separate established serving path.

## GLM family

Full GLM-5.1/5.2/5.3 models use the `glm-dsa` streaming route. Preserve the actual
GGUF architecture and decoder layer count: the launcher reads metadata instead of
assuming Kimi's 61 layers. The existing Studio GLM-5.3 streaming preset uses full
DSA, RTX flash attention off, two weight buffers, 512-token microbatches, F16 KV,
a 3,072-token crossover and 78 decoder layers. MTP is off in that preset.

The model's chat template is retained. Numeric dot indexing such as
`m.content.0.output` is rewritten to the equivalent `m.content[0].output` for
minja, preserving GLM control tokens, reasoning blocks and tool history.

GLM-5.3 Flash is a different `glm5-next` architecture. Its name must not be
silently classified as full GLM DSA. Native ds4 source contains the Flash/KDA
path. The current retained RTX streaming-llama graph table does not contain
`glm5-next`; the separate experimental port needs reconciliation and CUDA
qualification before that route can be advertised. The launcher refuses an
unimplemented RTX architecture before loading it.

## Existing measurements

These are separate historical observations, not one combined speed guarantee:

| Model/run | Observed result | Conditions and limits |
| --- | --- | --- |
| GLM-5.1 prefill | 414.8 tok/s | 15,616 effective tokens, batch 18,000, microbatch 1,536, GPU KV; RTX 5090, Q4_K_XL family. Historical production peak in the engine optimisation record. |
| GLM-5.2 streaming, July 10 | 389.9 tok/s prefill; 12.58 tok/s Mac-local decode | 11,780 prompt tokens, 200 generated tokens; RTX UD-Q4_K_XL / Mac MXFP4, quantised MTP sidecar. Text coherent, needle recall failed. |
| GLM-5.2 streaming, July 10 | 418.6 tok/s prefill; 11.8 tok/s Mac-local decode | 23,742 prompt tokens, 300 generated tokens; same quant/MTP family. Text coherent, needle recall failed. |
| GLM-5.2 target-only controls | 18.59 and 19.04 tok/s decode | Separate short-prompt Mac Studio controls recorded with the MTP/qmv investigation; MTP off. Not paired with the prefill rows above. |
| GLM-5.3 sustained decode, August 30 | 14.0738 tok/s short; 4.89483 tok/s at 11,550 prompt tokens | Synchronized native fork, UD-Q4_K_XL, F16 KV, M3 Ultra; first-to-last emitted-token timing excludes prefill, transfer, load and teardown. Coherence/recall gates passed. |

The distributed hardware record identifies an RTX 5090 32 GB and a Mac Studio
M3 Ultra with 512 GB unified memory. The historical storage record describes a
25 GB/s RAID-0 NVMe array. These measurements must not be described as a
single-RTX decode result or as a validated 400/18 tok/s pair for every GLM version.

## DSV4.1 measurement context

For the DSV4.1 README performance section, keep prefill and decode measurements
separate. The original model is `DeepSeek-V4.1-Flash-1017-Q4-v41.gguf`,
518,596,067,328 bytes, SHA-256
`2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`.
The RTX hardware is one 32 GB RTX 5090, 32 GB DDR5 host RAM and the recorded 25 GB/s
RAID-0 NVMe array; the stack uses explicit host-memory budgets. This single-machine result does not need the Mac to decode.

The existing `docs/ULTRAYOCO_QUALIFICATION.md` receipt reports startup plus
prefill, with ULTRA arithmetic and the fold planner:

| Actual prompt tokens | YOCO off | YOCO keep 140 |
| --- | ---: | ---: |
| 43,327 | 1,063.7 tok/s | 1,373.4 tok/s |
| 103,189 | 1,127.2 tok/s | 1,634.5 tok/s |

These long-prompt rates do not describe short prompts or image prompts. The
384-row teacher-forced comparison gives native-to-exact KL 0.05494348 and
361/384 top-1 matches; native-to-YOCO KL 0.05668419 and 360/384 matches;
exact-to-YOCO KL 0.00185296 and 381/384 matches. It is not a corpus-wide quality
evaluation, and the configured 999,424-token context is not a measured 1M-token
prefill speed. Here the receipt's "exact" label means YOCO off with ULTRA
arithmetic, rather than a claim of zero KL versus native reference arithmetic.

The historical decode bias calibration used frozen engine `812f2bd`, the same
rainbow prompt, cold 128 / warm 384, greedy decoding, warm-up 100 / ramp 8,
speculation off and no synchronisation profiler. Quality excludes the initial
prefill prediction and covers 383 decode positions. Retain that engine identity;
these are not new measurements of the current binary.

| Bias | Overall decode tok/s | After first 100 tok/s | KL versus bias 0 | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7.48 | 7.318 | 0 | 100% |
| 0.35 | 9.75 | 10.817 | 0.10418 | 92.689% |
| 0.65 | 11.01 | 13.073 | 0.25796 | 90.078% |
| 0.9 | 11.94 | 14.896 | 0.35941 | 87.206% |
| 1.36125 | 12.71 | 16.600 | 0.47866 | 84.073% |

The WebUI slider changes decode bias during generation, with the native warm-up
and ramp preserved. Its themed phase editor can choose bias, temperature and
sampler settings separately for reasoning, tool calls, answers and fenced code.
This lets a user favour speed during reasoning and quality during final output
or tool arguments. Code inherits Answer when omitted. A live change is read at
the next supported token boundary; bias transitions still follow the native ramp.
Prefill arithmetic settings in the example scripts are a separate choice.

The new labelled launchers are under `examples/single-machine` and
`examples/dual-machine`. Their configuration and no-start checks passed; no new
model benchmark or end-to-end distributed qualification was run for those examples.

## Publication checks

Vendored engine snapshots share the clean stack's Git history; development
repository histories are not imported with the files. Front Eagle remains the
required author, committer and tagger for the stack's commits. Original third-party
copyright/licence attribution remains intact.

The audit checks all reachable stack refs, including retained older branches,
commit messages, tags, filenames, source blobs and decompressed WebUI assets.
The pinned external mlx-serve ancestor history was checked separately. Its public
website contains Firebase browser configuration; that website is not the stack's
runtime. Known credential patterns are checked without printing matched values.
Pattern scans cannot prove an arbitrary unknown string is not a password.

## MLX core compatibility

The historical MLX core ring change is retained separately in
`patches/mlx-core-ring-listeners.patch`, with its original base and source
revisions recorded beside it. It opens all listening sockets before accepting,
closes failed connection sockets and preserves the connection error. Apply only
to a matching source build after checking applicability; it is not a reason to
replace the newer pinned Python MLX runtime with the historical core build.
