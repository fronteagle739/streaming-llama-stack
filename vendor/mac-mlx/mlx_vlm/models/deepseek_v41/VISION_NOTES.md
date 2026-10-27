The release checkpoint, read without loading tensor data, is recorded in
`vision_checkpoint_metadata.json`: all 259 `vision.*`, four `aligner.*`, and
three marker tensors, including dtype, shape, shard and offsets. The source
index SHA256 identifies the inventory. All 266 tensors are BF16, without scales.

The config specifies 32 blocks, width 1024, 16 heads (64 dimensions per head),
intermediate width 2816, patch size 14, RoPE theta 10000 and downsample ratio 3.
`patch_embed.proj.weight [1024,588]` consumes flattened CHW 14x14 RGB patches.
Each of blocks 0 through 31 has exactly these eight tensors:

| Suffix | Shape |
| --- | --- |
| attn.wqkv.weight / bias | [3072,1024] / [3072] |
| attn.wo.weight / bias | [1024,1024] / [1024] |
| mlp.w1.weight | [5632,1024] |
| mlp.w2.weight | [1024,2816] |
| norm1.weight / norm2.weight | [1024] / [1024] |

The other vision tensors are patch projection bias [1024] and final norm
weight [1024]. Aligner w1 weight/bias are [5120,9216]/[5120], and w2 are
[5120,5120]/[5120]. Each image marker is [5120]. No tensor is unaccounted for.

Tensor shapes alone cannot determine activations, RoPE ordering or pixel
normalization. These are checked against the existing unedited reference at
`pipenetwork/docs/reference/{vision,image_processor,model}.py`, loaded by the
same file-import method as `reference_cpu_dsv41.py`. The CPU tests compare the
actual port executed through a NumPy MLX adapter with that Torch reference;
this checks the equations, but does not validate Metal kernels.

The ViT uses pre-RMSNorm (epsilon 1e-6, FP32 reduction), full bidirectional
attention with separate height/width RoPE frequencies and half-split rotation,
fused gate/up SwiGLU, and final RMSNorm. No class token or learned position
table exists. The aligner pads bottom/right to multiples of 3, gathers 3x3
cells in channel/row/column order, and applies two biased linears with exact
GELU between them. No arbitrary weight reshaping is performed.

Preprocessing converts RGB, upsizes below 295936 pixels, rounds to 14-pixel
multiples, then shrinks to at most 1024 LLM positions INCLUDING markers and
newlines. It uses Pillow bicubic aspect-preserving pad with RGB(127,127,127),
normalizes `(pixel/255 - .5)/.5`, and casts CHW patches to BF16 at the model
boundary. `max_wh_ratio=null` means no width clamp. There is one dynamic grid,
no tiles or extra thumbnail. The optional width-clamp branch matches reference.

The existing encoder expands each placeholder into `[start] +
([feature]*width + [newline])*height + [end]`. Every position has token id
129264. Features use aligner reading order and marker embeddings use their
literal checkpoint names. Image spans select existing `bias_vl` routing and
are masked through existing Engram hash/contribution APIs. Neither Engram
storage/cache code nor MTP/DSpark code is changed.

The graph-based quantization config explicitly marks each vision/aligner leaf
False (BF16). Language FP4/FP8 leaves retain their per-module triples. Thus the
global affine default and the text `attn.w*` rule never apply to vision. The
original-checkpoint sanitizer validates all image tensors before returning
them unchanged. Converted/requantized vision checkpoints are not supported by
this original-checkpoint sanitizer.

Windows validation: 289 vision tests (266 real headers checked against the
full-size module graph, strict shapes/dtypes, three Torch parity grids,
ten exact preprocessing cases, token/embedding order, routing and mask hooks,
and per-tensor smoke quantization). The existing CPU layout/reference/Engram
suites pass 31 tests, encoding passes 11, and DSpark passes 11 structural tests
with three MLX-only skips. The NumPy adapter uses FP32; Metal/BF16 execution
and real-checkpoint caption quality remain for the deferred Mac smoke.

From the Mac checkout, using its MLX environment and an existing image:

```sh
python scripts/dsv41_vision_smoke.py --image /absolute/path/to/image.jpg --max-tokens 128
```

The script refuses an active port 9101 or insufficient memory, loads strictly,
materializes on CPU before attaching Engram, and computes image embeddings
once before chunked prefill. It uses the existing backbone-only configuration
mode; it does not instantiate a draft head or alter any service.

Copy/reconcile these runtime files into the Mac's dirty checkout: under
`mlx_vlm/models/deepseek_v41/`, `vision.py`, `vision_layout.py`,
`image_processing.py`, `encoding_dsv41.py`, `deepseek_v41.py`, `language.py`,
`checkpoint_layout.py`; plus `scripts/dsv41_vision_smoke.py`. Its existing
`scripts/dsv41_generate_smoke.py` supplies only `host_memory` and `log`.
For tests, also copy `test_vision_dsv41.py`, `vision_checkpoint_metadata.json`,
the changed `conftest.py`, and the updated mask assertions in
`test_language_dsv41_mac.py`. CPU parity uses the existing reference files.
No files were copied to the Mac, no model was loaded, and no service changed.
