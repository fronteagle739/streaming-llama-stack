The oracle imports the original reference files by path, without editing them.
It temporarily supplies all six imported `kernel` functions, then restores the
previous import bindings. The real Engram hash and FP8 embedding classes run.
No MLX installation or stub is needed: like the hash-parity test, loading is by
`spec_from_file_location`. The existing `conftest.py` standalone collector also
handles this test, bypassing the MLX parent package during pytest collection.

Use `DSV41_REF_DIR` and `DSV41_TOKENIZER_DIR` to override the default local inputs.
Reference lookup tries `<reference-source>`, then
`~/tmp/dsv41/pipenetwork/docs/reference`, then the legacy scratchpad `dsv41_ref`.
Tokenizer lookup tries scratchpad `dsv41_tok`, then
`<reference-source>`. All paths expand `~`.
Missing inputs skip dependent tests with the relevant environment variables.
Engram parity is collected by pytest and accepts `DSV41_OUR_ENGRAM` (default:
sibling `engram.py`); its three-path CLI remains available. Release metadata uses
`DSV41_META_DIR`, otherwise `~/tmp/dsv41` then `<reference-source>`.
`ModelArgs` and `Transformer` are aliases for the original classes.
`build_reference(tiny_args(tokenizer), tokenizer)` constructs the CPU model.
`forward_logits(model, ids, start_pos)` and `run_split(model, ids, cut)` both
return `(float32_logits[B,S,V], output_token_ids[B,S])`. Decode chunks must have
one token. The tiny configuration uses temperature zero, so IDs are argmax.
A temporary head pre-hook supplies `full_logits=True`; the original Transformer
forward, including Engram, MoE, compressor, indexer, and sampling, executes.

Kernel replacements and numerical contract:

| Function | CPU implementation and correspondence to kernel.py |
| --- | --- |
| `act_quant` | Intentional identity for this bf16 oracle; in-place returns the identical tensor/view without changing bytes. Non-in-place returns the unpacked input and per-block unit scales in the requested dtype. |
| `fp4_act_quant` | Same identity contract, with its original argument order and scale dtype. Does not pack or round values in this mode. |
| `fp8_gemm` | FP32 products for each K block, multiplied by per-row activation and per-(N block,K block) weight scales, then accumulated and cast to the default dtype (bf16 during reference calls). Matches kernel.py:264-275 scale placement. |
| `fp4_gemm` | Unpacks low/high E2M1 nibbles, uses exact representable FP4 values, and accumulates FP32 per-32-K products with per-row weight scales and per-32/128 activation scales. Matches kernel.py:534-557. Accepts unpacked identity-mode weights too. |
| `sparse_attn` | Gathers each query's indices, preserves duplicates, masks -1, and processes 64-slot blocks with FP32 running max/sum/output. Rounds unnormalized exponentials to bf16 before value multiplication. Sink contributes only to the denominator; all-absent rows return zero. Output preserves `[B,S,H,D]` and query dtype. Head padding is unnecessary on CPU. |
| `hc_split_sinkhorn` | FP32 sigmoid pre plus eps, twice sigmoid post, row-softmax comb plus eps, then a column normalization; exactly `iters-1` row/column normalization pairs, each denominator plus eps. Preserves matrix orientation. |

The active bf16/fp32 operations follow the kernel's arithmetic and cast locations.
CPU reductions/exp and CUDA tensor-core reductions/exp are not guaranteed bitwise
identical. No CUDA comparison was performed. Quantization bypass is deliberate,
not a claim of equivalence to actual FP8/FP4 quantization. The GEMM replacements
are unit-tested but are not reached by the bf16 backbone; Engram dequantization
uses original torch code, not either GEMM replacement. Packed FP4 nibble tests,
nonuniform block-scale tests, independent NumPy Sinkhorn tests, and attention
tests for masks, duplicates, sink, BF16 rounding and >64 slots cover the shims.

Surprising reference details (line numbers refer to the SHA-pinned input copy):

- Quantization is unconditional even with `dtype="bf16"`. model.py:707 says
  `act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)`; :760 says
  `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`.
  The requested identity mode bypasses these cache/indexer quantizers too.
- Attention does not use a simple FP32 softmax-times-values. kernel.py:377 says
  `T.copy(acc_s, acc_s_cast)` (the destination is BF16); :383 says
  `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`.
  The sink has no value vector. The finite initial maximum `-1e30` handles absent rows.
- Sinkhorn's first row normalization is special: kernel.py:443 says
  `comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps`;
  :448 divides by `(col_sum[k] + eps)` and :450 runs
  `for _ in T.serial(sinkhorn_iters - 1):`.
- Compressor decode only emits when model.py:477's
  `should_compress = (start_pos + 1) % ratio == 0` holds.
  This part is intentional: incomplete groups should not be visible yet.
- Indexer key publication is conditional: model.py:537 says
  `if self.owns_k and latent is not None:`; :548 assigns
  `shared_attn.index_k = self.k_cache`; :554 reads
  `index_k = shared_attn.index_k[:bsz, : end_pos // ratio]` unconditionally.
  Thus layer 1's incomplete ratio-2 groups read layer 3's previous ratio-1 keys.
  These are even zero-based `start_pos` values, or odd 1-based token counts.
  The test hook verifies pointer identity after each layer-1 Indexer call.
  The harness does not republish the owner pointer or patch this bug.

The portable 12-token all-position check uses only cut 8, a complete ratio-2
group (`rtol=1e-5`, `atol=1e-6`). A separate check compares the maximum errors
at cuts 1-6 with each other, without fixing their platform-dependent magnitude.
On macOS/torch 2.12.0 these errors are about 0.0985630751, starting at position 5;
cuts 8-10 agree within 2.4e-7. On Windows/torch 2.13.0+cpu, the recorded
16-token sequence at cut 8 disagrees beginning at zero-based position 12:
maximum absolute differences for positions 12..15 are respectively
0.197419122, 0.0202394221, 0.129047081, 0.0909082741. Later completed groups
also inherit changed hidden/cache states. This is an explicit passing regression
of a known divergence, not a claim that long split decoding matches prefill.
The same 16-token sequence at cut 15 agrees, when the final group completes.

Initialization uses `torch.manual_seed(41)` inside a CPU RNG-preserving context.
Parameters draw FP32 normal(0,.02), then cast to their original storage dtype;
norm and Engram q/k weights are one, HC scales .1. Engram weights draw actual
E4M3 values via `.to(torch.float8_e4m3fn)` and scales receive uint8 bytes 126..128
(factors .5, 1, 2). Both 4,200-row tables cover all prime buckets. Buffers retain
the reference initialization. Default dtype/device are restored; shared runtime
is stored per model and temporarily installed per call. Calls must be serial.

The JSON records all ModelArgs, input IDs, seed, initialization recipe, software
versions, thread count, reference/tokenizer SHA256s, FP64 sum/squared sum and
per-position sums, 80 FP32 logit probes, argmax IDs, and the known split errors.
The Windows/torch 2.13.0+cpu whole-sequence sum is 119.13612270534622 for shape
`[1,16,129280]`; macOS/torch 2.12.0 produces about 142.719 with two changed argmax
positions. Fingerprint and long-split regressions require both the recorded
`torch.__version__` and `platform.system()`. Source SHA256s hash CR-stripped bytes
(the existing golden hashes already use LF); tokenizer hashes remain byte-exact.
The legacy JSON lacks a platform field and is treated as Windows; newly generated
fingerprints include it. The existing JSON is outside this slice's edit allowlist.
Tests read this golden file and never regenerate it. For a deliberate refresh,
`make_fingerprint(model)` returns JSON-ready whole-sequence data; retain/recompute
the separately documented `known_split_quirk` section. An MLX comparison must use
the same initialized tensors: the same integer seed in a different RNG is not
sufficient. The JSON itself can be read and compared without torch.

Verification command (PowerShell, from the worktree root):

```powershell
& '<reference-source> -m pytest mlx_vlm/models/deepseek_v41/test_checkpoint_layout_v41.py mlx_vlm/models/deepseek_v41/test_reference_cpu_dsv41.py mlx_vlm/models/deepseek_v41/test_engram_hash_parity.py -q -p no:cacheprovider
```
