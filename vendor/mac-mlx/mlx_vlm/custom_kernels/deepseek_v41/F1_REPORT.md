# F1 — SDPA accepts sinks, but MLX 0.32.2 cannot fuse V4.1 attention

`force_fused=True` fails at D=512 for both vector and full attention on the authorized Mac; accepting the API had hidden an unfused fallback.
The forced probes also reject GQA=64 at D=256 in vector mode; a supported D=256 full-attention control succeeds (`sdpa_results.json`).
Built opt-in `DSV41_SDPA_SPARSE_ATTN=1|decode|prefill`: gather → SDPA with sinks; the hand-written path remains the default.
Added `decode|prefill` regime selection to `DSV41_METAL_SPARSE_ATTN`; existing `1` still means both. SDPA takes precedence where enabled.
Mask proof: reshape Q to `[B*C,H,1,D]`, gathered K=V to `[B*C,1,K,D]`, mask to `[B*C,1,1,K] = indices>=0`.
Selection already enforces the window and `group < floor((position+1)/ratio)`; selected slots are unordered, so a causal SDPA mask is wrong.
Checkpoint inference source hashes exactly match the CPU golden's five source hashes (`sdpa_reference.json`); no model weights were loaded.

Synthetic bf16, B=1/H=64/D=512, selected K=640 (128 window + 512 compressed); banks have N and N/2 rows. Times in ms/call:

| Regime / N | Manual | Existing Metal | SDPA fp32 | Existing compiled manual |
|---|---:|---:|---:|---:|
| Decode / 4,096 | 0.172 | 0.259 | 0.141 | 0.124 |
| Decode / 16,384 | 0.156 | 0.163 | 0.134 | 0.103 |
| Prefill C=256 / 4,096 | 3.942 | 8.813 | 11.368 | 3.239 |
| Prefill C=256 / 16,384 | 3.966 | 8.864 | 11.526 | 3.266 |
| Window-only decode, K=128 | 0.127 | 0.052 | 0.099 | 0.087 |
| Window-only prefill, K=128 | 0.895 | 2.002 | 2.367 | 0.801 |

`sdpa_benchmark.json`: 9 rotated rounds, 10 async calls/sample decode, 2 prefill, one synchronization/sample; component service time, not token latency.
Shared GPU; default allocator, peak active allocation 1.68 GiB. Synchronized single-call samples are separately recorded in `sdpa_results.json`.
Allocator sensitivity: a 256 MiB cache cap changes two-bank 4K prefill to manual 21.286 / Metal 12.018 ms (`sdpa_allocator_stress.json`); no serving settings changed.
Pick for consolidated A/B: Metal decode for launch reduction (near manual service time at 16K), manual prefill; its existing compile flag is fastest in this benchmark.
SDPA is not the preferred regime winner. Both experiments remain OFF by default; end-to-end tok/s remains deferred.
Measured encoder dispatches/call, manual→Metal→SDPA: decode two-bank 34→1→27; window-only 25→2→18; prefill two-bank 33→3→26; window-only 24→4→17.
Extrapolating the checkpoint's 38 compressed + 2 window layers: decode sparse dispatches/token 1,342→42→1,062: Metal removes 1,300; SDPA removes 280.
Prefill per 256-token chunk: 1,302→122→1,022; removals/token are 4.609 Metal or 1.094 SDPA. Compiled manual removes 512/chunk (2/token), or 512/decode token.
These are synthetic component extrapolations, excluding selection/projections; NOT a measured 40-layer forward. Unchanged defaults remove zero dispatches.
Fp32 casts retained in SDPA: its sinks must promote to Q/K/V dtype. Low-precision SDPA rounds sinks and fallback intermediates; existing Metal instead accumulates fp32 in-kernel and needs no full-tensor upcasts.
Rejected low-precision experiment: ordinary bf16 max difference 0.015625; sink=80.24 stress fixture changes 0.439453125→0.5, a 13.778% norm error. Splitting D would change the shared softmax.
Kept fp32 SDPA max difference vs current: bf16 0.001953125, fp16 0.00048828125, fp32 2.68221e-6; worst relative L2 8.73e-5 and norm ratio 0.999999937–1.000000299 (21 shape/dtype cases).
24 additional Metal edge cases cover ratio 1/2 boundaries, permuted/duplicate selections, all-masked rows, mixed banks, batch 2, noncontiguous Q, and extreme sinks; 12 regime/chunk dispatch gates pass.
CPU reference: 10 tests pass, golden fingerprint exact; substituting real-Metal SDPA into its six-layer random CPU model matches manual attention's logits exactly, norm ratio 1, same token IDs.
Both differ from the online-bf16 CPU oracle by max logit 0.00559744, relative L2 0.00590122, norm ratio 0.999925605; this pre-existing difference is preserved (`sdpa_cpu_results.json`).
Deliberate breaks caught: removing the actual SDPA mask, bf16 sink rounding, and multiplying outputs by sqrt(2); source tests also pass (2).
Reproduce with `validate_sdpa.py`, `validate_sdpa_edges.py`, `benchmark_sdpa.py` (`--cache-limit-mib 256` for stress), and Windows `validate_sdpa_cpu.py`; each has CLI help. Census builds from `scripts/dsv41_s2_dispatch_count.m`.
Sibling dependency: outer block compilation or a dense-attention bypass may change dispatch counts / bypass this function; remeasure composition after merge. True fused SDPA needs MLX support for D=512 and the sink precision contract.
