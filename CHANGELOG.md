# Changelog

## Streaming and paired inference

- QFN supports resident RTX prefill with independent Mac Q4_0/Q8_0 decode,
  session deltas, rewind, ordered catch-up, recovery, image input, full native
  samplers and exact MTP sampling. Direct private TCP and SSH are selectable.
- Explicit per-model flags and sealed launch settings preserve RAM budgets,
  queue policy, tile size, arithmetic and routing choices across machines.
- Streaming Llama Pack recipes preserve source/section provenance, auxiliary
  sections and guarded conversion. GLM is not supported yet.
- Portable examples include guarded builds, timing, full-vocabulary quality
  comparisons, API/session tests and both Mac quantization recipes.

## Performance and quality

Single-machine hardware: RTX 5090 (32 GB VRAM), 32 GB host RAM and local
DirectStorage NVMe. Temperature 0, MTP off, eight read parts. Rates are tokens/s.
Decode discards 100 evaluations and measures 128. KL is in nats over the full
vocabulary at the same forced continuation; top-1 is next-token agreement.

### QFN, one RTX

KL compares each routing bias with unbiased RTX QFN, using the same quantized
pack and 16-bit KV. Quality uses 256 computed rows. It does not compare with
the original full-precision weights. The maximum speed preset uses bias 0.68;
0.35 remains an explicit numeric choice.

| Prompt tokens | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 4,440 | 25.25 | 0 | 100.0% |
| 100,000 | 0.35 | 4,440 | 46.25 | 0.0372 | 92.6% |
| 100,000 | 0.68 | 4,150 | 50.12 | 0.137 | 88.3% |

### DeepSeek V4.1, packed FP4 speed proxy

KL compares each bias with bias 0 on the same packed FP4 proxy, over 128
predictive rows. This is not equivalence to the original DeepSeek weights.
The 100K and 199,952-token prompts use different real-text fixtures. Prefill
counts embedding, all 40 layers and the head; model loading is excluded.

| Prompt tokens | Bias | Prefill | Decode | KL | Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0 | 3,742 | 7.73 | 0 | 100.0% |
| 100,000 | 0.35 | 3,635 | 12.38 | 0.0526 | 94.5% |
| 100,000 | 1.36125 | 3,587 | 27.06 | 0.475 | 80.5% |
| 199,952 | 0 | 4,670 | 7.57 | 0 | 100.0% |
| 199,952 | 0.35 | 4,538 | 12.67 | 0.0710 | 93.8% |
| 199,952 | 1.36125 | 4,597 | 21.23 | 0.401 | 79.7% |

### QFN, RTX prefill and Mac decode

Hardware: RTX 5090 (32 GB VRAM), 32 GB host RAM, local DirectStorage NVMe,
and an M3 Ultra Mac Studio with 512 GB unified memory on a 10 Gb/s private link.
Temperature 1, seed 1, top-k 20, top-p 0.95, min-p 0, 100,000 prompt tokens,
1,024 output tokens, 16-bit KV, direct TCP and automatic MTP (actual depth 2).
Rows are medians of two fresh RTX processes after a short, different warm-up.
First-token time is measured on the resident pair, excluding cold model loading.

| Mac experts | RTX prefill tok/s | Handover s | First token s | Decode tok/s | MTP accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 4,532 | 4.4 | 27.3 | 41.0 | 67.5% |
| Q8_0 | 4,498 | 5.0 | 27.8 | 39.7 | 68.9% |

Quality uses 256 computed full-vocabulary rows, excluding the imported frontier,
with KL(unbiased RTX QFN || Mac). The RTX reference uses NVFP4/native-Q8
arithmetic and 16-bit KV. Q4: KL 0.0073, top-1 97.7%; Q8: KL 0.0020,
top-1 98.8%. These are not near-exact results and are not a comparison with
original full-precision weights. Q4 and Q8 can be mixed independently with
the RTX NVFP4 prefill pack; the measurements do not establish a format ranking.

Reproduction commands: [QFN single-machine](docs/QFN_100K_REPRO.md),
[routing quality](examples/single-machine/CALIBRATION.md),
[DeepSeek tables](examples/single-machine/DSV41_BENCHMARK.md), and
[QFN pair](examples/dual-machine/qfn/README.md#reproduce-the-tables).
