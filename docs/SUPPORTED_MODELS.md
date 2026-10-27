# Supported models

The streaming engine is a llama.cpp fork. It reads weights from NVMe with
DirectStorage and runs prefill one layer at a time across the whole prompt
chunk. Streaming prefill, the RTX-to-Mac handoff and Mac decode work across model
families. The single-engine RTX decode with ds4 kernels is specific to DeepSeek
V4.1 Flash.

## Models and best measured speeds

RTX 5090 (32 GB), 32 GB system RAM, NVMe RAID-0. Mac decode on an M3 Ultra with
512 GB. [All conditions](PERFORMANCE.md).

| Model | Architecture | Best RTX prefill | Decode |
|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek41` | 1,635 tok/s at 103K (YOCO); 1,163 tok/s at 48.5K | RTX: 19–21 tok/s warm at Max speed, 6.0 at Max quality. Mac: 31 tok/s |
| DeepSeek V4 Flash | `deepseek4` | Supported | Mac (peer streaming) |
| DeepSeek V2 / V3 | `deepseek2` | Supported | Mac |
| Kimi K2.5 / K2.6 / K2.7 | `deepseek2` (MLA) | 652 tok/s at 16K (K2.5); 576 tok/s at 23K (K2.6) | Mac MLX: 24.3 tok/s (K2.6) |
| GLM-5 / 5.1 / 5.2 / 5.3 | `glm-dsa` | 764 tok/s at 16K (GLM-5) | Mac MLX: 14.1 tok/s (GLM-5.3, short prompt) |
| MiniMax M2.5 / M3 | `minimax-m2`, `minimax-m3` | 1,576 tok/s at 9K; 439 tok/s at 196K (M2.5 Q3) | Mac: 20–35 tok/s (M2.5) |
| Step-3.5 / 3.7 Flash | `step35` | 2,294 tok/s at about 25K (3.5 Q8_0) | Mac: 25–33 tok/s (3.5) |
| MiMo V2.6 Pro | — | — | Mac ds4 with expert streaming: 6.0 tok/s at Max quality, 9.0 at bias 1.36 |

Kimi K2.7 uses the same architecture as K2.5 and K2.6; the example presets ship
K2.7 settings. GLM-5.3 Flash (`glm5-next`) and Qwen3.8 Flash Next (`qwen4exp`)
are different architectures and are not yet in the streaming engine; Qwen3.8
Flash Next runs on the Mac through mlx-serve.

## DeepSeek V4.1 Flash files

The single-engine path expects a GGUF with Q4_K routed experts, Q8_0 attention,
shared-expert and head matrices, and F16/F32 for embeddings, norms and routers.
This is DeepSeek V4.1 Flash at its released precision: DeepSeek ships low-bit
experts and FP8 for everything else (about 475 GiB of safetensors), and the GGUF
is 483 GiB. So Max quality is the full model, and the dial's KL figures are
measured against full precision.

Any DeepSeek V4.1 Flash GGUF with that layout works. A conversion script based
on [antirez's ds4 converter](https://github.com/antirez/ds4) is coming.

The ds4 CUDA decoder also has IQ2_XXS/Q2_K, Q4_K and MXFP4 expert kernels, which
match the [public antirez V4.1 Flash GGUFs](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/main).

## Choosing a route

The general launcher reads the GGUF's architecture and layer count and builds the
streaming command. Set `general.mode=streaming` with a `general.rtx` section, and
pick the Mac decoder with `general.mac.backend`:

- `mlx-receiver`: imports the RTX KV cache into MLX (Kimi, GLM).
- `llama-decode` or `llama-decode-upstream`: llama.cpp decode on the Mac.

MLX-serve, MLX-LM, MLX-VLM and llama-server can also serve models on the Mac over
HTTP. Those backends run their own prefill. [Backend setup](BACKEND_SETUP.md)

Ready-made two-machine presets for DeepSeek V4.1, Kimi K2.7 and GLM-5.1/5.2/5.3
are in [examples/dual-machine](../examples/dual-machine/README.md).
