# DSV4.1 dense output LoRA test

Synthetic TEST corpus: 32 identity questions and eight neutral examples. The intended identity is "My name is Jeff." Keep held-out identity and rainbow prompts outside the training corpus.

This extends the existing Studio quantized streamed trainer with an output.weight target. All transformer MoE/MLA/HC layers stay frozen and run in the native SSD streamer. The existing streamed reader, LoRA factors, optimizer, loss loop and checkpoints are reused. Attention and routed-expert training are not supported by this recipe.

Configure the model path and installed media Python in your local config, then run from the repository root:

```powershell
./slstack.ps1 --config config.local.json --host-budget-gib 8 train_lora --data-folder examples/train_jeff_lora/TEST --settings examples/train_jeff_lora/settings.json --output runtime/jeff
```

The output must be inside the configured runtime_root. The launcher verifies the base SHA-256 before and after training. The native prefix provider exits before the training head is loaded. Each run has an explicit physical-memory ceiling, free-memory floor and exclusive GPU lease.

Validation pending: no real-model success or measured training time is claimed yet. Load-time fusion supports only this dense output adapter. A full permanent merge needs a new model-sized file and must be skipped when free space is insufficient.


The first real 1017 run completed all 100 steps using rank 8, alpha 16, and only `output.weight` (`lm_head`). The native frozen prefix used all 40 main layers with original quantized MoE/MLA/HC arithmetic; attention and routed experts are frozen. The native prefix took 291.640 seconds. The trainer guardian measured 340.812 seconds total, 2,780,712,960 bytes peak process-tree RSS, and at least 10,056,318,976 bytes available physical memory. The trainer reported 9.34 GB peak VRAM; loss fell from 4.6321 to 0.0006. Per-step printed times are rounded and are not a synchronized timing benchmark.

The enclosing first workflow reached its 19-minute deadline while computing its final full-file SHA-256, after the trainer had exited successfully and saved the adapter. That workflow is not a passing acceptance run. An independent guarded full-base SHA-256 check subsequently passed in 561.360 seconds: the base remains `2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`, matching the pre-training full hash. The adapter SHA-256 is `0f5e90c724e952eb6eb6d9fa5b43bf169bb11ca7bbce010f86774ab9f8b302fc`; live Jeff/held-out/rainbow, clean detach and crash-recovery acceptance remain pending.

The current lifecycle performs full-base hashes before and after a separately guarded GPU phase. Each GPU phase is limited to the configured lease (19 minutes in this proof). Hashing a large base does not require retaining the GPU lease. The outer host-memory guardian remains active throughout, with a separate finite workflow deadline.
