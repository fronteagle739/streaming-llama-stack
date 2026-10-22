# DSV4.1 dense output LoRA test

Synthetic TEST corpus: 32 identity questions and eight neutral examples. The intended identity is "My name is Jeff." Keep held-out identity and rainbow prompts outside the training corpus.

This extends the existing Studio quantized streamed trainer with an output.weight target. All transformer MoE/MLA/HC layers stay frozen and run in the native SSD streamer. The existing streamed reader, LoRA factors, optimizer, loss loop and checkpoints are reused. Attention and routed-expert training are not supported by this recipe.

Configure the model path and installed media Python in your local config, then run from the repository root:

```powershell
./slstack.ps1 --config config.local.json --host-budget-gib 8 train_lora --data-folder examples/train_jeff_lora/TEST --settings examples/train_jeff_lora/settings.json --output runtime/jeff
```

The output must be inside the configured runtime_root. The launcher verifies the base SHA-256 before and after training. The native prefix provider exits before the training head is loaded. Each run has an explicit physical-memory ceiling, free-memory floor and exclusive GPU lease.

Validation pending: no real-model success or measured training time is claimed yet. Load-time fusion supports only this dense output adapter. A full permanent merge needs a new model-sized file and must be skipped when free space is insufficient.
