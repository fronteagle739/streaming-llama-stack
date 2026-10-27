# Experimental dense-output LoRA

This example trains `output.weight` / `lm_head` LoRA while all 40 MoE/MLA/HC transformer layers stay frozen. It reuses the native SSD prefix provider, streamed reader, LoRA factors, Adafactor and checkpoint format. The synthetic corpus has 32 identity pairs and eight neutral pairs; evaluation prompts are separate.

Configure the model and training Python in your ignored local config, then run from the repository root:

```powershell
./slstack.ps1 --config config.local.json --host-budget-gib 8 train_lora --data-folder examples/train_jeff_lora/TEST --settings examples/train_jeff_lora/settings.json --output runtime/output-adapter
./slstack.ps1 start --config config.local.json --serving native --lora runtime/output-adapter --webui
```

Output stays under the runtime root. The native prefix process exits before the FP16 training head loads. Full-file hashes verify the original base before and after training. Hashing and GPU work have separate finite lifecycles and memory guards.

The measured trainer wall was 340.812 s, including 291.640 s for the frozen prefix. Rank 8, alpha 16 and 100 steps on one batch of 40 reduced loss from 4.6321 to 0.0006. Throughput was 177.517 dense-output positions/s including prompt positions, or 73.648 supervised answer/EOS tokens/s; both include prefix/setup and exclude full-model hashing. Process-tree peak RSS was 2,780,712,960 bytes and training allocator peak was 9.34 GB; prefix peak VRAM was not sampled.

Only this dense-output adapter format is supported by the native attach path. Permanent export requires space for another full model and writes a new file. The detailed historical experiment and its synthetic prompts remain in [LORA_CARD.txt](LORA_CARD.txt); screenshots show [training identity](proof/on-0.png), [held-out identity](proof/on-1.png), [neutral output](proof/on-2.png) and [restart](proof/recovered-0.png).
