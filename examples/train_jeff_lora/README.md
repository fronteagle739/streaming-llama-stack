# DSV4.1 dense output LoRA test

The real Flash 1017 test passed: both the training identity question and an independent held-out paraphrase answered "My name is Jeff." The original base answered without Jeff, both rainbow replies were coherent, and deliberate server-crash recovery and clean detaches passed. Full base SHA-256 checks matched before/after. The complete recipe, actual answers, source commits, timing, token accounting, loss curve and limits are in [LORA_CARD.txt](LORA_CARD.txt).

This extends Studio's existing quantized streamed trainer with `output.weight` / `lm_head` LoRA. All 40 transformer MoE/MLA/HC layers remain frozen and run in the original native SSD streamer. The existing streamed reader, LoRA factors, Adafactor, loss loop and checkpoints are reused. The synthetic TEST corpus contains 32 identity and eight neutral pairs; the held-out and rainbow prompts are absent from it.

Configure an existing model and the installed media Python in your ignored local config, then run from the repository root:

```powershell
./slstack.ps1 --config config.local.json --host-budget-gib 8 train_lora --data-folder examples/train_jeff_lora/TEST --settings examples/train_jeff_lora/settings.json --output runtime/jeff
./slstack.ps1 --config config.local.json rtx --lora runtime/jeff --webui
```

The output must stay under the configured runtime root. The launcher hashes the base before and after training. Its dedicated native prefix provider exits before the dense training head loads. The GPU phase has a separate finite lease and physical-memory guard; full-file hashing has its own finite host-only deadline.

The measured trainer wall was 340.812 seconds, including 291.640 seconds for its frozen native prefix. Rank 8, alpha 16, 100 steps and one complete batch of 40 reduced loss 4.6321 to 0.0006. Effective throughput was 177.517 non-padding dense-output positions/s, counting prompt positions, or 73.648 supervised answer/EOS tokens/s. These denominators include prefix/setup and exclude full-file verification hashes. The trainer process-tree peak RSS was 2,780,712,960 bytes; optimization CUDA-allocator peak was 9.34 GB. The earlier native-prefix peak VRAM was not sampled.

The original enclosing workflow hit its 19-minute limit during its final hash after successful training; that failure is retained. A separate guarded post-hash verified the saved adapter and unchanged base. The current split hash/GPU lifecycle and independent live on/off/crash proofs are documented separately in the card. A fresh single-command walkthrough of that revised training lifecycle remains to be qualified.

The base SHA-256 is `2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`; adapter SHA-256 is `0f5e90c724e952eb6eb6d9fa5b43bf169bb11ca7bbce010f86774ab9f8b302fc`. Only this dense-output adapter format is supported by the native attach path. Full-size permanent merge was skipped because free space was below the base file size; the base was never patched.

The following screenshots show actual UI replies from isolated fresh chats. Only synthetic TEST conversations are present.

| Check | LoRA on | Original base |
|---|---|---|
| Training identity | [Reply](proof/on-0.png) | [Reply](proof/off-0.png) |
| Held-out identity | [Reply](proof/on-1.png) | [Reply](proof/off-1.png) |
| Rainbow | [Reply](proof/on-2.png) | [Reply](proof/off-2.png) |

[Post-crash restart reply](proof/recovered-0.png).
