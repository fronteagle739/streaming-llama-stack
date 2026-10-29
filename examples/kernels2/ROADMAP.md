# Remaining E19 work

- Make TileLang attention exact; preserve the negative post-EOS fixture as a regression gate and broaden real-prompt KL/recall coverage.
- Obtain licensed reference b12x fork sources, then compare b12x, FlashInfer and HYPER on identical captured100K routing/weights. Current component data only compares official FP4 with ABI4 FlashInfer.
- Finish native FP4 source/download/conversion and VRAM lifetime restructuring through NATIVEFP4 before adopting larger integrated expert batches or one100K sweep.
- Reproduce E4 bit-exact full-prefill output with unpadded2541-row cone shapes; current optional shape experiment is lossy.
- Replace per-tile lifetime/allocation metadata with stable per-sweep capture, prove bank/routing ownership, then add CUDA graph replay/PDL without weakening the allocator guards.
- Move host staging and next-layer preparation off the critical path after measuring exposed work; current HC metadata reduction does not implement this overlap.
- Benchmark licensed dense FP8, remaining HC/norm, copy/concat, QAT and TOP_K candidates in the ranked order in PERFORMANCE.md. Estimates are upper bounds/planning assumptions, not accepted speedups.
