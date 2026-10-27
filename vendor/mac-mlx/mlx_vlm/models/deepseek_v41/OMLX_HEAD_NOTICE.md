`omlx_head.py` derives from [oMLX head.py at b390b31](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v41/head.py).
oMLX is distributed under Apache License 2.0; this source file specifically declares MIT and its directory carries Copyright (c) 2023 DeepSeek.
Both license texts accompany this port as `OMLX_APACHE_LICENSE.txt` and `OMLX_DEEPSEEK_MIT_LICENSE.txt`. The source SPDX identifier is preserved.

Local modifications: provenance documentation, distinct Metal JIT symbol, default-off integration in `LanguageModel` under `DSV41_OMLX_HEAD=1`.
The port retains BF16 weights, FP32 inputs/accumulators/output, the upstream one-to-five-row eligibility limit, and the full FP32 GEMM fallback.
Promoted FP32 head weights, CPU execution and longer prefill/verify blocks fall back; biased, quantized or custom head modules retain their original call.
This is a numerical-performance experiment: the parallel reduction differs from ordinary GEMM and is not a repair for DSpark sampled equivalence.
