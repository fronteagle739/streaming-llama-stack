# Single-RTX execution modes

The default single engine combines streaming-llama ULTRA prefill and ds4 decode kernels in one process. The live conversation appends only new tokens. Start it with `scripts/start-dsv41-single-262k.ps1 -WebUI` or the 1M variant. [Quickstart](../README.md#quickstart)

The native server remains available with `start --serving native`, including native output adapters. The separate two-process coordinator exchanges native checkpoints between ULTRA and ds4 on the same RTX. Its mode-1 state contract and calibration tools are described in [DS4COMBO](DS4COMBO.md). This differs from the default single engine and from [RTX-to-Mac operation](TWO_MACHINE.md).

All paths preserve the original model, validate context and state identities, and use explicit host/device memory admission. A failed partial state import invalidates that session. Retirement completes before another process takes the GPU.
