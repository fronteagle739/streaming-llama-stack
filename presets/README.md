# Qualified native presets

`dsv41-affinity-spreads.txt` is the existing native 1017 calibration:40 scalar routing-spread medians,64 unbiased decode samples per layer. It contains no weights or prompt text. File SHA-256: `21ea6a0fbd6c9469a2318f8d1ebe8b5428ac96580f1ef4486b3e782ec80a0967`. Its source lineage is native `c65136f30f8c9a5536d929165dcbb2abca2e7a6f`; the same format is consumed by the current four-phase provider.

Its header records315,838,021,632 mapped main-weight bytes, excluding disk-only Engram, plus384 experts,40 layers and native header fingerprint `9bebbe7c246be2dd`. The full canonical GGUF is518,596,067,328 bytes. These sizes describe different spans of the same model. The launcher selects this default only when the full-file size and first64KiB FNV-1a fingerprint match; the native loader then independently verifies its mapped span, header and all40 rows. Renaming the same GGUF does not change this match. Other models can supply `--ds4-cache-affinity-spread-file`; their settings are not silently replaced.

Native warm-up100 and ramp8 remain unchanged. Max quality is bias0. The five names/range and optional reasoning/tool_call/answer/code arguments come from the delivered provider. This package reuses an existing calibration; no calibration or accepted parity experiment was repeated to add it.
