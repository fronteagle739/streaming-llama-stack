# Qwen3.8 Flash Next, single RTX, Max quality (the full model, router bias 0). Copy qfn.example.json to my.local.json
# and set "model" and "gpu_uuid" first.
& (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\start-qfn-single.ps1') -Mode max_quality @args
