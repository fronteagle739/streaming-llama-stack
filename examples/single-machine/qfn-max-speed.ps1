# Qwen3.8 Flash Next, single RTX, Max speed (router bias 0.68 toward experts already in VRAM). Copy qfn.example.json to
# my.local.json and set "model" and "gpu_uuid" first.
& (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\start-qfn-single.ps1') -MaxSpeed @args
