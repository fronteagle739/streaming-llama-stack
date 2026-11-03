# Qwen3.8 Flash Next, named Balanced mode; see qfn.settings.json for values and calibration status.
& (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\start-qfn-single.ps1') -Mode balanced @args
