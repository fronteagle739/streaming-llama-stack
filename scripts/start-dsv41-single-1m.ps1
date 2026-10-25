# DSV4.1 single engine, 1M context (999,424 tokens): bf16 context cache, growable.
# Settings: scripts/dsv41_single.py (preset 1m). Options: -Config, -Model, -WebUI, -DryRun, -Stop.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$DryRun, [switch]$Stop)
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 1m -Config $Config -Model $Model -WebUI:$WebUI -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
