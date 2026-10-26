# DSV4.1 single engine, 262K context: f32 context cache, growable, full L2 expert tier.
# Settings: scripts/dsv41_single.py (preset 262k). Options: -Config, -Model, -WebUI, -DryRun, -Stop.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$Vision, [switch]$DryRun, [switch]$Stop)
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 262k -Config $Config -Model $Model -WebUI:$WebUI -Vision:$Vision -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
