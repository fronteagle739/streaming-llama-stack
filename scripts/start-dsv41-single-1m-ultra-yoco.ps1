# Opt-in approximate YOCO: full layers 0..20, final 140 rows at layers 21..39.
# Local KV snapshots include YOCO state automatically. -Vision enables matching native image prefill.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$Vision, [switch]$DryRun, [switch]$Stop)
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 1m -YocoKeep 140 -Config $Config -Model $Model -WebUI:$WebUI -Vision:$Vision -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
