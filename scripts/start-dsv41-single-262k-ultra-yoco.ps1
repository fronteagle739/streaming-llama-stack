# Opt-in approximate YOCO: full layers 0..20, final 140 rows at layers 21..39.
# No RTX/Mac session export. Shipping memory budgets and -WebUI support are preserved.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$DryRun, [switch]$Stop)
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 262k -YocoKeep 140 -Config $Config -Model $Model -WebUI:$WebUI -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
