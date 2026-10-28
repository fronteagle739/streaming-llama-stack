# Compatibility launcher: exact cone by default; -ApproximateYoco selects keep 140.
# No RTX/Mac session export. Shipping memory budgets and -WebUI support are preserved.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$DryRun, [switch]$Stop, [switch]$ApproximateYoco)
$keep = if ($ApproximateYoco) { 140 } else { 2541 }
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 262k -YocoKeep $keep -Config $Config -Model $Model -WebUI:$WebUI -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
