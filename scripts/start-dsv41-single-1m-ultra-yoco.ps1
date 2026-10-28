# Compatibility launcher: exact cone by default; -ApproximateYoco selects keep 140.
# Local KV snapshots include YOCO state automatically. -Vision enables matching native image prefill.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$Vision, [switch]$DryRun, [switch]$Stop, [switch]$ApproximateYoco)
$keep = if ($ApproximateYoco) { 140 } else { 2541 }
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 1m -YocoKeep $keep -Config $Config -Model $Model -WebUI:$WebUI -Vision:$Vision -DryRun:$DryRun -Stop:$Stop
exit $LASTEXITCODE
