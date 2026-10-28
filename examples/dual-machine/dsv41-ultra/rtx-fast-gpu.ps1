# Run the RTX half; private paths and peer address come from -Config.
param([Parameter(Mandatory=$true)][string]$Config, [switch]$DryRun, [switch]$ApproximateYoco)
$ErrorActionPreference='Stop'
$stackRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$launchArgs=@('-B',(Join-Path $stackRoot 'examples/run_preset.py'),'--preset','dsv41-ultra','--machine','rtx','--config',(Resolve-Path -LiteralPath $Config).Path)
if ($DryRun) { $launchArgs+='--dry-run' }
if ($ApproximateYoco) { $launchArgs+='--approximate-yoco' }
& python @launchArgs
exit $LASTEXITCODE
