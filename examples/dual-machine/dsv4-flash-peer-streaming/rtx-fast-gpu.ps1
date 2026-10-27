# Run the RTX half; private paths and peer address come from -Config.
param([Parameter(Mandatory=$true)][string]$Config, [switch]$DryRun)
$ErrorActionPreference='Stop'
$stackRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$launchArgs=@('-B',(Join-Path $stackRoot 'examples/run_preset.py'),'--preset','dsv4-flash-peer-streaming','--machine','rtx','--config',(Resolve-Path -LiteralPath $Config).Path)
if ($DryRun) { $launchArgs+='--dry-run' }
& python @launchArgs
exit $LASTEXITCODE
