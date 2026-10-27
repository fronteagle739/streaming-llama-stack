# DSV4.1: accurate prefill, WebUI and API.
param([Parameter(Mandatory=$true)][string]$Config,
      [ValidateSet('262k','1m')][string]$Context='262k', [switch]$DryRun)
$ErrorActionPreference='Stop'
$entry=Join-Path (Split-Path -Parent $PSScriptRoot) 'run_preset.py'
$launchArgs=@('-B',$entry,'--single','accurate','--context',$Context,'--config',(Resolve-Path -LiteralPath $Config).Path)
$launchArgs+='--webui'
if ($DryRun) { $launchArgs+='--dry-run' }
& python @launchArgs
exit $LASTEXITCODE
