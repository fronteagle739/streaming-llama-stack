# DSV4.1: exact ULTRA dependency-cone prefill, API only.
# The 1m preset uses the qualified VRAM carry lifetime and N=1 weight passes.
param([Parameter(Mandatory=$true)][string]$Config,
      [ValidateSet('262k','1m')][string]$Context='262k', [switch]$DryRun)
$ErrorActionPreference='Stop'
$entry=Join-Path (Split-Path -Parent $PSScriptRoot) 'run_preset.py'
$launchArgs=@('-B',$entry,'--single','ultra-cone','--context',$Context,'--config',(Resolve-Path -LiteralPath $Config).Path)
if ($DryRun) { $launchArgs+='--dry-run' }
& python @launchArgs
exit $LASTEXITCODE
