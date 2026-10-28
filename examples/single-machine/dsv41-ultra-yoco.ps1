# DSV4.1: exact cone by default; -ApproximateYoco opts into YOCO. API only.
param([Parameter(Mandatory=$true)][string]$Config,
      [ValidateSet('262k','1m')][string]$Context='262k', [switch]$DryRun, [switch]$ApproximateYoco)
$ErrorActionPreference='Stop'
$entry=Join-Path (Split-Path -Parent $PSScriptRoot) 'run_preset.py'
$profile = if ($ApproximateYoco) { 'ultra-yoco' } else { 'ultra-cone' }
$launchArgs=@('-B',$entry,'--single',$profile,'--context',$Context,'--config',(Resolve-Path -LiteralPath $Config).Path)

if ($DryRun) { $launchArgs+='--dry-run' }
& python @launchArgs
exit $LASTEXITCODE
