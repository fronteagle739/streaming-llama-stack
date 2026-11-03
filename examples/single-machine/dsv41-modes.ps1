# Named decode modes are independent of the qualified prefill arithmetic profile.
param([Parameter(Mandatory=$true)][string]$Config,
      [ValidateSet('max_quality','balanced','max_speed')][string]$Mode='max_quality',
      [ValidateSet('262k','1m')][string]$Context='262k',
      [string]$Settings='',
      [switch]$Webui, [switch]$DryRun)
$ErrorActionPreference='Stop'
if (-not $Settings) { $Settings=Join-Path $PSScriptRoot 'dsv41.settings.json' }
$entry=Join-Path (Split-Path -Parent $PSScriptRoot) 'run_preset.py'
$launchArgs=@('-B',$entry,'--single','ultra-cone','--context',$Context,'--config',(Resolve-Path -LiteralPath $Config).Path,
              '--model-settings',(Resolve-Path -LiteralPath $Settings).Path,'--model-mode',$Mode)
if ($Webui) { $launchArgs+='--webui' }
if ($DryRun) { $launchArgs+='--dry-run' }
& python @launchArgs
exit $LASTEXITCODE
