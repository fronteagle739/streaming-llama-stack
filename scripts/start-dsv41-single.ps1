# Start the DSV4.1 single engine (streaming-llama ULTRA prefill + ds4-decode.dll decode) through
# `slstack start`, detached, with one of the gated configurations in scripts/dsv41_single.py.
# Use scripts/start-dsv41-single-262k.ps1 or scripts/start-dsv41-single-1m.ps1; they call this file.
#
#   -Config my.local.json   your local paths/ports/budgets (default: my.local.json in the clone, if present)
#   -Model <gguf>           DSV4.1 model path (overrides "model" in the config)
#   -WebUI                  also serve the stack's llama.cpp web UI (stack tools: quality/speed, media)
#   -DryRun                 print the start config and the resolved settings; start nothing
#   -Stop                   ask the running stack to shut down gracefully (POST /v1/stack/shutdown)
param(
    [Parameter(Mandatory = $true)][ValidateSet('262k', '1m')][string]$Preset,
    [string]$Config = '',
    [string]$Model = '',
    [switch]$WebUI,
    [switch]$DryRun,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction Stop).Source
if (-not $Config) {
    $default = Join-Path $Root 'my.local.json'
    if (Test-Path -LiteralPath $default) { $Config = $default }
}
if ($Config) { $Config = (Resolve-Path -LiteralPath $Config).Path }

$StartConfig = Join-Path $Root "runtime\dsv41-single-$Preset.local.json"
$genArgs = @('-B', (Join-Path $Root 'scripts\dsv41_single.py'), '--preset', $Preset, '--write', $StartConfig)
if ($Config) { $genArgs += @('--config', $Config) }
if ($Model) { $genArgs += @('--model', $Model) }
if ($DryRun) { $genArgs += '--dry-run' }
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Python @genArgs
if ($LASTEXITCODE -ne 0) { throw "dsv41_single.py failed ($LASTEXITCODE)" }

$cfg = Get-Content -LiteralPath $StartConfig -Raw | ConvertFrom-Json
$example = Get-Content -LiteralPath (Join-Path $Root 'config.example.json') -Raw | ConvertFrom-Json
$bindHost = $example.host; if ($cfg.host) { $bindHost = $cfg.host }
$port = $example.port; if ($cfg.port) { $port = $cfg.port }
$Url = "http://${bindHost}:$port"
$runtime = $example.runtime_root; if ($cfg.runtime_root) { $runtime = $cfg.runtime_root }
if (-not [System.IO.Path]::IsPathRooted($runtime)) { $runtime = Join-Path $Root $runtime }

if ($Stop) {
    # Graceful only: the router closes the single engine, whose producer exits through its own
    # shutdown (drains DirectStorage, frees VRAM and the GPU hold). Nothing is killed.
    Invoke-RestMethod -Method Post -Uri "$Url/v1/stack/shutdown" -ContentType 'application/json' -Body '{}'
    "Shutdown requested at $Url; the start process exits once the engine has stopped."
    exit 0
}

$startArgs = "start"
if ($WebUI) { $startArgs += " --webui" }
$command = "`"$Python`" -B `"$(Join-Path $Root 'scripts\bootstrap.py')`" --config `"$StartConfig`" $startArgs"
if ($DryRun) {
    "Start config: $StartConfig"
    "Would run (detached, from $Root): $command"
    "Router / web UI: $Url"
    exit 0
}

$logs = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ("dsv41-single-$Preset-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
# Detached (WMI Win32_Process.Create): the stack keeps running when this window closes.
$line = "cmd /c set PYTHONDONTWRITEBYTECODE=1&& $command > `"$log`" 2>&1"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $Root }
if ($r.ReturnValue -ne 0) { throw "Process creation failed ($($r.ReturnValue))" }
"Started DSV4.1 single engine ($Preset) pid $($r.ProcessId); log: $log"

# Wait while the launcher is alive and the router is not yet answering (no fixed deadline).
while ($true) {
    try {
        $health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2
        break
    } catch {
        if (-not (Get-Process -Id $r.ProcessId -ErrorAction SilentlyContinue)) {
            Get-Content -LiteralPath $log -Tail 40 -ErrorAction SilentlyContinue
            throw "The stack exited before its router answered; see $log"
        }
        Start-Sleep -Milliseconds 500
    }
}
"Router ready: $Url  ($($health | ConvertTo-Json -Compress))"
if ($WebUI) { "Web UI: open $Url in a browser. The producer loads with the first prompt." }
"Health:   Invoke-RestMethod $Url/health"
"Shutdown: powershell -File `"$PSCommandPath`" -Preset $Preset -Stop   (or POST $Url/v1/stack/shutdown with body {})"
