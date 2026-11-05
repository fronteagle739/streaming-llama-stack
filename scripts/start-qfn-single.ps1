# Start Qwen3.8 Flash Next on this RTX with streamed weights (qfn-engine behind `slstack qfn`, an OpenAI-compatible API).
#
#   -Model <pack.basalt>       the Qwen3.8 Flash Next .basalt pack (overrides "model" in the config); keep it on the fast
#                              NVMe volume: experts and the n-gram table are read from it while serving
#   -Config my.local.json      local paths/ports/budgets (default: my.local.json in the clone, if present)
#   -Port 8088                 HTTP port of the API
#   -Context 100480            engine context (KV, index caches and the prefill carry are sized from it)
#   -MaxSpeed                  router bias 0.68 toward experts already in VRAM (Max speed); otherwise Max quality
#   -Preload                   load the engine at start instead of on the first request
#   -Stop                      ask the running stack to shut down gracefully (POST /v1/stack/shutdown)
param(
    [string]$Model = '',
    [string]$Python = '',
    [string]$Config = '',
    [int]$Port = 0,
    [int]$Context = 0,
    [double]$HostBudgetGiB = 3,
    [double]$NativeHostBudgetGiB = 6,
    [ValidateRange(1,8)][int]$EngramScanRings = 1,
    [int]$Tile = 2048,
    [double]$MaxSpeedBias = 0.68,
    [double]$MinAvailableGiB = 6,
    [double]$JobCommitGiB = 40,
    [switch]$MaxSpeed,
    [ValidateSet('max_quality','balanced','max_speed')][string]$Mode = '',
    [string]$Settings = '',
    [switch]$DryRun,
    [switch]$Preload,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
if (-not $Config) {
    $default = Join-Path $Root 'my.local.json'
    if (Test-Path -LiteralPath $default) { $Config = $default }
}
if ($Stop) {
    $p = if ($Port) { $Port } else { 8088 }
    $headers = @{}
    if ($Config) {
        $c = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        if ($c.api_key) { $headers['Authorization'] = 'Bearer ' + $c.api_key }
    }
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$p/v1/stack/shutdown" -Headers $headers -Body '{}' -ContentType 'application/json'
    return
}
$argsList = @('-B', '-m', 'slstack')
if ($Config) { $argsList += @('--config', (Resolve-Path -LiteralPath $Config).Path) }
$argsList += 'qfn'
$argsList += @('--host-budget-gib', "$HostBudgetGiB", '--qfn-host-budget-gib', "$NativeHostBudgetGiB",
    '--qfn-engram-scan-rings', "$EngramScanRings", '--qfn-tile', "$Tile",
    '--min-available-gib', "$MinAvailableGiB", '--job-commit-gib', "$JobCommitGiB", '--qfn-max-speed-bias', "$MaxSpeedBias")
if ($Model) { $argsList += @('--model', $Model) }
if ($Port) { $argsList += @('--port', "$Port") }
if ($Context) { $argsList += @('--context', "$Context") }
if ($MaxSpeed -and $Mode -and $Mode -ne 'max_speed') { throw '-MaxSpeed conflicts with -Mode' }
if ($MaxSpeed) { $Mode = 'max_speed' }
if (-not $Settings -and $Config) {
    $local = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    if ($local.model_settings -is [string]) { $Settings = $local.model_settings }
}
if (-not $Settings) { $Settings = Join-Path $Root 'examples/single-machine/qfn.settings.json' }
if ($Mode) { $argsList += @('--quality-speed', $Mode) }
$argsList += @('--model-settings', (Resolve-Path -LiteralPath $Settings).Path)
if ($Preload) { $argsList += '--preload' }
if ($DryRun) { $argsList += '--dry-run' }
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location $Root
try { & $Python @argsList } finally { Pop-Location }
