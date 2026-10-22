param([string]$Config = 'config.example.json', [ValidateSet('ds4','llama','webui','all')][string]$Target='all')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP = 'runtime/ds4stack'
$env:TMP = $env:TEMP
$env:npm_config_cache = 'runtime/npm-cache'
if ($Target -in @('ds4','all')) {
    $ds4Path = Join-Path $PSScriptRoot 'engines/ds4-win'
    & python -B -m ds4stack --config $Config guarded --cwd $ds4Path -- powershell -NoProfile -ExecutionPolicy Bypass -File windows_build.ps1 -Target cuda -BuildTemp runtime/ds4stack
    if ($LASTEXITCODE) { throw 'ds4 build failed' }
}
if ($Target -in @('llama','all')) {
    & python -B -m ds4stack --config $Config guarded -- cmake -S engines/streaming-llama/prefill_llama.cpp -B build/llama -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    if ($LASTEXITCODE) { throw 'llama configure failed' }
    & python -B -m ds4stack --config $Config guarded -- cmake --build build/llama --config Release --parallel 2 --target llama-server
    if ($LASTEXITCODE) { throw 'llama build failed' }
}
if ($Target -in @('webui','all')) {
    # Use the pinned UI's own build pipeline. Never invoke its destructive reset/cleanup scripts.
    $uiPath = Join-Path $PSScriptRoot 'engines/streaming-llama/prefill_llama.cpp/tools/server/webui'
    & python -B -m ds4stack --config $Config guarded --cwd $uiPath -- npm.cmd ci --ignore-scripts
    if ($LASTEXITCODE) { throw 'webui dependency installation failed' }
    & python -B -m ds4stack --config $Config guarded --cwd $uiPath -- npx.cmd svelte-kit sync
    if ($LASTEXITCODE) { throw 'webui sync failed' }
    & python -B -m ds4stack --config $Config guarded --cwd $uiPath -- npx.cmd vite build
    if ($LASTEXITCODE) { throw 'webui build failed' }
}
