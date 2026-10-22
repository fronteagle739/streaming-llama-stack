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
    $llamaPath = Join-Path $PSScriptRoot 'engines/streaming-llama/prefill_llama.cpp'
    # Same pinned CMake build as the engine's build.bat, narrowed to the used
    # targets with two compile jobs. Enable its existing offline ULTRA producer.
    & python -B -m ds4stack --config $Config guarded -- cmake -S $llamaPath -B build/llama -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_DSV41_OFFLINE_PRODUCER=ON "-DLLAMA_DSV41_JSON_INCLUDE_DIR=$llamaPath/vendor" "-DLLAMA_DSV41_DSTORAGE_INCLUDE_DIR=$llamaPath/external/directstorage/native/include" "-DLLAMA_DSV41_DSTORAGE_LIBRARY=$llamaPath/external/directstorage/native/lib/x64/dstorage.lib"
    if ($LASTEXITCODE) { throw 'llama configure failed' }
    & python -B -m ds4stack --config $Config guarded -- cmake --build build/llama --config Release --parallel 2 --target llama-server llama-dsv41-prefill llama-export-lora
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
