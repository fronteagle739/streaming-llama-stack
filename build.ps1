param([string]$Config = 'config.example.json', [ValidateSet('ds4','llama','webui','all')][string]$Target='all')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeConfig = & python -B -c "import sys; from slstack.config import load; print(load(sys.argv[1])['runtime_root'])" $Config
$env:TEMP = Join-Path $runtimeConfig 'build-temp'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:TMP = $env:TEMP
$env:npm_config_cache = Join-Path $runtimeConfig 'npm-cache'
if ($Target -in @('ds4','all')) {
    $ds4Path = Join-Path $PSScriptRoot 'engines/ds4-win'
    & python -B -m slstack --config $Config guarded --cwd $ds4Path -- powershell -NoProfile -ExecutionPolicy Bypass -File windows_build.ps1 -Target cuda -BuildTemp $env:TEMP
    if ($LASTEXITCODE) { throw 'ds4 build failed' }
}
if ($Target -in @('llama','all')) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ Build Tools first.' }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vsEnvCmd = Join-Path $vsPath 'VC/Auxiliary/Build/vcvars64.bat'
    $env:PATH = (Split-Path -Parent $vswhere) + ';' + $env:PATH
    # Reuse the native engine's vcvars64 setup. Capture values without logging the environment.
    $compilerEnvironment = & cmd.exe /d /s /c "call `"$vsEnvCmd`" >nul && set"
    if ($LASTEXITCODE) { throw 'Visual Studio compiler environment failed' }
    foreach ($line in $compilerEnvironment) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2 -and $parts[0] -match '^[A-Za-z_][A-Za-z0-9_() -]*$') {
            [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
        }
    }
    if (!(Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        $ninjaPath = Join-Path $vsPath 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja'
        if (!(Test-Path -LiteralPath (Join-Path $ninjaPath 'ninja.exe'))) { throw 'Install Ninja or the Visual Studio CMake tools.' }
        $env:PATH = $ninjaPath + ';' + $env:PATH
    }
    $llamaPath = Join-Path $PSScriptRoot 'engines/streaming-llama/prefill_llama.cpp'
    # Same pinned CMake build as the engine's build.bat, narrowed to the used
    # targets with two compile jobs. Enable its existing offline ULTRA producer.
    & python -B -m slstack --config $Config guarded -- cmake -G Ninja -S $llamaPath -B build/llama -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_DSV41_OFFLINE_PRODUCER=ON "-DLLAMA_DSV41_JSON_INCLUDE_DIR=$llamaPath/vendor" "-DLLAMA_DSV41_DSTORAGE_INCLUDE_DIR=$llamaPath/external/directstorage/native/include" "-DLLAMA_DSV41_DSTORAGE_LIBRARY=$llamaPath/external/directstorage/native/lib/x64/dstorage.lib"
    if ($LASTEXITCODE) { throw 'llama configure failed' }
    & python -B -m slstack --config $Config guarded -- cmake --build build/llama --config Release --parallel 2 --target llama-server llama-dsv41-prefill llama-export-lora slstack-weight-fetcher
    if ($LASTEXITCODE) { throw 'llama build failed' }
}
if ($Target -in @('webui','all')) {
    # Use the pinned UI's own build pipeline. Never invoke its destructive reset/cleanup scripts.
    $uiPath = Join-Path $PSScriptRoot 'engines/streaming-llama/prefill_llama.cpp/tools/server/webui'
    & python -B -m slstack --config $Config guarded --cwd $uiPath -- npm.cmd ci --ignore-scripts
    if ($LASTEXITCODE) { throw 'webui dependency installation failed' }
    & python -B -m slstack --config $Config guarded --cwd $uiPath -- npx.cmd svelte-kit sync
    if ($LASTEXITCODE) { throw 'webui sync failed' }
    & python -B -m slstack --config $Config guarded --cwd $uiPath -- npx.cmd vite build
    if ($LASTEXITCODE) { throw 'webui build failed' }
}
