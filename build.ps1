param([string]$Config = 'config.example.json', [ValidateSet('ds4','llama','webui','all')][string]$Target='all')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeConfig = & python -B -c "import sys; from slstack.config import load; print(load(sys.argv[1])['runtime_root'])" $Config
$env:TEMP = Join-Path $runtimeConfig 'build-temp'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:TMP = $env:TEMP
$env:npm_config_cache = Join-Path $runtimeConfig 'npm-cache'
# Use the isolated core runtime (it has the launcher's packages), not whatever python is on PATH.
$py = Join-Path $runtimeConfig 'core-venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $py)) { $py = 'python' }
if ($Target -in @('webui','all')) {
    # Use the pinned UI's own build pipeline. Never invoke its destructive reset/cleanup scripts.
    $uiPath = Join-Path $PSScriptRoot 'engines/streaming-llama/prefill_llama.cpp/tools/server/webui'
    & $py -B -m slstack --config $Config guarded --cwd $uiPath -- npm.cmd ci --ignore-scripts
    if ($LASTEXITCODE) { throw 'webui dependency installation failed' }
    & $py -B -m slstack --config $Config guarded --cwd $uiPath -- npx.cmd svelte-kit sync
    if ($LASTEXITCODE) { throw 'webui sync failed' }
    & $py -B -m slstack --config $Config guarded --cwd $uiPath -- npx.cmd vite build
    if ($LASTEXITCODE) { throw 'webui build failed' }
}

if ($Target -in @('ds4','all')) {
    $ds4Path = Join-Path $PSScriptRoot 'engines/ds4-win'
    & $py -B -m slstack --config $Config guarded --cwd $ds4Path -- powershell -NoProfile -ExecutionPolicy Bypass -File windows_build.ps1 -Target cuda -BuildTemp $env:TEMP
    if ($LASTEXITCODE) { throw 'ds4 build failed' }
    # ds4-decode.dll: ds4-win's decode kernels for the single-engine ULTRA producer (loaded beside it).
    & $py -B -m slstack --config $Config guarded --cwd $ds4Path -- powershell -NoProfile -ExecutionPolicy Bypass -File windows_build.ps1 -Target cuda -ExternalDecode -BuildTemp $env:TEMP
    if ($LASTEXITCODE) { throw 'ds4-decode build failed' }
    $producerBin = Join-Path $PSScriptRoot 'build/llama/bin'
    New-Item -ItemType Directory -Force -Path $producerBin | Out-Null
    Copy-Item -LiteralPath (Join-Path $ds4Path 'ds4-decode.dll') -Destination $producerBin -Force
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
    # Pinned CUTLASS 3.9.2 for the split-F32 attention used by the Fast tier
    # (--split-f32-tensor-core-qk). Exact tag, commit-hash checked; no fallback.
    $cutlassCommit = 'ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e'
    $cutlassPath = Join-Path $PSScriptRoot 'build/cutlass-v3.9.2'
    if (!(Test-Path -LiteralPath (Join-Path $cutlassPath 'include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h'))) {
        & git clone --quiet --depth 1 --branch v3.9.2 https://github.com/NVIDIA/cutlass $cutlassPath
        if ($LASTEXITCODE) { throw 'Pinned CUTLASS 3.9.2 fetch failed' }
    }
    $cutlassHead = (& git -C $cutlassPath rev-parse HEAD).Trim()
    if ($cutlassHead -ne $cutlassCommit) { throw "CUTLASS checkout is $cutlassHead, expected pinned $cutlassCommit" }
    # Same pinned CMake build as the engine's build.bat, narrowed to the used
    # targets with two compile jobs. Enable its existing offline ULTRA producer.
    & $py -B -m slstack --config $Config guarded -- cmake -G Ninja -S $llamaPath -B build/llama -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_DSV41_OFFLINE_PRODUCER=ON -DLLAMA_DSV41_NATIVE_HALF_PROFILES=ON "-DLLAMA_DSV41_JSON_INCLUDE_DIR=$llamaPath/vendor" "-DLLAMA_DSV41_DSTORAGE_INCLUDE_DIR=$llamaPath/external/directstorage/native/include" "-DLLAMA_DSV41_DSTORAGE_LIBRARY=$llamaPath/external/directstorage/native/lib/x64/dstorage.lib" -DGGML_CUDA_DSV41_SPLIT_F32=ON "-DGGML_CUDA_CUTLASS_PATH=$cutlassPath"
    if ($LASTEXITCODE) { throw 'llama configure failed' }
    & $py -B -m slstack --config $Config guarded -- cmake --build build/llama --config Release --parallel 2 --target llama-server llama-dsv41-prefill llama-export-lora slstack-weight-fetcher
    if ($LASTEXITCODE) { throw 'llama build failed' }
    $profileCheck = Join-Path $PSScriptRoot 'build/llama/bin/llama-dsv41-prefill.exe'
    & $py -B -m slstack --config $Config guarded -- $profileCheck --check-native-profiles
    if ($LASTEXITCODE) { throw 'producer profile preflight failed' }
}
