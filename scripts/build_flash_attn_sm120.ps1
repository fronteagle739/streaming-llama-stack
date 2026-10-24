# FlashAttention-2 (Dao) source build for Blackwell sm_120 on Windows.
# Vendored from the proven 2026-07-05 local build (STATUS SUCCESS, cp313/torch2.11+cu128).
# Pinned: flash-attention 002cce0a1068f8c07dfccb5a1d232b9a3276947c, CUDA 12.8 redist (nvcc 12.8.93,
# cudart/cccl 12.8.90, nvrtc 12.8.93), VS 2022 BuildTools (CUDA 12.8 rejects VS 18).
# The target venv's torch must be a cu128 build. Large compile: MAX_JOBS defaults to 2 (32 GB RAM box).
param(
  [Parameter(Mandatory=$true)][string]$Python,            # target venv python.exe (e.g. the stack media venv, py3.12)
  [Parameter(Mandatory=$true)][string]$Root,              # work dir on a fast, roomy volume (needs ~15 GB)
  [string]$CudaHome = "",                                 # portable CUDA 12.8 toolchain; assembled from redist if empty
  [string]$FaCommit = "002cce0a1068f8c07dfccb5a1d232b9a3276947c",
  [switch]$Install
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Log = Join-Path $Root "build.log"
function Log($m) { $l = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; Add-Content $Log $l; Write-Output $l }
if (-not (Test-Path $Python)) { throw "Target python missing: $Python" }
& $Python -c "import torch,sys;assert torch.version.cuda.startswith('12.8'),torch.version.cuda;print('torch',torch.__version__,'py',sys.version.split()[0])" | ForEach-Object { Log $_ }

# CUDA 12.8 toolchain (matches torch cu128). Assemble from NVIDIA redist zips if not supplied.
if (-not $CudaHome) {
  $CudaHome = Join-Path $Root "cuda_home"
  if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) {
    $base = "https://developer.download.nvidia.com/compute/cuda/redist"
    $parts = @("cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.8.93-archive.zip",
               "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip",
               "cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.8.90-archive.zip",
               "cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.8.93-archive.zip")
    $redist = Join-Path $Root "redist"; New-Item -ItemType Directory -Force -Path $redist,$CudaHome | Out-Null
    foreach ($p in $parts) {
      $zip = Join-Path $redist (Split-Path $p -Leaf)
      if (-not (Test-Path $zip)) { Log "download $p"; Invoke-WebRequest "$base/$p" -OutFile $zip }
      $dir = $zip -replace '\.zip$',''
      if (-not (Test-Path $dir)) { Expand-Archive $zip -DestinationPath $redist }
      Copy-Item (Join-Path $dir "*") $CudaHome -Recurse -Force
    }
  }
}
if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) { throw "CUDA 12.8 toolchain incomplete at $CudaHome" }
$env:CUDA_HOME = $CudaHome; $env:CUDA_PATH = $CudaHome; $env:Path = (Join-Path $CudaHome "bin") + ";" + $env:Path

# VS 2022 only.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "VS 2022 BuildTools with VC tools required (CUDA 12.8 rejects VS 18)" }
foreach ($line in (cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set")) {
  if ($line -match "^([^=]+)=(.*)$") { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] }
}

$Fa = Join-Path $Root "flash-attention"
if (-not (Test-Path $Fa)) { git clone https://github.com/Dao-AILab/flash-attention $Fa }
Push-Location $Fa
git fetch --quiet origin; git checkout --quiet $FaCommit; git submodule update --init --recursive
if ((git rev-parse HEAD) -ne $FaCommit) { throw "flash-attention is not at the pinned commit" }

if (-not $env:MAX_JOBS) { $env:MAX_JOBS = "2" }
$env:FLASH_ATTENTION_FORCE_BUILD = "TRUE"; $env:FLASH_ATTN_CUDA_ARCHS = "120"; $env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:DISTUTILS_USE_SDK = "1"                         # VC env already imported; else torch _check_abi aborts
$env:NVCC_APPEND_FLAGS = "-Xcompiler /permissive-"   # MSVC 14.4x + CUDA 12.8 'std' C2872 in torch headers
& $Python -m pip install -U ninja packaging wheel setuptools | ForEach-Object { Log "pip: $_" }
Log "building wheel (30-120+ min)"
& $Python setup.py bdist_wheel 2>&1 | ForEach-Object { Log "build: $_" }
$wheel = Get-ChildItem (Join-Path $Fa "dist") -Filter "flash_attn-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
Pop-Location
if (-not $wheel) { throw "FA2 build produced no wheel; see $Log" }
$sha = (Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower()
Log "wheel $($wheel.FullName) sha256 $sha"
if ($Install) {
  & $Python -m pip install --force-reinstall --no-deps $wheel.FullName | ForEach-Object { Log "install: $_" }
  $probe = "import torch;from flash_attn import flash_attn_func;q=torch.randn(1,256,8,64,device='cuda',dtype=torch.bfloat16);o=flash_attn_func(q,q,q);torch.cuda.synchronize();print('FA2_OK')"
  $out = & $Python -c $probe 2>&1; $out | ForEach-Object { Log "probe: $_" }
  if (-not ($out -match "FA2_OK")) { throw "FA2 installed but the sm_120 probe failed" }
}
Write-Output $wheel.FullName
