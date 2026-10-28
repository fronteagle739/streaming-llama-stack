param([Parameter(Mandatory=$true)][string]$AotDirectory,
      [Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$CutlassRuntimeLibrary)
$ErrorActionPreference='Stop'
# Run from a Visual Studio x64 developer shell, under a host-memory guard.
$cudaRoot=$env:CUDA_PATH
if(!$cudaRoot){throw 'CUDA_PATH must name CUDA 13.2'}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$obj=Join-Path $OutputDirectory 'flashinfer-bridge.obj'
$dll=Join-Path $OutputDirectory 'slstack-flashinfer.dll'
& "$cudaRoot/bin/nvcc.exe" -std=c++17 -O3 -Xcompiler /EHsc,/Zc:preprocessor,/MD -gencode arch=compute_120a,code=sm_120a "-I$AotDirectory" -c "$PSScriptRoot/flashinfer_bridge.cu" -o $obj
if($LASTEXITCODE){throw 'FlashInfer bridge compile failed'}
& link.exe /nologo /DLL "/OUT:$dll" $obj "$AotDirectory/fi_mxfp4_2304_5120.o" "$AotDirectory/fi_mxfp4_5120_2304.o" $CutlassRuntimeLibrary "/LIBPATH:$cudaRoot/lib/x64" cudart.lib cuda.lib
if($LASTEXITCODE){throw 'FlashInfer bridge link failed'}
