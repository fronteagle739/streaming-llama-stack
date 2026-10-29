param([Parameter(Mandatory=$true)][string]$AotDirectory,
      [Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$CutlassRuntimeLibrary)
$ErrorActionPreference='Stop'
$cudaRoot=$env:CUDA_PATH
if(!$cudaRoot){throw 'CUDA_PATH must name CUDA13.2'}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$obj=Join-Path $OutputDirectory 'dense-fp8-bridge.obj'
$dll=Join-Path $OutputDirectory 'slstack-dense-fp8.dll'
& "$cudaRoot/bin/nvcc.exe" -std=c++17 -O3 -Xcompiler /EHsc,/Zc:preprocessor,/MD -gencode arch=compute_120a,code=sm_120a "-I$AotDirectory" -c "$PSScriptRoot/dense_fp8_bridge.cu" -o $obj
if($LASTEXITCODE){throw 'Dense FP8 bridge compile failed'}
$objects=@('1280_5120','32768_1280','512_5120','128_512','5120_8192','2304_5120','5120_2304') | ForEach-Object {Join-Path $AotDirectory ('fi_mxfp8_'+$_+'.o')}
& link.exe /nologo /DLL "/OUT:$dll" $obj @objects $CutlassRuntimeLibrary "/LIBPATH:$cudaRoot/lib/x64" cudart.lib cuda.lib
if($LASTEXITCODE){throw 'Dense FP8 bridge link failed'}
