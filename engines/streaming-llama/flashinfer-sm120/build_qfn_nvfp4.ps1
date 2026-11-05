param([Parameter(Mandatory=$true)][string]$AotDirectory,
      [Parameter(Mandatory=$true)][string]$CutlassRuntimeLibrary,
      [Alias('Workspace')][Parameter(Mandatory=$true)][string]$TaskRoot,
      [string]$BinDirectory='')
$ErrorActionPreference='Stop'
# Run in the existing serial guarded VS x64/CUDA13.2 build environment.
# Stable DLL name; all intermediates stay in the explicit workspace.
$taskRoot=[IO.Path]::GetFullPath($TaskRoot)
$stack=Split-Path (Split-Path (Split-Path $PSScriptRoot))
$output=Join-Path $taskRoot 'build/native-nvfp4'
$bin=if($BinDirectory){[IO.Path]::GetFullPath($BinDirectory)}else{Join-Path $stack 'bin'}
$cudaRoot=$env:CUDA_PATH
if(!$cudaRoot){throw 'CUDA_PATH must name CUDA13.2'}
if(!$bin.StartsWith($taskRoot.TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Stable DLL output must stay under the explicit workspace'
}
if(!(Test-Path -LiteralPath $AotDirectory -PathType Container) -or !(Test-Path -LiteralPath $CutlassRuntimeLibrary -PathType Leaf)){
    throw 'Existing read-only AOT inputs and runtime library are required'
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
New-Item -ItemType Directory -Path $bin -Force | Out-Null
$env:TEMP=Join-Path $taskRoot 'tmp'
$env:TMP=$env:TEMP
$env:CUDA_CACHE_PATH=$env:TEMP
$obj=Join-Path $output 'qfn_native_nvfp4_bridge.obj'
& "$cudaRoot/bin/nvcc.exe" -std=c++17 -O3 -Xcompiler /EHsc,/Zc:preprocessor,/MD -gencode arch=compute_120a,code=sm_120a "-I$AotDirectory" -c "$PSScriptRoot/qfn_native_nvfp4_bridge.cu" -o $obj
if($LASTEXITCODE){throw 'QFN native NVFP4 bridge compile failed'}
& link.exe /nologo /DLL "/OUT:$bin/qfn-native-nvfp4.dll" "/IMPLIB:$output/qfn-native-nvfp4.lib" $obj "$AotDirectory/fi_nvfp4_640_2560.o" "$AotDirectory/fi_nvfp4_2560_640.o" $CutlassRuntimeLibrary "/LIBPATH:$cudaRoot/lib/x64" cudart.lib cuda.lib
if($LASTEXITCODE){throw 'QFN native NVFP4 bridge link failed'}
