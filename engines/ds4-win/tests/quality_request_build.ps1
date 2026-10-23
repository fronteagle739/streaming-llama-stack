param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $Root).Path
$repo=Split-Path -Parent $PSScriptRoot
$env:TEMP=$root
$env:TMP=$root
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'MSVC tools unavailable' }
$names=@('ds4','ds4_help','ds4_ssd','ds4_win32','ds4_distributed_win32','ds4_gpu_args',
 'ds4_tp_win32','ds4_layer_pack','ds4_image','ds4_prompt_prefix','ds4_engram','ds4_peer','ds4_kvstore','rax')
$sources=($names | ForEach-Object { '"'+(Join-Path $repo "$_.c")+'"' }) -join ' '
$objects=$names | Where-Object { $_ -ne 'ds4' } | ForEach-Object { Join-Path $root "$_.obj" }
$objectArgs=($objects | ForEach-Object { '"'+$_+'"' }) -join ' '
$command=@"
@echo off
set "PATH=$([IO.Path]::GetDirectoryName($vswhere));%PATH%"
call "$vs\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /O2 /std:c11 /DDS4_NO_GPU /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_DEPRECATE /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I"$repo" /c /Fo"$root\\" $sources
if errorlevel 1 exit /b %errorlevel%
cl /nologo /O2 /std:c11 /DDS4_NO_GPU /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_DEPRECATE /I"$repo" /c /Fo"$root\quality-runtime.obj" "$repo\tests\quality_request_runtime.c"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /O2 /std:c11 /DDS4_NO_GPU /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_DEPRECATE /I"$repo" /Fo"$root\quality-request.obj" /Fe"$root\quality-request.exe" "$repo\tests\test_quality_request.c" "$root\quality-runtime.obj" $objectArgs /link ws2_32.lib
if errorlevel 1 exit /b %errorlevel%
"$root\quality-request.exe"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /O2 /std:c11 /DDS4_NO_GPU /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_DEPRECATE /I"$repo" /Fo"$root\quality-http.obj" /Fe"$root\quality-http.exe" "$repo\tests\test_quality_speed_http.c" "$root\ds4.obj" $objectArgs /link ws2_32.lib
if errorlevel 1 exit /b %errorlevel%
"$root\quality-http.exe"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /O2 /std:c11 /I"$repo" /Fo"$root\quality-phase.obj" /Fe"$root\quality-phase.exe" "$repo\tests\test_quality_phase.c"
if errorlevel 1 exit /b %errorlevel%
"$root\quality-phase.exe"
exit /b %errorlevel%
"@
$path=Join-Path $root 'quality-request-build.cmd'
[IO.File]::WriteAllText($path,$command)
& cmd.exe /d /c $path
if($LASTEXITCODE) { throw "Quality regression failed: $LASTEXITCODE" }
