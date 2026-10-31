# Builds linkbench.exe with MSVC (x64). macOS/Linux: cc -O2 -o linkbench linkbench.c
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$out = if ($args.Count) { $args[0] } else { Join-Path $PSScriptRoot 'linkbench.exe' }
$src = Join-Path $PSScriptRoot 'linkbench.c'
$objdir = Split-Path -Parent $out
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && cl /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS `"$src`" /Fo`"$objdir\linkbench.obj`" /Fe`"$out`" ws2_32.lib"
if ($LASTEXITCODE) { throw "cl failed: $LASTEXITCODE" }
