# Build a one-click DSV4.1 launcher .exe for this Windows (RTX) machine.
#
#   powershell -File launcher/pack_windows.ps1 -Mode single|dual [-OutDir <folder>] [-Python <python.exe>] [-Shortcuts]
#
# Reads launcher/config/<Mode>.json plus your private launcher/config/<Mode>.local.json,
# bakes the RTX section into the .exe (with the launcher runtime and the Python path) and
# writes "<OutDir>/DSV41 Single.exe" or "DSV41 Dual.exe", overwriting it in place.
# -Shortcuts adds Start-menu and Desktop shortcuts with the launcher icon.
# Uses only the C# compiler that ships with Windows (.NET Framework csc.exe).
param(
    [Parameter(Mandatory = $true)][ValidateSet('single', 'dual')][string]$Mode,
    [string]$OutDir = '',
    [string]$Python = '',
    [string]$StackRoot = '',
    [switch]$Shortcuts
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $here 'dist' }
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $csc)) { throw "C# compiler not found at $csc" }
$name = if ($Mode -eq 'single') { 'DSV41 Single' } else { 'DSV41 Dual' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$stage = Join-Path $here "dist\stage-$Mode"   # overwritten on every pack
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$bakeArgs = @('-B', (Join-Path $here 'app\bake.py'), '--mode', $Mode, '--side', 'rtx', '--python', $Python,
              '--out', (Join-Path $stage 'launcher.json'))
if ($StackRoot) { $bakeArgs += @('--stack-root', $StackRoot) }
& $Python @bakeArgs
if ($LASTEXITCODE -ne 0) { throw "Config bake failed ($LASTEXITCODE)" }
Copy-Item -LiteralPath (Join-Path $here 'app\dsv41_launcher.py') -Destination (Join-Path $stage 'dsv41_launcher.py') -Force
[System.IO.File]::WriteAllText((Join-Path $stage 'python.txt'), $Python, (New-Object System.Text.UTF8Encoding $false))

$exe = Join-Path $OutDir "$name.exe"
$icon = Join-Path $here "icons\built\$Mode.ico"
$cscArgs = @('/nologo', '/target:exe', '/optimize+', "/out:$exe", "/win32icon:$icon",
             "/resource:$(Join-Path $stage 'dsv41_launcher.py'),dsv41_launcher.py",
             "/resource:$(Join-Path $stage 'launcher.json'),launcher.json",
             "/resource:$(Join-Path $stage 'python.txt'),python.txt",
             (Join-Path $here 'app\LauncherStub.cs'))
& $csc @cscArgs
if ($LASTEXITCODE -ne 0) { throw "csc failed ($LASTEXITCODE)" }
"Built $exe"

if ($Shortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    $places = @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))
    foreach ($place in $places) {
        $link = $shell.CreateShortcut((Join-Path $place "$name.lnk"))
        $link.TargetPath = $exe
        $link.WorkingDirectory = $OutDir
        $link.IconLocation = "$exe,0"
        $link.Description = "Start $name and open its web UI"
        $link.Save()
        "Shortcut: $(Join-Path $place "$name.lnk")"
    }
    "Taskbar: Windows does not allow programs to pin themselves. Right-click the '$name' shortcut on the Desktop (or in Start) and choose 'Pin to taskbar' (on Windows 11: Show more options > Pin to taskbar)."
}
