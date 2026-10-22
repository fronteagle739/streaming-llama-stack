$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
& python -B (Join-Path $PSScriptRoot 'scripts/bootstrap.py') @args
exit $LASTEXITCODE
